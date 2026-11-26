"""Training-time Knowledge Distillation (Feature CWD/MSE + Detection Logit KD) for YOLOv9."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import Tensor, nn

from yolo.model.yolo import YOLO, create_model
from yolo.utils.logger import logger


class FrozenTeacherWrapper(nn.Module):
    """Wraps a YOLO teacher model in permanent eval/no-grad mode."""

    def __init__(self, teacher: YOLO):
        super().__init__()
        self.teacher = teacher
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad_(False)

    def train(self, mode: bool = True):
        # Keep BatchNorm and any stateful layers strictly in eval mode during student training.
        return super().train(False)

    @torch.no_grad()
    def forward(self, x: Tensor):
        self.teacher.eval()
        return self.teacher(x)


def _find_tagged_layers(model: YOLO, tags: Sequence[str]) -> Dict[str, Tuple[nn.Module, int]]:
    found: Dict[str, Tuple[nn.Module, int]] = {}
    tag_set = set(tags)
    for layer in model.model:
        layer_tag = getattr(layer, "tags", None)
        if layer_tag in tag_set:
            out_c = getattr(layer, "out_c", None)
            if isinstance(out_c, int):
                found[layer_tag] = (layer, out_c)
    return found


def build_teacher_model(
    teacher_model_name: str,
    teacher_weights: Optional[str],
    class_num: int,
) -> YOLO:
    """Load a teacher YOLO model by YAML config name (or path) without re-initializing Hydra."""
    candidate_path = Path(teacher_model_name)
    if not candidate_path.is_file():
        config_dir = Path(__file__).resolve().parent.parent / "config" / "model"
        candidate_path = config_dir / f"{teacher_model_name}.yaml"
    if not candidate_path.is_file():
        raise FileNotFoundError(f"Teacher model config not found: {teacher_model_name} ({candidate_path})")

    teacher_cfg = OmegaConf.load(candidate_path)
    weight_arg = teacher_weights if teacher_weights else None
    return create_model(teacher_cfg, weight_path=weight_arg, class_num=class_num, partial_load=False)


class KnowledgeDistillationLoss(nn.Module):
    """Computes multi-scale neck feature KD + detection-head logit KD without modifying the student graph."""

    def __init__(
        self,
        student: YOLO,
        teacher: YOLO,
        *,
        lambda_kd: float = 1.0,
        feature_weight: float = 0.5,
        logit_weight: float = 0.5,
        temperature: float = 1.0,
        feature_tags: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.lambda_kd = float(lambda_kd)
        self.feature_weight = float(feature_weight)
        self.logit_weight = float(logit_weight)
        self.temperature = max(float(temperature), 1e-4)

        tags = list(feature_tags or ("P3", "P4", "P5"))
        student_layers = _find_tagged_layers(student, tags)
        teacher_layers = _find_tagged_layers(teacher, tags)
        self.matched_tags = [t for t in tags if t in student_layers and t in teacher_layers]

        self.teacher_wrapper = FrozenTeacherWrapper(teacher)
        self.adapters = nn.ModuleDict()
        for tag in self.matched_tags:
            s_c = student_layers[tag][1]
            t_c = teacher_layers[tag][1]
            if s_c == t_c:
                self.adapters[tag] = nn.Identity()
            else:
                self.adapters[tag] = nn.Conv2d(s_c, t_c, kernel_size=1, bias=False)

        self._capturing = False
        self._student_feats: Dict[str, Tensor] = {}
        self._teacher_feats: Dict[str, Tensor] = {}
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []

        for tag in self.matched_tags:
            s_mod = student_layers[tag][0]
            t_mod = teacher_layers[tag][0]
            self._hooks.append(s_mod.register_forward_hook(self._make_hook(tag, is_teacher=False)))
            self._hooks.append(t_mod.register_forward_hook(self._make_hook(tag, is_teacher=True)))

        logger.info(
            f":mortar_board: Enabled Knowledge Distillation on tags={self.matched_tags} "
            f"(lambda={self.lambda_kd}, feat_w={self.feature_weight}, logit_w={self.logit_weight})"
        )

    def _make_hook(self, tag: str, is_teacher: bool):
        def _hook(_module, _inputs, output):
            if not self._capturing:
                return
            if is_teacher:
                self._teacher_feats[tag] = output.detach()
            else:
                self._student_feats[tag] = output

        return _hook

    def begin_capture(self) -> None:
        self._student_feats.clear()
        self._teacher_feats.clear()
        self._capturing = True

    def end_capture(self) -> None:
        self._capturing = False
        self._student_feats.clear()
        self._teacher_feats.clear()

    def close(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self.end_capture()

    def _feature_kd_loss(self) -> Tensor:
        if not self.matched_tags or self.feature_weight <= 0:
            device = next(self.teacher_wrapper.parameters()).device
            return torch.zeros((), device=device)

        losses: List[Tensor] = []
        T = self.temperature
        for tag in self.matched_tags:
            s_feat = self._student_feats.get(tag)
            t_feat = self._teacher_feats.get(tag)
            if s_feat is None or t_feat is None:
                continue

            s_proj = self.adapters[tag](s_feat)
            if s_proj.shape[-2:] != t_feat.shape[-2:]:
                s_proj = F.interpolate(s_proj, size=t_feat.shape[-2:], mode="bilinear", align_corners=False)

            s_flat = s_proj.float().flatten(2)
            t_flat = t_feat.float().flatten(2)

            # Channel-normalized MSE + Channel-Wise Distillation (CWD) KL divergence
            s_norm = F.normalize(s_flat, p=2, dim=1)
            t_norm = F.normalize(t_flat, p=2, dim=1)
            mse_loss = F.mse_loss(s_norm, t_norm)

            s_log_prob = F.log_softmax(s_flat / T, dim=-1)
            t_prob = F.softmax(t_flat / T, dim=-1)
            cwd_loss = F.kl_div(s_log_prob, t_prob, reduction="batchmean") * (T * T) / max(t_feat.shape[1], 1)

            losses.append(mse_loss + cwd_loss)

        if not losses:
            device = next(self.teacher_wrapper.parameters()).device
            return torch.zeros((), device=device)
        return torch.stack(losses).mean()

    def _logit_kd_loss(self, student_main: List[Tuple[Tensor, ...]], teacher_main: List[Tuple[Tensor, ...]]) -> Tensor:
        device = student_main[0][0].device
        if self.logit_weight <= 0:
            return torch.zeros((), device=device)

        teacher_by_hw = {head[0].shape[-2:]: head for head in teacher_main}
        T = self.temperature
        head_losses: List[Tensor] = []

        for s_head in student_main:
            s_cls, s_anc, _ = s_head
            hw = s_cls.shape[-2:]
            t_head = teacher_by_hw.get(hw)
            if t_head is None:
                continue
            t_cls, t_anc, _ = t_head

            loss_terms: List[Tensor] = []
            if s_cls.shape[1] == t_cls.shape[1]:
                t_cls_prob = torch.sigmoid(t_cls.float() / T)
                cls_kd = F.binary_cross_entropy_with_logits(s_cls.float() / T, t_cls_prob) * (T * T)
                loss_terms.append(cls_kd)

            if s_anc.shape[1] == t_anc.shape[1]:
                # anchor_x shape: [B, reg_max, 4, H, W]
                s_anc_flat = s_anc.float().permute(0, 2, 3, 4, 1).reshape(-1, s_anc.shape[1])
                t_anc_flat = t_anc.float().permute(0, 2, 3, 4, 1).reshape(-1, t_anc.shape[1])
                anc_kd = (
                    F.kl_div(
                        F.log_softmax(s_anc_flat / T, dim=-1),
                        F.softmax(t_anc_flat / T, dim=-1),
                        reduction="batchmean",
                    )
                    * (T * T)
                )
                loss_terms.append(anc_kd)

            if loss_terms:
                head_losses.append(sum(loss_terms))

        if not head_losses:
            return torch.zeros((), device=device)
        return torch.stack(head_losses).mean()

    def forward(self, images: Tensor, student_outputs: Dict[str, List[Tuple[Tensor, ...]]]) -> Tuple[Tensor, Dict[str, float]]:
        try:
            self._capturing = True
            teacher_outputs = self.teacher_wrapper(images)
            feat_loss = self._feature_kd_loss()
            logit_loss = self._logit_kd_loss(student_outputs["Main"], teacher_outputs["Main"])
            total_kd = self.lambda_kd * (self.feature_weight * feat_loss + self.logit_weight * logit_loss)
            metrics = {
                "Loss/KDFeatLoss": float(feat_loss.detach().item()),
                "Loss/KDLogitLoss": float(logit_loss.detach().item()),
                "Loss/KDTotalLoss": float(total_kd.detach().item()),
            }
            return total_kd, metrics
        finally:
            self.end_capture()
