from math import ceil
from pathlib import Path
import random

import torch
from lightning import LightningModule
from torchmetrics.detection import MeanAveragePrecision

from yolo.config.config import Config
from yolo.model.yolo import create_model
from yolo.tools.data_loader import create_dataloader
from yolo.tools.distillation import KnowledgeDistillationLoss, build_teacher_model
from yolo.tools.drawer import draw_bboxes
from yolo.tools.loss_functions import create_loss_function
from yolo.utils.bounding_box_utils import create_converter, to_metrics_format
from yolo.utils.logger import logger
from yolo.utils.model_utils import PostProcess, create_optimizer, create_scheduler


class BaseModel(LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        partial_weights = getattr(cfg, "partial_weights", None)
        partial_load = bool(getattr(cfg, "partial_load", False))
        weight_path = partial_weights if partial_weights else cfg.weight
        if isinstance(weight_path, (str, Path)) and Path(weight_path).is_file():
            ckpt = torch.load(Path(weight_path), map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "pruned_model" in ckpt:
                self.model = ckpt["pruned_model"]
                logger.info(f":scissors: Loaded standalone pruned model from {weight_path}")
                return
        self.model = create_model(
            cfg.model,
            class_num=cfg.dataset.class_num,
            weight_path=weight_path,
            partial_load=partial_load or bool(partial_weights),
        )

    def forward(self, x):
        return self.model(x)


class ValidateModel(BaseModel):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        if self.cfg.task.task == "validation":
            self.validation_cfg = self.cfg.task
        else:
            self.validation_cfg = self.cfg.task.validation
        self.metric = MeanAveragePrecision(iou_type="bbox", box_format="xyxy", backend="faster_coco_eval")
        self.metric.warn_on_many_detections = False
        self.val_loader = create_dataloader(self.validation_cfg.data, self.cfg.dataset, self.validation_cfg.task)
        self.ema = self.model

    def setup(self, stage):
        self.vec2box = create_converter(
            self.cfg.model.name, self.model, self.cfg.model.anchor, self.cfg.image_size, self.device
        )
        self.post_process = PostProcess(self.vec2box, self.validation_cfg.nms)

    def val_dataloader(self):
        return self.val_loader

    def validation_step(self, batch, batch_idx):
        batch_size, images, targets, rev_tensor, img_paths = batch
        H, W = images.shape[2:]
        predicts = self.post_process(self.ema(images), image_size=[W, H])
        mAP = self.metric(
            [to_metrics_format(predict) for predict in predicts], [to_metrics_format(target) for target in targets]
        )
        return predicts, mAP

    def on_validation_epoch_end(self):
        epoch_metrics = self.metric.compute()
        del epoch_metrics["classes"]
        self.log_dict(epoch_metrics, prog_bar=True, sync_dist=True, rank_zero_only=True)
        self.log_dict(
            {"PyCOCO/AP @ .5:.95": epoch_metrics["map"], "PyCOCO/AP @ .5": epoch_metrics["map_50"]},
            sync_dist=True,
            rank_zero_only=True,
        )
        self.metric.reset()


class TrainModel(ValidateModel):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.train_loader = create_dataloader(self.cfg.task.data, self.cfg.dataset, self.cfg.task.task)
        self._mosaic_closed = False

        distill_cfg = getattr(self.cfg.task, "distill", None)
        if distill_cfg and getattr(distill_cfg, "enabled", False):
            teacher = build_teacher_model(
                teacher_model_name=getattr(distill_cfg, "teacher_model", "v9-m"),
                teacher_weights=getattr(distill_cfg, "teacher_weights", None),
                class_num=self.cfg.dataset.class_num,
            )
            self.kd_loss = KnowledgeDistillationLoss(
                student=self.model,
                teacher=teacher,
                lambda_kd=getattr(distill_cfg, "lambda_kd", 1.0),
                feature_weight=getattr(distill_cfg, "feature_weight", 0.5),
                logit_weight=getattr(distill_cfg, "logit_weight", 0.5),
                temperature=getattr(distill_cfg, "temperature", 1.0),
                feature_tags=getattr(distill_cfg, "feature_tags", ["P3", "P4", "P5"]),
            )
        else:
            self.kd_loss = None

    def setup(self, stage):
        super().setup(stage)
        self.loss_fn = create_loss_function(self.cfg, self.vec2box)

    def train_dataloader(self):
        return self.train_loader

    def on_train_epoch_start(self):
        self.trainer.optimizers[0].next_epoch(
            ceil(len(self.train_loader) / self.trainer.world_size), self.current_epoch
        )

        close_mosaic = int(getattr(self.cfg.task, "close_mosaic", 0) or 0)
        max_epochs = int(getattr(self.cfg.task, "epoch", 0) or 0)
        if (
            not self._mosaic_closed
            and close_mosaic > 0
            and max_epochs > 0
            and self.current_epoch >= max(0, max_epochs - close_mosaic)
        ):
            dataset = getattr(self.train_loader, "dataset", None)
            composer = getattr(dataset, "transform", None)
            if composer is not None and hasattr(composer, "transforms"):
                for t in composer.transforms:
                    if t.__class__.__name__ in {"Mosaic", "MixUp"} and hasattr(t, "prob"):
                        t.prob = 0.0
                self._mosaic_closed = True
                logger.info(f":broom: Closed Mosaic/MixUp augmentations at epoch {self.current_epoch}")

        active_size = list(self.cfg.image_size)
        if getattr(self.cfg.task, "multi_scale", False):
            scale_range = getattr(self.cfg.task, "multi_scale_range", None) or [0.8, 1.2]
            lo, hi = float(scale_range[0]), float(scale_range[1])
            factor = random.uniform(lo, hi)
            active_size = [max(128, int(round((s * factor) / 32.0) * 32)) for s in self.cfg.image_size]
            dataset = getattr(self.train_loader, "dataset", None)
            if dataset is not None and hasattr(dataset, "transform"):
                dataset.image_size = active_size
                dataset.transform.base_size = sum(active_size) // 2
                if hasattr(dataset.transform, "pad_resize"):
                    dataset.transform.pad_resize.set_size(active_size)

        self.vec2box.update(active_size)

    def training_step(self, batch, batch_idx):
        lr_dict = self.trainer.optimizers[0].next_batch()
        batch_size, images, targets, *_ = batch
        self.vec2box.update([images.shape[3], images.shape[2]])
        if self.kd_loss is not None:
            self.kd_loss.begin_capture()
        predicts = self(images)
        aux_predicts = self.vec2box(predicts["AUX"]) if "AUX" in predicts else None
        main_predicts = self.vec2box(predicts["Main"])
        loss, loss_item = self.loss_fn(aux_predicts, main_predicts, targets)
        if self.kd_loss is not None:
            kd_loss, kd_metrics = self.kd_loss(images, predicts)
            loss = loss + kd_loss
            loss_item.update(kd_metrics)
        self.log_dict(
            loss_item,
            prog_bar=True,
            on_epoch=True,
            batch_size=batch_size,
            rank_zero_only=True,
        )
        self.log_dict(lr_dict, prog_bar=False, logger=True, on_epoch=False, rank_zero_only=True)
        return loss * batch_size

    def on_save_checkpoint(self, checkpoint):
        if "state_dict" in checkpoint:
            checkpoint["state_dict"] = {
                k: v for k, v in checkpoint["state_dict"].items() if not k.startswith("kd_loss.")
            }

    def configure_optimizers(self):
        extra_params = list(self.kd_loss.adapters.parameters()) if self.kd_loss is not None else None
        optimizer = create_optimizer(self.model, self.cfg.task.optimizer, extra_params=extra_params)
        scheduler = create_scheduler(optimizer, self.cfg.task.scheduler)
        return [optimizer], [scheduler]


class InferenceModel(BaseModel):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        # TODO: Add FastModel
        self.predict_loader = create_dataloader(cfg.task.data, cfg.dataset, cfg.task.task)

    def setup(self, stage):
        self.vec2box = create_converter(
            self.cfg.model.name, self.model, self.cfg.model.anchor, self.cfg.image_size, self.device
        )
        self.post_process = PostProcess(self.vec2box, self.cfg.task.nms)

    def predict_dataloader(self):
        return self.predict_loader

    def predict_step(self, batch, batch_idx):
        images, rev_tensor, origin_frame = batch
        predicts = self.post_process(self(images), rev_tensor=rev_tensor)
        img = draw_bboxes(origin_frame, predicts, idx2label=self.cfg.dataset.class_list)
        if getattr(self.predict_loader, "is_stream", None):
            fps = self._display_stream(img)
        else:
            fps = None
        if getattr(self.cfg.task, "save_predict", None):
            self._save_image(img, batch_idx)
        return img, fps

    def _save_image(self, img, batch_idx):
        save_image_path = Path(self.trainer.default_root_dir) / f"frame{batch_idx:03d}.png"
        img.save(save_image_path)
        print(f"💾 Saved visualize image at {save_image_path}")
