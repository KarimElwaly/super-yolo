"""End-to-end smoke test suite covering all YOLOv9 CPU-upgrade deliverables (Phases 1-6)."""

from __future__ import annotations

import gc
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import supervision as sv
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from lightning import Trainer
from omegaconf import OmegaConf
from PIL import Image
from torch import nn

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from inference.demo_stream_count import SimpleByteTrack, run_stream_pipeline
from inference.detector_ov import OpenVINODetector
from scripts.benchmark_compare import evaluate_go_no_go, run_benchmark_suite
from scripts.check_licenses import _evaluate_package
from scripts.export_openvino import export_pipeline
from yolo.model.yolo import create_model, load_partial_weights, strip_auxiliary_branches
from yolo.tools.data_loader import YoloDataset
from yolo.tools.solver import BaseModel, TrainModel
from yolo.utils.bounding_box_utils import Vec2Box
from yolo.utils.logging_utils import setup


def _hydra_compose(overrides):
    GlobalHydra.instance().clear()
    with initialize(config_path="../yolo/config", version_base=None):
        return compose(config_name="config", overrides=overrides)


def _count_activations(model: nn.Module):
    counts = {"SiLU": 0, "ReLU": 0, "Hardswish": 0}
    for m in model.modules():
        if isinstance(m, nn.SiLU):
            counts["SiLU"] += 1
        elif isinstance(m, nn.ReLU):
            counts["ReLU"] += 1
        elif isinstance(m, nn.Hardswish):
            counts["Hardswish"] += 1
    return counts


def _make_coco_split_dataset(base_dir: Path, num_images: int = 3, img_size: int = 256):
    for split in ("train", "val"):
        img_dir = base_dir / "images" / split
        ann_dir = base_dir / "annotations"
        img_dir.mkdir(parents=True, exist_ok=True)
        ann_dir.mkdir(parents=True, exist_ok=True)
        images = []
        annotations = []
        for idx in range(1, num_images + 1):
            fname = f"{idx:012d}.jpg"
            arr = np.full((img_size, img_size, 3), 80 + idx * 20, dtype=np.uint8)
            cv2.rectangle(arr, (30, 30), (110, 120), (0, 200, 50), -1)
            cv2.imwrite(str(img_dir / fname), arr)
            images.append({"id": idx, "file_name": fname, "width": img_size, "height": img_size})
            annotations.append(
                {
                    "id": idx,
                    "image_id": idx,
                    "category_id": (idx - 1) % 2,
                    "bbox": [30.0, 30.0, 80.0, 90.0],
                    "area": 7200.0,
                    "iscrowd": 0,
                }
            )
        coco = {
            "categories": [{"id": 0, "name": "obj0"}, {"id": 1, "name": "obj1"}],
            "images": images,
            "annotations": annotations,
        }
        (ann_dir / f"instances_{split}.json").write_text(json.dumps(coco), encoding="utf-8")


# 1. License audit
def test_01_check_licenses_clean_and_policy_enforced():
    proc = subprocess.run(
        [sys.executable, str(project_root / "scripts" / "check_licenses.py")],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"License check failed:\n{proc.stdout}\n{proc.stderr}"
    assert "0 GPL/AGPL, 0 banned" in proc.stdout

    # Verify policy unit checks reject banned packages and strong GPL
    fake_ultralytics = SimpleNamespace(
        name="ultralytics",
        version="8.0.0",
        metadata={"Name": "ultralytics", "License": "AGPL-3.0"},
        files=[],
    )
    res_ban = _evaluate_package(fake_ultralytics, strict_unknown=True)
    assert res_ban.status == "banned"

    fake_gpl = SimpleNamespace(
        name="some-gpl-pkg",
        version="1.0.0",
        metadata={"Name": "some-gpl-pkg", "License": "GPL-3.0-only", "License-Expression": "GPL-3.0-only"},
        files=[],
    )
    res_gpl = _evaluate_package(fake_gpl, strict_unknown=True)
    assert res_gpl.status == "copyleft"


# 2. Ultralytics YOLO -> COCO conversion + --verify
def test_02_convert_yolo_to_coco_and_verify(tmp_path: Path):
    yolo_root = tmp_path / "yolo_ds"
    coco_root = tmp_path / "coco_ds"
    for split in ("train", "val"):
        (yolo_root / "images" / split).mkdir(parents=True)
        (yolo_root / "labels" / split).mkdir(parents=True)
        Image.new("RGB", (120, 100), (40, 80, 120)).save(yolo_root / f"images/{split}/sample_0.jpg")
        (yolo_root / f"labels/{split}/sample_0.txt").write_text("0 0.5 0.5 0.4 0.6\n1 0.25 0.25 0.2 0.2\n")

    (yolo_root / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\nnames: [pothole, crack]\n")
    hydra_yaml = coco_root / "dataset.yaml"

    proc = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "convert_yolo_to_coco.py"),
            "--data-yaml",
            str(yolo_root / "data.yaml"),
            "--output-dir",
            str(coco_root),
            "--hydra-yaml",
            str(hydra_yaml),
            "--verify",
            "--verify-samples",
            "1",
            "--verify-dir",
            str(tmp_path / "verify_vis"),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert (coco_root / "annotations/instances_train.json").is_file()
    assert (coco_root / "annotations/instances_val.json").is_file()
    assert (tmp_path / "verify_vis/train/verify_sample_0.jpg").is_file()

    ds_cfg = OmegaConf.load(hydra_yaml)
    data_cfg = OmegaConf.create({"image_size": [128, 128], "batch_size": 1, "data_augment": {}, "dynamic_shape": False})
    yds = YoloDataset(data_cfg, ds_cfg, phase="train")
    assert len(yds) == 1


# 3. COCO class filter -> contiguous IDs + Hydra YAML
def test_03_filter_coco_classes(tmp_path: Path):
    src = tmp_path / "src_coco"
    dst = tmp_path / "filtered_coco"
    (src / "images/val").mkdir(parents=True)
    (src / "annotations").mkdir(parents=True)
    for i in range(3):
        Image.new("RGB", (100, 100), (i * 50, 20, 30)).save(src / f"images/val/img_{i}.jpg")

    coco = {
        "categories": [
            {"id": 1, "name": "person"},
            {"id": 3, "name": "car"},
            {"id": 6, "name": "bus"},
            {"id": 17, "name": "cat"},
        ],
        "images": [
            {"id": 10, "file_name": "img_0.jpg", "width": 100, "height": 100},
            {"id": 11, "file_name": "img_1.jpg", "width": 100, "height": 100},
            {"id": 12, "file_name": "img_2.jpg", "width": 100, "height": 100},
        ],
        "annotations": [
            {"id": 1, "image_id": 10, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0},
            {"id": 2, "image_id": 10, "category_id": 17, "bbox": [30, 30, 20, 20], "area": 400, "iscrowd": 0},
            {"id": 3, "image_id": 11, "category_id": 6, "bbox": [15, 15, 30, 30], "area": 900, "iscrowd": 0},
            {"id": 4, "image_id": 12, "category_id": 17, "bbox": [20, 20, 25, 25], "area": 625, "iscrowd": 0},
        ],
    }
    (src / "annotations/instances_val.json").write_text(json.dumps(coco))
    out_yaml = dst / "subset.yaml"

    proc = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "filter_coco_classes.py"),
            "--source-dir",
            str(src),
            "--splits",
            "val",
            "--classes",
            "person,bus",
            "--output-dir",
            str(dst),
            "--hydra-yaml",
            str(out_yaml),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    filtered = json.loads((dst / "annotations/instances_val.json").read_text())
    assert [c["name"] for c in filtered["categories"]] == ["person", "bus"]
    assert [c["id"] for c in filtered["categories"]] == [0, 1]
    assert len(filtered["images"]) == 2
    assert len(filtered["annotations"]) == 2

    cfg = OmegaConf.load(out_yaml)
    assert cfg.class_num == 2 and list(cfg.class_list) == ["person", "bus"]


# 4. Vanilla v9-t, v9-s, v9-m regression check
def test_04_vanilla_models_load_and_infer():
    x = torch.randn(1, 3, 256, 256)
    for name in ("v9-t", "v9-s", "v9-m"):
        cfg = _hydra_compose([f"model={name}", "weight=False"])
        model = create_model(cfg.model, weight_path=None, class_num=80).eval()
        acts = _count_activations(model)
        assert acts["SiLU"] > 0 and acts["ReLU"] == 0 and acts["Hardswish"] == 0
        with torch.no_grad():
            out = model(x)
        assert "Main" in out and "AUX" in out and len(out["Main"]) == 3
        del model
    gc.collect()


# 5. Upgraded -opt / -p2 architectures + partial weight loading + PGI stripping
def test_05_upgraded_models_partial_load_and_strip(tmp_path: Path):
    x = torch.randn(1, 3, 256, 256)
    specs = [
        ("v9-t-opt", "ReLU", 3, [8, 16, 32]),
        ("v9-s-opt", "ReLU", 3, [8, 16, 32]),
        ("v9-m-opt", "Hardswish", 3, [8, 16, 32]),
        ("v9-t-opt-p2", "ReLU", 4, [4, 8, 16, 32]),
        ("v9-s-opt-p2", "ReLU", 4, [4, 8, 16, 32]),
    ]
    for name, exp_act, exp_heads, exp_strides in specs:
        cfg = _hydra_compose([f"model={name}", "weight=False"])
        m = create_model(cfg.model, weight_path=None, class_num=5).eval()
        acts = _count_activations(m)
        assert acts["SiLU"] == 0 and acts[exp_act] > 0
        with torch.no_grad():
            out_pre = m(x)
        assert len(out_pre["Main"]) == exp_heads
        v2b = Vec2Box(m, cfg.model.anchor, [256, 256], torch.device("cpu"))
        assert v2b.strides == exp_strides

        strip_info = strip_auxiliary_branches(m)
        assert strip_info["removed_layers"] > 0
        with torch.no_grad():
            out_post = m(x)
        assert list(out_post.keys()) == ["Main"]
        for a, b in zip(out_pre["Main"], out_post["Main"]):
            for ta, tb in zip(a, b):
                assert torch.allclose(ta, tb, atol=1e-6)
        del m

    # Partial weight loading from vanilla v9-t -> v9-t-opt and v9-t-opt-p2
    cfg_v9t = _hydra_compose(["model=v9-t", "weight=False"])
    m_v9t = create_model(cfg_v9t.model, weight_path=None, class_num=80)
    ckpt_path = tmp_path / "v9t_vanilla.pt"
    torch.save(m_v9t.model.state_dict(), ckpt_path)
    del m_v9t

    cfg_opt = _hydra_compose(
        ["model=v9-t-opt", "dataset.class_num=3", f"partial_weights={ckpt_path.as_posix()}"]
    )
    bm = BaseModel(cfg_opt)
    assert bm.model.num_classes == 3
    del bm

    cfg_p2 = _hydra_compose(["model=v9-t-opt-p2", "weight=False"])
    m_p2 = create_model(cfg_p2.model, weight_path=None, class_num=3)
    stats_p2 = load_partial_weights(m_p2, ckpt_path)
    assert stats_p2["loaded"] > 0 and stats_p2["reinitialized"] > 0
    del m_p2
    gc.collect()


# 6. Knowledge Distillation + close_mosaic + multi_scale (2-step dummy batch)
def test_06_knowledge_distillation_2step(tmp_path: Path):
    ds_dir = tmp_path / "kd_ds"
    runs_dir = tmp_path / "runs"
    _make_coco_split_dataset(ds_dir, num_images=2, img_size=256)

    cfg = _hydra_compose(
        [
            "task=train",
            "model=v9-t-opt",
            "dataset=custom_template",
            f"dataset.path={ds_dir.as_posix()}",
            "dataset.class_num=2",
            "dataset.class_list=[obj0,obj1]",
            "weight=False",
            "use_wandb=False",
            "+quiet=True",
            "accelerator=cpu",
            "device=cpu",
            "task.epoch=2",
            "task.close_mosaic=1",
            "task.multi_scale=True",
            "task.data.cpu_num=0",
            "task.validation.data.cpu_num=0",
            "task.data.batch_size=1",
            "task.validation.data.batch_size=1",
            "image_size=[256,256]",
            "task.distill.enabled=True",
            "task.distill.teacher_model=v9-m",
            "task.distill.teacher_weights=null",
            "task.distill.lambda_kd=1.0",
            "task.distill.feature_weight=0.5",
            "task.distill.logit_weight=0.5",
            f"out_path={runs_dir.as_posix()}",
        ]
    )
    callbacks, _, save_path = setup(cfg)
    model = TrainModel(cfg)
    assert model.kd_loss is not None
    assert all(not p.requires_grad for p in model.kd_loss.teacher_wrapper.parameters())
    teacher_before = next(model.kd_loss.teacher_wrapper.parameters()).detach().clone()

    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=2,
        limit_train_batches=1,
        limit_val_batches=1,
        callbacks=callbacks,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        default_root_dir=save_path,
    )
    trainer.fit(model)
    assert model._mosaic_closed is True
    assert torch.equal(teacher_before, next(model.kd_loss.teacher_wrapper.parameters()))
    assert not any("teacher" in n or "adapters" in n for n, _ in model.model.named_modules())

    ckpt_dict = {"state_dict": model.state_dict()}
    model.on_save_checkpoint(ckpt_dict)
    assert not any(k.startswith("kd_loss.") for k in ckpt_dict["state_dict"])
    del model, trainer
    gc.collect()


# 7. Structured channel pruning via torch-pruning
def test_07_structured_channel_pruning(tmp_path: Path):
    cfg = _hydra_compose(["model=v9-t-opt", "weight=False"])
    m = create_model(cfg.model, weight_path=None, class_num=3)
    in_ckpt = tmp_path / "v9t_opt_init.pt"
    out_ckpt = tmp_path / "v9t_opt_pruned.pt"
    torch.save(m.model.state_dict(), in_ckpt)
    orig_bytes = in_ckpt.read_bytes()
    del m

    proc = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "prune_model.py"),
            "--model",
            "v9-t-opt",
            "--weights",
            str(in_ckpt),
            "--output",
            str(out_ckpt),
            "--class-num",
            "3",
            "--pruning-ratio",
            "0.20",
            "--image-size",
            "256",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert in_ckpt.read_bytes() == orig_bytes
    assert out_ckpt.is_file()

    payload = torch.load(out_ckpt, map_location="cpu", weights_only=False)
    assert payload["stats"]["param_reduction_pct"] > 10.0
    assert payload["stats"]["macs_reduction_pct"] > 10.0

    cfg_reload = _hydra_compose(["model=v9-t-opt", "dataset.class_num=3", f"weight={out_ckpt.as_posix()}"])
    bm = BaseModel(cfg_reload).eval()
    with torch.no_grad():
        out = bm(torch.randn(1, 3, 256, 256))
    assert "Main" in out and len(out["Main"]) == 3
    del bm, payload
    gc.collect()


# 8, 9, 10. OpenVINO export (FP16 + INT8 + PPP), OpenVINODetector, stream demo, and benchmark_compare
def test_08_09_10_openvino_export_detector_stream_and_benchmark(tmp_path: Path):
    ds_dir = tmp_path / "eval_ds"
    _make_coco_split_dataset(ds_dir, num_images=4, img_size=256)

    # 8a: Export vanilla v9-t -> ONNX (0 aux, parity) -> OpenVINO FP16
    res_fp16 = export_pipeline(
        model_arg="v9-t",
        class_num_arg=2,
        class_names_arg="obj0,obj1",
        image_size=(256, 256),
        output_dir=tmp_path / "ov_fp16",
        name="v9t_fp16",
    )
    assert Path(res_fp16["xml_path"]).is_file()
    assert res_fp16["meta"]["precision"] == "FP16"
    assert res_fp16["meta"]["stripped_aux"]["removed_layers"] > 0
    assert res_fp16["meta"]["onnx_info"]["onnx_convs"] <= res_fp16["meta"]["onnx_info"]["expected_max_convs"]

    # 8b: Export upgraded v9-t-opt -> ONNX -> OpenVINO INT8 (--protect-heads) + PrePostProcessor
    res_int8 = export_pipeline(
        model_arg="v9-t-opt",
        class_num_arg=2,
        class_names_arg="obj0,obj1",
        image_size=(256, 256),
        output_dir=tmp_path / "ov_int8_ppp",
        name="v9t_opt_int8_ppp",
        int8=True,
        calib_dir=ds_dir / "images/val",
        calib_samples=4,
        protect_heads=True,
        embed_preprocess=True,
    )
    assert Path(res_int8["xml_path"]).is_file()
    assert res_int8["meta"]["precision"] == "INT8"
    assert res_int8["meta"]["embedded_preprocess"] is True
    assert res_int8["meta"]["protect_heads"] is True
    assert res_int8["meta"]["protected_ops_count"] > 0
    assert (tmp_path / "ov_int8_ppp/export_meta.json").is_file()

    # 9: OpenVINODetector on both FP16 (manual letterbox) and INT8+PPP (raw uint8 BGR)
    sample_frame = cv2.imread(str(ds_dir / "images/val/000000000001.jpg"))
    for xml in (res_fp16["xml_path"], res_int8["xml_path"]):
        det = OpenVINODetector(xml, num_threads=2, conf_threshold=0.25, iou_threshold=0.45)
        sv_out = det(sample_frame)
        assert isinstance(sv_out, sv.Detections)
        assert sv_out.xyxy.ndim == 2 and sv_out.xyxy.shape[1] == 4

        _, geom = det.prepare_input(sample_frame)
        fake_boxes = np.array([[[20, 20, 120, 120], [22, 22, 122, 122], [140, 140, 220, 220]]], dtype=np.float32)
        fake_scores = np.zeros((1, 3, 2), dtype=np.float32)
        fake_scores[0, 0, 0] = 0.91
        fake_scores[0, 1, 0] = 0.65
        fake_scores[0, 2, 1] = 0.84
        nms_dets = det.postprocess(fake_boxes, fake_scores, geom)
        assert len(nms_dets) == 2 and set(nms_dets.class_id.tolist()) == {0, 1}

    # Test SimpleByteTrack + demo_stream_count on synthetic video
    trk = SimpleByteTrack(high_thresh=0.4)
    d1 = sv.Detections(xyxy=np.array([[20, 20, 80, 80]], dtype=np.float32), confidence=np.array([0.9], dtype=np.float32), class_id=np.array([0]))
    d2 = sv.Detections(xyxy=np.array([[24, 24, 84, 84]], dtype=np.float32), confidence=np.array([0.85], dtype=np.float32), class_id=np.array([0]))
    t1 = trk.update_with_detections(d1)
    t2 = trk.update_with_detections(d2)
    assert t1.tracker_id[0] == t2.tracker_id[0]

    vid_path = tmp_path / "stream.mp4"
    vw = cv2.VideoWriter(str(vid_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (256, 256))
    for _ in range(3):
        vw.write(sample_frame)
    vw.release()
    stream_stats = run_stream_pipeline(
        model_xml=Path(res_int8["xml_path"]),
        source=str(vid_path),
        num_threads=2,
        detect_every_n=2,
        max_frames=3,
        output_video=tmp_path / "stream_out.mp4",
        display=False,
        drop_stale=False,
    )
    assert stream_stats["frames_processed"] == 3
    assert (tmp_path / "stream_out.mp4").is_file()

    # 10: Automated A/B benchmark suite + Markdown/JSON reports + Go/No-Go gate
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        f"baseline: v9t-fp16\nmodels:\n"
        f"  - name: v9t-fp16\n    model_xml: {res_fp16['xml_path']}\n"
        f"  - name: v9t-opt-int8-ppp\n    model_xml: {res_int8['xml_path']}\n",
        encoding="utf-8",
    )
    out_md = tmp_path / "bench.md"
    out_json = tmp_path / "bench.json"
    report = run_benchmark_suite(
        manifest_path=manifest,
        val_images=ds_dir / "images/val",
        val_ann=ds_dir / "annotations/instances_val.json",
        max_eval_images=4,
        threads=2,
        warmup=3,
        iters=10,
        output_md=out_md,
        output_json=out_json,
        isolated=False,
    )
    assert out_md.is_file() and out_json.is_file()
    parsed = json.loads(out_json.read_text(encoding="utf-8"))
    assert len(parsed["results"]) == 2 and len(parsed["comparisons"]) == 1
    assert parsed["comparisons"][0]["verdict"] in ("PASS (Adopt Upgraded)", "ROLLBACK (Keep Vanilla)")

    base = {"name": "b", "map_50_95": 40.0, "ap_small": 22.0, "latency_p95_ms": 50.0, "cpu_sec_per_100_frames": 10.0, "peak_rss_mb": 300.0}
    good = {"name": "g", "map_50_95": 39.7, "ap_small": 21.5, "latency_p95_ms": 30.0, "cpu_sec_per_100_frames": 6.0, "peak_rss_mb": 310.0}
    bad = {"name": "x", "map_50_95": 38.0, "ap_small": 20.0, "latency_p95_ms": 48.0, "cpu_sec_per_100_frames": 9.8, "peak_rss_mb": 390.0}
    assert evaluate_go_no_go(base, good)["verdict"] == "PASS (Adopt Upgraded)"
    assert evaluate_go_no_go(base, bad)["verdict"] == "ROLLBACK (Keep Vanilla)"
    gc.collect()
