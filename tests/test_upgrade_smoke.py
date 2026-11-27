"""End-to-end smoke tests covering every phase of the MIT-clean YOLOv9 CPU & accuracy upgrade."""

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
from scripts.check_licenses import _evaluate_package, scan_installed_licenses
from scripts.convert_yolo_to_coco import main as convert_yolo_main
from scripts.export_openvino import export_pipeline
from scripts.filter_coco_classes import main as filter_coco_main
from scripts.prune_model import main as prune_main
from yolo.model.yolo import create_model, load_partial_weights, strip_auxiliary_branches
from yolo.model.module import RepConv
from yolo.tools.data_loader import YoloDataset
from yolo.tools.solver import BaseModel, TrainModel
from yolo.utils.bounding_box_utils import Vec2Box
from yolo.utils.logging_utils import setup


def _count_activations(model: nn.Module) -> dict[str, int]:
    counts = {"SiLU": 0, "ReLU": 0, "Hardswish": 0}
    for mod in model.modules():
        if isinstance(mod, nn.SiLU):
            counts["SiLU"] += 1
        elif isinstance(mod, nn.ReLU):
            counts["ReLU"] += 1
        elif isinstance(mod, nn.Hardswish):
            counts["Hardswish"] += 1
    return counts


def _cleanup(*objs) -> None:
    for obj in objs:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def test_repconv_fusion_preserves_eval_output():
    module = RepConv(8, 8, 3).eval()
    inputs = torch.randn(1, 8, 32, 32)
    with torch.no_grad():
        expected = module(inputs)
        module.switch_to_deploy()
        actual = module(inputs)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert module.reparam is not None


def test_01_license_audit_clean():
    report = scan_installed_licenses(strict_unknown=True)
    assert report["clean"] is True, f"Violations: {report['violations']}"
    assert report["total_packages"] > 0

    proc = subprocess.run(
        [sys.executable, str(project_root / "scripts" / "check_licenses.py")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "0 GPL/AGPL, 0 banned" in proc.stdout

    fake_banned = SimpleNamespace(
        name="ultralytics",
        version="8.0.0",
        metadata={"Name": "ultralytics", "License": "AGPL-3.0"},
        files=[],
    )
    assert _evaluate_package(fake_banned, strict_unknown=True).status == "banned"

    fake_gpl = SimpleNamespace(
        name="some-gpl-pkg",
        version="1.0.0",
        metadata={"Name": "some-gpl-pkg", "License": "GPL-3.0-only", "License-Expression": "GPL-3.0-only"},
        files=[],
    )
    assert _evaluate_package(fake_gpl, strict_unknown=True).status == "copyleft"


def test_02_convert_yolo_to_coco_and_verify(tmp_path: Path):
    yolo_root = tmp_path / "yolo_ds"
    for split in ("train", "val"):
        (yolo_root / "images" / split).mkdir(parents=True)
        (yolo_root / "labels" / split).mkdir(parents=True)
        for idx in range(2):
            Image.new("RGB", (128, 96), (20 + idx * 40, 80, 120)).save(
                yolo_root / "images" / split / f"sample_{idx}.jpg"
            )
        (yolo_root / "labels" / split / "sample_0.txt").write_text(
            "0 0.5 0.5 0.4 0.5\n1 0.25 0.25 0.2 0.2\n", encoding="utf-8"
        )
        (yolo_root / "labels" / split / "sample_1.txt").write_text(
            "2 0.7 0.6 0.3 0.3\n", encoding="utf-8"
        )
    data_yaml = yolo_root / "data.yaml"
    data_yaml.write_text(
        "path: .\ntrain: images/train\nval: images/val\nnames: [person, car, bus]\n",
        encoding="utf-8",
    )

    coco_out = tmp_path / "coco_ds"
    hydra_yaml = coco_out / "dataset.yaml"
    verify_dir = tmp_path / "verify_samples"
    rc = convert_yolo_main(
        [
            "--data-yaml",
            str(data_yaml),
            "--output-dir",
            str(coco_out),
            "--hydra-yaml",
            str(hydra_yaml),
            "--verify",
            "--verify-samples",
            "1",
            "--verify-dir",
            str(verify_dir),
        ]
    )
    assert rc == 0
    assert (coco_out / "annotations" / "instances_train.json").is_file()
    assert (coco_out / "annotations" / "instances_val.json").is_file()
    assert hydra_yaml.is_file()
    assert (verify_dir / "train" / "verify_sample_0.jpg").is_file()

    data_cfg = OmegaConf.create({"image_size": [128, 128], "batch_size": 1, "data_augment": {}, "dynamic_shape": False})
    yds = YoloDataset(data_cfg, OmegaConf.load(hydra_yaml), phase="train")
    assert len(yds) == 2


def test_03_filter_coco_classes_subset(tmp_path: Path):
    src_dir = tmp_path / "src_coco"
    for split in ("train", "val"):
        (src_dir / "images" / split).mkdir(parents=True)
        (src_dir / "annotations").mkdir(parents=True, exist_ok=True)
        for i in range(3):
            Image.new("RGB", (100, 100), (i * 50, 40, 20)).save(src_dir / "images" / split / f"img_{i}.jpg")
        coco = {
            "categories": [
                {"id": 1, "name": "person"},
                {"id": 2, "name": "bicycle"},
                {"id": 3, "name": "car"},
                {"id": 6, "name": "bus"},
            ],
            "images": [
                {"id": 1, "file_name": "img_0.jpg", "width": 100, "height": 100},
                {"id": 2, "file_name": "img_1.jpg", "width": 100, "height": 100},
                {"id": 3, "file_name": "img_2.jpg", "width": 100, "height": 100},
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 30, 30], "area": 900, "iscrowd": 0},
                {"id": 2, "image_id": 2, "category_id": 2, "bbox": [20, 20, 20, 20], "area": 400, "iscrowd": 0},
                {"id": 3, "image_id": 3, "category_id": 6, "bbox": [30, 30, 40, 40], "area": 1600, "iscrowd": 0},
            ],
        }
        (src_dir / "annotations" / f"instances_{split}.json").write_text(json.dumps(coco), encoding="utf-8")

    out_dir = tmp_path / "filtered_coco"
    hydra_yaml = out_dir / "subset.yaml"
    rc = filter_coco_main(
        [
            "--source-dir",
            str(src_dir),
            "--splits",
            "train",
            "val",
            "--classes",
            "person,bus",
            "--output-dir",
            str(out_dir),
            "--hydra-yaml",
            str(hydra_yaml),
        ]
    )
    assert rc == 0
    ds_cfg = OmegaConf.load(hydra_yaml)
    assert ds_cfg.class_num == 2
    assert list(ds_cfg.class_list) == ["person", "bus"]

    val_ann = json.loads((out_dir / "annotations" / "instances_val.json").read_text(encoding="utf-8"))
    assert [c["id"] for c in val_ann["categories"]] == [0, 1]
    assert len(val_ann["images"]) == 2
    assert len(val_ann["annotations"]) == 2


@pytest.mark.parametrize("model_name,expected_silu", [("v9-t", 237), ("v9-s", 237), ("v9-m", 206)])
def test_04_vanilla_models_hydra_and_forward(model_name: str, expected_silu: int):
    x = torch.randn(1, 3, 128, 128)
    with initialize(config_path="../yolo/config", version_base=None):
        cfg = compose(config_name="config", overrides=[f"model={model_name}", "weight=False"])
    model = create_model(cfg.model, weight_path=None, class_num=80).eval()
    acts = _count_activations(model)
    assert acts["SiLU"] == expected_silu and acts["ReLU"] == 0 and acts["Hardswish"] == 0
    with torch.no_grad():
        out = model(x)
    assert "Main" in out and "AUX" in out and len(out["Main"]) == 3
    _cleanup(model, out, x)


@pytest.mark.parametrize(
    "model_name,expected_act,expected_heads,expected_strides",
    [
        ("v9-t-opt", "ReLU", 3, [8, 16, 32]),
        ("v9-s-opt", "ReLU", 3, [8, 16, 32]),
        ("v9-m-opt", "Hardswish", 3, [8, 16, 32]),
        ("v9-t-opt-p2", "ReLU", 4, [4, 8, 16, 32]),
        ("v9-s-opt-p2", "ReLU", 4, [4, 8, 16, 32]),
    ],
)
def test_05_upgraded_models_partial_load_and_strip_aux(
    tmp_path: Path, model_name: str, expected_act: str, expected_heads: int, expected_strides: list[int]
):
    x = torch.randn(1, 3, 128, 128)
    with initialize(config_path="../yolo/config", version_base=None):
        vanilla_base = "v9-m" if "v9-m" in model_name else ("v9-s" if "v9-s" in model_name else "v9-t")
        v_cfg = compose(config_name="config", overrides=[f"model={vanilla_base}", "weight=False"])
        v_model = create_model(v_cfg.model, weight_path=None, class_num=80)
        ckpt_path = tmp_path / f"{vanilla_base}.pt"
        torch.save(v_model.model.state_dict(), ckpt_path)
        _cleanup(v_model)

        opt_cfg = compose(
            config_name="config",
            overrides=[f"model={model_name}", "dataset.class_num=3", f"partial_weights={ckpt_path.as_posix()}"],
        )

    bm = BaseModel(opt_cfg)
    model = bm.model.eval()
    acts = _count_activations(model)
    assert acts["SiLU"] == 0 and acts[expected_act] > 0

    # Verify load_partial_weights helper directly too
    stats = load_partial_weights(model, ckpt_path, strict=False)
    assert stats["loaded"] > 0 and stats["reinitialized"] > 0

    vec2box = Vec2Box(model, opt_cfg.model.anchor, [128, 128], torch.device("cpu"))
    assert vec2box.strides == expected_strides

    with torch.no_grad():
        out_before = model(x)
    assert len(out_before["Main"]) == expected_heads
    strip_info = strip_auxiliary_branches(model)
    assert strip_info["removed_layers"] > 0
    with torch.no_grad():
        out_after = model(x)
    assert "AUX" not in out_after and list(out_after.keys()) == ["Main"]
    for h_before, h_after in zip(out_before["Main"], out_after["Main"]):
        for tb, ta in zip(h_before, h_after):
            assert torch.allclose(tb, ta, atol=1e-6)

    _cleanup(bm, model, out_before, out_after, x)


def test_06_knowledge_distillation_two_steps(tmp_path: Path):
    ds_root = tmp_path / "kd_ds"
    for split in ("train", "val"):
        (ds_root / "images" / split).mkdir(parents=True)
        (ds_root / "annotations").mkdir(parents=True, exist_ok=True)
        for idx in range(2):
            Image.new("RGB", (192, 192), (40 + idx * 30, 90, 120)).save(
                ds_root / "images" / split / f"00000000000{idx+1}.jpg"
            )
        coco = {
            "categories": [{"id": 0, "name": "obj0"}, {"id": 1, "name": "obj1"}],
            "images": [
                {"id": 1, "file_name": "000000000001.jpg", "width": 192, "height": 192},
                {"id": 2, "file_name": "000000000002.jpg", "width": 192, "height": 192},
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 0, "bbox": [30, 30, 60, 60], "area": 3600, "iscrowd": 0},
                {"id": 2, "image_id": 2, "category_id": 1, "bbox": [45, 45, 70, 50], "area": 3500, "iscrowd": 0},
            ],
        }
        (ds_root / "annotations" / f"instances_{split}.json").write_text(json.dumps(coco), encoding="utf-8")

    with initialize(config_path="../yolo/config", version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "task=train",
                "model=v9-t-opt",
                "dataset=custom_template",
                f"dataset.path={ds_root.as_posix()}",
                "dataset.class_num=2",
                "dataset.class_list=[obj0,obj1]",
                "weight=False",
                "use_wandb=False",
                "+quiet=True",
                "accelerator=cpu",
                "device=cpu",
                "task.epoch=2",
                "task.close_mosaic=1",
                "task.data.cpu_num=0",
                "task.validation.data.cpu_num=0",
                "task.data.batch_size=1",
                "task.validation.data.batch_size=1",
                "image_size=[192,192]",
                "task.distill.enabled=True",
                "task.distill.teacher_model=v9-m",
                "task.distill.teacher_weights=null",
                "task.distill.lambda_kd=1.0",
                "task.distill.feature_weight=0.5",
                "task.distill.logit_weight=0.5",
                f"out_path={(tmp_path / 'runs').as_posix()}",
            ],
        )

    callbacks, _, save_path = setup(cfg)
    model = TrainModel(cfg)
    assert model.kd_loss is not None
    assert all(not p.requires_grad for p in model.kd_loss.teacher_wrapper.parameters())

    teacher_snapshot = {
        k: v.detach().clone()
        for k, v in list(model.kd_loss.teacher_wrapper.state_dict().items())[:5]
    }
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

    # Teacher weights remain completely frozen; student state_dict excludes teacher/adapters
    for k, snap in teacher_snapshot.items():
        assert torch.equal(snap, model.kd_loss.teacher_wrapper.state_dict()[k])
    assert not any("teacher" in n or "adapters" in n for n, _ in model.model.named_modules())
    fake_ckpt = {"state_dict": model.state_dict()}
    model.on_save_checkpoint(fake_ckpt)
    assert not any(k.startswith("kd_loss.") for k in fake_ckpt["state_dict"])

    _cleanup(model, trainer)


def test_07_structured_pruning(tmp_path: Path):
    pruned_pt = tmp_path / "v9t_opt_pruned.pt"
    rc = prune_main(
        [
            "--model",
            "v9-t-opt",
            "--class-num",
            "2",
            "--pruning-ratio",
            "0.20",
            "--image-size",
            "192",
            "--output",
            str(pruned_pt),
        ]
    )
    assert rc == 0 and pruned_pt.is_file()
    payload = torch.load(pruned_pt, map_location="cpu", weights_only=False)
    assert payload["stats"]["param_reduction_pct"] > 0
    assert payload["stats"]["macs_reduction_pct"] > 0
    pm = payload["pruned_model"].eval()
    with torch.no_grad():
        out = pm(torch.randn(1, 3, 192, 192))
    assert "Main" in out and len(out["Main"]) == 3

    with initialize(config_path="../yolo/config", version_base=None):
        cfg_reload = compose(
            config_name="config",
            overrides=["model=v9-t-opt", "dataset.class_num=2", f"weight={pruned_pt.as_posix()}"],
        )
    bm = BaseModel(cfg_reload).eval()
    with torch.no_grad():
        out_bm = bm(torch.randn(1, 3, 192, 192))
    assert "Main" in out_bm and len(out_bm["Main"]) == 3
    _cleanup(pm, bm, payload, out, out_bm)


def test_08_09_10_export_detector_and_benchmark_e2e(tmp_path: Path):
    calib_dir = tmp_path / "val_imgs"
    calib_dir.mkdir(parents=True)
    images_meta = []
    anns_meta = []
    for i in range(4):
        fname = f"img_{i:04d}.jpg"
        frame = np.random.randint(0, 255, (192, 256, 3), dtype=np.uint8)
        cv2.imwrite(str(calib_dir / fname), frame)
        images_meta.append({"id": i + 1, "file_name": fname, "width": 256, "height": 192})
        anns_meta.append(
            {"id": i + 1, "image_id": i + 1, "category_id": 0, "bbox": [20, 20, 60, 60], "area": 3600, "iscrowd": 0}
        )

    ann_path = tmp_path / "val_ann.json"
    ann_path.write_text(
        json.dumps(
            {
                "categories": [{"id": 0, "name": "obj0"}, {"id": 1, "name": "obj1"}],
                "images": images_meta,
                "annotations": anns_meta,
            }
        ),
        encoding="utf-8",
    )

    res_van = export_pipeline(
        model_arg="v9-t",
        class_num_arg=2,
        image_size=(192, 192),
        output_dir=tmp_path / "exp_van_fp16",
        name="v9t_fp16",
    )
    assert res_van["meta"]["stripped_aux"]["removed_layers"] > 0
    assert res_van["meta"]["onnx_info"]["onnx_convs"] <= 230
    assert res_van["meta"]["parity"]["onnx_max_score_diff"] < 1e-3

    res_opt = export_pipeline(
        model_arg="v9-t-opt",
        class_num_arg=2,
        image_size=(192, 192),
        output_dir=tmp_path / "exp_opt_int8_ppp",
        name="v9t_opt_int8_ppp",
        int8=True,
        calib_dir=calib_dir,
        calib_samples=4,
        calib_exclude_ann=ann_path,
        calib_exclude_first_images=1,
        protect_heads=True,
        embed_preprocess=True,
        static_preprocess=True,
    )
    assert res_opt["meta"]["precision"] == "INT8"
    assert res_opt["meta"]["embedded_preprocess"] is True
    assert res_opt["meta"]["protected_ops_count"] > 0
    assert res_opt["meta"]["fused_repconv_count"] > 0
    assert res_opt["meta"]["calibration_excluded_images"] == 1
    assert res_opt["meta"]["calibration_samples"] == 3

    sample_bgr = cv2.imread(str(calib_dir / "img_0000.jpg"))
    for xml_str, exp_ppp in [(res_van["xml_path"], False), (res_opt["xml_path"], True)]:
        det = OpenVINODetector(xml_str, num_threads=2, conf_threshold=0.25, iou_threshold=0.45)
        assert det.embedded_preprocess is exp_ppp
        dets = det(sample_bgr)
        assert isinstance(dets, sv.Detections)
        assert dets.xyxy.ndim == 2 and dets.xyxy.shape[1] == 4

        _, geom = det.prepare_input(sample_bgr)
        fake_boxes = np.array([[[10.0, 10.0, 90.0, 90.0], [11.0, 11.0, 91.0, 91.0]]], dtype=np.float32)
        fake_scores = np.array([[[0.9, 0.1], [0.7, 0.05]]], dtype=np.float32)
        nms_dets = det.postprocess(fake_boxes, fake_scores, geom)
        assert len(nms_dets) == 1 and int(nms_dets.class_id[0]) == 0

    tracker = SimpleByteTrack(high_thresh=0.4)
    d1 = sv.Detections(xyxy=np.array([[20, 20, 80, 80]], dtype=np.float32), confidence=np.array([0.9], dtype=np.float32), class_id=np.array([0]))
    d2 = sv.Detections(xyxy=np.array([[24, 24, 84, 84]], dtype=np.float32), confidence=np.array([0.85], dtype=np.float32), class_id=np.array([0]))
    assert tracker.update_with_detections(d1).tracker_id[0] == tracker.update_with_detections(d2).tracker_id[0]

    vid_path = tmp_path / "stream.mp4"
    stream_out = tmp_path / "stream_out.mp4"
    vw = cv2.VideoWriter(str(vid_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (sample_bgr.shape[1], sample_bgr.shape[0]))
    for _ in range(3):
        vw.write(sample_bgr)
    vw.release()
    stream_stats = run_stream_pipeline(
        model_xml=Path(res_opt["xml_path"]),
        source=str(vid_path),
        num_threads=2,
        async_requests=2,
        detect_every_n=2,
        max_frames=3,
        output_video=stream_out,
        display=False,
        drop_stale=False,
    )
    assert stream_stats["frames_processed"] == 3 and stream_out.is_file()

    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        f"models:\n"
        f"  - name: v9t-fp16\n    model_xml: {Path(res_van['xml_path']).as_posix()}\n    baseline: true\n"
        f"  - name: v9t-opt-int8-ppp\n    model_xml: {Path(res_opt['xml_path']).as_posix()}\n",
        encoding="utf-8",
    )
    out_md = tmp_path / "bench.md"
    out_json = tmp_path / "bench.json"
    report = run_benchmark_suite(
        manifest_path=manifest,
        val_images=calib_dir,
        val_ann=ann_path,
        max_eval_images=4,
        threads=2,
        async_requests=2,
        warmup=2,
        iters=5,
        output_md=out_md,
        output_json=out_json,
        isolated=False,
    )
    assert out_md.is_file() and out_json.is_file()
    assert len(report["results"]) == 2 and len(report["comparisons"]) == 1

    base = {"name": "b", "map_50_95": 41.0, "ap_small": 22.0, "latency_p95_ms": 40.0, "cpu_sec_per_100_frames": 13.0, "peak_rss_mb": 500.0}
    pass_cand = {"name": "c1", "map_50_95": 40.7, "ap_small": 21.2, "latency_p95_ms": 20.0, "cpu_sec_per_100_frames": 6.5, "peak_rss_mb": 520.0}
    fail_cand = {"name": "c2", "map_50_95": 39.0, "ap_small": 20.0, "latency_p95_ms": 35.0, "cpu_sec_per_100_frames": 11.0, "peak_rss_mb": 600.0}
    assert evaluate_go_no_go(base, pass_cand)["verdict"] == "PASS (Adopt Upgraded)"
    assert evaluate_go_no_go(base, fail_cand)["verdict"] == "ROLLBACK (Keep Vanilla)"
    _cleanup()
