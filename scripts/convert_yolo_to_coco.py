"""Convert Ultralytics YOLO .txt datasets into COCO JSON format with verification."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import yaml
from PIL import Image, ImageDraw


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def _parse_names(names_raw) -> List[str]:
    if isinstance(names_raw, list):
        return [str(x) for x in names_raw]
    if isinstance(names_raw, dict):
        return [str(names_raw[k]) for k in sorted(names_raw.keys(), key=lambda x: int(x))]
    raise ValueError(f"Unsupported 'names' format in data.yaml: {type(names_raw)}")


def _resolve_split_dirs(dataset_root: Path, split_value: Optional[str], split_name: str) -> Tuple[Optional[Path], Optional[Path]]:
    """Resolve (images_dir, labels_dir) across common Ultralytics & Roboflow directory layouts."""
    candidates: List[Path] = []
    if split_value:
        p = Path(split_value)
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.append((dataset_root / p).resolve())
            candidates.append((dataset_root.parent / p).resolve())
    candidates.extend(
        [
            dataset_root / "images" / split_name,
            dataset_root / split_name / "images",
            dataset_root / split_name,
        ]
    )

    images_dir: Optional[Path] = None
    for cand in candidates:
        if cand.is_dir():
            images_dir = cand
            break
    if images_dir is None:
        return None, None

    # Infer corresponding labels directory
    label_candidates: List[Path] = []
    parts = list(images_dir.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        replaced = parts.copy()
        replaced[idx] = "labels"
        label_candidates.append(Path(*replaced))
    label_candidates.extend(
        [
            dataset_root / "labels" / split_name,
            images_dir.parent / "labels",
        ]
    )

    labels_dir = next((c for c in label_candidates if c.is_dir()), None)
    return images_dir, labels_dir


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or mode == "none":
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    elif mode == "symlink":
        try:
            os.symlink(src.resolve(), dst)
        except OSError:
            shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown link mode: {mode}")


def _yolo_line_to_coco_bbox(parts: Sequence[str], img_w: int, img_h: int) -> Tuple[int, List[float]]:
    if len(parts) < 5:
        raise ValueError(f"Expected >= 5 fields in YOLO label line, got {len(parts)}: {' '.join(parts)}")
    cls_id = int(float(parts[0]))
    vals = [float(v) for v in parts[1:]]

    if len(vals) == 4:
        xc, yc, bw, bh = vals
        x1 = (xc - bw / 2.0) * img_w
        y1 = (yc - bh / 2.0) * img_h
        x2 = (xc + bw / 2.0) * img_w
        y2 = (yc + bh / 2.0) * img_h
    elif len(vals) >= 6 and len(vals) % 2 == 0:
        xs = [vals[i] * img_w for i in range(0, len(vals), 2)]
        ys = [vals[i] * img_h for i in range(1, len(vals), 2)]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
    else:
        raise ValueError(f"Malformed YOLO coordinates count ({len(vals)}): {' '.join(parts)}")

    x1 = max(0.0, min(float(img_w), x1))
    y1 = max(0.0, min(float(img_h), y1))
    x2 = max(0.0, min(float(img_w), x2))
    y2 = max(0.0, min(float(img_h), y2))
    w = x2 - x1
    h = y2 - y1
    return cls_id, [round(x1, 4), round(y1, 4), round(w, 4), round(h, 4)]


def convert_split(
    images_dir: Path,
    labels_dir: Optional[Path],
    output_json: Path,
    class_names: List[str],
    target_images_dir: Optional[Path] = None,
    link_mode: str = "copy",
    start_id: int = 0,
) -> Dict[str, int]:
    """Convert one dataset split from YOLO .txt labels to a COCO JSON annotation file."""
    image_files = sorted(
        p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_files:
        raise FileNotFoundError(f"No images found in {images_dir}")

    categories = [
        {"id": start_id + idx, "name": name, "supercategory": "object"}
        for idx, name in enumerate(class_names)
    ]

    images_list = []
    annotations_list = []
    ann_id = 1
    skipped_boxes = 0

    for img_id, img_path in enumerate(image_files, start=1):
        with Image.open(img_path) as im:
            img_w, img_h = im.size

        if target_images_dir is not None and target_images_dir.resolve() != images_dir.resolve():
            _link_or_copy(img_path, target_images_dir / img_path.name, link_mode)

        images_list.append(
            {
                "id": img_id,
                "file_name": img_path.name,
                "width": img_w,
                "height": img_h,
            }
        )

        if labels_dir is not None:
            label_file = labels_dir / f"{img_path.stem}.txt"
            if label_file.is_file():
                with open(label_file, "r", encoding="utf-8") as fh:
                    for raw_line in fh:
                        line = raw_line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        cls_idx, bbox = _yolo_line_to_coco_bbox(parts, img_w, img_h)
                        if cls_idx < 0 or cls_idx >= len(class_names):
                            raise ValueError(
                                f"Class index {cls_idx} in {label_file} out of range [0, {len(class_names) - 1}]"
                            )
                        bw, bh = bbox[2], bbox[3]
                        if bw <= 0.0 or bh <= 0.0:
                            skipped_boxes += 1
                            continue
                        area = round(bw * bh, 4)
                        annotations_list.append(
                            {
                                "id": ann_id,
                                "image_id": img_id,
                                "category_id": start_id + cls_idx,
                                "bbox": bbox,
                                "area": area,
                                "iscrowd": 0,
                            }
                        )
                        ann_id += 1

    coco_dict = {
        "info": {"description": "Converted from Ultralytics YOLO format"},
        "licenses": [],
        "categories": categories,
        "images": images_list,
        "annotations": annotations_list,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as fh:
        json.dump(coco_dict, fh, indent=2)

    return {
        "images": len(images_list),
        "annotations": len(annotations_list),
        "skipped_boxes": skipped_boxes,
    }


def verify_coco_dataset(
    coco_json: Path,
    images_dir: Path,
    expected_classes: Optional[Sequence[str]] = None,
    sample_count: int = 0,
    verify_dir: Optional[Path] = None,
) -> Dict[str, object]:
    """Validate COCO JSON integrity: image existence/dimensions, box bounds, areas, and class IDs."""
    with open(coco_json, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    categories = {c["id"]: c["name"] for c in data.get("categories", [])}
    if not categories:
        raise ValueError(f"No categories defined in {coco_json}")

    if expected_classes is not None:
        sorted_names = [categories[cid] for cid in sorted(categories.keys())]
        if list(sorted_names) != list(expected_classes):
            raise ValueError(f"Category mismatch: JSON {sorted_names} != expected {list(expected_classes)}")

    images_by_id: Dict[int, dict] = {}
    errors: List[str] = []
    for img in data.get("images", []):
        img_path = images_dir / img["file_name"]
        if not img_path.is_file():
            errors.append(f"Missing image file: {img_path}")
            continue
        with Image.open(img_path) as im:
            actual_w, actual_h = im.size
        if actual_w != img["width"] or actual_h != img["height"]:
            errors.append(
                f"Dimension mismatch for {img['file_name']}: JSON=({img['width']},{img['height']}) actual=({actual_w},{actual_h})"
            )
        images_by_id[img["id"]] = img

    anns_by_img: Dict[int, List[dict]] = {}
    for ann in data.get("annotations", []):
        img = images_by_id.get(ann["image_id"])
        if img is None:
            errors.append(f"Annotation {ann['id']} references unknown image_id {ann['image_id']}")
            continue
        if ann["category_id"] not in categories:
            errors.append(f"Annotation {ann['id']} has invalid category_id {ann['category_id']}")
        x, y, w, h = ann["bbox"]
        img_w, img_h = img["width"], img["height"]
        if w <= 0 or h <= 0 or ann.get("area", 0) <= 0:
            errors.append(f"Annotation {ann['id']} has non-positive bbox/area: {ann['bbox']}, area={ann.get('area')}")
        if x < -1e-3 or y < -1e-3 or (x + w) > img_w + 1e-3 or (y + h) > img_h + 1e-3:
            errors.append(
                f"Annotation {ann['id']} out of bounds [0,{img_w}]x[0,{img_h}]: {ann['bbox']}"
            )
        anns_by_img.setdefault(img["id"], []).append(ann)

    if errors:
        raise ValueError(f"COCO verification failed with {len(errors)} error(s):\n" + "\n".join(errors[:20]))

    saved_samples: List[str] = []
    if sample_count > 0:
        out_dir = verify_dir or (coco_json.parent / "verify_samples")
        out_dir.mkdir(parents=True, exist_ok=True)
        for img in data.get("images", [])[:sample_count]:
            img_path = images_dir / img["file_name"]
            with Image.open(img_path).convert("RGB") as im:
                draw = ImageDraw.Draw(im)
                for ann in anns_by_img.get(img["id"], []):
                    x, y, w, h = ann["bbox"]
                    cls_name = categories[ann["category_id"]]
                    draw.rectangle([x, y, x + w, y + h], outline=(0, 255, 0), width=2)
                    draw.text((x + 2, max(0, y - 12)), cls_name, fill=(255, 255, 0))
                out_file = out_dir / f"verify_{img['file_name']}"
                im.save(out_file)
                saved_samples.append(str(out_file))

    return {
        "images_verified": len(images_by_id),
        "annotations_verified": len(data.get("annotations", [])),
        "categories": len(categories),
        "saved_samples": saved_samples,
    }


def write_hydra_dataset_yaml(
    yaml_path: Path,
    dataset_root: Path,
    train_split: str,
    val_split: str,
    class_names: List[str],
) -> None:
    payload = {
        "path": dataset_root.as_posix(),
        "train": train_split,
        "validation": val_split,
        "class_num": len(class_names),
        "class_list": class_names,
        "auto_download": None,
    }
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False, default_flow_style=None)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Convert Ultralytics YOLO .txt dataset to COCO JSON.")
    parser.add_argument("--data-yaml", type=Path, help="Path to Ultralytics data.yaml.")
    parser.add_argument("--input-dir", type=Path, help="Root directory of YOLO dataset (if no data.yaml).")
    parser.add_argument("--output-dir", type=Path, help="Target COCO dataset root (defaults to input root).")
    parser.add_argument("--classes", type=str, help="Comma-separated class list (overrides data.yaml names).")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="Splits to convert (default: train val).")
    parser.add_argument(
        "--link-mode",
        choices=["copy", "hardlink", "symlink", "none"],
        default="copy",
        help="How to populate output images directory when output-dir differs from input-dir.",
    )
    parser.add_argument("--start-id", type=int, default=0, help="Starting COCO category_id (default: 0).")
    parser.add_argument("--hydra-yaml", type=Path, help="Optional output path for Hydra dataset YAML.")
    parser.add_argument("--verify", action="store_true", help="Verify generated COCO JSON bounds, areas, and classes.")
    parser.add_argument("--verify-samples", type=int, default=0, help="Save N annotated verification images.")
    parser.add_argument("--verify-dir", type=Path, help="Directory to save verification sample images.")
    args = parser.parse_args(argv)

    if not args.data_yaml and not args.input_dir:
        parser.error("Provide at least one of --data-yaml or --input-dir.")

    yaml_cfg = {}
    if args.data_yaml:
        with open(args.data_yaml, "r", encoding="utf-8") as fh:
            yaml_cfg = yaml.safe_load(fh) or {}
        if args.input_dir:
            dataset_root = args.input_dir
        elif "path" in yaml_cfg and yaml_cfg["path"]:
            p = Path(yaml_cfg["path"])
            dataset_root = p if p.is_absolute() else (args.data_yaml.parent / p).resolve()
        else:
            dataset_root = args.data_yaml.parent.resolve()
    else:
        dataset_root = args.input_dir.resolve()

    if args.classes:
        class_names = [c.strip() for c in args.classes.split(",") if c.strip()]
    elif "names" in yaml_cfg:
        class_names = _parse_names(yaml_cfg["names"])
    else:
        raise ValueError("Class names must be provided via --classes or data.yaml 'names'.")

    output_root = (args.output_dir or dataset_root).resolve()
    split_aliases = {"val": yaml_cfg.get("val") or yaml_cfg.get("valid"), "train": yaml_cfg.get("train"), "test": yaml_cfg.get("test")}

    converted_splits: List[str] = []
    for split in args.splits:
        images_dir, labels_dir = _resolve_split_dirs(dataset_root, split_aliases.get(split), split)
        if images_dir is None:
            print(f"[SKIP] Split '{split}' not found under {dataset_root}")
            continue

        target_images_dir = output_root / "images" / split
        output_json = output_root / "annotations" / f"instances_{split}.json"
        stats = convert_split(
            images_dir=images_dir,
            labels_dir=labels_dir,
            output_json=output_json,
            class_names=class_names,
            target_images_dir=target_images_dir if args.link_mode != "none" else None,
            link_mode=args.link_mode,
            start_id=args.start_id,
        )
        converted_splits.append(split)
        print(
            f"[OK] Converted split '{split}': {stats['images']} images, "
            f"{stats['annotations']} boxes -> {output_json}"
        )

        if args.verify:
            check_images_dir = target_images_dir if target_images_dir.is_dir() else images_dir
            vstats = verify_coco_dataset(
                coco_json=output_json,
                images_dir=check_images_dir,
                expected_classes=class_names,
                sample_count=args.verify_samples,
                verify_dir=(args.verify_dir / split) if args.verify_dir else None,
            )
            print(
                f"[VERIFIED] '{split}': {vstats['images_verified']} images, "
                f"{vstats['annotations_verified']} boxes, {vstats['categories']} classes"
            )

    if not converted_splits:
        raise FileNotFoundError(f"No requested splits {args.splits} found under {dataset_root}")

    if args.hydra_yaml:
        train_split = "train" if "train" in converted_splits else converted_splits[0]
        val_split = "val" if "val" in converted_splits else converted_splits[-1]
        write_hydra_dataset_yaml(args.hydra_yaml, output_root, train_split, val_split, class_names)
        print(f"[OK] Wrote Hydra dataset config -> {args.hydra_yaml}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
