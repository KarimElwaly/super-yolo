"""Filter a COCO JSON dataset down to target classes and emit a Hydra dataset YAML."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import yaml


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
        raise ValueError(f"Unsupported link mode: {mode}")


def filter_coco_split(
    input_json: Path,
    images_dir: Optional[Path],
    output_dir: Path,
    split_name: str,
    target_classes: Sequence[str],
    link_mode: str = "copy",
    keep_empty_images: bool = False,
    start_id: int = 0,
) -> Dict[str, object]:
    """Filter annotations/images to target_classes and remap category IDs contiguously."""
    with open(input_json, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    cat_by_lower_name = {c["name"].strip().lower(): c for c in data.get("categories", [])}
    requested = [c.strip() for c in target_classes if c.strip()]
    missing = [c for c in requested if c.lower() not in cat_by_lower_name]
    if missing:
        available = [c["name"] for c in data.get("categories", [])]
        raise ValueError(f"Requested classes {missing} not found in {input_json}. Available: {available}")

    old_to_new_id: Dict[int, int] = {}
    new_categories: List[dict] = []
    canonical_names: List[str] = []
    for idx, req_name in enumerate(requested):
        orig_cat = cat_by_lower_name[req_name.lower()]
        new_id = start_id + idx
        old_to_new_id[orig_cat["id"]] = new_id
        canonical_names.append(orig_cat["name"])
        new_categories.append(
            {
                "id": new_id,
                "name": orig_cat["name"],
                "supercategory": orig_cat.get("supercategory", "object"),
            }
        )

    filtered_annotations: List[dict] = []
    matched_image_ids = set()
    for ann in data.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        old_cat_id = ann["category_id"]
        if old_cat_id not in old_to_new_id:
            continue
        new_ann = dict(ann)
        new_ann["id"] = len(filtered_annotations) + 1
        new_ann["category_id"] = old_to_new_id[old_cat_id]
        filtered_annotations.append(new_ann)
        matched_image_ids.add(ann["image_id"])

    filtered_images: List[dict] = []
    kept_image_ids = set()
    target_images_dir = output_dir / "images" / split_name
    for img in data.get("images", []):
        if not keep_empty_images and img["id"] not in matched_image_ids:
            continue
        filtered_images.append(img)
        kept_image_ids.add(img["id"])
        if images_dir is not None and link_mode != "none":
            src_img = images_dir / img["file_name"]
            if src_img.is_file():
                _link_or_copy(src_img, target_images_dir / img["file_name"], link_mode)

    # Re-ensure annotations only reference kept images
    filtered_annotations = [a for a in filtered_annotations if a["image_id"] in kept_image_ids]

    out_json = output_dir / "annotations" / f"instances_{split_name}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "info": data.get("info", {"description": "Filtered COCO subset"}),
                "licenses": data.get("licenses", []),
                "categories": new_categories,
                "images": filtered_images,
                "annotations": filtered_annotations,
            },
            fh,
            indent=2,
        )

    return {
        "split": split_name,
        "output_json": str(out_json),
        "images": len(filtered_images),
        "annotations": len(filtered_annotations),
        "classes": canonical_names,
    }


def write_hydra_dataset_yaml(
    yaml_path: Path,
    dataset_root: Path,
    train_split: str,
    val_split: str,
    class_names: Sequence[str],
) -> None:
    payload = {
        "path": dataset_root.as_posix(),
        "train": train_split,
        "validation": val_split,
        "class_num": len(class_names),
        "class_list": list(class_names),
        "auto_download": None,
    }
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False, default_flow_style=None)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Filter COCO JSON dataset to selected classes.")
    parser.add_argument("--classes", required=True, help="Comma-separated target class names.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination directory for filtered dataset.")
    parser.add_argument("--source-dir", type=Path, help="COCO root dir containing annotations/ and images/.")
    parser.add_argument("--splits", nargs="+", default=["train2017", "val2017"], help="Splits under --source-dir.")
    parser.add_argument("--input-json", type=Path, help="Single COCO JSON path (alternative to --source-dir).")
    parser.add_argument("--images-dir", type=Path, help="Images dir for single --input-json mode.")
    parser.add_argument("--split-name", default="val", help="Split name for single --input-json mode (default: val).")
    parser.add_argument(
        "--link-mode",
        choices=["copy", "hardlink", "symlink", "none"],
        default="copy",
        help="How to transfer matching images into output-dir/images/<split>.",
    )
    parser.add_argument("--keep-empty-images", action="store_true", help="Keep images with zero matching objects.")
    parser.add_argument("--start-id", type=int, default=0, help="Starting remapped category_id (default: 0).")
    parser.add_argument("--hydra-yaml", type=Path, help="Path to write matching Hydra dataset YAML.")
    args = parser.parse_args(argv)

    target_classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    if not target_classes:
        parser.error("--classes must contain at least one class name.")

    output_dir = args.output_dir.resolve()
    processed_splits: List[str] = []
    canonical_names: List[str] = []

    if args.input_json:
        res = filter_coco_split(
            input_json=args.input_json.resolve(),
            images_dir=args.images_dir.resolve() if args.images_dir else None,
            output_dir=output_dir,
            split_name=args.split_name,
            target_classes=target_classes,
            link_mode=args.link_mode,
            keep_empty_images=args.keep_empty_images,
            start_id=args.start_id,
        )
        processed_splits.append(args.split_name)
        canonical_names = res["classes"]
        print(f"[OK] Filtered '{args.split_name}': {res['images']} images, {res['annotations']} annotations -> {res['output_json']}")
    elif args.source_dir:
        src_root = args.source_dir.resolve()
        for split in args.splits:
            ann_path = src_root / "annotations" / f"instances_{split}.json"
            if not ann_path.is_file():
                print(f"[SKIP] Annotation JSON not found: {ann_path}")
                continue
            img_dir = src_root / "images" / split
            res = filter_coco_split(
                input_json=ann_path,
                images_dir=img_dir if img_dir.is_dir() else None,
                output_dir=output_dir,
                split_name=split,
                target_classes=target_classes,
                link_mode=args.link_mode,
                keep_empty_images=args.keep_empty_images,
                start_id=args.start_id,
            )
            processed_splits.append(split)
            canonical_names = res["classes"]
            print(f"[OK] Filtered '{split}': {res['images']} images, {res['annotations']} annotations -> {res['output_json']}")
    else:
        parser.error("Provide either --input-json or --source-dir.")

    if not processed_splits:
        raise FileNotFoundError("No matching COCO annotation files were processed.")

    yaml_path = args.hydra_yaml or (output_dir / "dataset.yaml")
    train_split = next((s for s in processed_splits if "train" in s.lower()), processed_splits[0])
    val_split = next((s for s in processed_splits if "val" in s.lower()), processed_splits[-1])
    write_hydra_dataset_yaml(yaml_path, output_dir, train_split, val_split, canonical_names)
    print(f"[OK] Wrote Hydra dataset config -> {yaml_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
