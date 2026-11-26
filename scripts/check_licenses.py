"""Verify that all installed Python packages comply with MIT/Apache-2.0/BSD licensing."""

from __future__ import annotations

import argparse
import importlib.metadata as importlib_metadata
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


BANNED_PACKAGES = ("ultralytics", "boxmot")

# Token patterns to identify strong copyleft vs permissive licenses.
STRONG_GPL_RE = re.compile(r"(?<![A-Za-z])(A?GPL(?:[-_ ]?v?\d+(?:\.\d+)?)?)(?![A-Za-z])", re.IGNORECASE)
LGPL_RE = re.compile(r"(?<![A-Za-z])LGPL(?:[-_ ]?v?\d+(?:\.\d+)?)?(?![A-Za-z])", re.IGNORECASE)

PERMISSIVE_MARKERS = (
    "MIT",
    "APACHE",
    "BSD",
    "PSF",
    "PYTHON SOFTWARE FOUNDATION",
    "ISC",
    "LGPL",
    "0BSD",
    "BSL-1.0",
    "BOOST SOFTWARE LICENSE",
    "FREEBSD",
    "ZLIB",
    "CC0",
    "UNLICENSE",
    "PUBLIC DOMAIN",
    "MPL-2.0",
    "MOZILLA PUBLIC LICENSE 2.0",
)

LICENSE_FILE_SIGNATURES: Tuple[Tuple[str, str], ...] = (
    ("Apache License", "Apache-2.0"),
    ("APACHE LICENSE", "Apache-2.0"),
    ("MIT License", "MIT"),
    ("Permission is hereby granted, free of charge", "MIT"),
    ("Redistribution and use in source and binary forms", "BSD"),
    ("Mozilla Public License Version 2.0", "MPL-2.0"),
    ("GNU LESSER GENERAL PUBLIC LICENSE", "LGPL"),
    ("GNU AFFERO GENERAL PUBLIC LICENSE", "AGPL"),
    ("GNU GENERAL PUBLIC LICENSE", "GPL"),
)


@dataclass
class PackageLicenseResult:
    name: str
    version: str
    license: str
    source: str
    status: str  # "ok", "banned", "copyleft", "unknown"
    reason: str = ""


def _normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower().strip()


def _match_license_text(text: str) -> Optional[str]:
    lower = text.lower()
    for sig, label in LICENSE_FILE_SIGNATURES:
        if sig.lower() in lower:
            return label
    return None


def _extract_metadata_license(dist: importlib_metadata.Distribution) -> Tuple[str, str]:
    meta = dist.metadata
    license_expr = (meta.get("License-Expression") or "").strip()
    if license_expr and license_expr.upper() != "UNKNOWN":
        return license_expr, "metadata:License-Expression"

    classifiers = meta.get_all("Classifier") or []
    license_classifiers = [
        c.split("License ::")[-1].strip().lstrip(": ")
        for c in classifiers
        if c.startswith("License ::") and "OSI Approved" not in c.split("::")[-1]
    ]
    if not license_classifiers:
        license_classifiers = [
            c.split("::")[-1].strip()
            for c in classifiers
            if c.startswith("License ::")
        ]

    license_field = (meta.get("License") or "").strip()
    if license_field and license_field.upper() != "UNKNOWN":
        lines = [ln.strip() for ln in license_field.splitlines() if ln.strip()]
        first_line = lines[0] if lines else ""
        if len(lines) == 1 and len(first_line) <= 100 and any(
            m in first_line.upper() for m in (*PERMISSIVE_MARKERS, "GPL", "AGPL")
        ):
            return first_line, "metadata:License"
        matched = _match_license_text(license_field)
        if matched:
            return matched, "metadata:License-Text"
        if license_classifiers:
            return "; ".join(dict.fromkeys(license_classifiers)), "metadata:Classifier"
        if len(first_line) <= 120:
            return first_line, "metadata:License"

    if license_classifiers:
        return "; ".join(dict.fromkeys(license_classifiers)), "metadata:Classifier"

    # Fall back to inspecting license files inside .dist-info
    dist_files = dist.files or []
    for rel_path in dist_files:
        parts_lower = [p.lower() for p in rel_path.parts]
        name_lower = rel_path.name.lower()
        if "licenses" in parts_lower or name_lower.startswith(("license", "copying")):
            try:
                located = Path(dist.locate_file(rel_path))
                content = located.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            matched = _match_license_text(content)
            if matched:
                return matched, f"dist-info:{rel_path.name}"

    return "UNKNOWN", "none"


def _has_permissive_alternative(license_str: str) -> bool:
    """Return True if an OR expression includes an allowed permissive license."""
    if not re.search(r"\bOR\b", license_str, re.IGNORECASE):
        return False
    parts = re.split(r"\bOR\b", license_str, flags=re.IGNORECASE)
    for part in parts:
        part_upper = part.upper()
        if STRONG_GPL_RE.search(part_upper) and not LGPL_RE.search(part_upper):
            continue
        if any(marker in part_upper for marker in PERMISSIVE_MARKERS):
            return True
    return False


def _evaluate_package(dist: importlib_metadata.Distribution, strict_unknown: bool) -> PackageLicenseResult:
    raw_name = dist.metadata.get("Name") or dist.name or "unknown"
    norm_name = _normalize_name(raw_name)
    version = dist.version or "0.0.0"

    for banned in BANNED_PACKAGES:
        if norm_name == banned or norm_name.startswith(f"{banned}-"):
            return PackageLicenseResult(
                name=raw_name,
                version=version,
                license="BANNED",
                source="package-name",
                status="banned",
                reason=f"Explicitly banned package '{banned}'",
            )

    lic_str, source = _extract_metadata_license(dist)
    lic_upper = lic_str.upper()

    # Check for AGPL or strong GPL (excluding LGPL unless paired without permissive OR)
    scrubbed = LGPL_RE.sub("", lic_upper)
    if STRONG_GPL_RE.search(scrubbed) and not _has_permissive_alternative(lic_str):
        return PackageLicenseResult(
            name=raw_name,
            version=version,
            license=lic_str,
            source=source,
            status="copyleft",
            reason=f"Strong copyleft license detected: {lic_str}",
        )

    if lic_upper == "UNKNOWN":
        status = "unknown" if strict_unknown else "ok"
        reason = "License metadata missing and no recognizable LICENSE file found" if strict_unknown else ""
        return PackageLicenseResult(
            name=raw_name,
            version=version,
            license=lic_str,
            source=source,
            status=status,
            reason=reason,
        )

    if any(marker in lic_upper for marker in PERMISSIVE_MARKERS):
        return PackageLicenseResult(
            name=raw_name,
            version=version,
            license=lic_str,
            source=source,
            status="ok",
        )

    return PackageLicenseResult(
        name=raw_name,
        version=version,
        license=lic_str,
        source=source,
        status="unknown" if strict_unknown else "ok",
        reason=f"Unrecognized non-standard license string: {lic_str}" if strict_unknown else "",
    )


def scan_installed_licenses(strict_unknown: bool = True) -> Dict[str, object]:
    """Scan all distributions in the active Python environment and classify compliance."""
    seen: Dict[str, PackageLicenseResult] = {}
    for dist in importlib_metadata.distributions():
        res = _evaluate_package(dist, strict_unknown=strict_unknown)
        key = _normalize_name(res.name)
        if key not in seen:
            seen[key] = res

    results = sorted(seen.values(), key=lambda r: r.name.lower())
    violations = [r for r in results if r.status != "ok"]
    return {
        "total_packages": len(results),
        "compliant_count": len(results) - len(violations),
        "violations": [asdict(v) for v in violations],
        "packages": [asdict(r) for r in results],
        "clean": len(violations) == 0,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Audit .venv package licenses for MIT/Apache-2.0/BSD compliance.")
    parser.add_argument("--json", action="store_true", help="Emit full JSON audit report.")
    parser.add_argument("--verbose", action="store_true", help="List all compliant packages as well.")
    parser.add_argument(
        "--allow-unknown",
        action="store_true",
        help="Do not fail on packages whose license string is missing/unknown.",
    )
    args = parser.parse_args(argv)

    report = scan_installed_licenses(strict_unknown=not args.allow_unknown)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Scanned {report['total_packages']} installed packages.")
        if args.verbose:
            for pkg in report["packages"]:
                print(f"  [OK] {pkg['name']}=={pkg['version']} ({pkg['license']} via {pkg['source']})")
        if report["violations"]:
            print(f"FAILED: Found {len(report['violations'])} license/package policy violation(s):")
            for v in report["violations"]:
                print(f"  [FAIL] {v['name']}=={v['version']}: {v['status']} - {v['reason']}")
        else:
            print(f"PASSED: All {report['total_packages']} packages are compliant (0 GPL/AGPL, 0 banned packages).")

    return 0 if report["clean"] else 1


if __name__ == "__main__":
    sys.exit(main())
