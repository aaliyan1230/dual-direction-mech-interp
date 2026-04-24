#!/usr/bin/env python3
"""Bundle experiment outputs into a portable archive.

Designed for Kaggle runs where artifacts are generated inside a temporary
working directory and need to be downloaded back to a local checkout for
inspection and commit.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


DEFAULT_PATTERNS = [
    "artifacts/directions/**/*.json",
    "artifacts/cross_ablation/**/*.json",
    "artifacts/quantization/**/*.json",
    "artifacts/cross_model/**/*.json",
    "artifacts/cross_model/**/*.csv",
    "artifacts/cross_model/**/*.pdf",
    "artifacts/cross_model/**/*.png",
    "artifacts/figures/**/*.json",
    "artifacts/figures/**/*.csv",
    "artifacts/figures/**/*.pdf",
    "artifacts/figures/**/*.png",
    "artifacts/figures/**/*.svg",
    "configs/**/*.yaml",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bundle run outputs for download")
    parser.add_argument(
        "--output",
        required=True,
        help="Output archive path, for example /kaggle/working/ddmi_export.zip",
    )
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Repository root to bundle from",
    )
    parser.add_argument(
        "--include-pattern",
        action="append",
        default=[],
        help="Additional glob pattern relative to repo root",
    )
    parser.add_argument(
        "--include-path",
        action="append",
        default=[],
        help="Additional file or directory path relative to repo root",
    )
    parser.add_argument(
        "--include-paper",
        action="store_true",
        help="Include paper/main.tex if present",
    )
    parser.add_argument(
        "--label",
        default="",
        help="Optional run label stored in the manifest",
    )
    return parser.parse_args()


def iter_pattern_matches(repo_root: Path, patterns: Iterable[str]) -> list[Path]:
    matches: set[Path] = set()
    for pattern in patterns:
        matches.update(path for path in repo_root.glob(pattern) if path.is_file())
    return sorted(matches)


def iter_extra_paths(repo_root: Path, include_paths: Iterable[str]) -> list[Path]:
    matches: set[Path] = set()
    for rel_path in include_paths:
        candidate = (repo_root / rel_path).resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Requested path does not exist: {rel_path}")
        if not candidate.is_relative_to(repo_root):
            raise ValueError(f"Requested path escapes repo root: {rel_path}")
        if candidate.is_dir():
            matches.update(path for path in candidate.rglob("*") if path.is_file())
        else:
            matches.add(candidate)
    return sorted(matches)


def resolve_git_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def build_manifest(
    repo_root: Path,
    files: list[Path],
    patterns: list[str],
    include_paths: list[str],
    label: str,
) -> dict[str, object]:
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "label": label,
        "repo_root": str(repo_root),
        "git_commit": resolve_git_commit(repo_root),
        "requested_patterns": patterns,
        "requested_paths": include_paths,
        "files": [str(path.relative_to(repo_root)) for path in files],
    }


def write_archive(
    output_path: Path,
    repo_root: Path,
    files: list[Path],
    manifest: dict[str, object],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with ZipFile(output_path, mode="w", compression=ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, arcname=str(path.relative_to(repo_root)))
        archive.writestr(
            "manifest.json",
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_path = Path(args.output).resolve()

    patterns = [*DEFAULT_PATTERNS, *args.include_pattern]
    include_paths = list(args.include_path)
    if args.include_paper:
        include_paths.append("paper/main.tex")

    files = iter_pattern_matches(repo_root, patterns)
    files.extend(iter_extra_paths(repo_root, include_paths))
    files = sorted(set(files))

    if not files:
        raise SystemExit("No files matched the requested bundle contents.")

    manifest = build_manifest(repo_root, files, patterns, include_paths, args.label)
    write_archive(output_path, repo_root, files, manifest)

    print(f"Wrote {len(files)} files to {output_path}")


if __name__ == "__main__":
    main()
