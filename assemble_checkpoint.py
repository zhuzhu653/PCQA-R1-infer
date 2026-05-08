#!/usr/bin/env python3
"""Verify and assemble split checkpoint shards into model.safetensors."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


def parse_sha256_file(manifest_path: Path) -> dict[str, str]:
    expected: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        digest, name = stripped.split(None, 1)
        expected[name.strip()] = digest.strip().lower()
    return expected


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def find_parts(checkpoint_dir: Path) -> list[Path]:
    parts = sorted(checkpoint_dir.glob("model.safetensors.part-*"))
    if not parts:
        raise FileNotFoundError(f"No shard files found under {checkpoint_dir}")
    return parts


def verify_parts(parts: list[Path], expected_hashes: dict[str, str] | None) -> None:
    if not expected_hashes:
        return
    missing = [part.name for part in parts if part.name not in expected_hashes]
    if missing:
        raise ValueError(f"Missing SHA256 entries for: {', '.join(missing)}")

    for part in parts:
        actual = sha256_file(part)
        expected = expected_hashes[part.name]
        if actual != expected:
            raise ValueError(
                f"SHA256 mismatch for {part.name}: expected {expected}, got {actual}"
            )
        print(f"[OK] {part.name} sha256={actual}")


def assemble(parts: list[Path], output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")
    with output_path.open("wb") as out_handle:
        for part in parts:
            print(f"[MERGE] {part.name}")
            with part.open("rb") as in_handle:
                for chunk in iter(lambda: in_handle.read(1024 * 1024), b""):
                    out_handle.write(chunk)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify and assemble split checkpoint shards")
    parser.add_argument(
        "checkpoint_dir",
        type=Path,
        help="Directory containing model.safetensors.part-* files",
    )
    parser.add_argument(
        "--sha256-file",
        type=Path,
        default=Path(__file__).resolve().parent / "CHECKPOINT_210_SHA256SUMS_1GB.txt",
        help="Optional SHA256 manifest to verify part files before assembly",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for merged model.safetensors (default: <checkpoint_dir>/model.safetensors)",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip SHA256 verification even if a manifest file exists",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output model.safetensors if it already exists",
    )
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir.resolve()
    output_path = args.output.resolve() if args.output else checkpoint_dir / "model.safetensors"
    parts = find_parts(checkpoint_dir)
    expected_hashes = None
    if not args.skip_verify and args.sha256_file.exists():
        expected_hashes = parse_sha256_file(args.sha256_file)
        verify_parts(parts, expected_hashes)
    elif not args.skip_verify:
        print(f"[WARN] SHA256 manifest not found: {args.sha256_file}; continuing without verification")

    assemble(parts, output_path, overwrite=args.overwrite)
    print(f"[DONE] Wrote merged checkpoint to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())