#!/usr/bin/env python3
"""Generate deterministic artifact snapshots for CI regression checks."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from seed_factory import DeterministicContext, SeedFactory

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKSUM_PATH = REPO_ROOT / "config" / "metadata_checksums.json"


def load_checksums() -> Dict[str, str]:
    with CHECKSUM_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def generate_payload(seed: int, checksums: Dict[str, str]) -> Dict[str, Any]:
    random_values = [random.random() for _ in range(5)]
    return {
        "seed": seed,
        "checksums": checksums,
        "random_probe": random_values,
    }


def main(output_dir: Path) -> None:
    checksums = load_checksums()
    factory = SeedFactory()
    seed = factory.create_deterministic_seed(
        correlation_id="ci-artifacts",
        file_checksums=checksums,
        context={"variant": "baseline"},
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "deterministic_snapshot.json"

    with DeterministicContext("ci-artifacts", file_checksums=checksums, context={"variant": "baseline"}) as ctx_seed:
        payload = generate_payload(ctx_seed, checksums)

    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Directory where artifacts will be written")
    args = parser.parse_args()

    main(args.output)
