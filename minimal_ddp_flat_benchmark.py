from __future__ import annotations

import sys

from naive_ddp_benchmark import main


def add_default_argument(flag: str, values: list[str]) -> None:
    if flag not in sys.argv:
        sys.argv.extend([flag, *values])


if __name__ == "__main__":
    add_default_argument("--sync-modes", ["individual", "flat"])
    add_default_argument("--output", ["minimal_ddp_flat_benchmark.csv"])
    main()
