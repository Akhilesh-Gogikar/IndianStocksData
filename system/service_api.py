"""Composable FastAPI entrypoint for Indian market-intelligence APIs.

This module now supports multiple API profiles so the repository can publish
separate market-data and agent-runtime APIs from the same codebase.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from system.api.factory import create_app, list_profiles


DEFAULT_DB_PATH = Path("./system/market_intel.db").resolve()
DEFAULT_VECTOR_INDEX_DIR = Path("./local_repository/vector_indexes").resolve()
app = create_app(DEFAULT_DB_PATH, profile_name="full", vector_index_dir=DEFAULT_VECTOR_INDEX_DIR)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run composable market-intelligence APIs")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--vector-index-dir",
        default=str(DEFAULT_VECTOR_INDEX_DIR),
        help="Local directory for TurboVec index files.",
    )
    parser.add_argument(
        "--profile",
        default="full",
        choices=list_profiles(),
        help="API surface to expose from this repository.",
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="Print the available API profiles and exit.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_profiles:
        for profile_name in list_profiles():
            print(profile_name)
        return

    runtime_app = create_app(
        Path(args.db_path).resolve(),
        profile_name=args.profile,
        vector_index_dir=Path(args.vector_index_dir).resolve(),
    )

    import uvicorn

    uvicorn.run(runtime_app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
