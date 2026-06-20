from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from riskops_ai.pipelines.generate_synthetic_data import generate_all
from riskops_ai.pipelines.build_marts_duckdb import build_all
from riskops_ai.rag.policy_index import build_index


def main() -> None:
    print("[1/3] Generating synthetic NBFC data...")
    counts = generate_all()
    for name, count in counts.items():
        print(f"  - {name}: {count:,}")

    print("[2/3] Building silver/gold marts and data-quality report...")
    marts = build_all()
    for name, count in sorted(marts.items()):
        print(f"  - {name}: {count:,}")

    print("[3/3] Building policy retrieval index...")
    chunks = build_index()
    print(f"  - policy chunks indexed: {chunks:,}")

    print("\nDone. Run: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
