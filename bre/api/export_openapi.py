"""Write ``api/openapi.json`` from the FastAPI app (build-time; no server, no database needed).

    python api/export_openapi.py            # from bre/, with src/ and . on the path
    make api                                # regenerates it before starting uvicorn

``tests/test_api.py`` checks that the committed file equals ``app.openapi()``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from api.main import app  # noqa: E402

OUT = HERE / "openapi.json"


def spec() -> dict:
    return app.openapi()


def main() -> int:
    OUT.write_text(json.dumps(spec(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(spec()['paths'])} paths)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
