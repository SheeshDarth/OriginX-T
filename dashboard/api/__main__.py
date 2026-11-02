import argparse
from pathlib import Path

from .app import create_app, openapi_json

SPEC = Path(__file__).with_name("openapi.json")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m dashboard.api", description="Serve the ORIGIN-T API.")
    parser.add_argument("--openapi", action="store_true", help=f"write {SPEC.name} from the code and exit")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.openapi:
        SPEC.write_text(openapi_json(), encoding="utf-8", newline="\n")
        print(f"wrote {SPEC}")
    else:
        import uvicorn

        uvicorn.run(create_app(), host=args.host, port=args.port)


# Guarded: on Windows (spawn), ProcessPoolExecutor workers re-import this module as
# __mp_main__; unguarded, each one started a second server and the audit's leak pool hung.
if __name__ == "__main__":
    main()
