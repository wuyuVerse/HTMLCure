"""Entry point for  python -m htmleval.viewer"""

import argparse
import signal
import sys

from htmleval.viewer import launch_viewer


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m htmleval.viewer",
        description="Launch the htmleval browser-based HTML viewer.",
    )
    parser.add_argument(
        "--results", default="eval_results",
        metavar="DIR",
        help="Path to eval_results directory (default: ./eval_results)",
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Bind address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", type=int, default=7860,
        help="Port to listen on (default: 7860)",
    )
    parser.add_argument(
        "--no-browser", action="store_true",
        help="Do not open a browser window automatically",
    )
    parser.add_argument(
        "--log", action="store_true",
        help="Log every HTTP request",
    )
    args = parser.parse_args()

    try:
        server = launch_viewer(
            results_dir   = args.results,
            host          = args.host,
            port          = args.port,
            open_browser  = not args.no_browser,
            log_requests  = args.log,
        )
        # Block until Ctrl-C
        signal.signal(signal.SIGINT, lambda *_: (server.stop(), sys.exit(0)))
        server.serve_forever()
    except FileNotFoundError as e:
        print(f"[viewer] Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[viewer] Stopped.")


if __name__ == "__main__":
    main()
