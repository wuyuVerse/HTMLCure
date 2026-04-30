"""
htmleval.viewer — browser-based HTML preview & comparison tool.

Quick start:
    python -m htmleval.viewer

Programmatic:
    from htmleval.viewer import launch_viewer
    launch_viewer("eval_results/")
"""

from __future__ import annotations

import sys
import webbrowser
from pathlib import Path

from htmleval.viewer.scanner import ResultsScanner
from htmleval.viewer.server import ViewerServer

__all__ = ["launch_viewer"]


def launch_viewer(
    results_dir: str | Path = "eval_results",
    host: str = "127.0.0.1",
    port: int = 7860,
    open_browser: bool = True,
    log_requests: bool = False,
) -> ViewerServer:
    """
    Start the HTML viewer server and (optionally) open it in a browser.

    Args:
        results_dir:  Path to the eval_results directory (default: ./eval_results).
        host:         Bind address (default: 127.0.0.1).
        port:         Port to listen on (default: 7860).
        open_browser: Automatically open the URL in the default browser.
        log_requests: Log every HTTP request to stdout.

    Returns:
        A running ViewerServer.  Call .stop() to shut it down.
        This function blocks until Ctrl-C when called from __main__.
    """
    results_dir = Path(results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"results_dir not found: {results_dir}")

    print(f"[viewer] Scanning {results_dir} …", flush=True)
    scanner = ResultsScanner(results_dir)
    scanner.load()
    n = sum(1 for _ in scanner.query(limit=1)[1:2]) or scanner.query()[1]
    print(f"[viewer] Indexed {scanner.query()[1]} records across datasets: "
          f"{scanner.datasets}", flush=True)

    server = ViewerServer(scanner, host=host, port=port, log_requests=log_requests)
    url = server.url

    server.start()
    print(f"[viewer] Serving at  {url}", flush=True)
    print(f"[viewer] Press Ctrl-C to stop.", flush=True)

    if open_browser:
        webbrowser.open(url)

    return server
