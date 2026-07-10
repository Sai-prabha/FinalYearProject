"""
Start the Trading Model Server for live trading signal streaming.

Supports version switching via --model-version flag:
    python start_model_server.py                    # default: v4.15
    python start_model_server.py --model-version v4.16

This server:
1. Loads the XGBClassifier from models/v4_14_production/
2. Instantiates the signal generator for the selected version
3. Restores portfolio state from persisted trade history
4. Fetches 1000 historical candles from Binance REST API (warm-up)
5. Connects to Binance WebSocket for live BTC/ETH 1m klines
6. Generates signals with probability-based hysteresis + circuit breaker
7. Streams signals to the frontend via WebSocket

Features:
- Auto-restarts on crash (5-second cooldown)
- Single-instance guard: prevents duplicate servers on the same port
- Ctrl+C exits cleanly without restart
"""
import argparse
import sys
import os
import signal
import socket
import time
import atexit
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

PORT = 8888
PID_FILE = ROOT / ".model_server.pid"
RESTART_DELAY = 5  # seconds between restart attempts

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_port_in_use(port: int) -> bool:
    """Check whether a TCP port is already bound on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def is_pid_alive(pid: int) -> bool:
    """Check whether a process with the given PID is still running."""
    try:
        os.kill(pid, 0)  # signal 0 = existence check, no actual signal sent
        return True
    except (OSError, ProcessLookupError):
        return False


def check_single_instance() -> None:
    """Abort if another model server instance is already running."""
    # Check PID file
    if PID_FILE.exists():
        try:
            old_pid = int(PID_FILE.read_text().strip())
            if is_pid_alive(old_pid):
                print(f"[GUARD] Model server already running (PID {old_pid}).")
                if is_port_in_use(PORT):
                    print(f"[GUARD] Port {PORT} is in use. Exiting to avoid duplicates.")
                    sys.exit(0)
                else:
                    print(f"[GUARD] Port {PORT} is free — stale PID file. Cleaning up.")
                    PID_FILE.unlink(missing_ok=True)
            else:
                # PID file exists but process is dead — stale
                PID_FILE.unlink(missing_ok=True)
        except (ValueError, OSError):
            PID_FILE.unlink(missing_ok=True)

    # Double-check port even without PID file
    if is_port_in_use(PORT):
        print(f"[GUARD] Port {PORT} is already in use by another process. Exiting.")
        sys.exit(0)


def write_pid() -> None:
    """Write the current PID to the lock file."""
    PID_FILE.write_text(str(os.getpid()))


def cleanup_pid() -> None:
    """Remove the PID file on exit."""
    try:
        PID_FILE.unlink(missing_ok=True)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    # ── CLI argument parsing ───────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Start the Trading Model Server")
    parser.add_argument(
        "--model-version",
        type=str,
        default=os.environ.get("MODEL_VERSION", "v4.15"),
        choices=["v4.15", "v4.16", "v4.17", "v4.18"],
        help="Strategy version to run (default: v4.15)",
    )
    args = parser.parse_args()
    model_version = args.model_version

    # Set env var so model_server.py picks it up
    os.environ["MODEL_VERSION"] = model_version

    # ── Single-instance guard ──────────────────────────────────────────────
    check_single_instance()
    write_pid()
    atexit.register(cleanup_pid)

    # ── Graceful stop flag (Ctrl+C / SIGTERM) ──────────────────────────────
    _stop_requested = False

    def _handle_stop(signum, frame):
        global _stop_requested
        _stop_requested = True
        print(f"\n[SERVER] Received signal {signum} — shutting down (no restart).")

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    # ── Banner ─────────────────────────────────────────────────────────────
    print("=" * 80)
    print(f"{model_version.upper()} TRADING MODEL SERVER")
    print("=" * 80)
    print(f"\nStarting model server ({model_version}) for live trading signals...")
    print("\nServer will:")
    print(f"  1. Load XGBClassifier + {model_version} signal generator")
    print("  2. Restore portfolio from trade history (if exists)")
    print("  3. Fetch 1000 historical candles from REST API")
    print("  4. Connect to WebSocket (BTC/ETH 1m klines)")
    print("  5. Stream predictions to frontend at ws://localhost:8888/ws/signals")
    print("\nAccess:")
    print("  - API: http://localhost:8888")
    print("  - Version: http://localhost:8888/version")
    print("  - Status: http://localhost:8888/status")
    print("  - WebSocket: ws://localhost:8888/ws/signals")
    print("\nPress Ctrl+C to stop (will NOT auto-restart)")
    print("=" * 80)
    print()

    # ── Restart loop ───────────────────────────────────────────────────────
    restart_count = 0

    while not _stop_requested:
        try:
            if restart_count > 0:
                print(f"\n{'=' * 80}")
                print(f"[SERVER] Restart #{restart_count}")
                print(f"{'=' * 80}\n")

            # Reset signal handlers to default during uvicorn so it
            # can handle its own graceful shutdown, then restore ours.
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            signal.signal(signal.SIGTERM, signal.SIG_DFL)

            uvicorn.run(
                "api.model_server:app",
                host="0.0.0.0",
                port=PORT,
                log_level="info",
                reload=False,
            )

        except KeyboardInterrupt:
            # Ctrl+C during uvicorn — exit cleanly
            _stop_requested = True
            print("\n[SERVER] Stopped by user (Ctrl+C).")
            break

        except SystemExit as exc:
            # uvicorn may call sys.exit(); treat code 0 as clean stop
            if exc.code == 0:
                # Check if we caused it via Ctrl+C
                if _stop_requested:
                    break
                # Otherwise treat as unexpected clean exit → restart
            # Non-zero exit → restart

        except Exception as e:
            print(f"\n[SERVER] Crashed with error: {e}")

        # Restore our signal handlers for the cooldown period
        signal.signal(signal.SIGINT, _handle_stop)
        signal.signal(signal.SIGTERM, _handle_stop)

        if _stop_requested:
            break

        restart_count += 1
        print(f"\n[SERVER] Server exited. Restarting in {RESTART_DELAY} seconds...")
        print(f"[SERVER] Press Ctrl+C now to cancel restart.\n")

        # Interruptible sleep
        for _ in range(RESTART_DELAY):
            if _stop_requested:
                break
            time.sleep(1)

    # ── Cleanup ────────────────────────────────────────────────────────────
    cleanup_pid()
    print("\n[SERVER] Model server shut down. PID file cleaned up.")
    sys.exit(0)
