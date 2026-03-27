from __future__ import annotations

import subprocess
import sys
import time


def _spawn_server(app_target: str, port: int) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        app_target,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--reload",
    ]
    return subprocess.Popen(cmd)


def main() -> None:
    public_proc = _spawn_server("app.main_public:app", 8000)
    admin_proc = _spawn_server("app.main_admin:app", 8001)

    print("Started local dual-server mode for testing.")
    print("Public docs: http://127.0.0.1:8000/docs")
    print("Admin docs:  http://127.0.0.1:8001/docs")
    print("Press Ctrl+C once to stop both servers.")

    try:
        while True:
            if public_proc.poll() is not None:
                raise RuntimeError("Public server exited unexpectedly.")
            if admin_proc.poll() is not None:
                raise RuntimeError("Admin server exited unexpectedly.")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Stopping both servers...")
    finally:
        for proc in (public_proc, admin_proc):
            if proc.poll() is None:
                proc.terminate()
        for proc in (public_proc, admin_proc):
            if proc.poll() is None:
                proc.wait(timeout=10)


if __name__ == "__main__":
    main()
