from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _run(cmd: list[str], timeout: int = 600, cwd: Path | None = None) -> tuple[int, str]:
    process = subprocess.run(
        cmd,
        cwd=str(cwd or ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        text=True,
        check=False,
    )
    return process.returncode, process.stdout


def _check_env_vars() -> bool:
    required = ["API_BASE_URL", "MODEL_NAME", "HF_TOKEN"]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        print(f"[FAIL] missing env vars: {', '.join(missing)}")
        return False
    print("[PASS] required env vars are set")
    return True


def _check_openenv_validate() -> bool:
    openenv_path = shutil.which("openenv")
    if openenv_path is None:
        scripts_dir = Path(sys.executable).resolve().parent
        candidate = scripts_dir / ("openenv.exe" if os.name == "nt" else "openenv")
        if candidate.exists():
            openenv_path = str(candidate)
    if openenv_path is None:
        print("[FAIL] openenv CLI not found in PATH")
        return False
    code, output = _run([openenv_path, "validate"], timeout=300)
    if code != 0:
        print("[FAIL] openenv validate failed")
        print(output)
        return False
    print("[PASS] openenv validate")
    return True


def _check_reset_ping(port: int = 8000) -> bool:
    python_bin = sys.executable
    cmd = [python_bin, "-m", "uvicorn", "server.app:app", "--host", "127.0.0.1", "--port", str(port)]
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.time() + 30
        url = f"http://127.0.0.1:{port}/reset"
        payload = json.dumps({}).encode("utf-8")
        request = urllib.request.Request(url, data=payload, method="POST")
        request.add_header("Content-Type", "application/json")

        while time.time() < deadline:
            try:
                with urllib.request.urlopen(request, timeout=5) as response:
                    if response.status == 200:
                        print("[PASS] POST /reset returned 200")
                        return True
            except urllib.error.URLError:
                time.sleep(1)
                continue
        print("[FAIL] POST /reset did not return 200 in time")
        return False
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def _check_docker(skip: bool) -> bool:
    if skip:
        print("[SKIP] docker build check")
        return True
    docker_path = shutil.which("docker")
    if docker_path is None:
        print("[FAIL] docker not found")
        return False
    code, output = _run([docker_path, "build", "-t", "customer-support-openenv", "."], timeout=1200)
    if code != 0:
        print("[FAIL] docker build failed")
        print(output)
        return False
    print("[PASS] docker build")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Local pre-submission checks for OpenEnv benchmark")
    parser.add_argument("--skip-docker", action="store_true", help="Skip docker build check")
    args = parser.parse_args()

    checks = [
        _check_env_vars(),
        _check_openenv_validate(),
        _check_reset_ping(),
        _check_docker(args.skip_docker),
    ]
    passed = all(checks)
    print("[DONE] pre-submission checks passed" if passed else "[DONE] pre-submission checks failed")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())