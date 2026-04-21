"""
Compatibility wrapper for the symmetric eta-window Step 2 run.

This script forwards to step2_pisr_mgn.py with:
  --eta-min -1
  --eta-max 1
  --out-subdir eta_sym
"""
import subprocess
import sys
from pathlib import Path


def main():
    script_path = Path(__file__).resolve().with_name("step2_pisr_mgn.py")
    cmd = [
        sys.executable,
        str(script_path),
        *sys.argv[1:],
        "--eta-min",
        "-1",
        "--eta-max",
        "1",
        "--out-subdir",
        "eta_sym",
    ]
    print("Forwarding to step2_pisr_mgn.py --eta-min -1 --eta-max 1 --out-subdir eta_sym")
    completed = subprocess.run(cmd, cwd=str(script_path.parent))
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
