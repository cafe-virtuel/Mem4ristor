import os
import sys
import subprocess
import time

def run_script(path):
    print(f"[RUN] {os.path.basename(path)}...")

    try:
        result = subprocess.run([sys.executable, path], capture_output=True, text=True, check=True)
        print(f"[OK] {os.path.basename(path)} passed.")

        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {os.path.basename(path)} failed.")

        print(e.stdout)
        print(e.stderr)
        return None

def main():
    print("[INIT] MEM4RISTOR v2.3.0 | FULL REPRODUCIBILITY SUITE")

    print("====================================================")
    
    # 1. Industrial Verification (Basic)
    cert_out = run_script("reproduction/verify_v204.py")
    
    # 2. Formal Robustness Verification (Stress Test)
    nuke_out = run_script("reproduction/nuclear_verif_v204.py")
    
    # 3. Deep Time Stability Test
    dt_out = run_script("reproduction/verify_v204_deep_time.py")
    
    # Generate Summary Report
    with open("REPRODUCIBILITY_REPORT.txt", "w") as f:
        f.write("MEM4RISTOR v2.3.0 REPRODUCIBILITY REPORT\n")
        f.write("==========================================\n")
        f.write(f"Timestamp: {time.ctime()}\n\n")
        f.write("--- PHASE 1: INDUSTRIAL VERIFICATION ---\n")
        f.write(cert_out if cert_out else "FAILED\n")
        f.write("\n--- PHASE 2: FORMAL ROBUSTNESS VERIFICATION ---\n")
        f.write(nuke_out if nuke_out else "FAILED\n")
        f.write("\n--- PHASE 3: DEEP TIME STABILITY ---\n")
        f.write(dt_out if dt_out else "FAILED\n")

    print("\n📦 All tests completed. Report generated: REPRODUCIBILITY_REPORT.txt")

if __name__ == "__main__":
    main()
