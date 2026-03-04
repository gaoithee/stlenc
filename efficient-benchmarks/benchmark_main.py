import subprocess
import json
import pandas as pd
import os
from pathlib import Path
import sys

# directory dove vive benchmark_main.py
BASE_DIR = Path(__file__).resolve().parent

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(RESULTS_DIR, "benchmark_final_clean_pt2.csv")

import subprocess
import sys
import json
from pathlib import Path

def run_worker(script, B, N):
    script_path = Path(__file__).parent / script
    process = subprocess.Popen(
        [sys.executable, str(script_path), str(B), str(N)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    json_line = None
    for line in process.stdout:
        print(line, end="")  # log live
        line = line.strip()
        if line.startswith("RESULT_JSON:"):
            json_line = line[len("RESULT_JSON:"):].strip()

    process.wait()

    if json_line is None:
        raise RuntimeError(f"{script} non ha prodotto output JSON valido.")
    return json.loads(json_line)

def run_benchmark(B_list, N_list):

    rows = []

    for B in B_list:
        for N in N_list:

            print("\n" + "="*120)
            print(f"Running isolated benchmark | B={B} | N={N}")
            print("="*120)

            # ---- Kernel ----
            # k = run_worker("kernel_worker.py", B, N)

            # ---- Transformer ----
            tf = run_worker("transformer_worker.py", B, N)

            row = {
                "B": B,
                "N": N,

                # Kernel
                # "K_T_Emb": k["T_Emb"],
                # "K_RAM_Emb": k["RAM_Emb"],
                # "K_VRAM_Emb": k["VRAM_Emb"],
                # "K_T_Sim": k["T_Sim"],
                # "K_RAM_Sim": k["RAM_Sim"],
                # "K_VRAM_Sim": k["VRAM_Sim"],

                # Transformer
                "TF_T_Emb": tf["T_Emb"],
                "TF_RAM_Emb": tf["RAM_Emb"],
                "TF_VRAM_Emb": tf["VRAM_Emb"],
                "TF_T_Sim": tf["T_Sim"],
                "TF_RAM_Sim": tf["RAM_Sim"],
                "TF_VRAM_Sim": tf["VRAM_Sim"],
            }

            rows.append(row)

            # salvataggio incrementale (importante se crasha)
            pd.DataFrame(rows).to_csv(RESULTS_FILE, index=False)

            print(f"✔ Completed B={B}, N={N}")

    print("\nBenchmark completato.")
    return pd.DataFrame(rows)


# -----------------------------
# USA LE TUE LISTE ORIGINALI
# -----------------------------
if __name__ == "__main__":

    B_LIST = [4000]
    N_LIST = [16000]
    # B_LIST = [500, 1000, 2000, 4000]
    # N_LIST = [500, 1000, 2000, 4000, 8000, 16000]

    run_benchmark(B_LIST, N_LIST)
