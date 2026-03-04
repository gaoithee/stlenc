# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# # --------------------------------------------------
# # Setup
# # --------------------------------------------------
# os.makedirs("figures", exist_ok=True)

# plt.rcParams.update({
#     "font.size": 13,
#     "figure.figsize": (7,5),
#     "axes.grid": True,
#     "grid.alpha": 0.35,
#     "grid.linestyle": "--",
#     "lines.linewidth": 2.2,
#     "axes.spines.top": False,
#     "axes.spines.right": False,
# })

# df = pd.read_csv("results_speedup/benchmark_pairwise_final.csv")

# # helper per tick leggibili
# def nice_ticks(vals):
#     return [f"{int(v/1000)}k" if v>=1000 else str(int(v)) for v in vals]


# # --------------------------------------------------
# # 1) Embedding time vs N (B fixed)
# # --------------------------------------------------
# B_FIXED = 2000
# sub = df[df["B"] == B_FIXED].sort_values("N")

# N_vals = sub["N"].values

# plt.figure()

# plt.plot(N_vals, sub["T_Emb_K"],
#          marker="o", markersize=7,
#          label="STL Kernel")

# plt.plot(N_vals, sub["T_Emb_TF"],
#          marker="s", markersize=7,
#          label="Neural Encoder")

# plt.xscale("log")
# plt.yscale("log")

# plt.xticks(N_vals, nice_ticks(N_vals))
# plt.xlabel("Signal resolution $N$")
# plt.ylabel("Embedding time (s)")
# plt.legend(frameon=False)

# plt.tight_layout()
# plt.savefig("figures/embedding_vs_N.png", dpi=350, bbox_inches="tight")
# plt.close()


# # --------------------------------------------------
# # 2) Kernel memory vs N
# # --------------------------------------------------
# plt.figure()

# plt.plot(N_vals, sub["M_Emb_K"],
#          marker="o", markersize=7)

# plt.xscale("log")
# plt.xticks(N_vals, nice_ticks(N_vals))

# plt.xlabel("Signal resolution $N$")
# plt.ylabel("GPU memory usage (MB)")

# plt.tight_layout()
# plt.savefig("figures/kernel_memory_vs_N.png", dpi=350, bbox_inches="tight")
# plt.close()


# # --------------------------------------------------
# # 3) Embedding time vs B (N fixed)
# # --------------------------------------------------
# N_FIXED = 16000
# subB = df[df["N"] == N_FIXED].sort_values("B")

# B_vals = subB["B"].values

# plt.figure()

# plt.plot(B_vals, subB["T_Emb_K"],
#          marker="o", markersize=7,
#          label="STL Kernel")

# plt.plot(B_vals, subB["T_Emb_TF"],
#          marker="s", markersize=7,
#          label="Neural Encoder")

# plt.xscale("log")
# plt.yscale("log")

# plt.xticks(B_vals, nice_ticks(B_vals))
# plt.xlabel("Number of formulas $B$")
# plt.ylabel("Embedding time (s)")
# plt.legend(frameon=False)

# plt.tight_layout()
# plt.savefig("figures/embedding_vs_B.png", dpi=350, bbox_inches="tight")
# plt.close()


# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib.cm as cm

# # 1. Caricamento dati
# df = pd.read_csv("./results_speedup/benchmark_pairwise_final.csv")

# # Calcoli derivati
# df["T_Tot_K"] = df["T_Emb_K"] + df["T_Sim_K"]
# df["T_Tot_TF"] = df["T_Emb_TF"] + df["T_Sim_TF"]
# df["M_Tot_K"] = df["M_Emb_K"] + df["M_Sim_K"]
# df["M_Tot_TF"] = df["M_Emb_TF"] + df["M_Sim_TF"]

# B_list = sorted(df["B"].unique())
# N_list = sorted(df["N"].unique())

# # 2. Setup Figure (Orizzontale per massima chiarezza)
# fig, axes = plt.subplots(1, 2, figsize=(18, 7)) 
# plt.style.use('seaborn-v0_8-whitegrid')

# # Gradienti: Blu per Kernel, Arancio/Marrone per Transformer (come nel tuo screen)
# colors_k = cm.Blues(np.linspace(0.5, 0.9, len(B_list)))
# colors_tf = cm.Oranges(np.linspace(0.5, 0.9, len(B_list)))

# # 3. Plotting
# for i, B in enumerate(B_list):
#     df_B = df[df["B"] == B].sort_values("N")
    
#     # --- Subplot Tempo ---
#     axes[0].plot(df_B["N"], df_B["T_Tot_K"], marker='o', markersize=4, 
#                  color=colors_k[i], label=f"Kernel B={B}", linewidth=1.5)
#     axes[0].plot(df_B["N"], df_B["T_Tot_TF"], marker='s', markersize=4, 
#                  color=colors_tf[i], label=f"TF B={B}", linewidth=1.5)

#     # --- Subplot Memoria ---
#     axes[1].plot(df_B["N"], df_B["M_Tot_K"], marker='o', markersize=4, 
#                  color=colors_k[i], label=f"Kernel B={B}", linewidth=1.5)
#     axes[1].plot(df_B["N"], df_B["M_Tot_TF"], marker='s', markersize=4, 
#                  color=colors_tf[i], label=f"TF B={B}", linewidth=1.5)

# # 4. Formattazione Professionale
# titles = ["Total Execution Time (s)", "Total Memory Usage (MB)"]

# for j, ax in enumerate(axes):
#     ax.set_xscale('log') # Aggiunto: asse X logaritmico
#     ax.set_yscale('log')
    
#     # Definiamo esplicitamente i tick della X basandoci sui tuoi valori di N
#     # Questo evita che matplotlib metta tick strani come 10^3, 10^4
#     ax.set_xticks(N_list)
#     ax.get_xaxis().set_major_formatter(plt.ScalarFormatter()) # Numeri normali, non scientifici
    
#     ax.set_xlabel("Signal Resolution (N)", fontweight='bold', fontsize=12)
#     ax.set_title(titles[j], fontsize=14, pad=15, fontweight='bold')
    
#     # Griglia migliorata per scale logaritmiche
#     ax.grid(True, which="major", ls="-", alpha=0.6)
#     ax.grid(True, which="minor", ls=":", alpha=0.3)
    
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)

# # Sostituisci la vecchia riga axes[1].legend(...) con questa:
# leg = axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), 
#                      title="Configuration", 
#                      frameon=True, fontsize=10)

# # Rendiamo il titolo della legenda grassetto manualmente per compatibilità
# leg.get_title().set_fontweight('bold')

# plt.tight_layout()

# # 5. Salvataggio
# output_path = "./results_speedup/pairwise_scaling_refined.png"
# plt.savefig(output_path, dpi=300, bbox_inches='tight')
# print(f"Grafico salvato con successo in: {output_path}")

# plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from matplotlib.ticker import ScalarFormatter

# 1. Caricamento dati
df = pd.read_csv("./results_speedup/benchmark_pairwise_final.csv")
df["T_Tot_K"] = df["T_Emb_K"] + df["T_Sim_K"]
df["T_Tot_TF"] = df["T_Emb_TF"] + df["T_Sim_TF"]

B_list = sorted(df["B"].unique())
N_list = sorted(df["N"].unique())

# 2. Setup Figure
fig, ax = plt.subplots(figsize=(12, 8))
plt.style.use('seaborn-v0_8-whitegrid')

# Gradienti di colore
colors_k = cm.Blues(np.linspace(0.4, 1.0, len(B_list)))
colors_tf = cm.Oranges(np.linspace(0.4, 1.0, len(B_list)))

# 3. Plotting
for i, B in enumerate(B_list):
    df_B = df[df["B"] == B].sort_values("N")
    
    # Kernel
    ax.plot(df_B["N"], df_B["T_Tot_K"], marker='o', markersize=5, 
            color=colors_k[i], label=f"Kernel B={B}", linewidth=1.8)
    
    # Transformer
    ax.plot(df_B["N"], df_B["T_Tot_TF"], marker='s', markersize=5, 
            color=colors_tf[i], label=f"TF B={B}", linewidth=1.8)
    
    # Area di vantaggio
    ax.fill_between(df_B["N"], df_B["T_Tot_K"], df_B["T_Tot_TF"], 
                    where=(df_B["T_Tot_K"] > df_B["T_Tot_TF"]),
                    interpolate=True, color='green', alpha=0.05)

# 4. Formattazione Assi (Log-Log)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xticks(N_list)
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_minor_formatter(plt.NullFormatter()) 

ax.set_xlabel("Signal Resolution (N)", fontweight='bold', fontsize=12)
ax.set_ylabel("Total Execution Time (s)", fontweight='bold', fontsize=12)
ax.set_title("Execution Time Comparison: Kernel vs Transformer Scaling", 
             fontsize=16, fontweight='bold', pad=20)

# 5. Legenda INTERNA
# 'ncol=2' divide le linee Kernel e TF su due colonne per risparmiare spazio verticale
leg = ax.legend(loc='upper left', fontsize=10, frameon=True, 
               framealpha=0.9, edgecolor='gray', ncol=2,
               title="System Configuration")
leg.get_title().set_fontweight('bold')

plt.tight_layout()

# 6. Salvataggio
output_path = "./results_speedup/execution_time_internal_legend.png"
plt.savefig(output_path, dpi=300)
plt.show()


# new

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from matplotlib.ticker import ScalarFormatter

# --- Dati ---
N_list = np.array([500, 1000, 2000, 4000, 8000, 16000])
B_list = [500, 1000, 2000, 4000]

# Tempi di esecuzione (s)
kernel_times = {
    500:  [1.58, 2.50, 4.38, 8.21, 16.81, 40.24],
    1000: [3.64, 6.17, 11.05, 21.57, 42.86, 98.51],
    2000: [10.54, 18.33, 33.87, 66.37, 132.29, 434.57],
    4000: [32.65, 60.11, 114.98, 227.25, 552.04, 1089.28]  # ultimo punto OOM projection
}
tf_times = {B: [t]*6 for B, t in zip(B_list, [0.73, 1.21, 2.29, 4.59])}

# Memoria (MB)
kernel_mem = {
    500:  [2059, 3061, 5051, 8955, 16995, 32442],
    1000: [3027, 4960, 8829, 16505, 32008, 62338],
    2000: [4907, 8737, 16464, 31696, 62384, 123384],
    4000: [16776, 32112, 62653, 123651, 245724, 489898]  # ultimo punto OOM
}
tf_mem_full = [2136, 2136, 2136, 2136, 2136, 2136]  # TF Full Pipeline

# --- Palette gradienti ---
colors_k = cm.Blues(np.linspace(0.4, 1.0, len(B_list)))
color_tf = cm.Oranges(0.8)

# --- Figura unica 1x2 ---
fig, axes = plt.subplots(1, 2, figsize=(22, 8))
plt.style.use('seaborn-v0_8-whitegrid')

# -------------------------
# Grafico Tempi
# -------------------------
ax = axes[0]
for i, B in enumerate(B_list):
    if B == 4000:
        # Linea normale fino al penultimo punto
        ax.plot(N_list[:-1], kernel_times[B][:-1], marker='o', markersize=5,
                color=colors_k[i], label=f"Kernel B={B}", linewidth=1.8)
        # Ultimo tratto tratteggiato (OOM projection)
        ax.plot(N_list[-2:], kernel_times[B][-2:], marker='o', markersize=5,
                color=colors_k[i], linestyle=':', linewidth=1.8)
        ax.scatter(N_list[-1], kernel_times[B][-1], marker='X', s=150, color=colors_k[i])
    else:
        ax.plot(N_list, kernel_times[B], marker='o', markersize=5,
                color=colors_k[i], label=f"Kernel B={B}", linewidth=1.8)

    # TF line
    ax.plot(N_list, tf_times[B], marker='s', markersize=5,
            color=cm.Oranges(0.4 + 0.15*i), label=f"TF B={B}", linewidth=1.8)

    # Area verde vantaggio Kernel
    ax.fill_between(N_list, kernel_times[B], tf_times[B],
                    where=(np.array(kernel_times[B]) > np.array(tf_times[B])),
                    interpolate=True, color='green', alpha=0.05)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xticks(N_list)
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_minor_formatter(plt.NullFormatter())
ax.set_xlabel("Signal Resolution (N)", fontweight='bold', fontsize=12)
ax.set_ylabel("Total Execution Time (s)", fontweight='bold', fontsize=12)
ax.set_title("Execution Time: Kernel vs Transformer", fontsize=16, fontweight='bold', pad=15)

leg = ax.legend(loc='upper left', fontsize=10, frameon=True, framealpha=0.9,
                edgecolor='gray', ncol=2, title="System Configuration")
leg.get_title().set_fontweight('bold')

# -------------------------
# Grafico Memoria
# -------------------------
ax = axes[1]
for i, B in enumerate(B_list):
    if B == 4000:
        ax.plot(N_list[:-1], kernel_mem[B][:-1], marker='o', markersize=5,
                color=colors_k[i], label=f"Kernel B={B}", linewidth=1.8)
        ax.plot(N_list[-2:], kernel_mem[B][-2:], marker='o', markersize=5,
                color=colors_k[i], linestyle=':', linewidth=1.8)
        ax.scatter(N_list[-1], kernel_mem[B][-1], marker='X', s=150, color=colors_k[i])
    else:
        ax.plot(N_list, kernel_mem[B], marker='o', markersize=5,
                color=colors_k[i], label=f"Kernel B={B}", linewidth=1.8)

# TF Full Pipeline
ax.plot(N_list, tf_mem_full, marker='s', markersize=5, color=color_tf,
        linewidth=1.8, label="TF Full Pipeline")

# Aree di vantaggio Kernel > TF
for i, B in enumerate(B_list):
    ax.fill_between(N_list, kernel_mem[B], tf_mem_full,
                    where=(np.array(kernel_mem[B]) > np.array(tf_mem_full)),
                    interpolate=True, color='green', alpha=0.05)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xticks(N_list)
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x/1024) if x>=1024 else x} {"GB" if x>=1024 else "MB"}'))
ax.yaxis.set_minor_formatter(plt.NullFormatter())
ax.set_xlabel("Signal Resolution (N)", fontweight='bold', fontsize=16)
ax.set_ylabel("Memory Usage", fontweight='bold', fontsize=16)
ax.set_title("Memory Usage: Kernel vs Transformer", fontsize=20, fontweight='bold', pad=15)

leg = ax.legend(loc='upper left', fontsize=14, frameon=True, framealpha=0.9,
                edgecolor='gray', ncol=2, title="System Configuration")
leg.get_title().set_fontweight('bold')

plt.tight_layout()

# Salvataggio ad altissima risoluzione
plt.savefig(
    "kernel_vs_transformer_high_res.png",
    dpi=800,                 # altissima risoluzione
    bbox_inches="tight",
    format="png"
)

plt.show()