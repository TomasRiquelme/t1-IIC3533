"""
Experimento para la pregunta (i) de la Tarea 1.

Usa bs_numpy y varia SIMULTANEAMENTE:
  - p: numero de procesos (joblib.Parallel n_jobs=p)
  - t: numero de threads internos de BLAS por proceso (threadpool_limits)
con la restriccion p * t <= pmax (pmax = tus cores logicos reales).

Guarda:
  - i_results.csv   (columnas: p, t, tiempo_s)
  - i_plot.png      (T(p) para cada valor de t, con el mejor punto marcado)

Uso:
    cd t1-IIC3533   (la carpeta con variables_config.py)
    python3 exp_i.py
"""
import csv
import multiprocessing
import time

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from variables_config import returnCoeficients, returnDataMatrix, returnYVector

N_LOGICAL_CORES = multiprocessing.cpu_count()
print("N_LOGICAL_CORES:", N_LOGICAL_CORES)
B = 48

dataMatrix = returnDataMatrix()
Yvector = returnYVector()
coeficients = returnCoeficients()


def bootstrap_regression(X, y, seed, t):
    """Un resample de bootstrap, corriendo con a lo mas t threads BLAS."""
    with threadpool_limits(limits=t, user_api="blas"):
        rng = np.random.default_rng(seed)
        Nrows = X.shape[0]
        indices = rng.choice(Nrows, size=Nrows, replace=True)
        Xb = X[indices]
        yb = y[indices]
        return np.dot((np.dot((np.linalg.inv(np.dot(Xb.T, Xb))), Xb.T)), yb)


def run_once(p, t):
    print(f"  p={p:2d}  t={t:2d}  (p*t={p*t:2d}) ...", end=" ", flush=True)
    t0 = time.perf_counter()
    Parallel(n_jobs=p)(
        delayed(bootstrap_regression)(dataMatrix, Yvector, b, t)
        for b in range(B)
    )
    elapsed = time.perf_counter() - t0
    print(f"{elapsed:.2f}s")
    return elapsed


def main():
    pmax = N_LOGICAL_CORES
    print(f"Cores logicos detectados: {pmax}\n")

    combos = [(p, t) for p in range(1, pmax + 1) for t in range(1, pmax + 1)
              if p * t <= pmax]
    combos.sort()

    print(f"Se probaran {len(combos)} combinaciones (p,t) con p*t<={pmax}:")
    print(combos, "\n")

    rows = []
    for p, t in combos:
        elapsed = run_once(p, t)
        rows.append({"p": p, "t": t, "tiempo_s": round(elapsed, 3)})

    with open("i_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["p", "t", "tiempo_s"])
        writer.writeheader()
        writer.writerows(rows)
    print("\nResultados guardados en i_results.csv")

    # --- Grafico: una linea por cada valor de t, tiempo vs p ---
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ts_present = sorted(set(r["t"] for r in rows))
    cmap = plt.get_cmap("viridis")
    for i, t in enumerate(ts_present):
        sub = sorted([r for r in rows if r["t"] == t], key=lambda r: r["p"])
        # color = cmap(i / max(1, len(ts_present) - 1))
        ax.plot([r["p"] for r in sub], [r["tiempo_s"] for r in sub],
                marker="o", linewidth=2, markersize=6, label=f"t={t}")

    best = min(rows, key=lambda r: r["tiempo_s"])
    ax.scatter([best["p"]], [best["tiempo_s"]], s=220, facecolors="none",
               edgecolors="#e34948", linewidths=2.5, zorder=5,
               label=f"mejor: p={best['p']}, t={best['t']} ({best['tiempo_s']}s)")

    ax.set_xlabel("p (procesos, joblib n_jobs)")
    ax.set_ylabel("Tiempo de ejecución (s)")
    ax.set_title(f"bs_numpy: T(p,t) con p·t ≤ {pmax}")
    ax.set_xticks(range(1, pmax + 1))
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig("i_plot.png", dpi=150)
    print("Grafico guardado en i_plot.png")
    print(f"\nMejor combinacion encontrada: p={best['p']}, t={best['t']} -> {best['tiempo_s']}s")

if __name__ == "__main__":
    main()
