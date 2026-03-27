import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

df = pd.read_csv("/users/eleves-b/2023/abdelbar.ghassoub/Downloads/evt/returns_btc_eth.csv", index_col=0, parse_dates=True)

# On garde que les pertes positives
btc_losses = df["btc_loss"][df["btc_loss"] > 0].values
eth_losses = df["eth_loss"][df["eth_loss"] > 0].values

#1-Kurtosis
print("Diagnostic de la queue :")
print("_"*30)
print("\n")

for name, series in [("BTC returns", df["btc_return"]), ("ETH returns", df["eth_return"])]:
    k = stats.kurtosis(series, fisher=True)
    sk = stats.skew(series)
    jb_stat, jb_p = stats.jarque_bera(series)
    print(f"\n{name}")
    print(f"  Excès de kurtosis : {k:.3f}  (normale = 0)")
    print(f"  Skewness          : {sk:.3f}  (normale = 0)")
    print(f"  Test Jarque-Bera  : stat={jb_stat:.1f}, p-value={jb_p:.2e}")
    if jb_p < 0.05:
        print("  → Rejet de la normalité (p < 0.05) ✓")
        

#2-Figure complète 
fig,axes= plt.subplots(2,3, figsize=(18,10))
fig.suptitle("Diagnostic des queues — BTC et ETH 2025", fontsize=14, fontweight="bold")

for row, (name, ret, losses) in enumerate([
    ("BTC", df["btc_return"].dropna().values, btc_losses),
    ("ETH", df["eth_return"].dropna().values, eth_losses)
]):
    mu, sigma = ret.mean(), ret.std()
    # -- Histogramme + overlay normale ---
    ax = axes[row, 0]
    ax.hist(ret, bins=40, density=True, alpha=0.6, color="steelblue", label="Empirique")
    x = np.linspace(ret.min(), ret.max(), 300)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), "r-", lw=2, label="Loi normale")
    ax.set_title(f"{name} — Distribution des rendements")
    ax.set_xlabel("Log-rendement")
    ax.set_ylabel("Densité")
    ax.legend()

    # --- QQ-plot vs Normale ---
    ax = axes[row, 1]
    (osm, osr), (slope, intercept, _) = stats.probplot(ret, dist="norm")
    ax.plot(osm, osr, "o", markersize=3, alpha=0.6, color="steelblue")
    ax.plot(osm, slope * np.array(osm) + intercept, "r-", lw=2)
    ax.set_title(f"{name} — QQ-plot vs Normale")
    ax.set_xlabel("Quantiles théoriques (Normale)")
    ax.set_ylabel("Quantiles empiriques")

    # --- Zoom sur la queue gauche (pertes) ---
    ax = axes[row, 2]
    sorted_losses = np.sort(losses)[::-1]  # Pertes triées décroissantes
    ax.plot(range(1, len(sorted_losses)+1), sorted_losses, "o",
            markersize=3, alpha=0.7, color="crimson")
    ax.axhline(y=np.quantile(losses, 0.95), color="orange",
               linestyle="--", label="Seuil 95%")
    ax.axhline(y=np.quantile(losses, 0.90), color="green",
               linestyle="--", label="Seuil 90%")
    ax.set_title(f"{name} — Pertes classées (queue droite)")
    ax.set_xlabel("Rang")
    ax.set_ylabel("Perte")
    ax.legend()

plt.tight_layout()
plt.savefig("/users/eleves-b/2023/abdelbar.ghassoub/Downloads/evt/figures/evt_step2_diagnostics.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nFigure sauvegardée : evt_step2_diagnostics.png")
