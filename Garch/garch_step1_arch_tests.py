# garch_step1_arch_tests.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.graphics.tsaplots import plot_acf

DATA = "/users/eleves-b/2023/abdelbar.ghassoub/Downloads/evt/returns_btc_eth.csv"
FIGS = "/users/eleves-b/2023/abdelbar.ghassoub/Downloads/Garch/figures/"

df = pd.read_csv(DATA, index_col=0, parse_dates=True)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Tests d'effets ARCH — BTC et ETH, données horaires 2025",
             fontsize=13, fontweight="bold")

for row, (asset, ret_col) in enumerate([("BTC", "btc_return"),
                                         ("ETH", "eth_return")]):
    r = df[ret_col].dropna()

    # ACF rendements
    plot_acf(r, lags=48, ax=axes[row, 0], alpha=0.05, zero=False,
             title=f"{asset} — ACF des rendements $r_t$")
    axes[row, 0].set_xlabel("Lag (heures)")

    # ACF rendements au carré
    plot_acf(r**2, lags=48, ax=axes[row, 1], alpha=0.05, zero=False,
             color="crimson",
             title=f"{asset} — ACF de $r_t^2$ (proxy volatilité)")
    axes[row, 1].set_xlabel("Lag (heures)")

    # Test ARCH : p-values vs lag
    lags = range(1, 25)
    pvals = [het_arch(r.values, nlags=l)[1] for l in lags]
    axes[row, 2].semilogy(lags, pvals, "o-", color="darkorange", lw=2)
    axes[row, 2].axhline(0.05, color="red", ls="--", label="Seuil 5%")
    axes[row, 2].set_title(f"{asset} — Test ARCH de Engle (p-values)")
    axes[row, 2].set_xlabel("Nombre de lags")
    axes[row, 2].set_ylabel("p-value (échelle log)")
    axes[row, 2].legend()
    axes[row, 2].grid(True, alpha=0.3)

    # Stats console
    lb_r  = acorr_ljungbox(r,    lags=[10, 20], return_df=True)
    lb_r2 = acorr_ljungbox(r**2, lags=[10, 20], return_df=True)
    arch_stat, arch_p, _, _ = het_arch(r.values, nlags=10)
    print(f"\n{asset}")
    print(f"  Ljung-Box r_t   (lag 10) : p = {lb_r['lb_pvalue'].iloc[0]:.2e}")
    print(f"  Ljung-Box r_t^2 (lag 10) : p = {lb_r2['lb_pvalue'].iloc[0]:.2e}")
    print(f"  Test ARCH (lag 10)        : stat={arch_stat:.2f}, p={arch_p:.2e}")

plt.tight_layout()
plt.savefig(f"{FIGS}garch_step1_arch_tests.png", dpi=150, bbox_inches="tight")
print(f"\nFigure : {FIGS}garch_step1_arch_tests.png")