import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
import warnings
warnings.filterwarnings("ignore")

DATA = "/users/eleves-b/2023/abdelbar.ghassoub/Downloads/evt/returns_btc_eth.csv"
FIGS = "/users/eleves-b/2023/abdelbar.ghassoub/Downloads/Garch/figures/"

df = pd.read_csv(DATA, index_col=0, parse_dates=True)

specs = {
    "GARCH(1,1)"    : dict(vol="GARCH", p=1, o=0, q=1),
    "GJR-GARCH(1,1)": dict(vol="GARCH", p=1, o=1, q=1),
    "EGARCH(1,1)"   : dict(vol="EGARCH", p=1, o=1, q=1),
}

results = {}

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Estimation GARCH — BTC et ETH, données horaires 2025",
             fontsize=13, fontweight="bold")

print(f"{'Actif':<6} {'Modèle':<18} {'AIC':>10} {'BIC':>10} "
      f"{'alpha':>8} {'gamma':>8} {'beta':>8} {'nu':>6}")
print("-" * 78)

for row, (asset, col) in enumerate([("BTC", "btc_return"),
                                      ("ETH", "eth_return")]):
    r = df[col].dropna() * 100  # en %
    best_aic, best_name, best_res = np.inf, None, None

    for name, kwargs in specs.items():
        m   = arch_model(r, mean="AR", lags=1,
                         dist="studentst", **kwargs)
        res = m.fit(disp="off", options={"maxiter": 500})
        aic, bic = res.aic, res.bic
        p = res.params
        gamma = p.get("gamma[1]", 0)
        print(f"{asset:<6} {name:<18} {aic:>10.1f} {bic:>10.1f} "
              f"{p.get('alpha[1]',0):>8.4f} {gamma:>8.4f} "
              f"{p.get('beta[1]',0):>8.4f} {p.get('nu',0):>6.2f}")
        if aic < best_aic:
            best_aic, best_name, best_res = aic, name, res

    results[asset] = best_res
    print(f"  → Meilleur modèle {asset} : {best_name} (AIC={best_aic:.1f})\n")

    # Volatilité conditionnelle
    ax = axes[row, 0]
    cond_vol = best_res.conditional_volatility
    ax.plot(cond_vol.index, cond_vol, color="steelblue", lw=0.8)
    ax.set_title(f"{asset} — Volatilité conditionnelle $\\hat{{\\sigma}}_t$ "
                 f"({best_name})")
    ax.set_ylabel("Volatilité (%)")
    ax.grid(True, alpha=0.25)

    # Distribution résidus standardisés
    ax = axes[row, 1]
    z = best_res.std_resid.dropna()
    ax.hist(z, bins=80, density=True, alpha=0.6,
            color="steelblue", label="Résidus $z_t$")
    from scipy import stats
    x = np.linspace(z.min(), z.max(), 300)
    nu = best_res.params.get("nu", 5)
    ax.plot(x, stats.t.pdf(x, df=nu), "r-", lw=2,
            label=f"Student-t (ν={nu:.1f})")
    ax.plot(x, stats.norm.pdf(x), "k--", lw=1.5,
            label="Normale")
    ax.set_title(f"{asset} — Distribution des résidus $z_t$")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig(f"{FIGS}garch_step2_estimation.png", dpi=150, bbox_inches="tight")
print(f"Figure : {FIGS}garch_step2_estimation.png")