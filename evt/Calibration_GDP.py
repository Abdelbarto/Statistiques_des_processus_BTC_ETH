import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import genpareto

df = pd.read_csv("/users/eleves-b/2023/abdelbar.ghassoub/Downloads/evt/garch_residuals.csv", index_col=0, parse_dates=True)

# Seuils retenus à l'étape 3
THRESHOLDS = {"BTC": 0.0071, "ETH": 0.0114}

results = {}

for asset, col, u in [("BTC", "btc_loss", THRESHOLDS["BTC"]),
                      ("ETH", "eth_loss", THRESHOLDS["ETH"])]:

    losses = df[col][df[col] > 0].values
    n      = len(losses)
    excesses = losses[losses > u] - u
    n_u    = len(excesses)

    # ── MLE GPD ──────────────────────────────────────────────
    # genpareto(c=xi, loc=0, scale=sigma)
    xi_hat, _, sigma_hat = genpareto.fit(excesses, floc=0)

    print(f"\n{'='*45}")
    print(f"{asset} — Ajustement GPD (seuil u = {u})")
    print(f"{'='*45}")
    print(f"  n total        : {n}")
    print(f"  n_u (excès)    : {n_u}")
    print(f"  ξ̂  (shape)    : {xi_hat:.4f}")
    print(f"  σ̂  (scale)    : {sigma_hat:.6f}")
    print(f"  ξ > 0 ?        : {'✓ Queue lourde (Fréchet)' if xi_hat > 0 else '✗'}")

    # ── VaR et ES ────────────────────────────────────────────
    def var_gpd(alpha):
        return u + (sigma_hat / xi_hat) * ((n / n_u * (1 - alpha))**(-xi_hat) - 1)

    def es_gpd(alpha):
        v = var_gpd(alpha)
        return (v + sigma_hat - xi_hat * u) / (1 - xi_hat)

    print(f"\n  Mesures de risque :")
    for alpha in [0.95, 0.99, 0.999]:
        v = var_gpd(alpha)
        e = es_gpd(alpha)
        print(f"    α={alpha:.3f} → VaR = {v*100:.2f}%  |  ES = {e*100:.2f}%")

    results[asset] = {
        "xi": xi_hat, "sigma": sigma_hat,
        "u": u, "n": n, "n_u": n_u,
        "losses": losses, "excesses": excesses
    }

# ── FIGURES ──────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Étape 4 — Ajustement GPD  |  BTC & ETH 2025",
             fontsize=13, fontweight="bold")

for row, asset in enumerate(["BTC", "ETH"]):
    r = results[asset]
    xi, sigma, u = r["xi"], r["sigma"], r["u"]
    excesses = r["excesses"]
    n, n_u   = r["n"], r["n_u"]

    # --- QQ-plot GPD ---
    ax = axes[row, 0]
    emp_quantiles = np.sort(excesses)
    p = (np.arange(1, n_u + 1)) / (n_u + 1)
    theo_quantiles = genpareto.ppf(p, c=xi, loc=0, scale=sigma)
    ax.plot(theo_quantiles, emp_quantiles, "o", markersize=3,
            alpha=0.5, color="steelblue")
    lim = max(theo_quantiles.max(), emp_quantiles.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r-", lw=2, label="Diagonale parfaite")
    ax.set_title(f"{asset} — QQ-plot GPD (ξ={xi:.3f}, σ={sigma:.5f})")
    ax.set_xlabel("Quantiles théoriques GPD")
    ax.set_ylabel("Quantiles empiriques")
    ax.legend(fontsize=9)

    # --- Return Level Plot ---
    ax = axes[row, 1]
    # Périodes de retour en heures → conversion en jours
    T_hours = np.logspace(1, 5, 200)          # 10h à 100 000h
    T_days  = T_hours / 24
    # P(dépasser RL) = 1/T_hours → α = 1 - 1/T_hours
    alpha_T = 1 - 1 / T_hours
    # Garde uniquement les α valides
    valid = alpha_T > (1 - n_u / n)
    rl = np.where(
        valid,
        u + (sigma / xi) * ((n / n_u * (1 - alpha_T))**(-xi) - 1),
        np.nan
    )
    ax.semilogx(T_days, rl * 100, color="crimson", lw=2)
    # Lignes de référence
    for label, days in [("1 semaine", 7), ("1 mois", 30),
                         ("6 mois", 180), ("1 an", 365)]:
        ax.axvline(days, linestyle=":", color="gray", alpha=0.5)
        ax.text(days * 1.05, ax.get_ylim()[0] if not np.isnan(rl).all() else 0,
                label, fontsize=7, rotation=90, va="bottom", color="gray")
    ax.set_title(f"{asset} — Return Level Plot")
    ax.set_xlabel("Période de retour (jours, échelle log)")
    ax.set_ylabel("Perte extrême (%)")

plt.tight_layout()
plt.savefig("/users/eleves-b/2023/abdelbar.ghassoub/Downloads/evt/figures/evt_step4_gpd_residuals.png", dpi=150, bbox_inches="tight")
plt.show()
