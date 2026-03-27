import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import genpareto, kstest

df = pd.read_csv("/users/eleves-b/2023/abdelbar.ghassoub/Downloads/evt/garch_residuals.csv", index_col=0, parse_dates=True)

THRESHOLDS = {"BTC": 0.0071, "ETH": 0.0114}
ASSETS = [("BTC", "btc_loss"), ("ETH", "eth_loss")]

# ── Ré-estimation GPD ────────────────────────────────────────
results = {}
for asset, col in ASSETS:
    u      = THRESHOLDS[asset]
    losses = df[col][df[col] > 0].values
    n      = len(losses)
    exc    = losses[losses > u] - u
    n_u    = len(exc)
    xi, _, sigma = genpareto.fit(exc, floc=0)
    results[asset] = dict(xi=xi, sigma=sigma, u=u, n=n, n_u=n_u,
                          losses=losses, exc=exc)

# ════════════════════════════════════════════════════════════
# ÉTAPE 5 — VALIDATION
# ════════════════════════════════════════════════════════════

print("=" * 55)
print("ÉTAPE 5 — VALIDATION STATISTIQUE")
print("=" * 55)

for asset, _ in ASSETS:
    r = results[asset]
    xi, sigma, exc = r["xi"], r["sigma"], r["exc"]

    # Test KS
    ks_stat, ks_p = kstest(exc, "genpareto",
                           args=(xi, 0, sigma))
    print(f"\n{asset}")
    print(f"  Test KS  : stat = {ks_stat:.4f}, p-value = {ks_p:.4f}")
    print(f"  {'→ Non-rejet de GPD ✓ (p > 0.05)' if ks_p > 0.05 else '→ Rejet GPD ✗ (p < 0.05)'}")

# ── Figure validation ────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle("Étape 5 — Validation GPD  |  BTC & ETH 2025",
             fontsize=13, fontweight="bold")

for row, (asset, _) in enumerate(ASSETS):
    r = results[asset]
    xi, sigma, exc, n_u = r["xi"], r["sigma"], r["exc"], r["n_u"]
    p_emp  = np.arange(1, n_u + 1) / (n_u + 1)
    exc_s  = np.sort(exc)
    p_theo = genpareto.cdf(exc_s, c=xi, loc=0, scale=sigma)
    q_theo = genpareto.ppf(p_emp, c=xi, loc=0, scale=sigma)

    # QQ-plot
    ax = axes[row, 0]
    ax.plot(q_theo, exc_s, "o", ms=3, alpha=0.5, color="steelblue")
    lim = max(q_theo.max(), exc_s.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r-", lw=2)
    ax.set_title(f"{asset} — QQ-plot GPD")
    ax.set_xlabel("Quantiles théoriques GPD")
    ax.set_ylabel("Quantiles empiriques")

    # PP-plot
    ax = axes[row, 1]
    ax.plot(p_theo, p_emp, "o", ms=3, alpha=0.5, color="darkorange")
    ax.plot([0, 1], [0, 1], "r-", lw=2)
    ax.set_title(f"{asset} — PP-plot GPD")
    ax.set_xlabel("Probabilités théoriques GPD")
    ax.set_ylabel("Probabilités empiriques")

plt.tight_layout()
plt.savefig("/users/eleves-b/2023/abdelbar.ghassoub/Downloads/evt/figures/evt_step5_validation_residuals.png", dpi=150, bbox_inches="tight")
plt.show()


# ════════════════════════════════════════════════════════════
# ÉTAPE 6 — VaR, ES ET SYNTHÈSE FINALE
# ════════════════════════════════════════════════════════════

def var_gpd(alpha, xi, sigma, u, n, n_u):
    return u + (sigma / xi) * ((n / n_u * (1 - alpha))**(-xi) - 1)

def es_gpd(alpha, xi, sigma, u, n, n_u):
    v = var_gpd(alpha, xi, sigma, u, n, n_u)
    return (v + sigma - xi * u) / (1 - xi)

print("\n" + "=" * 55)
print("ÉTAPE 6 — MESURES DE RISQUE FINALES")
print("=" * 55)

alphas = [0.95, 0.99, 0.999, 0.9999]
risk_table = {}

for asset, _ in ASSETS:
    r = results[asset]
    xi, sigma, u, n, n_u = r["xi"], r["sigma"], r["u"], r["n"], r["n_u"]
    print(f"\n{asset}  (ξ={xi:.4f}, σ={sigma:.6f}, u={u})")
    print(f"  {'Alpha':<10} {'VaR (%)':>10} {'ES (%)':>10}")
    print(f"  {'-'*32}")
    risk_table[asset] = {}
    for alpha in alphas:
        v = var_gpd(alpha, xi, sigma, u, n, n_u) * 100
        e = es_gpd(alpha, xi, sigma, u, n, n_u) * 100
        print(f"  {alpha:<10.4f} {v:>10.2f} {e:>10.2f}")
        risk_table[asset][alpha] = (v, e)


# ── Figure synthèse finale ───────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Étape 6 — Synthèse des risques extrêmes  |  BTC & ETH 2025",
             fontsize=13, fontweight="bold")

# --- Densités empiriques + queue GPD ajustée ---
ax = axes[0]
for asset, col, color in [("BTC","btc_loss","steelblue"),
                           ("ETH","eth_loss","crimson")]:
    r = results[asset]
    losses = r["losses"]
    u, xi, sigma = r["u"], r["xi"], r["sigma"]
    x = np.linspace(0, losses.max(), 500)
    # Densité empirique (KDE)
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(losses)
    ax.plot(x, kde(x), color=color, lw=2, label=f"{asset} empirique")
    # Densité GPD dans la queue
    x_tail = np.linspace(u, losses.max(), 300)
    gpd_density = genpareto.pdf(x_tail - u, c=xi, loc=0, scale=sigma)
    # Normalisation pour superposition
    gpd_density *= (losses > u).mean()
    ax.plot(x_tail, gpd_density, "--", color=color, lw=1.5,
            label=f"{asset} GPD ajustée")
ax.set_xlim(0, 0.06)
ax.set_title("Densités — Queue droite (pertes)")
ax.set_xlabel("Perte horaire")
ax.set_ylabel("Densité")
ax.legend(fontsize=8)

# --- Return Level Plot comparatif BTC vs ETH ---
ax = axes[1]
T_hours = np.logspace(np.log10(50), 5, 300)
T_days  = T_hours / 24
for asset, color in [("BTC", "steelblue"), ("ETH", "crimson")]:
    r = results[asset]
    xi, sigma, u, n, n_u = r["xi"], r["sigma"], r["u"], r["n"], r["n_u"]
    alpha_T = 1 - 1 / T_hours
    valid   = alpha_T > (1 - n_u / n)
    rl = np.where(valid,
                  u + (sigma/xi)*((n/n_u*(1-alpha_T))**(-xi) - 1),
                  np.nan) * 100
    ax.semilogx(T_days, rl, color=color, lw=2, label=asset)

# Événement flash crash 10 oct 2025
ax.axhline(8.93, color="steelblue", linestyle=":", alpha=0.6,
           label="BTC pire jour (-8.93%)")
ax.axhline(15.85, color="crimson", linestyle=":", alpha=0.6,
           label="ETH pire jour (-15.85%)")
ax.set_title("Return Level Plot — BTC vs ETH")
ax.set_xlabel("Période de retour (jours, log)")
ax.set_ylabel("Perte extrême (%)")
ax.legend(fontsize=8)

# --- Barres comparatives VaR et ES ---
ax = axes[2]
x     = np.arange(len(alphas))
width = 0.2
labels = [f"{a*100:.1f}%" for a in alphas]
for i, (asset, color) in enumerate([("BTC","steelblue"),("ETH","crimson")]):
    vars_ = [risk_table[asset][a][0] for a in alphas]
    ess_  = [risk_table[asset][a][1] for a in alphas]
    offset = (i - 0.5) * width * 2
    bars = ax.bar(x + offset,          vars_, width, color=color,
                  alpha=0.8, label=f"{asset} VaR")
    ax.bar(x + offset + width, ess_,  width, color=color,
           alpha=0.4, label=f"{asset} ES", hatch="//")

ax.set_xticks(x + width * 0.5)
ax.set_xticklabels(labels)
ax.set_title("VaR et ES par niveau de confiance")
ax.set_xlabel("Niveau de confiance α")
ax.set_ylabel("Perte (%)")
ax.legend(fontsize=7, ncol=2)

plt.tight_layout()
plt.savefig("/users/eleves-b/2023/abdelbar.ghassoub/Downloads/evt/figures/evt_step6_synthesis_residuals.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nFigures sauvegardées : /users/eleves-b/2023/abdelbar.ghassoub/Downloads/evt/figures/evt_step5_validation.png, /users/eleves-b/2023/abdelbar.ghassoub/Downloads/evt/figures/evt_step6_synthesis.png")