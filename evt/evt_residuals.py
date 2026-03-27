import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

# ============================================================
# CHARGEMENT
# ============================================================

df = pd.read_csv(
    "/users/eleves-b/2023/abdelbar.ghassoub/Downloads/evt/garch_residuals.csv",
    index_col=0, parse_dates=True
)

btc_std_losses = df["btc_std_loss"][df["btc_std_loss"] > 0].values
eth_std_losses = df["eth_std_loss"][df["eth_std_loss"] > 0].values

# ============================================================
# 1. DIAGNOSTIC
# ============================================================

print("Diagnostic des résidus standardisés GARCH")
print("=" * 45)

for name, col in [("BTC résidus z_t", "btc_std_resid"),
                  ("ETH résidus z_t", "eth_std_resid")]:
    s = df[col].dropna()
    k          = stats.kurtosis(s, fisher=True)
    sk         = stats.skew(s)
    jb_stat, jb_p = stats.jarque_bera(s)
    print(f"\n{name}")
    print(f"  Excès de kurtosis : {k:.3f}  (normale = 0)")
    print(f"  Skewness          : {sk:.3f}  (normale = 0)")
    print(f"  Test Jarque-Bera  : stat={jb_stat:.1f}, p-value={jb_p:.2e}")
    if jb_p < 0.05:
        print("  → Rejet de la normalité (p < 0.05) ✓")

# ============================================================
# 2. FIGURE : 2 lignes (BTC, ETH) x 3 colonnes
#    Col 1 : Histogramme + overlay normale
#    Col 2 : QQ-plot vs normale
#    Col 3 : Pertes classées par rang
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(
    "Diagnostic des queues — Résidus standardisés GARCH $z_t$\n"
    "BTC et ETH, données horaires 2025",
    fontsize=14, fontweight="bold"
)

BLUE  = "#2171b5"
FILL  = "#6baed6"

ASSETS = [
    ("BTC", df["btc_std_resid"].dropna().values, btc_std_losses),
    ("ETH", df["eth_std_resid"].dropna().values, eth_std_losses),
]

for row, (name, resid, losses) in enumerate(ASSETS):

    mu, sig = resid.mean(), resid.std()

    # ── Panel 1 : Histogramme + normale ──────────────────────
    ax = axes[row, 0]
    ax.hist(resid, bins=80, density=True, alpha=0.65,
            color=FILL, edgecolor="white", linewidth=0.3,
            label=r"$z_t$ empirique")
    x = np.linspace(resid.min(), resid.max(), 400)
    ax.plot(x, stats.norm.pdf(x, mu, sig),
            color="crimson", lw=2, label="Loi normale ajustée")
    ax.set_title(f"{name} — Distribution de $z_t$", fontsize=11)
    ax.set_xlabel("Résidu standardisé $z_t$")
    ax.set_ylabel("Densité")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # ── Panel 2 : QQ-plot vs normale ─────────────────────────
    ax = axes[row, 1]
    (osm, osr), (slope, intercept, _) = stats.probplot(resid, dist="norm")
    ax.plot(osm, osr, "o", markersize=2.5, alpha=0.55,
            color=BLUE, label=r"$z_t$")
    ax.plot(osm, slope * np.array(osm) + intercept,
            "-", color="crimson", lw=2, label="Diagonale théorique")
    ax.set_title(f"{name} — QQ-plot de $z_t$ vs Normale", fontsize=11)
    ax.set_xlabel("Quantiles théoriques (Normale)")
    ax.set_ylabel("Quantiles empiriques de $z_t$")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # ── Panel 3 : Pertes classées par rang ───────────────────
    ax = axes[row, 2]
    sl = np.sort(losses)[::-1]
    ax.plot(range(1, len(sl) + 1), sl, "o",
            markersize=2.5, alpha=0.65, color=BLUE,
            label=r"Pertes $\ell_t^z = -z_t$")
    q90 = np.quantile(losses, 0.90)
    q95 = np.quantile(losses, 0.95)
    ax.axhline(q95, color="crimson",  ls="--", lw=1.5,
               label=f"Seuil 95% ({q95:.2f})")
    ax.axhline(q90, color="darkorange", ls=":",  lw=1.5,
               label=f"Seuil 90% ({q90:.2f})")
    ax.set_title(f"{name} — Pertes $\\ell_t^z$ classées par rang",
                 fontsize=11)
    ax.set_xlabel("Rang")
    ax.set_ylabel(r"Perte $\ell_t^z$")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

plt.tight_layout()
out = ("/users/eleves-b/2023/abdelbar.ghassoub/Downloads/evt/"
       "figures/evt_step2_diagnostics_residuals.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"\nFigure sauvegardée : {out}")