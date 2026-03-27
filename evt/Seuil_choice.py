import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("/users/eleves-b/2023/abdelbar.ghassoub/Downloads/evt/garch_residuals.csv", index_col=0, parse_dates=True)

btc_losses = np.sort(df["btc_std_loss"][df["btc_loss"] > 0].values)
eth_losses = np.sort(df["eth_std_loss"][df["eth_loss"] > 0].values)

# Fonctions

def mean_excess_plot_data(losses):
    """Calcule e(u) pour chaque valeur de u = x_i"""
    losses_sorted = np.sort(losses)
    thresholds, mean_excess, ci_upper, ci_lower = [], [], [], []
    for i in range(len(losses_sorted) - 10):  # garde au moins 10 pts au-dessus
        u = losses_sorted[i]
        excesses = losses_sorted[i+1:] - u
        n_u = len(excesses)
        me = excesses.mean()
        se = excesses.std() / np.sqrt(n_u)
        thresholds.append(u)
        mean_excess.append(me)
        ci_upper.append(me + 1.96 * se)
        ci_lower.append(me - 1.96 * se)
    return np.array(thresholds), np.array(mean_excess), np.array(ci_upper), np.array(ci_lower)

def hill_estimator(losses, max_k=None):
    """Calcule l'estimateur de Hill pour k = 1, ..., max_k"""
    x = np.sort(losses)[::-1]  
    n = len(x)
    if max_k is None:
        max_k = n // 2
    ks = np.arange(1, max_k + 1)
    hill = np.array([
        np.mean(np.log(x[:k])) - np.log(x[k])
        for k in ks
    ])
    return ks, hill


fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Étape 3 — Sélection du seuil u  |  BTC & ETH 2025",
             fontsize=13, fontweight="bold")



#Calcul 
btc_thresholds, btc_me, btc_ci_u, btc_ci_l = mean_excess_plot_data(btc_losses)
eth_thresholds, eth_me, eth_ci_u, eth_ci_l = mean_excess_plot_data(eth_losses)

btc_ks, btc_hill = hill_estimator(btc_losses)
eth_ks, eth_hill = hill_estimator(eth_losses)



for row, (name, thresholds, me, ci_u, ci_l, ks, hill, losses) in enumerate([
    ("BTC", btc_thresholds, btc_me, btc_ci_u, btc_ci_l, btc_ks, btc_hill, btc_losses),
    ("ETH", eth_thresholds, eth_me, eth_ci_u, eth_ci_l, eth_ks, eth_hill, eth_losses),
]):
    q90 = np.quantile(losses, 0.90)
    q95 = np.quantile(losses, 0.95)

    # --- Mean Excess Plot ---
    ax = axes[row, 0]
    ax.plot(thresholds, me, color="steelblue", lw=1.5, label="e(u) empirique")
    ax.fill_between(thresholds, ci_l, ci_u, alpha=0.2, color="steelblue", label="IC 95%")
    ax.axvline(q90, color="green",  linestyle="--", lw=1.5,
               label=f"Quantile 90% ({q90:.4f})")
    ax.axvline(q95, color="orange", linestyle="--", lw=1.5,
               label=f"Quantile 95% ({q95:.4f})")
    ax.set_title(f"{name} — Mean Excess Plot")
    ax.set_xlabel("Seuil u")
    ax.set_ylabel("Excès moyen e(u)")
    ax.legend(fontsize=8)

    # --- Hill Plot ---
    ax = axes[row, 1]
    ax.plot(ks, hill, color="crimson", lw=1.2)
    ax.axhline(np.median(hill[5:30]), color="black", linestyle=":",
               lw=1.5, label=f"Plateau ≈ {np.median(hill[5:30]):.3f}")
    ax.set_xlim(0, min(80, len(ks)))
    ax.set_title(f"{name} — Hill Plot (estimateur de ξ)")
    ax.set_xlabel("k (nombre de valeurs extrêmes)")
    ax.set_ylabel("ξ estimé (Hill)")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("/users/eleves-b/2023/abdelbar.ghassoub/Downloads/evt/figures/evt_step3_threshold_residuals.png", dpi=150, bbox_inches="tight")
plt.show()
