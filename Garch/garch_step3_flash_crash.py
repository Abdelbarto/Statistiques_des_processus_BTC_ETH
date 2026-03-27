import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")

DATA        = "/users/eleves-b/2023/abdelbar.ghassoub/Downloads/evt/returns_btc_eth.csv"
RESID       = "/users/eleves-b/2023/abdelbar.ghassoub/Downloads/evt/garch_residuals.csv"
FIGS        = "/users/eleves-b/2023/abdelbar.ghassoub/Downloads/Garch/figures/"
CRASH_START = "2025-10-06"
CRASH_END   = "2025-10-14"

df  = pd.read_csv(DATA,  index_col=0, parse_dates=True)
res = pd.read_csv(RESID, index_col=0, parse_dates=True)

# ── Strip timezone sur les deux index ────────────────────
if df.index.tz  is not None: df.index  = df.index.tz_localize(None)
if res.index.tz is not None: res.index = res.index.tz_localize(None)

fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle("Analyse du pic de volatilité — Flash Crash du 10 octobre 2025",
             fontsize=13, fontweight="bold")

RED   = "#d62728"
cspan = dict(alpha=0.12, color=RED)

for col_idx, (asset, ret_col, vol_col) in enumerate([
    ("BTC", "btc_return", "btc_cond_vol"),
    ("ETH", "eth_return", "eth_cond_vol"),
]):
    r       = df[ret_col].dropna()
    σ_garch = res[vol_col].dropna() * 100
    σ_roll  = r.rolling(24).std().dropna() * 100

    # Aligner σ_roll sur l'index de σ_garch (une seule fois)
    σ_roll_aligned = σ_roll.reindex(σ_garch.index, method="nearest")

    # Aligner r sur l'index de σ_garch
    r_aligned = r.reindex(σ_garch.index, method="nearest")

    # Masque sur σ_garch (index propre, sans tz)
    mask    = (σ_garch.index >= CRASH_START) & (σ_garch.index <= CRASH_END)
    sg_zoom = σ_garch.loc[mask]
    sr_zoom = σ_roll_aligned.loc[mask]
    r_zoom  = r_aligned.loc[mask] * 100

    # ── Panel 1 : vue annuelle ──────────────────────────────
    ax = axes[0, col_idx]
    ax.plot(σ_garch.index,        σ_garch,        lw=0.7,
            color="steelblue",   label="GARCH $\\hat{\\sigma}_t$")
    ax.plot(σ_roll_aligned.index, σ_roll_aligned, lw=0.7, alpha=0.6,
            color="darkorange",  label="Vol. roulante 24h")
    ax.axvspan(pd.Timestamp(CRASH_START), pd.Timestamp(CRASH_END),
               label="Flash crash", **cspan)
    ax.set_title(f"{asset} — Vue annuelle")
    ax.set_ylabel("Volatilité (%)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # ── Panel 2 : zoom flash crash ──────────────────────────
    ax = axes[1, col_idx]
    ax.plot(sg_zoom.index, sg_zoom, lw=1.5,
            color="steelblue", label="GARCH $\\hat{\\sigma}_t$")
    ax.plot(sr_zoom.index, sr_zoom, lw=1.0, alpha=0.6,
            color="darkorange", label="Vol. roulante 24h")

    peak_t = sg_zoom.idxmax()
    peak_v = sg_zoom.max()
    ax.annotate(
        f"Pic : {peak_v:.2f}%\n{peak_t.strftime('%d oct. %Hh UTC')}",
        xy=(peak_t, peak_v),
        xytext=(peak_t, peak_v * 0.72),
        arrowprops=dict(arrowstyle="->", color=RED),
        color=RED, fontsize=9, ha="center"
    )
    ax.axvspan(pd.Timestamp("2025-10-10"), pd.Timestamp("2025-10-11"),
               alpha=0.18, color=RED, label="10 oct.")
    ax.set_title(f"{asset} — Zoom flash crash (6–14 oct.)")
    ax.set_ylabel("Volatilité (%)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    # ── Panel 3 : rendements + σ_t ─────────────────────────
    ax  = axes[2, col_idx]
    ax2 = ax.twinx()

    ax.bar(r_zoom.index, r_zoom,
           color=np.where(r_zoom < 0, RED, "steelblue"),
           width=0.03, alpha=0.7, label="Rendements (%)")
    ax2.plot(sg_zoom.index, sg_zoom, color="darkorange",
             lw=1.5, label="$\\hat{\\sigma}_t$ GARCH")

    ax.set_title(f"{asset} — Rendements & volatilité conditionnelle")
    ax.set_ylabel("Rendement (%)")
    ax2.set_ylabel("Volatilité (%)", color="darkorange")
    ax.legend(loc="upper left",  fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    σ_annuel = σ_garch.mean()
    print(f"\n{asset} — Statistiques du flash crash")
    print(f"  Pic de σ_t GARCH  : {peak_v:.4f}% le {peak_t}")
    print(f"  Moyenne annuelle  : {σ_annuel:.4f}%")
    print(f"  Ratio pic/moyenne : {peak_v/σ_annuel:.1f}x")

plt.tight_layout()
plt.savefig(f"{FIGS}garch_step3_flash_crash.png", dpi=150, bbox_inches="tight")
print(f"\nFigure : {FIGS}garch_step3_flash_crash.png")