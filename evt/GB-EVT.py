# evt_gbevt.py
# GB-EVT conditionnel (Velthoen et al., 2023)
# Placer dans evt/ -- lire returns_btc_eth.csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from scipy.stats import genpareto
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

DATA_PATH   = "/users/eleves-b/2023/abdelbar.ghassoub/Downloads/evt/returns_btc_eth.csv"
FIGURES_DIR = "/users/eleves-b/2023/abdelbar.ghassoub/Downloads/evt/figures/"

STATIC = {
    "BTC": {"xi": 0.1827, "sigma": 0.003913, "u": 0.0071, "n_u": 429},
    "ETH": {"xi": 0.1754, "sigma": 0.006543, "u": 0.0114, "n_u": 428},
}

ALPHA_0     = 0.90
ALPHA       = 0.999
B_ITER      = 150
LR_XI       = 0.008
LR_SIG      = 0.04
CRASH_START = "2025-10-06"
CRASH_END   = "2025-10-14"

# ============================================================
# 1. CHARGEMENT
# ============================================================

def load_returns(path):
    df = pd.read_csv(path)

    # Index datetime
    for col in ["datetime", "date", "time", "timestamp", "Date", "Datetime"]:
        if col in df.columns:
            df.index = pd.to_datetime(df[col], utc=True, errors="coerce")
            df = df.drop(columns=[col])
            break
    else:
        df.index = pd.to_datetime(df.iloc[:, 0], utc=True, errors="coerce")
        df = df.iloc[:, 1:]

    df = df.sort_index().dropna(how="all")

    # Colonnes returns
    btc_col = next((c for c in df.columns if "btc" in c.lower() and "return" in c.lower()), None)
    eth_col = next((c for c in df.columns if "eth" in c.lower() and "return" in c.lower()), None)
    if btc_col is None:
        btc_col = next(c for c in df.columns if "btc" in c.lower())
    if eth_col is None:
        eth_col = next(c for c in df.columns if "eth" in c.lower())

    print(f"  Colonnes : BTC='{btc_col}', ETH='{eth_col}'")

    out = pd.DataFrame(index=df.index)
    out["btc_return"] = pd.to_numeric(df[btc_col], errors="coerce")
    out["eth_return"] = pd.to_numeric(df[eth_col], errors="coerce")
    out["btc_loss"]   = -out["btc_return"]
    out["eth_loss"]   = -out["eth_return"]

    # Volume optionnel
    btc_vol = next((c for c in df.columns if "btc" in c.lower() and "vol" in c.lower()), None)
    eth_vol = next((c for c in df.columns if "eth" in c.lower() and "vol" in c.lower()), None)
    if btc_vol:
        out["btc_volume"] = pd.to_numeric(df[btc_vol], errors="coerce")
    if eth_vol:
        out["eth_volume"] = pd.to_numeric(df[eth_vol], errors="coerce")

    return out.dropna(subset=["btc_loss", "eth_loss"])


# ============================================================
# 2. FEATURES
# ============================================================

def build_features(df, asset="btc"):
    L = pd.to_numeric(df[f"{asset}_loss"],   errors="coerce")
    R = pd.to_numeric(df[f"{asset}_return"], errors="coerce")

    # Strip timezone pour eviter desalignements
    if hasattr(L.index, "tz") and L.index.tz is not None:
        L.index = L.index.tz_localize(None)
        R.index = R.index.tz_localize(None)

    cols = {"loss": L, "ret": R}

    # Volatilites roulantes
    for h in [5, 24, 168]:
        cols[f"vol_{h}h"] = L.rolling(h, min_periods=1).std()

    # Ratios volatilite
    cols["vol_r_5_24"]   = cols["vol_5h"]  / (cols["vol_24h"]  + 1e-10)
    cols["vol_r_24_168"] = cols["vol_24h"] / (cols["vol_168h"] + 1e-10)

    # Retards
    for lag in [1, 2, 3, 6, 12, 24]:
        cols[f"lag_{lag}h"] = L.shift(lag)

    # Momentum
    for h in [6, 24, 72]:
        cols[f"mom_{h}h"] = L - L.shift(h)

    # Drawdown et cumret
    cols["dd_24h"]     = L.rolling(24, min_periods=1).max()
    cols["dd_72h"]     = L.rolling(72, min_periods=1).max()
    cols["cumret_6h"]  = R.rolling(6,  min_periods=1).sum()
    cols["cumret_24h"] = R.rolling(24, min_periods=1).sum()

    # Volume
    vol_col = f"{asset}_volume"
    if vol_col in df.columns:
        V = pd.to_numeric(df[vol_col], errors="coerce")
        if hasattr(V.index, "tz") and V.index.tz is not None:
            V.index = V.index.tz_localize(None)
        cols["vol_ratio"] = V / (V.rolling(24, min_periods=1).mean() + 1e-10)

    out = pd.DataFrame(cols)

    # Supprimer les NaN des retards (les 24 premieres lignes)
    out = out.dropna(subset=["lag_24h", "mom_72h"])

    # Verifier et forcer zero NaN avec ffill/bfill sur les residuels
    out = out.ffill().bfill()

    # Verification finale stricte
    n_nan = out.isna().sum().sum()
    if n_nan > 0:
        print(f"  WARNING : {n_nan} NaN residuels, suppression des lignes")
        out = out.dropna()

    feat_cols = [c for c in out.columns if c not in ("loss", "ret")]
    print(f"  Features : {len(out)} obs x {len(feat_cols)} covariables")
    return out


# ============================================================
# 3. GB-EVT
# ============================================================

class GBEVT:
    """
    Velthoen, Dombry, Cai, Engelke (2023).
    Gradient boosting for extreme quantile regression.
    Extremes 26, 639-667.
    """

    def __init__(self, alpha_0=0.90, alpha=0.999,
                 B=150, lr_xi=0.008, lr_sigma=0.04):
        self.alpha_0  = alpha_0
        self.alpha    = alpha
        self.B        = B
        self.lr_xi    = lr_xi
        self.lr_sigma = lr_sigma

    @staticmethod
    def _nll(xi, lsig, z):
        sig = np.exp(lsig)
        arg = np.maximum(1.0 + xi * z / sig, 1e-10)
        return np.mean(np.log(sig) + (1.0 + 1.0 / xi) * np.log(arg))

    @staticmethod
    def _grad(xi, lsig, z):
        sig = np.exp(lsig)
        arg = np.maximum(1.0 + xi * z / sig, 1e-10)
        g_s = 1.0 - (1.0 + 1.0 / xi) * (xi * z / sig) / arg
        g_x = -(1.0 / xi**2) * np.log(arg) + (1.0 + 1.0 / xi) * z / (sig * arg)
        return g_x, g_s

    def fit(self, X, Y, verbose=True):
        # Securite NaN absolue avant tout
        valid = np.isfinite(X).all(axis=1) & np.isfinite(Y)
        X, Y  = X[valid], Y[valid]
        print(f"  Observations valides : {len(Y)}")

        self.scaler_ = StandardScaler()
        Xs = self.scaler_.fit_transform(X)

        # ── Etape 1 : seuil conditionnel q_{x, alpha_0} ──────────
        if verbose: print("  [1/4] Quantile conditionnel initial...")
        gbq = GradientBoostingRegressor(
            loss="quantile", alpha=self.alpha_0,
            n_estimators=200, learning_rate=0.05,
            max_depth=3, subsample=0.8, random_state=42
        )
        gbq.fit(Xs, Y)
        self.gbq_    = gbq
        self.q_init_ = gbq.predict(Xs)

        # ── Etape 2 : excedances conditionnelles Z_i ─────────────
        self.mask_ = Y > self.q_init_
        Z          = Y[self.mask_] - self.q_init_[self.mask_]
        Xe         = Xs[self.mask_]
        if verbose:
            print(f"  [2/4] Excedances : {self.mask_.sum()} / {len(Y)} "
                  f"({self.mask_.mean()*100:.1f}%)")

        # ── Etape 3 : GPD initiale ────────────────────────────────
        self.xi0_, _, self.s0_ = genpareto.fit(Z, floc=0)
        if verbose:
            print(f"  [3/4] GPD init : xi={self.xi0_:.4f}, sigma={self.s0_:.6f}")

        # ── Etape 4 : gradient boosting sur xi(x) et sigma(x) ────
        xi_v  = np.full(len(Z), self.xi0_)
        ls_v  = np.full(len(Z), np.log(self.s0_))
        self.nll_hist_ = []

        if verbose: print(f"  [4/4] Boosting ({self.B} iters)...")
        for b in range(self.B):
            gx, gs = self._grad(xi_v, ls_v, Z)

            tx = GradientBoostingRegressor(
                n_estimators=1, max_depth=2,
                subsample=0.8, learning_rate=1.0, random_state=b)
            ts = GradientBoostingRegressor(
                n_estimators=1, max_depth=2,
                subsample=0.8, learning_rate=1.0, random_state=b + 9999)
            tx.fit(Xe, -gx)
            ts.fit(Xe, -gs)

            xi_v = np.clip(xi_v + self.lr_xi    * tx.predict(Xe), -0.49, 2.0)
            ls_v = np.clip(ls_v + self.lr_sigma * ts.predict(Xe), -15.,  15.)

            self.nll_hist_.append(self._nll(xi_v, ls_v, Z))
            if verbose and b % 30 == 0:
                print(f"    iter {b:3d} | NLL={self.nll_hist_[-1]:.5f} "
                      f"| xi_mean={xi_v.mean():.4f}")

        self.xi_  = xi_v
        self.sig_ = np.exp(ls_v)
        return self

    def predict_var(self):
        q0    = self.q_init_[self.mask_]
        ratio = (1 - self.alpha) / (1 - self.alpha_0)
        return q0 + self.sig_ / self.xi_ * (ratio ** (-self.xi_) - 1)


# ============================================================
# 4. FIGURE
# ============================================================

def plot_step7(dates_exc, var_cond, xi_cond,
               dates_all, losses_all,
               var_static, xi_static, nll_hist,
               asset, alpha, save_path):

    fig = plt.figure(figsize=(15, 12))
    gs  = gridspec.GridSpec(3, 2, fig, hspace=0.42, wspace=0.30,
                            height_ratios=[1.4, 1.0, 1.0])
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[2, 1])

    RED, ORA, PUR = "#d62728", "#ff7f0e", "#9467bd"
    cspan = dict(alpha=0.13, color=RED)

    y = losses_all.values if hasattr(losses_all, "values") else np.array(losses_all)

    # Panel 1 : Pertes + VaR
    ax1.fill_between(dates_all, y * 100, 0,
                     where=y > 0, color="steelblue", alpha=0.22,
                     label="Pertes horaires (%)")
    ax1.axhline(var_static * 100, color=RED, lw=2, ls="--",
                label=f"VaR {alpha*100:.1f}% POT statique = {var_static*100:.3f}%")
    ax1.scatter(dates_exc, var_cond * 100, s=5, color=ORA, alpha=0.85,
                zorder=5, label=f"VaR {alpha*100:.1f}% GB-EVT conditionnel")
    ax1.axvspan(pd.Timestamp(CRASH_START), pd.Timestamp(CRASH_END),
                label="Flash crash", **cspan)
    ax1.set_ylabel("Perte (%)", fontsize=11)
    ax1.set_title(f"{asset} — VaR {alpha*100:.1f}% : POT statique vs GB-EVT conditionnel",
                  fontsize=12, fontweight="bold")
    ax1.legend(fontsize=8.5, loc="upper left", ncol=2)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax1.set_ylim(bottom=-0.2)

    # Panel 2 : xi(x) dans le temps
    ax2.scatter(dates_exc, xi_cond, s=4, color=PUR, alpha=0.7,
                label=r"$\hat{\xi}(x)$ GB-EVT")
    ax2.axhline(xi_static, color=RED, lw=2, ls="--",
                label=f"xi statique = {xi_static:.4f}")
    ax2.axhline(0, color="black", lw=0.6, ls=":")
    ax2.axvspan(pd.Timestamp(CRASH_START), pd.Timestamp(CRASH_END), **cspan)
    ax2.set_ylabel("xi(x)", fontsize=11)
    ax2.set_title("Indice de queue conditionnel xi(x) dans le temps",
                  fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # Panel 3 : Convergence NLL
    ax3.plot(nll_hist, color="steelblue", lw=1.5)
    ax3.set_xlabel("Iteration", fontsize=10)
    ax3.set_ylabel("NLL", fontsize=10)
    ax3.set_title("Convergence du gradient boosting", fontsize=10, fontweight="bold")
    ax3.grid(True, alpha=0.3)

    # Panel 4 : Boxplot par regime
    cm        = ((dates_exc >= pd.Timestamp(CRASH_START)) &
                 (dates_exc <= pd.Timestamp(CRASH_END)))
    calm_var  = var_cond[~cm] * 100
    crash_var = var_cond[cm]  * 100 if cm.sum() > 0 else np.array([var_static * 100])
    data_bp   = [calm_var, crash_var, np.array([var_static * 100])]

    bp = ax4.boxplot(data_bp,
                     labels=["GB-EVT\nCalme", "GB-EVT\nCrash", "POT\nStatique"],
                     patch_artist=True, widths=0.5,
                     medianprops=dict(color="black", lw=2))
    for patch, c in zip(bp["boxes"], ["#2ca02c", RED, ORA]):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax4.set_ylabel(f"VaR {alpha*100:.1f}% (%)", fontsize=10)
    ax4.set_title("VaR par regime de marche", fontsize=10, fontweight="bold")
    ax4.grid(True, alpha=0.3, axis="y")

    plt.suptitle(f"GB-EVT conditionnel — {asset}/USDT 2025",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure : {save_path}")


# ============================================================
# 5. TABLEAU LATEX
# ============================================================

def print_latex(results):
    print("\n" + "=" * 62)
    print("TABLEAU LATEX")
    print("=" * 62)
    pct = f"{ALPHA * 100:.1f}"
    print(r"\begin{table}[h!]")
    print(r"\centering\small\renewcommand{\arraystretch}{1.3}")
    print(f"\\caption{{Comparaison VaR~{pct}\\% : POT statique vs GB-EVT conditionnel}}")
    print(r"\label{tab:gbevt_comparison}")
    print(r"\begin{tabular}{lcccc}\toprule")
    print(r" & \multicolumn{2}{c}{\textbf{BTC}} & \multicolumn{2}{c}{\textbf{ETH}} \\")
    print(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}")
    print(r"\textbf{Regime} & $\hat{\xi}$ & VaR (\%) & $\hat{\xi}$ & VaR (\%) \\\midrule")
    for regime in ["Calme", "Flash crash", "Statique (POT)"]:
        b = results.get("BTC", {}).get(regime, {})
        e = results.get("ETH", {}).get(regime, {})
        def fmt(d, k, mult=1, dec=4):
            v = d.get(k)
            return f"{v * mult:.{dec}f}" if v is not None else "--"
        print(f"{regime:<22} & ${fmt(b,'xi')}$ & ${fmt(b,'var',100,3)}$ "
              f"& ${fmt(e,'xi')}$ & ${fmt(e,'var',100,3)}$ \\\\")
    print(r"\bottomrule\end{tabular}\end{table}")


# ============================================================
# 6. PIPELINE
# ============================================================

def run(asset, df):
    print(f"\n{'=' * 55}")
    print(f"  GB-EVT — {asset}")
    print(f"{'=' * 55}")

    df_feat   = build_features(df, asset=asset.lower())
    feat_cols = [c for c in df_feat.columns if c not in ("loss", "ret")]
    Y         = df_feat["loss"].values.astype(float)
    X         = df_feat[feat_cols].values.astype(float)
    dates     = df_feat.index
    n_total   = len(Y)

    model = GBEVT(alpha_0=ALPHA_0, alpha=ALPHA,
                  B=B_ITER, lr_xi=LR_XI, lr_sigma=LR_SIG)
    model.fit(X, Y, verbose=True)

    var_cond  = model.predict_var()
    xi_cond   = model.xi_
    dates_exc = dates[model.mask_]

    st         = STATIC[asset]
    var_static = (st["u"] + st["sigma"] / st["xi"] *
                  ((n_total / st["n_u"] * (1 - ALPHA)) ** (-st["xi"]) - 1))

    cm = ((dates_exc >= pd.Timestamp(CRASH_START)) &
          (dates_exc <= pd.Timestamp(CRASH_END)))

    results = {
        "Calme":          {"xi":  xi_cond[~cm].mean()  if (~cm).sum() > 0 else None,
                           "var": var_cond[~cm].mean()  if (~cm).sum() > 0 else None},
        "Flash crash":    {"xi":  xi_cond[cm].mean()   if cm.sum() > 0  else None,
                           "var": var_cond[cm].mean()   if cm.sum() > 0  else None},
        "Statique (POT)": {"xi":  st["xi"], "var": var_static},
    }

    print(f"\n  VaR {ALPHA * 100:.1f}%")
    print(f"  {'Regime':<22} {'xi':>8} {'VaR (%)':>10}")
    print(f"  {'-' * 42}")
    for regime, v in results.items():
        xs = f"{v['xi']:.4f}"       if v["xi"]  is not None else "---"
        vs = f"{v['var']*100:.3f}%" if v["var"] is not None else "---"
        print(f"  {regime:<22} {xs:>8} {vs:>10}")

    if results["Flash crash"]["var"] and var_static > 0:
        ratio = results["Flash crash"]["var"] / var_static
        print(f"\n  => VaR crash = {ratio:.2f}x la VaR statique POT")

    plot_step7(
        dates_exc=dates_exc, var_cond=var_cond, xi_cond=xi_cond,
        dates_all=dates,     losses_all=df_feat["loss"],
        var_static=var_static, xi_static=st["xi"],
        nll_hist=model.nll_hist_,
        asset=asset, alpha=ALPHA,
        save_path=f"{FIGURES_DIR}evt_step7_gbevt_{asset.lower()}.png"
    )
    return results


# ============================================================
# 7. MAIN
# ============================================================

if __name__ == "__main__":
    print("Chargement...")
    df = load_returns(DATA_PATH)
    print(f"  {len(df)} obs | {df.index[0].date()} -> {df.index[-1].date()}")

    all_results = {}
    for asset in ["BTC", "ETH"]:
        try:
            all_results[asset] = run(asset, df)
        except Exception as e:
            import traceback
            print(f"  Erreur {asset} : {e}")
            traceback.print_exc()

    if all_results:
        print_latex(all_results)