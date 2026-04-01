# ============================================================
# DEMO INFERENCE CELL — Regime-Aware ML + 1 Extra Model
# ============================================================
# Expected files in ./src/
#   global_catboost_20260401_050144.joblib
#   global_extratrees_20260401_050144.joblib
#   global_lightgbm_20260401_050144.joblib
#   global_randomforest_20260401_050144.joblib
#   regime_regime_0_20260401_050244.joblib
#   regime_regime_1_20260401_050244.joblib
#   regime_regime_2_20260401_050244.joblib
# ============================================================

import os, glob, joblib, datetime, warnings
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

# ── CONFIG ──────────────────────────────────────────────────
MODEL_DIR   = "./src"          # folder where all .joblib files live
START_DATE  = "2020-01-01"     # enough history for 52-week rolling windows
END_DATE    = datetime.date.today().strftime("%Y-%m-%d")
BUY_THRESH  = 0.57
SELL_THRESH = 0.43
MIN_SPREAD  = 0.08
TOP_K       = 8

NIFTY50_TICKERS = [
    "ADANIENT.NS","ADANIPORTS.NS","APOLLOHOSP.NS","ASIANPAINT.NS",
    "AXISBANK.NS","BAJAJ-AUTO.NS","BAJAJFINSV.NS","BAJFINANCE.NS",
    "BHARTIARTL.NS","BPCL.NS","BRITANNIA.NS","CIPLA.NS",
    "COALINDIA.NS","DIVISLAB.NS","DRREDDY.NS","EICHERMOT.NS",
    "GRASIM.NS","HCLTECH.NS","HDFCBANK.NS","HDFCLIFE.NS",
    "HINDALCO.NS","HINDUNILVR.NS","ICICIBANK.NS","INFY.NS",
    "ITC.NS","JSWSTEEL.NS","KOTAKBANK.NS","LT.NS","MARUTI.NS",
    "NESTLEIND.NS","NTPC.NS","ONGC.NS","POWERGRID.NS",
    "RELIANCE.NS","SBIN.NS","SBILIFE.NS","SHRIRAMFIN.NS",
    "SUNPHARMA.NS","TATACONSUM.NS","TATASTEEL.NS",
    "TCS.NS","TECHM.NS","TITAN.NS","ULTRACEMCO.NS","WIPRO.NS",
]

SECTOR_MAP = {
    "ADANIENT.NS":"Energy","ADANIPORTS.NS":"Infra","RELIANCE.NS":"Energy",
    "ONGC.NS":"Energy","BPCL.NS":"Energy","COALINDIA.NS":"Energy",
    "POWERGRID.NS":"Utilities","NTPC.NS":"Utilities",
    "HDFCBANK.NS":"Financials","ICICIBANK.NS":"Financials","SBIN.NS":"Financials",
    "KOTAKBANK.NS":"Financials","AXISBANK.NS":"Financials","BAJFINANCE.NS":"Financials",
    "BAJAJFINSV.NS":"Financials","SBILIFE.NS":"Financials","HDFCLIFE.NS":"Financials",
    "SHRIRAMFIN.NS":"Financials","INFY.NS":"IT","TCS.NS":"IT","WIPRO.NS":"IT",
    "TECHM.NS":"IT","HCLTECH.NS":"IT","HINDUNILVR.NS":"FMCG","ITC.NS":"FMCG",
    "NESTLEIND.NS":"FMCG","BRITANNIA.NS":"FMCG","TATACONSUM.NS":"FMCG",
    "MARUTI.NS":"Auto","EICHERMOT.NS":"Auto","BAJAJ-AUTO.NS":"Auto",
    "ULTRACEMCO.NS":"Cement","GRASIM.NS":"Cement","ASIANPAINT.NS":"Consumer",
    "TITAN.NS":"Consumer","APOLLOHOSP.NS":"Healthcare","SUNPHARMA.NS":"Pharma",
    "DRREDDY.NS":"Pharma","DIVISLAB.NS":"Pharma","CIPLA.NS":"Pharma",
    "JSWSTEEL.NS":"Metals","TATASTEEL.NS":"Metals","HINDALCO.NS":"Metals",
    "BHARTIARTL.NS":"Telecom","LT.NS":"Infra",
}

FEATURE_COLS = [
    "ret_1w","ret_2w","ret_4w","ret_8w","ret_12w","ret_26w","ret_52w",
    "mom_4w","mom_8w","mom_26w",
    "vol_4w","vol_8w","vol_26w",
    "bb_pct","macd_norm","prox_52w","prox_52w_mom",
    "ma_cross","price_to_ma26",
    "vol_mom","vol_ratio","sec_mom_rel","sec_nifty_rel",
    "mom_8w_rank","vol_26w_rank","ret_4w_rank",
    "beta_52w","idio_vol","mkt_ret_1w","mkt_vol_4w",
    "regime_id",          # always last — matches training order
]

plt.rcParams.update({
    "figure.dpi": 110,
    "axes.facecolor": "#0d1117", "figure.facecolor": "#0d1117",
    "axes.labelcolor": "#c9d1d9", "text.color": "#c9d1d9",
    "xtick.color": "#c9d1d9", "ytick.color": "#c9d1d9",
    "axes.edgecolor": "#30363d", "grid.color": "#21262d",
    "legend.facecolor": "#161b22", "legend.edgecolor": "#30363d",
})
C = ["#4f9cff","#00d084","#ff4757","#ffd60a","#a855f7","#00c8d7","#f59e0b","#ec4899"]


# ── HELPERS ─────────────────────────────────────────────────

def load_latest(prefix: str) -> tuple:
    """
    Find the most recent .joblib whose filename starts with `prefix`.
    
    Naming convention saved by the training script:
        global_catboost_YYYYMMDD_HHMMSS.joblib
        global_extratrees_YYYYMMDD_HHMMSS.joblib
        global_lightgbm_YYYYMMDD_HHMMSS.joblib
        global_randomforest_YYYYMMDD_HHMMSS.joblib
        regime_regime_0_YYYYMMDD_HHMMSS.joblib
        regime_regime_1_YYYYMMDD_HHMMSS.joblib
        regime_regime_2_YYYYMMDD_HHMMSS.joblib

    Returns (model_object, filename_string).
    """
    pattern = os.path.join(MODEL_DIR, f"{prefix}*.joblib")
    matches = sorted(glob.glob(pattern))   # lexicographic sort → latest timestamp last
    if not matches:
        raise FileNotFoundError(
            f"No file matching '{pattern}'.\n"
            f"Files in {MODEL_DIR}: {sorted(os.listdir(MODEL_DIR))}"
        )
    path  = matches[-1]
    model = joblib.load(path)
    return model, os.path.basename(path)


# ── MUST be defined before any joblib.load call ─────────────
# The regime models were saved as RegimeBlender instances.
# Pickle needs this class present in __main__ to deserialise them.
class RegimeBlender:
    """Blends one bagger (RF/ET) and one booster (XGB/LGB/CB) 50/50."""
    def __init__(self, model_bagger, model_booster, name_bagger, name_booster):
        self.model_bagger  = model_bagger
        self.model_booster = model_booster
        self.name_bagger   = name_bagger
        self.name_booster  = name_booster

    def _proba(self, model, X):
        X = np.asarray(X, dtype=np.float32)
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X)
            return p[:, 1] if (p.ndim == 2 and p.shape[1] >= 2) else p.ravel().astype(float)
        if hasattr(model, "decision_function"):
            s = model.decision_function(X)
            return 1.0 / (1.0 + np.exp(-s))
        return model.predict(X).astype(float)

    def predict_proba(self, X):
        return (self._proba(self.model_bagger, X) * 0.5 +
                self._proba(self.model_booster, X) * 0.5)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def get_params(self, deep=True):
        return {}


def predict_proba_safe(model, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        return p[:, 1] if (p.ndim == 2 and p.shape[1] >= 2) else p.ravel().astype(float)
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-s))
    return model.predict(X).astype(float)


def rating(p: float) -> str:
    if p >= BUY_THRESH:   return "🟢 BUY"
    if p <= SELL_THRESH:  return "🔴 SELL"
    return "🟡 HOLD"


def conviction(p: float) -> str:
    if p >= 0.76: return "★★★ VERY HIGH"
    if p >= 0.66: return "★★★ HIGH"
    if p >= 0.58: return "★★☆ MEDIUM"
    if p >= 0.50: return "★☆☆ LOW"
    if p >= 0.43: return "☆☆☆ HOLD"
    return "▼▼▼ EXIT"


# ── FEATURE ENGINEERING (mirrors training exactly) ──────────

def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))

def compute_bb_pct(series, w=20, k=2):
    mid = series.rolling(w, min_periods=w // 2).mean()
    std = series.rolling(w, min_periods=w // 2).std()
    return (series - (mid - k * std)) / (2 * k * std + 1e-9)

def compute_macd_norm(series):
    m = series.ewm(span=12, adjust=False).mean() - series.ewm(span=26, adjust=False).mean()
    s = m.ewm(span=9, adjust=False).mean()
    return (m - s) / (series.abs().rolling(12, min_periods=6).mean() + 1e-9)

def engineer_features(stocks: pd.DataFrame, nifty_idx: pd.DataFrame) -> pd.DataFrame:
    stocks = stocks.copy()
    g = stocks.groupby("stock", sort=False)

    for n in [1, 2, 4, 8, 12, 26, 52]:
        stocks[f"ret_{n}w"] = g["price"].pct_change(n)
    for n, mp in [(4, 3), (8, 4), (26, 8)]:
        stocks[f"mom_{n}w"] = g["ret_1w"].transform(
            lambda x, n=n, mp=mp: x.shift(1).rolling(n, min_periods=mp).sum())
    for n, mp in [(4, 3), (8, 4), (26, 8)]:
        stocks[f"vol_{n}w"] = g["ret_1w"].transform(
            lambda x, n=n, mp=mp: x.rolling(n, min_periods=mp).std())
    for n, mp in [(4, 3), (26, 8)]:
        stocks[f"ma_{n}w"] = g["price"].transform(
            lambda x, n=n, mp=mp: x.rolling(n, min_periods=mp).mean())
    stocks["price_to_ma4"]  = stocks["price"] / stocks["ma_4w"]
    stocks["price_to_ma26"] = stocks["price"] / stocks["ma_26w"]
    stocks["ma_cross"]      = stocks["ma_4w"]  / stocks["ma_26w"]
    stocks["rsi_14"]        = g["price"].transform(compute_rsi)
    stocks["bb_pct"]        = g["price"].transform(compute_bb_pct)
    stocks["macd_norm"]     = g["price"].transform(compute_macd_norm)
    stocks["hi_52w"]        = g["price"].transform(lambda x: x.rolling(52, min_periods=26).max())
    stocks["prox_52w"]      = (stocks["price"] / stocks["hi_52w"]).clip(0, 1.5)
    stocks["prox_52w_mom"]  = g["prox_52w"].transform(lambda x: x.pct_change(4))
    stocks["vol_mom"]       = g["volume"].transform(lambda x: x.pct_change(4))
    stocks["vol_ratio"]     = g["volume"].transform(
        lambda x: x / (x.rolling(8, min_periods=4).mean() + 1e-9))
    stocks["sector"]        = stocks["stock"].map(SECTOR_MAP).fillna("Other")
    stocks["sec_mom_rel"]   = stocks.groupby(["date", "sector"])["ret_8w"].transform(
        lambda x: x - x.median())

    nret_map             = nifty_idx["nifty_ret"].to_dict()
    nvol_map             = nifty_idx["nifty_vol4"].to_dict()
    stocks["mkt_ret_1w"] = stocks["date"].map(nret_map)
    stocks["mkt_vol_4w"] = stocks["date"].map(nvol_map)
    stocks["nifty_ret"]  = stocks["date"].map(nret_map)

    beta_s = (
        stocks.groupby("stock", sort=False, group_keys=False)
        .apply(lambda df: (
            df["ret_1w"].rolling(52, min_periods=26).cov(df["nifty_ret"]) /
            df["nifty_ret"].rolling(52, min_periods=26).var().replace(0, np.nan)
        ))
    )
    stocks["beta_52w"] = beta_s.reset_index(level=0, drop=True)
    stocks["idio_vol"] = stocks["ret_1w"] - stocks["beta_52w"] * stocks["nifty_ret"]
    stocks["idio_vol"] = stocks.groupby("stock")["idio_vol"].transform(
        lambda x: x.rolling(26, min_periods=12).std())
    stocks["sec_nifty_rel"] = stocks.groupby(["date", "sector"])["ret_4w"].transform(
        lambda x: x.mean()) - stocks["nifty_ret"].rolling(4, min_periods=4).sum()
    for col in ["mom_8w", "vol_26w", "rsi_14", "ret_4w"]:
        stocks[f"{col}_rank"] = stocks.groupby("date")[col].rank(pct=True)

    return stocks


# ── STEP 1 — DOWNLOAD ───────────────────────────────────────
print("=" * 65)
print("  REGIME-AWARE ML — DEMO INFERENCE")
print("=" * 65)
print(f"\n📥  Downloading market data ({START_DATE} → {END_DATE}) …")

raw_nifty = yf.download("^NSEI", start=START_DATE, end=END_DATE,
                         interval="1d", auto_adjust=True,
                         progress=False, threads=False)
if isinstance(raw_nifty.columns, pd.MultiIndex):
    raw_nifty.columns = raw_nifty.columns.get_level_values(0)
nifty_w   = raw_nifty["Close"].resample("W-FRI").last().dropna()
nifty_idx = pd.DataFrame({
    "nifty_close": nifty_w,
    "nifty_vol":   raw_nifty["Volume"].resample("W-FRI").sum(),
})
nifty_idx["nifty_ret"]  = nifty_idx["nifty_close"].pct_change()
nifty_idx["nifty_vol4"] = nifty_idx["nifty_ret"].rolling(4).std()
nifty_idx.dropna(subset=["nifty_ret"], inplace=True)
print(f"   NIFTY : {len(nifty_idx)} weekly rows | latest {nifty_idx.index[-1].date()}")

all_data = yf.download(NIFTY50_TICKERS, start=START_DATE, end=END_DATE,
                        interval="1d", auto_adjust=True, progress=False)
closes  = all_data["Close"]
volumes = all_data["Volume"]

frames = []
for sym in NIFTY50_TICKERS:
    try:
        cd = closes[sym].dropna()
        cd.index = pd.to_datetime(cd.index)
        cw = cd.resample("W-FRI").last().dropna()
        if len(cw) < 52:
            continue
        df = pd.DataFrame({"price": cw}); df.index.name = "date"
        if sym in volumes.columns:
            vd = volumes[sym].dropna(); vd.index = pd.to_datetime(vd.index)
            df["volume"] = vd.resample("W-FRI").sum().reindex(cw.index).fillna(0)
        else:
            df["volume"] = 0.0
        df["stock"] = sym
        df = df.reset_index(); df["date"] = pd.to_datetime(df["date"])
        frames.append(df)
    except Exception:
        pass

stocks = pd.concat(frames, ignore_index=True)
stocks.sort_values(["stock", "date"], inplace=True)
stocks.reset_index(drop=True, inplace=True)
print(f"   Stocks: {stocks['stock'].nunique()} tickers | {len(stocks):,} weekly rows")


# ── STEP 2 — FEATURE ENGINEERING ────────────────────────────
print("\n🔧  Engineering features …")
stocks = engineer_features(stocks, nifty_idx)


# ── STEP 3 — AUTO-DETECT CURRENT REGIME ─────────────────────
# Maps to the 3 saved files: regime_regime_0 / _1 / _2
latest_nifty_ret = float(nifty_idx["nifty_ret"].iloc[-1])
latest_nifty_vol = float(nifty_idx["nifty_vol4"].iloc[-1])

if   latest_nifty_ret >  0.005 and latest_nifty_vol < 0.018:
    current_regime = 2          # BULL / BREAKOUT
elif latest_nifty_ret < -0.005 or latest_nifty_vol > 0.025:
    current_regime = 0          # BEAR / CRASH
else:
    current_regime = 1          # SIDEWAYS

REGIME_LABELS = {0: "BEAR/CRASH", 1: "SIDEWAYS", 2: "BULL/BREAKOUT"}
print(f"   NIFTY  ret={latest_nifty_ret:+.4f}  vol4={latest_nifty_vol:.4f}")
print(f"   Regime → {current_regime}  ({REGIME_LABELS[current_regime]})")


# ── STEP 4 — LIVE FEATURES + SCALER ─────────────────────────
signal_date = stocks["date"].max()
live_df     = stocks[stocks["date"] == signal_date].copy()
live_df["regime_id"] = current_regime
print(f"\n   Signal date : {signal_date.date()}")

# Refit scaler on downloaded history (no serialised scaler in this project)
feat_no_regime = [c for c in FEATURE_COLS if c != "regime_id"]
tmp            = stocks.dropna(subset=feat_no_regime).copy()
tmp["regime_id"] = current_regime
scaler = RobustScaler().fit(tmp[FEATURE_COLS])
print("   Scaler      : RobustScaler refitted on downloaded history")


# ── STEP 5 — LOAD MODELS ────────────────────────────────────
print(f"\n📦  Loading models from: {os.path.abspath(MODEL_DIR)}")
all_files = sorted(f for f in os.listdir(MODEL_DIR) if f.endswith(".joblib"))
print(f"    Joblib files found ({len(all_files)}): {all_files}\n")

# ── PRIMARY — regime_regime_<current_regime>_YYYYMMDD_HHMMSS.joblib ──
primary_prefix = f"regime_regime_{current_regime}_"
try:
    primary_model, primary_fname = load_latest(primary_prefix)
    print(f"   ✅ Primary : {primary_fname}")
except FileNotFoundError as e:
    print(f"   ⚠️  Regime file not found: {e}")
    # Fallback: try every regime file, then any global
    loaded = False
    for rid in [r for r in [0, 1, 2] if r != current_regime]:
        try:
            primary_model, primary_fname = load_latest(f"regime_regime_{rid}_")
            primary_prefix = f"regime_regime_{rid}_"
            print(f"   ↳ Fallback regime {rid}: {primary_fname}")
            loaded = True; break
        except FileNotFoundError:
            pass
    if not loaded:
        for prefix in ["global_catboost_", "global_lightgbm_",
                       "global_randomforest_", "global_extratrees_"]:
            try:
                primary_model, primary_fname = load_latest(prefix)
                primary_prefix = prefix
                print(f"   ↳ Global fallback: {primary_fname}")
                break
            except FileNotFoundError:
                pass

primary_label = (primary_fname
                 .replace("regime_regime_", "Regime-")
                 .replace("global_", "Global/")
                 .rsplit("_", 2)[0]          # strip timestamp
                 .replace("_", " ").title())

# ── EXTRA — best global model different from the primary family ──
# Priority order: CatBoost → LightGBM → RandomForest → ExtraTrees
GLOBAL_PRIORITY = [
    ("global_catboost_",     "CatBoost"),
    ("global_lightgbm_",     "LightGBM"),
    ("global_randomforest_", "RandomForest"),
    ("global_extratrees_",   "ExtraTrees"),
]

extra_model, extra_fname, extra_label = None, None, None
for gprefix, glabel in GLOBAL_PRIORITY:
    # Skip if the same file was already chosen as primary
    try:
        candidate_path = sorted(glob.glob(os.path.join(MODEL_DIR, f"{gprefix}*.joblib")))[-1]
    except IndexError:
        continue
    if os.path.basename(candidate_path) == primary_fname:
        continue                              # same file → skip
    try:
        extra_model, extra_fname = load_latest(gprefix)
        extra_label = glabel
        print(f"   ✅ Extra   : {extra_fname}")
        break
    except FileNotFoundError:
        pass

if extra_model is None:
    print("   ⚠️  No distinct extra model — mirroring primary.")
    extra_model, extra_fname, extra_label = primary_model, primary_fname, primary_label


# ── STEP 6 — SCORE ──────────────────────────────────────────
print("\n🎯  Scoring …")
clean = live_df.dropna(subset=FEATURE_COLS).copy()
if len(clean) == 0:
    raise RuntimeError(
        "All rows dropped after NaN filter. "
        "Extend START_DATE further back (e.g. 2019-01-01)."
    )

X_live = scaler.transform(clean[FEATURE_COLS].astype(np.float32))

prob_primary   = predict_proba_safe(primary_model, X_live)
prob_extra     = predict_proba_safe(extra_model,   X_live)
spread_primary = float(prob_primary.max() - prob_primary.min())
spread_extra   = float(prob_extra.max()   - prob_extra.min())

# Collapse guard
if spread_primary < MIN_SPREAD:
    print(f"   ⚠️  Primary spread={spread_primary:.4f} < {MIN_SPREAD} → switching to global fallback")
    for gprefix, glabel in GLOBAL_PRIORITY:
        try:
            primary_model, primary_fname = load_latest(gprefix)
            primary_label  = glabel + " (collapse fallback)"
            prob_primary   = predict_proba_safe(primary_model, X_live)
            spread_primary = float(prob_primary.max() - prob_primary.min())
            print(f"   ↳ {primary_fname}  spread={spread_primary:.4f}")
            break
        except FileNotFoundError:
            pass

clean["prob_primary"] = prob_primary
clean["prob_extra"]   = prob_extra
clean["rating"]       = clean["prob_primary"].apply(rating)
clean["conviction"]   = clean["prob_primary"].apply(conviction)
clean["sector"]       = clean["stock"].map(SECTOR_MAP).fillna("Other")

signals = (
    clean.sort_values("prob_primary", ascending=False)
    [["stock","sector","rating","conviction",
      "prob_primary","prob_extra","mom_8w","rsi_14","beta_52w"]]
    .reset_index(drop=True)
)
buys  = signals[signals["rating"] == "🟢 BUY"]
holds = signals[signals["rating"] == "🟡 HOLD"]
sells = signals[signals["rating"] == "🔴 SELL"]


# ── STEP 7 — PRINT ──────────────────────────────────────────
print(f"\n{'='*105}")
print(f"  INFERENCE RESULTS — {signal_date.date()}")
print(f"  Primary : {primary_label:<40} spread={spread_primary:.4f} "
      f"{'✅ OK' if spread_primary >= MIN_SPREAD else '⚠️ LOW'}")
print(f"  Extra   : {extra_label:<40} spread={spread_extra:.4f} "
      f"{'✅ OK' if spread_extra >= MIN_SPREAD else '⚠️ LOW'}")
print(f"  Regime  : {current_regime} — {REGIME_LABELS[current_regime]}"
      f"  (ret={latest_nifty_ret:+.4f}  vol={latest_nifty_vol:.4f})")
print(f"{'='*105}")

print(f"\n  {'#':<3} {'Stock':<18} {'Sector':<13} {'Rating':<14} {'Conviction':<16}"
      f" {'Primary%':>10} {'Extra%':>8} {'RSI':>6} {'Beta':>6}")
print(f"  {'─'*100}")
for i, row in signals.iterrows():
    print(f"  {i+1:<3} {row['stock']:<18} {row['sector']:<13} {row['rating']:<14}"
          f" {row['conviction']:<16}"
          f" {row['prob_primary']*100:>9.1f}% {row['prob_extra']*100:>7.1f}%"
          f" {row['rsi_14']:>6.1f} {row['beta_52w']:>6.2f}")

print(f"\n  BUY: {len(buys)}  |  HOLD: {len(holds)}  |  SELL: {len(sells)}")
print(f"{'='*105}")

print(f"\n  🟢  TOP BUY PICKS  [{primary_label}]")
print(f"  {'─'*70}")
if len(buys) > 0:
    for _, row in buys.head(8).iterrows():
        agree = "✅ BOTH AGREE" if row["prob_extra"] >= BUY_THRESH else "⚪ ONLY PRIMARY"
        print(f"  {row['stock']:<18} | {row['conviction']:<16} |"
              f" Primary={row['prob_primary']*100:.1f}%"
              f"  Extra={row['prob_extra']*100:.1f}%  {agree}")
else:
    print(f"  No BUY signals — regime is {REGIME_LABELS[current_regime]}")

if len(sells) > 0:
    print(f"\n  🔴  SELL / AVOID")
    print(f"  {'─'*70}")
    for _, row in sells.iterrows():
        print(f"  {row['stock']:<18} | Primary={row['prob_primary']*100:.1f}%"
              f"  Extra={row['prob_extra']*100:.1f}%")


# ── STEP 8 — PLOTS ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(22, 8))

# Panel 1 — Primary model probability bars
ax = axes[0]
bar_cols = [C[1] if r == "🟢 BUY" else C[2] if r == "🔴 SELL" else C[3]
            for r in signals["rating"]]
ax.barh(signals["stock"][::-1], signals["prob_primary"][::-1] * 100,
        color=bar_cols[::-1], edgecolor="#30363d", linewidth=0.8, alpha=0.9)
ax.axvline(BUY_THRESH * 100,  color=C[1], lw=2, ls="--",
           label=f"BUY ≥ {BUY_THRESH*100:.0f}%")
ax.axvline(SELL_THRESH * 100, color=C[2], lw=2, ls="--",
           label=f"SELL ≤ {SELL_THRESH*100:.0f}%")
ax.axvline(50, color="#445f7a", lw=1, ls=":", alpha=0.6, label="50%")
for bar, val in zip(ax.patches, signals["prob_primary"][::-1] * 100):
    ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", fontsize=7, color="#c9d1d9")
ax.set_xlabel("Outperform Probability (%)")
ax.set_title(f"Primary Model: {primary_label}\n{signal_date.date()}",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=8); ax.grid(axis="x", alpha=0.3)

# Panel 2 — Model agreement scatter
ax = axes[1]
both_buy  = (signals["prob_primary"] >= BUY_THRESH) & (signals["prob_extra"] >= BUY_THRESH)
only_prim = (signals["prob_primary"] >= BUY_THRESH) & (signals["prob_extra"] <  BUY_THRESH)
neither   = ~both_buy & ~only_prim

ax.scatter(signals.loc[neither,   "prob_primary"] * 100,
           signals.loc[neither,   "prob_extra"]   * 100,
           color="#445f7a", s=55, alpha=0.7, label="HOLD / SELL")
ax.scatter(signals.loc[only_prim, "prob_primary"] * 100,
           signals.loc[only_prim, "prob_extra"]   * 100,
           color=C[3], s=80, alpha=0.9, zorder=3, label="Primary only ⚠️")
ax.scatter(signals.loc[both_buy,  "prob_primary"] * 100,
           signals.loc[both_buy,  "prob_extra"]   * 100,
           color=C[1], s=100, alpha=0.95, zorder=4, label="Both BUY ✅")
for _, row in signals[both_buy].iterrows():
    ax.annotate(row["stock"].replace(".NS", ""),
                (row["prob_primary"] * 100, row["prob_extra"] * 100),
                fontsize=7, color=C[1], xytext=(4, 2), textcoords="offset points")
ax.axvline(BUY_THRESH * 100, color=C[1], lw=1.5, ls="--", alpha=0.6)
ax.axhline(BUY_THRESH * 100, color=C[1], lw=1.5, ls="--", alpha=0.6)
ax.set_xlabel(f"Primary: {primary_label} (%)")
ax.set_ylabel(f"Extra: {extra_label} (%)")
ax.set_title("Model Agreement\nGreen quadrant = both BUY",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Panel 3 — Signal donut
ax = axes[2]
rc = {}
for r in signals["rating"]:
    k = r.split()[-1]
    rc[k] = rc.get(k, 0) + 1
dv = [rc.get("BUY", 0), rc.get("HOLD", 0), rc.get("SELL", 0)]
dl = [f"BUY\n({rc.get('BUY',0)})", f"HOLD\n({rc.get('HOLD',0)})",
      f"SELL\n({rc.get('SELL',0)})"]
if sum(dv) > 0:
    ax.pie(dv, labels=dl, autopct="%1.0f%%",
           colors=[C[1], C[3], C[2]], startangle=90,
           wedgeprops=dict(width=0.55), textprops={"fontsize": 10})
ax.set_title(
    f"Signal Distribution\nRegime {current_regime}: "
    f"{REGIME_LABELS[current_regime]}\n{signal_date.date()}",
    fontsize=10, fontweight="bold")

plt.suptitle(
    f"Demo Inference — {signal_date.date()}\n"
    f"Primary: {primary_label}   |   Extra: {extra_label}",
    fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

print("\n⚠️  DISCLAIMER: Research only — NOT financial advice.")
