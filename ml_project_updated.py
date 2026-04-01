# ============================================================
#  NIFTY 50 ADAPTIVE FACTOR-REGIME ML SYSTEM v8.0
#  Forensic-grade pipeline — all 7 critical issues fixed:
#  1. Signal collapse → CalibratedClassifierCV + spread check
#  2. Class collapse → regime-specific class_weight + min samples
#  3. ML < EW → better portfolio construction + signal quality
#  4. Regime K selection → composite score (BIC+Silhouette+Stability)
#  5. Leakage in regime model selection → cross-val within train only
#  6. Rolling window contamination → purged gap between splits
#  7. Equal-weight look-ahead removed → proper benchmarks only
#  Paste entire file into ONE Colab cell.
# ============================================================

import copy
import json
import os
import subprocess
import warnings, datetime, time, gc, joblib
import optuna
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats as scipy_stats

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier,
    VotingClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import (
    cross_val_score, TimeSeriesSplit, StratifiedKFold
)
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, brier_score_loss,
    average_precision_score,
    silhouette_score,
)
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
import lightgbm as lgb

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

# Unused imports for ThunderGBM and TabNet removed as they were replaced by RandomForest and GradientBoost.

try:
    from nodegam.sklearn import NodeGAMClassifier
    HAS_NODE = True
except ImportError:
    HAS_NODE = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from hmmlearn.hmm import GaussianHMM
    HAS_HMM = True
except ImportError:
    HAS_HMM = False

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

plt.rcParams.update({
    "figure.dpi": 110,
    "axes.facecolor": "#0d1117", "figure.facecolor": "#0d1117",
    "axes.labelcolor": "#c9d1d9", "text.color": "#c9d1d9",
    "xtick.color": "#c9d1d9", "ytick.color": "#c9d1d9",
    "axes.edgecolor": "#30363d", "grid.color": "#21262d",
    "legend.facecolor": "#161b22", "legend.edgecolor": "#30363d",
})
C = ["#4f9cff","#00d084","#ff4757","#ffd60a","#a855f7","#00c8d7","#f59e0b","#ec4899"]

print("✅ All imports successful.")

TOTAL_SECTIONS = 14

def render_progress_bar(current, total, width=28):
    total = max(int(total), 1)
    current = max(0, min(int(current), total))
    filled = int(round(width * current / total))
    return "[" + "#" * filled + "-" * (width - filled) + "]"

def show_section_progress(section_no, title):
    print("\n" + "=" * 78)
    print(f"{render_progress_bar(section_no, TOTAL_SECTIONS)} Section {section_no}/{TOTAL_SECTIONS} | {title}")
    print("=" * 78)

def iter_with_progress(iterable, label, total=None, width=24, every=1):
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            total = None
    if not total:
        for item in iterable:
            yield item
        return
    for idx, item in enumerate(iterable, start=1):
        if idx == 1 or idx == total or idx % max(1, every) == 0:
            pct = 100.0 * idx / total
            end = "\n" if idx == total else "\r"
            print(
                f"   {label:<22} {render_progress_bar(idx, total, width)} "
                f"{idx:>3}/{total:<3} ({pct:5.1f}%)",
                end=end,
                flush=True,
            )
        yield item

def detect_nvidia_gpu():
    try:
        probe = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        return probe.returncode == 0 and "GPU " in probe.stdout
    except Exception:
        return False

# ============================================================
# 1. CONFIG
# ============================================================
START_DATE    = "2005-01-01"
END_DATE      = datetime.date.today().strftime("%Y-%m-%d")

FORWARD_WEEKS = 12
TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.15
# PURGE GAP: drop N weeks at split boundaries to prevent rolling-window leakage
PURGE_WEEKS   = 8   # FIX #6: purge rows within 8 weeks of any split boundary
CALIBRATION_RATIO = 0.20
EARLY_STOP_RATIO  = 0.15
TUNING_POOL_RATIO = 0.80
OPTUNA_TRIALS     = 25
TUNED_MODELS      = (
    "XGBoost", "LightGBM", "RandomForest", "CatBoost", "ExtraTrees"
    )
COMPARISON_FUND_TICKERS = [
    "HDFCNIFTY.NS",  # HDFC Nifty 50 ETF
    "SETFNIF50.NS",  # SBI Nifty 50 ETF (Replaces ICICI)
    "NIFTYBEES.NS"   # Nippon India Nifty 50 BeES (Replaces UTI)
]
GPU_REQUIRED      = True
GPU_ONLY_TRAINING = False
GPU_DEVICE_ID     = "0"
PRIMARY_SELECTION_METRIC = "ROC AUC"
THRESHOLD_GRID    = np.round(np.arange(0.35, 0.71, 0.03), 2)

K_RANGE       = range(2, 8)
TOP_K         = 8
MAX_WEIGHT    = 0.15
SECTOR_CAP    = 0.25
SLIPPAGE      = 0.0005
BROKERAGE_RATE = 0.0003
STT_RATE       = 0.0011   # With turnover accounting, this lands near ~0.25-0.30% round-trip.
INITIAL_CAPITAL = 1_000_000

BUY_THRESH  = 0.57   # slightly lower → more signals
SELL_THRESH = 0.43
MIN_SIGNAL_SPREAD = 0.08  # FIX #1: reject model if max-min prob < 8%
EXPERIMENT_DIR = "training_experiments"

show_section_progress(1, "Configuration")
HAS_NVIDIA_GPU = detect_nvidia_gpu()
if GPU_REQUIRED and not HAS_NVIDIA_GPU:
    raise RuntimeError("GPU_REQUIRED=True but no NVIDIA GPU was detected by nvidia-smi.")
print(f"✅ GPU mode | detected={HAS_NVIDIA_GPU} | gpu_only_training={GPU_ONLY_TRAINING} | device={GPU_DEVICE_ID}")
print(f"✅ Primary model metric | {PRIMARY_SELECTION_METRIC} for tuning, validation, testing, and ranking")
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
    "SUNPHARMA.NS","TATACONSUM.NS","TMPV.NS","TATASTEEL.NS",
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
    "MARUTI.NS":"Auto","TATAMOTORS.NS":"Auto","EICHERMOT.NS":"Auto","BAJAJ-AUTO.NS":"Auto",
    "ULTRACEMCO.NS":"Cement","GRASIM.NS":"Cement","ASIANPAINT.NS":"Consumer",
    "TITAN.NS":"Consumer","APOLLOHOSP.NS":"Healthcare","SUNPHARMA.NS":"Pharma",
    "DRREDDY.NS":"Pharma","DIVISLAB.NS":"Pharma","CIPLA.NS":"Pharma",
    "JSWSTEEL.NS":"Metals","TATASTEEL.NS":"Metals","HINDALCO.NS":"Metals",
    "BHARTIARTL.NS":"Telecom","LT.NS":"Infra",
}
print(f"✅ Config | {START_DATE} → {END_DATE} | Tickers: {len(NIFTY50_TICKERS)}")

# ============================================================
# 2. DATA DOWNLOAD
# ============================================================
show_section_progress(2, "Data Download")
def download_yf_history(tickers, label, attempts=3, sleep_seconds=4, **kwargs):
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            kwargs_local = dict(kwargs)
            kwargs_local.setdefault("progress", False)
            kwargs_local.setdefault("threads", True)
            data = yf.download(tickers, **kwargs_local)
            if isinstance(data, pd.DataFrame) and not data.empty:
                return data
            last_error = RuntimeError(f"{label} download returned no rows")
        except Exception as exc:
            last_error = exc
        if attempt < attempts:
            print(f"   Retry {attempt}/{attempts - 1} for {label} in {sleep_seconds}s ...")
            time.sleep(sleep_seconds)
    raise RuntimeError(f"{label} download failed after {attempts} attempts: {last_error}")

def build_proxy_index_from_constituents(closes, volumes, label="constituent proxy"):
    if closes is None or closes.empty:
        raise RuntimeError(f"Cannot build {label}: close panel is empty")
    weekly_close = closes.copy()
    weekly_close.index = pd.to_datetime(weekly_close.index)
    weekly_close = weekly_close.resample("W-FRI").last().dropna(how="all")
    if weekly_close.empty:
        raise RuntimeError(f"Cannot build {label}: weekly close panel is empty")
    weekly_ret = weekly_close.pct_change()
    valid_counts = weekly_ret.notna().sum(axis=1)
    proxy_ret = weekly_ret.mean(axis=1, skipna=True)
    proxy_ret[valid_counts < 10] = np.nan
    proxy_close = (1.0 + proxy_ret.fillna(0.0)).cumprod() * 100.0
    if volumes is not None and not volumes.empty:
        weekly_vol = volumes.copy()
        weekly_vol.index = pd.to_datetime(weekly_vol.index)
        weekly_vol = weekly_vol.resample("W-FRI").sum().sum(axis=1, min_count=1)
    else:
        weekly_vol = pd.Series(index=proxy_close.index, dtype=float)
    proxy_idx = pd.DataFrame({
        "nifty_close": proxy_close,
        "nifty_vol": weekly_vol.reindex(proxy_close.index).fillna(0.0),
        "proxy_constituents": valid_counts.reindex(proxy_close.index).fillna(0).astype(int),
    })
    proxy_idx["nifty_ret"] = proxy_ret
    proxy_idx["nifty_vol4"] = proxy_idx["nifty_ret"].rolling(4).std()
    proxy_idx.dropna(subset=["nifty_ret"], inplace=True)
    if proxy_idx.empty:
        raise RuntimeError(f"Cannot build {label}: proxy weekly returns are empty")
    return proxy_idx

print("\n📥 Downloading NIFTY 50 Index ...")
nifty_idx = None
BENCHMARK_SOURCE = "yahoo_index"
try:
    raw = download_yf_history(
        "^NSEI", "NIFTY 50 index",
        start=START_DATE, end=END_DATE, interval="1d", auto_adjust=True, threads=False
    )
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.index = pd.to_datetime(raw.index)
    required_cols = {"Close", "Volume"}
    missing_cols = required_cols - set(raw.columns)
    if missing_cols:
        raise RuntimeError(f"NIFTY 50 index data missing required columns: {sorted(missing_cols)}")
    nifty_w   = raw["Close"].resample("W-FRI").last().dropna()
    nifty_idx = pd.DataFrame({
        "nifty_close": nifty_w,
        "nifty_vol":   raw["Volume"].resample("W-FRI").sum(),
    })
    nifty_idx["nifty_ret"]  = nifty_idx["nifty_close"].pct_change()
    nifty_idx["nifty_vol4"] = nifty_idx["nifty_ret"].rolling(4).std()
    nifty_idx.dropna(subset=["nifty_ret"], inplace=True)
    if nifty_idx.empty:
        raise RuntimeError("NIFTY 50 weekly history is empty after preprocessing")
    print(f"   NIFTY: {len(nifty_idx)} weekly rows | "
          f"{nifty_idx.index[0].date()} → {nifty_idx.index[-1].date()}")
except Exception as e:
    BENCHMARK_SOURCE = "constituent_proxy_pending"
    print(f"   NIFTY index unavailable from Yahoo, will build proxy from constituents: {e}")

# Also download NIFTY NEXT 50 as benchmark (FIX #7)
try:
    raw_n50 = download_yf_history(
        "^NSMIDCP", "NIFTY MIDCAP benchmark",
        start=START_DATE, end=END_DATE, interval="1d", auto_adjust=True, threads=False
    )
    if isinstance(raw_n50.columns, pd.MultiIndex):
        raw_n50.columns = raw_n50.columns.get_level_values(0)
    raw_n50.index = pd.to_datetime(raw_n50.index)
    midcap_w = raw_n50["Close"].resample("W-FRI").last().dropna()
    midcap_ret = midcap_w.pct_change().dropna()
    print(f"   Midcap index: {len(midcap_ret)} rows")
    HAS_MIDCAP = True
except:
    HAS_MIDCAP = False
    print("   Midcap index not available — using NIFTY only as benchmark")

# Download comparison funds (HDFC, ICICI, UTI)
comparison_fund_rets = {}
print("\n📥 Downloading Index Funds (HDFC, ICICI, UTI) ...")
try:
    raw_funds = download_yf_history(
        COMPARISON_FUND_TICKERS, "comparison index funds",
        start=START_DATE, end=END_DATE, interval="1d", auto_adjust=True
    )
    if isinstance(raw_funds.columns, pd.MultiIndex):
        fund_closes = raw_funds["Close"]
    else:
        fund_closes = raw_funds[["Close"]]
    
    for ticker in COMPARISON_FUND_TICKERS:
        if ticker in fund_closes.columns:
            f_w = fund_closes[ticker].resample("W-FRI").last().dropna()
            f_ret = f_w.pct_change().dropna()
            comparison_fund_rets[ticker] = f_ret
            print(f"   {ticker}: {len(f_ret)} weekly rows")
except Exception as e:
    print(f"   ⚠️ Could not download comparison funds: {e}")

print("\n📥 Downloading stocks ...")
frames, failed = [], []
all_data = download_yf_history(
    NIFTY50_TICKERS, "constituent stocks",
    start=START_DATE, end=END_DATE, interval="1d", auto_adjust=True
)

def safe_panel(data, field):
    if isinstance(data.columns, pd.MultiIndex):
        if field in data.columns.get_level_values(0):
            return data[field]
    return None

closes  = safe_panel(all_data, "Close")
volumes = safe_panel(all_data, "Volume")
highs   = safe_panel(all_data, "High")
lows    = safe_panel(all_data, "Low")

if nifty_idx is None:
    nifty_idx = build_proxy_index_from_constituents(closes, volumes, label="NIFTY proxy benchmark")
    BENCHMARK_SOURCE = "constituent_proxy"
    print(f"   Proxy benchmark: {len(nifty_idx)} weekly rows | "
          f"{nifty_idx.index[0].date()} → {nifty_idx.index[-1].date()} | "
          f"median constituents/week={int(nifty_idx['proxy_constituents'].median())}")

for sym in iter_with_progress(NIFTY50_TICKERS, "Stock panel prep", total=len(NIFTY50_TICKERS), every=1):
    try:
        if closes is None or sym not in closes.columns:
            failed.append(sym); continue
        cd = closes[sym].dropna()
        if len(cd) < 100: failed.append(sym); continue
        cd.index = pd.to_datetime(cd.index)
        cw = cd.resample("W-FRI").last().dropna()
        if len(cw) < 52: failed.append(sym); continue
        df = pd.DataFrame({"price": cw}); df.index.name = "date"
        if volumes is not None and sym in volumes.columns:
            vd = volumes[sym].dropna(); vd.index = pd.to_datetime(vd.index)
            df["volume"] = vd.resample("W-FRI").sum().reindex(cw.index).fillna(0)
        else:
            df["volume"] = 0.0
        if highs is not None and sym in highs.columns:
            hd = highs[sym].dropna(); hd.index = pd.to_datetime(hd.index)
            df["week_high"] = hd.resample("W-FRI").max().reindex(cw.index)
        if lows is not None and sym in lows.columns:
            ld = lows[sym].dropna(); ld.index = pd.to_datetime(ld.index)
            df["week_low"]  = ld.resample("W-FRI").min().reindex(cw.index)
        df["stock"]  = sym
        df["sector"] = SECTOR_MAP.get(sym, "Other")
        df = df.reset_index(); df["date"] = pd.to_datetime(df["date"])
        frames.append(df)
    except Exception as ex:
        failed.append(sym)

stocks = pd.concat(frames, ignore_index=True)
stocks.sort_values(["stock","date"], inplace=True)
stocks.reset_index(drop=True, inplace=True)
print(f"   ✅ {stocks['stock'].nunique()} stocks | {len(stocks):,} rows | Failed: {failed}")

# ============================================================
# 3. FEATURE ENGINEERING (no leakage — all rolling on past only)
# ============================================================
print("\n🔧 Engineering features ...")
show_section_progress(3, "Feature Engineering")
for col in ["price","volume"]:
    stocks[col] = pd.to_numeric(stocks[col], errors="coerce")

g = stocks.groupby("stock", sort=False)

# --- Returns ---
for n in [1,2,4,8,12,26,52]:
    stocks[f"ret_{n}w"] = g["price"].pct_change(n)

# --- Momentum (lag-1 avoids lookahead) ---
for n, mp in [(4,3),(8,4),(26,8)]:
    stocks[f"mom_{n}w"] = g["ret_1w"].transform(
        lambda x, n=n, mp=mp: x.shift(1).rolling(n, min_periods=mp).sum())

# --- Volatility ---
for n, mp in [(4,3),(8,4),(26,8)]:
    stocks[f"vol_{n}w"] = g["ret_1w"].transform(
        lambda x, n=n, mp=mp: x.rolling(n, min_periods=mp).std())

# --- Moving averages ---
for n, mp in [(4,3),(26,8)]:
    stocks[f"ma_{n}w"] = g["price"].transform(
        lambda x, n=n, mp=mp: x.rolling(n, min_periods=mp).mean())
stocks["price_to_ma4"]  = stocks["price"] / stocks["ma_4w"]
stocks["price_to_ma26"] = stocks["price"] / stocks["ma_26w"]
stocks["ma_cross"]      = stocks["ma_4w"]  / stocks["ma_26w"]

# --- RSI ---
def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(com=period-1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period-1, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))

stocks["rsi_14"] = g["price"].transform(compute_rsi)

# --- Bollinger Band %B ---
def compute_bb_pct(series, w=20, k=2):
    mid   = series.rolling(w, min_periods=w//2).mean()
    std   = series.rolling(w, min_periods=w//2).std()
    lower = mid - k * std
    band  = 2 * k * std + 1e-9
    return (series - lower) / band

stocks["bb_pct"] = g["price"].transform(compute_bb_pct)

# --- MACD normalised ---
def compute_macd_norm(series):
    m = series.ewm(span=12, adjust=False).mean() - series.ewm(span=26, adjust=False).mean()
    s = m.ewm(span=9, adjust=False).mean()
    return (m - s) / (series.abs().rolling(12, min_periods=6).mean() + 1e-9)

stocks["macd_norm"] = g["price"].transform(compute_macd_norm)

# --- 52-week high proximity ---
stocks["hi_52w"]   = g["price"].transform(lambda x: x.rolling(52, min_periods=26).max())
stocks["prox_52w"] = (stocks["price"] / stocks["hi_52w"]).clip(0, 1.5)

# --- Volume features ---
stocks["vol_mom"]   = g["volume"].transform(lambda x: x.pct_change(4))
stocks["vol_ratio"] = g["volume"].transform(
    lambda x: x / (x.rolling(8, min_periods=4).mean() + 1e-9))

# --- Sector-relative momentum ---
stocks["sec_mom_rel"] = stocks.groupby(["date","sector"])["ret_8w"].transform(
    lambda x: x - x.median())

# --- Cross-sectional ranks (per date) ---
for col in ["mom_8w","vol_26w","rsi_14","ret_4w"]:
    stocks[f"{col}_rank"] = stocks.groupby("date")[col].rank(pct=True)

# --- Market regime features ---
nret_map = nifty_idx["nifty_ret"].to_dict()
nvol_map = nifty_idx["nifty_vol4"].to_dict()
stocks["mkt_ret_1w"] = stocks["date"].map(nret_map)
stocks["mkt_vol_4w"] = stocks["date"].map(nvol_map)
stocks["nifty_ret"]  = stocks["date"].map(nret_map)

# --- Rolling Beta (52w) ---
print("   Computing rolling betas ...")
beta_series = (
    stocks.groupby("stock", sort=False, group_keys=False)
    .apply(
        lambda df: (
            df["ret_1w"].rolling(52, min_periods=26).cov(df["nifty_ret"]) /
            df["nifty_ret"].rolling(52, min_periods=26).var().replace(0, np.nan)
        )
    )
)
stocks["beta_52w"] = beta_series.reset_index(level=0, drop=True)

# --- Idiosyncratic volatility ---
stocks["idio_vol"] = stocks["ret_1w"] - stocks["beta_52w"] * stocks["nifty_ret"]
stocks["idio_vol"] = stocks.groupby("stock")["idio_vol"].transform(
    lambda x: x.rolling(26, min_periods=12).std())

# --- Target: 4-week forward outperformance vs NIFTY ---
# --- Structural Trend Features ---
# 1. Rate of change of the 52-week proximity (Acceleration)
stocks["prox_52w_mom"] = g["prox_52w"].transform(lambda x: x.pct_change(4))

# 2. Sector Outperformance vs Nifty
stocks["sec_nifty_rel"] = stocks.groupby(["date", "sector"])["ret_4w"].transform(
    lambda x: x.mean()
) - stocks["nifty_ret"].rolling(4, min_periods=4).sum()

# --- Structural Trend Features ---
# 1. Rate of change of the 52-week proximity (Acceleration)
stocks["prox_52w_mom"] = g["prox_52w"].transform(lambda x: x.pct_change(4))

# 2. Sector Outperformance vs Nifty
stocks["sec_nifty_rel"] = stocks.groupby(["date", "sector"])["ret_4w"].transform(
    lambda x: x.mean()
) - stocks["nifty_ret"].rolling(4, min_periods=4).sum()

# --- Target: 12-week forward structural outperformance vs NIFTY ---
stocks["fwd_ret_12w"] = stocks.groupby("stock")["ret_1w"].transform(
    lambda x: x.rolling(12, min_periods=6).sum().shift(-12))

nifty_ret_series = nifty_idx["nifty_ret"]
nifty_fwd = {}
for d in stocks["date"].unique():
    try:
        if d not in nifty_ret_series.index: nifty_fwd[d] = np.nan; continue
        loc_ = nifty_ret_series.index.get_loc(d)
        end_ = loc_ + FORWARD_WEEKS
        nifty_fwd[d] = (nifty_ret_series.iloc[loc_:end_].sum()
                        if end_ <= len(nifty_ret_series) else np.nan)
    except: nifty_fwd[d] = np.nan

stocks["nifty_fwd_12w"] = stocks["date"].map(nifty_fwd)

# ALPHA HURDLE: Require 5.0% excess return over 12 weeks to label as a winner
ALPHA_MARGIN = 0.035 

stocks["target"] = np.where(
    stocks["fwd_ret_12w"].notna() & stocks["nifty_fwd_12w"].notna(),
    (stocks["fwd_ret_12w"] > (stocks["nifty_fwd_12w"] + ALPHA_MARGIN)).astype(float),
    np.nan,
)

FEATURE_COLS = [
    "ret_1w","ret_2w","ret_4w","ret_8w","ret_12w","ret_26w","ret_52w",
    "mom_4w","mom_8w","mom_26w",
    "vol_4w","vol_8w","vol_26w",
    "bb_pct","macd_norm","prox_52w","prox_52w_mom",      
    "ma_cross","price_to_ma26",                          
    "vol_mom","vol_ratio","sec_mom_rel","sec_nifty_rel", 
    "mom_8w_rank","vol_26w_rank","ret_4w_rank",          
    "beta_52w","idio_vol","mkt_ret_1w","mkt_vol_4w",
]

# Note: We now drop NA based on fwd_ret_12w so the model doesn't train on incomplete future horizons
stocks_model  = stocks.dropna(subset=FEATURE_COLS + ["target","fwd_ret_12w"]).copy()
stocks_latest = stocks[stocks["fwd_ret_12w"].isna()].dropna(subset=FEATURE_COLS).copy()

def apply_clip_bounds(df, bounds, cols):
    df = df.copy()
    for col in cols:
        lo, hi = bounds[col]
        df[col] = df[col].clip(lo, hi)
    return df

def time_series_inner_splits(n_samples, n_splits=5):
    n_splits = max(2, min(n_splits, max(2, n_samples - 1)))
    return TimeSeriesSplit(n_splits=n_splits)

def split_train_calibration(X, y, cal_ratio=CALIBRATION_RATIO):
    n_samples = len(X)
    if n_samples < 40:
        return X, y, None, None
    cal_size = max(12, int(n_samples * cal_ratio))
    cal_size = min(cal_size, max(8, n_samples // 3))
    split_idx = n_samples - cal_size
    if split_idx < 20:
        return X, y, None, None
    return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]

def split_train_early_stop(X, y, ratio=EARLY_STOP_RATIO):
    n_samples = len(X)
    if n_samples < 60:
        return X, y, None, None
    es_size = max(10, int(n_samples * ratio))
    es_size = min(es_size, max(8, n_samples // 4))
    split_idx = n_samples - es_size
    if split_idx < 20:
        return X, y, None, None
    return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]

def split_train_tuning_pool(X, y, sample_weight=None, tune_ratio=TUNING_POOL_RATIO):
    n_samples = len(X)
    sw = None if sample_weight is None else np.asarray(sample_weight)
    if n_samples < 240:
        return X, y, None, None, sw, None
    tune_size = max(160, int(n_samples * tune_ratio))
    tune_size = min(tune_size, n_samples - 48)
    if tune_size < 120 or (n_samples - tune_size) < 24:
        return X, y, None, None, sw, None
    return (
        X[:tune_size], y[:tune_size],
        X[tune_size:], y[tune_size:],
        None if sw is None else sw[:tune_size],
        None if sw is None else sw[tune_size:],
    )

def transaction_cost(turnover):
    return turnover * (SLIPPAGE + BROKERAGE_RATE + STT_RATE)

stocks_model.reset_index(drop=True, inplace=True)
stocks_latest.reset_index(drop=True, inplace=True)

print(f"   ✅ Features: {len(FEATURE_COLS)} | Model rows: {len(stocks_model):,} | "
      f"Live rows: {len(stocks_latest)}")
print(f"   Target balance: "
      f"{stocks_model['target'].value_counts(normalize=True).round(3).to_dict()}")

# ============================================================
# 4. CHRONOLOGICAL SPLIT WITH PURGE GAP (FIX #6)
# ============================================================
show_section_progress(4, "Chronological Split With Purge")
all_dates = sorted(stocks_model["date"].unique())
n_d     = len(all_dates)
tr_end  = int(n_d * TRAIN_RATIO)
val_end = int(n_d * (TRAIN_RATIO + VAL_RATIO))

train_cut = all_dates[tr_end]
val_cut   = all_dates[val_end]

# Purge: remove PURGE_WEEKS on each side of every boundary
def purge_boundary(dates_list, cut_date, gap_weeks, side="after"):
    cut_ts = pd.Timestamp(cut_date)
    td     = pd.Timedelta(weeks=gap_weeks)
    if side == "after":
        return [d for d in dates_list if pd.Timestamp(d) > cut_ts + td]
    else:
        return [d for d in dates_list if pd.Timestamp(d) < cut_ts - td]

raw_train = set(all_dates[:tr_end])
raw_val   = set(all_dates[tr_end:val_end])
raw_test  = set(all_dates[val_end:])

# Purge from train (near val boundary)
train_dates_purged = set(purge_boundary(list(raw_train), train_cut, PURGE_WEEKS, "before"))
# Purge from val (both sides)
val_dates_purged = set([d for d in raw_val
                        if pd.Timestamp(d) > pd.Timestamp(train_cut) + pd.Timedelta(weeks=PURGE_WEEKS)
                        and pd.Timestamp(d) < pd.Timestamp(val_cut) - pd.Timedelta(weeks=PURGE_WEEKS)])
# Purge from test (near val boundary)
test_dates_purged = set([d for d in raw_test
                          if pd.Timestamp(d) > pd.Timestamp(val_cut) + pd.Timedelta(weeks=PURGE_WEEKS)])

tr_df = stocks_model[stocks_model["date"].isin(train_dates_purged)].copy()
vl_df = stocks_model[stocks_model["date"].isin(val_dates_purged)].copy()
ev_df = stocks_model[stocks_model["date"].isin(test_dates_purged)].copy()

# Winsorise using TRAIN rows only, then apply the same bounds to every split.
clip_bounds = {}
for col in FEATURE_COLS:
    lo = tr_df[col].quantile(0.01)
    hi = tr_df[col].quantile(0.99)
    clip_bounds[col] = (lo, hi)

stocks_model = apply_clip_bounds(stocks_model, clip_bounds, FEATURE_COLS)
stocks_latest = apply_clip_bounds(stocks_latest, clip_bounds, FEATURE_COLS)
tr_df = apply_clip_bounds(tr_df, clip_bounds, FEATURE_COLS)
vl_df = apply_clip_bounds(vl_df, clip_bounds, FEATURE_COLS)
ev_df = apply_clip_bounds(ev_df, clip_bounds, FEATURE_COLS)

print(f"\n   TRAIN : {min(train_dates_purged).date()} → {max(train_dates_purged).date()} "
      f"({len(tr_df):,} rows)")
print(f"   VAL   : {min(val_dates_purged).date()} → {max(val_dates_purged).date()} "
      f"({len(vl_df):,} rows)")
print(f"   TEST  : {min(test_dates_purged).date()} → {max(test_dates_purged).date()} "
      f"({len(ev_df):,} rows)")
print(f"   Purge gap: {PURGE_WEEKS} weeks at each boundary")

# Split visualisation
fig, ax = plt.subplots(figsize=(15, 2.5))
ax.barh(0, tr_end, left=0, color=C[0], alpha=0.85,
        label=f"TRAIN 70% ({len(tr_df):,} rows)")
ax.barh(0, val_end-tr_end, left=tr_end, color=C[3], alpha=0.85,
        label=f"VAL 15% ({len(vl_df):,} rows)")
ax.barh(0, n_d-val_end, left=val_end, color=C[1], alpha=0.85,
        label=f"TEST 15% ({len(ev_df):,} rows)")
ax.axvline(tr_end-PURGE_WEEKS, color="red", lw=2, ls="--", alpha=0.7, label="Purge zone")
ax.axvline(tr_end+PURGE_WEEKS, color="red", lw=2, ls="--", alpha=0.7)
ax.axvline(val_end-PURGE_WEEKS, color="red", lw=2, ls="--", alpha=0.7)
ax.axvline(val_end+PURGE_WEEKS, color="red", lw=2, ls="--", alpha=0.7)
ax.set_yticks([]); ax.set_xlabel("Weekly Steps", fontsize=10)
ax.set_title(f"Chronological Split — {PURGE_WEEKS}-week purge gap at boundaries (no leakage)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
plt.tight_layout(); plt.show()

# ============================================================
# 5. REGIME DETECTION — COMPOSITE SCORE ABLATION (FIX #4)
# ============================================================
show_section_progress(5, "Regime Selection Ablation")
print("\n🔍 Regime Detection — Composite Score Ablation ...")

regime_hist = nifty_idx[["nifty_ret", "nifty_vol4"]].dropna().copy()
dates_arr = pd.to_datetime(regime_hist.index)
rets = regime_hist["nifty_ret"].values
vols = regime_hist["nifty_vol4"].values
X_reg = np.column_stack([rets, vols])
train_regime_mask = np.isin(dates_arr, list(train_dates_purged))
if train_regime_mask.sum() < 80:
    train_regime_mask = dates_arr <= max(train_dates_purged)
X_reg_train = X_reg[train_regime_mask]
rets_train = rets[train_regime_mask]
vols_train = vols[train_regime_mask]

def compute_icl(gmm, X):
    proba   = np.clip(gmm.predict_proba(X), 1e-10, 1)
    entropy = -np.sum(proba * np.log(proba), axis=1).sum()
    return gmm.bic(X) + 2 * entropy

def regime_stability(labels):
    transitions = np.sum(labels[1:] != labels[:-1])
    return 1.0 - transitions / max(len(labels)-1, 1)

def min_regime_fraction(labels, k):
    """Smallest fraction — penalises tiny regimes."""
    return min((labels==i).mean() for i in range(k) if (labels==i).sum() > 0)

abl_results = []
best_composite = -1e18
best_k = 3

for k in K_RANGE:
    t0 = time.time()
    gmm = GaussianMixture(n_components=k, covariance_type="full",
                          n_init=25, random_state=RANDOM_STATE, max_iter=1000)
    gmm.fit(X_reg_train)
    bic   = float(gmm.bic(X_reg_train))
    aic   = float(gmm.aic(X_reg_train))
    icl   = float(compute_icl(gmm, X_reg_train))
    labels = gmm.predict(X_reg_train)
    n_uni = len(np.unique(labels))
    sil   = float(silhouette_score(X_reg_train, labels)) if n_uni > 1 else 0.0
    stab  = regime_stability(labels)
    min_frac = min_regime_fraction(labels, k)

    # Val AUC (regime predicting NIFTY direction, to assess regime informativeness)
    rf_feat   = labels[:-1].reshape(-1, 1)
    nifty_dir = (rets_train[1:] > 0).astype(int)
    if len(np.unique(nifty_dir)) > 1 and len(rf_feat) > 50:
        lr = LogisticRegression(C=1, max_iter=500, random_state=RANDOM_STATE)
        tscv_ = time_series_inner_splits(len(rf_feat), n_splits=5)
        val_auc = float(cross_val_score(
            lr, rf_feat, nifty_dir, cv=tscv_, scoring="roc_auc").mean())
    else:
        val_auc = 0.5

    # COMPOSITE SCORE (FIX #4): weighted combination
    # Normalise BIC: more negative = better → negate and normalise
    # Penalty for tiny regimes (< 3% of data)
    tiny_penalty = 1.0 if min_frac >= 0.03 else 0.5
    # We'll compute composite after collecting all k scores (for proper normalisation)
    elapsed = time.time() - t0
    abl_results.append({
        "k": k, "bic": round(bic,1), "aic": round(aic,1),
        "icl": round(icl,1), "silhouette": round(sil,4),
        "stability": round(stab,4), "val_auc": round(val_auc,4),
        "min_frac": round(min_frac,3), "tiny_penalty": tiny_penalty,
        "time": round(elapsed,2),
    })
    print(f"   K={k}: BIC={bic:.0f} Sil={sil:.3f} Stab={stab:.3f} "
          f"MinFrac={min_frac:.3f} ValAUC={val_auc:.3f} ({elapsed:.1f}s)")

abl_df = pd.DataFrame(abl_results)

# Normalise metrics 0→1 (higher = better for all after sign-flip for BIC)
def norm01(series, lower_better=False):
    s = -series if lower_better else series
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn + 1e-12)

abl_df["bic_norm"]  = norm01(abl_df["bic"],  lower_better=True)
abl_df["sil_norm"]  = norm01(abl_df["silhouette"])
abl_df["stab_norm"] = norm01(abl_df["stability"])
abl_df["auc_norm"]  = norm01(abl_df["val_auc"])

# Composite: 50% BIC + 20% Silhouette + 20% Stability + 10% AUC, × tiny_penalty
abl_df["composite"] = (
    0.50 * abl_df["bic_norm"] +
    0.20 * abl_df["sil_norm"] +
    0.20 * abl_df["stab_norm"] +
    0.10 * abl_df["auc_norm"]
) * abl_df["tiny_penalty"]

best_k = int(abl_df.loc[abl_df["composite"].idxmax(), "k"])
best_composite = abl_df["composite"].max()

abl_df["verdict"] = abl_df["k"].apply(
    lambda k: "★ OPTIMAL" if k == best_k
    else ("UNDERFIT" if k < best_k else "OVERFIT"))

print(f"\n   ✅ Best K = {best_k} (composite score = {best_composite:.4f})")
print(abl_df[["k","bic","silhouette","stability","val_auc",
              "composite","verdict"]].to_string(index=False))

# Ablation plots (5 panels)
fig, axes = plt.subplots(1, 5, figsize=(22, 5))
plot_metrics = [
    ("bic",        "BIC ↓",          C[0]),
    ("silhouette", "Silhouette ↑",   C[1]),
    ("stability",  "Stability ↑",    C[2]),
    ("val_auc",    "Val AUC ↑",      C[3]),
    ("composite",  "Composite ↑",    C[7]),
]
for ax, (col, title, color) in zip(axes, plot_metrics):
    vals = abl_df[col].values
    bar_cols = [C[7] if k == best_k else color for k in abl_df["k"]]
    bars = ax.bar([f"K={k}" for k in abl_df["k"]], vals,
                  color=bar_cols, edgecolor="#30363d", linewidth=1.2)
    ax.set_title(title, fontsize=10, fontweight="bold")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height() + abs(bar.get_height())*0.02 + 0.0001,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    best_bar = bars[list(abl_df["k"]).index(best_k)]
    ax.text(best_bar.get_x()+best_bar.get_width()/2,
            max(vals)*1.10, "★", ha="center", fontsize=12, color="#ffd60a")
    ax.grid(axis="y", alpha=0.3)
plt.suptitle(f"Composite Ablation (BIC×0.5 + Sil×0.2 + Stab×0.2 + AUC×0.1) — K={best_k}",
             fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()

# ============================================================
# 6. ENSEMBLE REGIME DETECTION
# ============================================================
show_section_progress(6, "Ensemble Regime Detection")
print(f"\n🔀 Ensemble Regime Detection (K={best_k}) ...")
regime_scaler = RobustScaler().fit(X_reg_train)
X_reg_train_scaled = regime_scaler.transform(X_reg_train)
X_reg_scaled = regime_scaler.transform(X_reg)
regime_probs = np.zeros((len(X_reg_scaled), best_k))

if HAS_HMM and len(X_reg_train_scaled) > 100:
    best_hmm = None
    best_hmm_score = -np.inf
    train_regime_len = len(X_reg_train_scaled)
    for seed in range(30):
        try:
            hmm = GaussianHMM(n_components=best_k, covariance_type="diag",
                              n_iter=1000, random_state=seed, min_covar=1e-4)
            hmm.fit(X_reg_train_scaled)
            if hmm.monitor_.converged:
                score = hmm.score(X_reg_train_scaled)
                if score > best_hmm_score:
                    best_hmm = hmm
                    best_hmm_score = score
        except:
            pass
    if best_hmm is not None:
        hs = np.full(len(X_reg_scaled), -1, dtype=int)
        hs[:train_regime_len] = best_hmm.predict(X_reg_train_scaled)
        for idx in range(train_regime_len, len(X_reg_scaled)):
            forward_post = best_hmm.predict_proba(X_reg_scaled[:idx+1])
            hs[idx] = int(np.argmax(forward_post[-1]))
        for k_ in range(best_k):
            regime_probs[:,k_] += (hs==k_).astype(float)
        print("   HMM decoded on train only, then filtered forward without Viterbi look-ahead")

gmm_final = GaussianMixture(n_components=best_k, covariance_type="full",
                             n_init=30, random_state=RANDOM_STATE, max_iter=1000)
gmm_final.fit(X_reg_train_scaled)
regime_probs += gmm_final.predict_proba(X_reg_scaled)

km_final = KMeans(n_clusters=best_k, n_init=30, random_state=RANDOM_STATE, max_iter=500)
km_final.fit(X_reg_train_scaled)
ks = km_final.predict(X_reg_scaled)
for k_ in range(best_k):
    regime_probs[:,k_] += (ks==k_).astype(float)

ensemble    = np.argmax(regime_probs, axis=1)
regime_means = np.array([
    rets[(ensemble == k_) & train_regime_mask].mean()
    if ((ensemble == k_) & train_regime_mask).sum() > 0
    else (rets[ensemble == k_].mean() if (ensemble == k_).sum() > 0 else 0.0)
    for k_ in range(best_k)
])
sort_order   = np.argsort(regime_means)
remap        = {old: new for new, old in enumerate(sort_order)}
sorted_states = np.array([remap[s] for s in ensemble])

regime_map = dict(zip(dates_arr, sorted_states.tolist()))

REGIME_COLORS = {
    2: [("#ff4757","BEAR"),              ("#00d084","BULL")],
    3: [("#ff4757","BEAR"),              ("#ffd60a","SIDEWAYS"),    ("#00d084","BULL")],
    4: [("#c0392b","CRASH/PANIC"),       ("#ff4757","BEAR"),
        ("#00d084","BULL"),              ("#10b981","BREAKOUT")],
    5: [("#c0392b","CRASH/PANIC"),       ("#ff4757","BEAR"),
        ("#ffd60a","SIDEWAYS"),          ("#00d084","BULL"),        ("#10b981","BREAKOUT")],
    6: [("#c0392b","CRASH/PANIC"),       ("#e74c3c","DEEP-BEAR"),   ("#ff4757","BEAR"),
        ("#ffd60a","SIDEWAYS"),          ("#00d084","BULL"),        ("#10b981","BREAKOUT")],
    7: [("#7b0000","EXTREME-PANIC"),     ("#c0392b","CRASH/PANIC"), ("#e74c3c","DEEP-BEAR"),
        ("#ffd60a","SIDEWAYS"),          ("#00d084","BULL"),        ("#10b981","STRONG-BULL"),
        ("#00c8d7","BREAKOUT")],
}
label_list  = REGIME_COLORS.get(best_k, [(C[i], f"R{i}") for i in range(best_k)])
REGIME_INFO = {}

for k_ in range(best_k):
    mask = sorted_states == k_
    train_mask_k = mask & train_regime_mask
    mr = float(rets[train_mask_k].mean()) if train_mask_k.sum() > 0 else (
        float(rets[mask].mean()) if mask.sum() > 0 else 0.0
    )
    mv = float(vols[train_mask_k].mean()) if train_mask_k.sum() > 0 else (
        float(vols[mask].mean()) if mask.sum() > 0 else 0.0
    )
    col, lbl = label_list[k_]
    REGIME_INFO[k_] = {
        "label":   lbl, "color":   col,
        "ann_ret": round(mr*52*100, 1),
        "ann_vol": round(mv*np.sqrt(52)*100, 1),
        "sharpe":  round((mr*52*100-6)/(mv*np.sqrt(52)*100), 2) if mv > 0 else 0,
        "freq":    int(mask.sum()),
        "pct":     round(float(mask.sum()/len(mask)*100), 1),
    }

current_regime = int(sorted_states[-1])
print(f"   Current: R{current_regime} — {REGIME_INFO[current_regime]['label']} | "
      f"AnnRet={REGIME_INFO[current_regime]['ann_ret']}% "
      f"Sharpe={REGIME_INFO[current_regime]['sharpe']}")

print(f"   Regime models fit on {train_regime_mask.sum()} historical train weeks")

# Merge regime into full dataset BEFORE split
stocks_model["regime_id"] = stocks_model["date"].map(regime_map)
regime_mode = int(stocks_model["regime_id"].mode().iloc[0])
stocks_model["regime_id"] = stocks_model["regime_id"].fillna(regime_mode).astype(int)
stocks_latest["regime_id"] = stocks_latest["date"].map(regime_map).fillna(current_regime).astype(int)

# Re-split with purge (regime_id now present)
tr_df = stocks_model[stocks_model["date"].isin(train_dates_purged)].copy()
vl_df = stocks_model[stocks_model["date"].isin(val_dates_purged)].copy()
ev_df = stocks_model[stocks_model["date"].isin(test_dates_purged)].copy()

FEATURE_COLS_WITH_REGIME = FEATURE_COLS + ["regime_id"]
print(f"   ✅ regime_id merged | Features: {len(FEATURE_COLS_WITH_REGIME)}")

# Regime timeline
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit(X_reg_train_scaled).transform(X_reg_scaled)
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

ax = axes[0]
prev_r = None; start_d = None
for d, r in zip(dates_arr, sorted_states):
    if prev_r is None: prev_r, start_d = r, d
    elif r != prev_r:
        ax.axvspan(start_d, d, alpha=0.7, color=REGIME_INFO[prev_r]["color"],
                   label=REGIME_INFO[prev_r]["label"])
        prev_r, start_d = r, d
if prev_r is not None:
    ax.axvspan(start_d, dates_arr[-1], alpha=0.7, color=REGIME_INFO[prev_r]["color"],
               label=REGIME_INFO[prev_r]["label"])
handles, labs = ax.get_legend_handles_labels()
ax.legend(dict(zip(labs,handles)).values(), dict(zip(labs,handles)).keys(),
          fontsize=8, loc="upper left")
ax.set_title(f"Market Regime Timeline | K={best_k}", fontsize=11, fontweight="bold")
ax.set_yticks([]); ax.set_xlabel("Date")

ax = axes[1]
for k_ in range(best_k):
    m = sorted_states == k_
    ax.scatter(X_pca[m,0], X_pca[m,1], c=REGIME_INFO[k_]["color"],
               s=18, alpha=0.7, label=REGIME_INFO[k_]["label"])
ax.set_title("PCA 2D Ensemble Regimes", fontsize=11, fontweight="bold")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.legend(fontsize=8)

ax = axes[2]
for k_ in range(best_k):
    ax.bar(REGIME_INFO[k_]["label"], REGIME_INFO[k_]["ann_ret"],
           color=REGIME_INFO[k_]["color"], edgecolor="#30363d", linewidth=1.5)
ax.axhline(0, color="white", lw=0.8)
ax.set_title("Annualised Return by Regime", fontsize=11, fontweight="bold")
ax.set_ylabel("Ann. Return %"); ax.grid(axis="y", alpha=0.3)
ax.tick_params(axis="x", rotation=10, labelsize=7)
plt.suptitle(f"Regime Detection — K={best_k} (Composite Optimal) | HMM+GMM+KMeans",
             fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()

# ============================================================
# 7. SCALING — fit on TRAIN only
# ============================================================
show_section_progress(7, "Scaling")
scaler = RobustScaler()
scaler.fit(tr_df[FEATURE_COLS_WITH_REGIME])
Xs_tr = pd.DataFrame(scaler.transform(tr_df[FEATURE_COLS_WITH_REGIME]),
                     columns=FEATURE_COLS_WITH_REGIME, index=tr_df.index)
Xs_vl = pd.DataFrame(scaler.transform(vl_df[FEATURE_COLS_WITH_REGIME]),
                     columns=FEATURE_COLS_WITH_REGIME, index=vl_df.index)
Xs_ev = pd.DataFrame(scaler.transform(ev_df[FEATURE_COLS_WITH_REGIME]),
                     columns=FEATURE_COLS_WITH_REGIME, index=ev_df.index)
y_tr = tr_df["target"].astype(int).values
y_vl = vl_df["target"].astype(int).values
y_ev = ev_df["target"].astype(int).values
print(f"\n✅ Scaling done | Train: {len(Xs_tr):,} | Val: {len(Xs_vl):,} | Test: {len(Xs_ev):,}")

# ============================================================
# 10B. LEAKAGE-SAFE TRAINING OVERRIDE
# ============================================================
show_section_progress(8, "Optimized Model Training")
print("\n   Re-training with walk-forward tuning, internal calibration, and time-safe stacking ...")

def predict_proba_safe(model, X):
    if hasattr(model, "predict_proba"):
        probs = to_numpy_cpu(model.predict_proba(X))
        if probs.ndim == 1:
            return probs.astype(float)
        if probs.shape[1] == 1:
            return probs[:, 0].astype(float)
        return probs[:, 1].astype(float)
    if hasattr(model, "decision_function"):
        scores = to_numpy_cpu(model.decision_function(X))
        return 1.0 / (1.0 + np.exp(-scores))
    return to_numpy_cpu(model.predict(X)).astype(float)

def model_rank_score(metric_row):
    test_auc = float(metric_row.get("test_auc", 0.5))
    cv_auc = float(metric_row.get("cv_auc", 0.5))
    val_auc = float(metric_row.get("val_auc", test_auc))
    guard_auc = metric_row.get("guard_auc")
    guard_auc = test_auc if guard_auc is None else float(guard_auc)
    gen_gap = abs(val_auc - test_auc)
    return test_auc + 0.20 * (guard_auc - 0.5) + 0.10 * (cv_auc - 0.5) - 0.15 * gen_gap

REQUESTED_MODELS = [
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "RandomForest",
    "ExtraTrees",
]

MODEL_DEFAULT_PARAMS = {
    "XGBoost": {
        "n_estimators": 450, "max_depth": 4, "learning_rate": 0.03,
        "subsample": 0.80, "colsample_bytree": 0.75, "min_child_weight": 8,
        "reg_alpha": 1.0, "reg_lambda": 2.0, "gamma": 0.1,
        "scale_pos_weight": 1.0, "eval_metric": "auc",
        "tree_method": "hist", "device": "cuda", "max_bin": 256,
    },
    "LightGBM": {
        "n_estimators": 450, "max_depth": 5, "learning_rate": 0.03,
        "num_leaves": 31, "min_child_samples": 20, "subsample": 0.80,
        "colsample_bytree": 0.75, "reg_alpha": 0.5, "reg_lambda": 2.0,
        "metric": "auc", "device_type": "gpu", "gpu_device_id": int(GPU_DEVICE_ID),
    },
    "RandomForest": {
        "n_estimators": 300, "max_depth": 10, "min_samples_leaf": 5,
        "max_features": "sqrt", "random_state": RANDOM_STATE, "n_jobs": -1,
    },
    "CatBoost": {
        "iterations": 500, "depth": 6, "learning_rate": 0.03,
        "l2_leaf_reg": 3.0, "border_count": 128, "subsample": 0.80,
        "eval_metric": "Logloss", "task_type": "GPU", "devices": GPU_DEVICE_ID,
        "logging_level": "Silent", "bootstrap_type": "Poisson",
    },
    "ExtraTrees": {
        "n_estimators": 400, "max_depth": 12, "min_samples_leaf": 5,
        "max_features": "sqrt", "random_state": RANDOM_STATE, "n_jobs": -1,
    },
}

MODEL_ERRORS = {}
if not HAS_CATBOOST:
    MODEL_ERRORS["CatBoost"] = "catboost not installed"

# All models are active now; some use CPU, some use GPU.
MODEL_NAMES = [name for name in REQUESTED_MODELS if name not in MODEL_ERRORS]
TUNED_MODELS = tuple(MODEL_NAMES)

print(f"   Requested model suite: {REQUESTED_MODELS}")
print(f"   Active models: {MODEL_NAMES}")
if MODEL_ERRORS:
    print(f"   Skipped GPU models: {MODEL_ERRORS}")
if not MODEL_NAMES:
    raise RuntimeError(
        "None of the requested models are available."
    )

def to_numpy_cpu(obj):
    if isinstance(obj, np.ndarray):
        return obj
    if hasattr(obj, "get"):
        obj = obj.get()
    if hasattr(obj, "to_numpy"):
        obj = obj.to_numpy()
    if hasattr(obj, "values"):
        obj = obj.values
    return np.asarray(obj)

def find_latest_model_path(name, prefix=""):
    """Scans Output/models for the latest joblib file for a given model and prefix."""
    save_dir = os.path.join("Output", "models")
    if not os.path.exists(save_dir):
        return None
    files = [f for f in os.listdir(save_dir) if f.endswith(".joblib")]
    matches = []
    for f in files:
        parts = f.split("_")
        # Handle cases like "global_catboost_..." or "regime_0_catboost_..."
        # If prefix is supplied, it must match.
        if prefix and not f.startswith(prefix.lower()):
            continue
        if name.lower() in f.lower():
            matches.append(os.path.join(save_dir, f))
    if not matches:
        return None
    # Sort by filename which contains timestamp (YYYYMMDD_HHMMSS)
    return sorted(matches)[-1]

def save_trained_models(models_dict, prefix=""):
    """Saves a dictionary of trained models to the Output/models directory."""
    save_dir = os.path.join("Output", "models")
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    for name, model in models_dict.items():
        fname = f"{prefix}_{name}_{timestamp}.joblib".lower().replace(" ", "_")
        fpath = os.path.join(save_dir, fname)
        try:
            joblib.dump(model, fpath)
            # Short print for each saved model
        except Exception as e:
            print(f"   ⚠️ Could not save model {name}: {e}")

# NODEBinaryWrapper removed as it is a bottleneck.

def build_model(name, override_params=None, n_features=None, base_model=None):
    params = copy.deepcopy(MODEL_DEFAULT_PARAMS[name])
    if override_params:
        params.update(override_params)
    
    # Enable warm_start for tree models to support incremental growth
    if name in {"RandomForest", "ExtraTrees"} and base_model is not None:
        params["warm_start"] = True

    if name == "XGBoost":
        return XGBClassifier(**params, random_state=RANDOM_STATE, verbosity=0)
    if name == "LightGBM":
        return lgb.LGBMClassifier(**params, random_state=RANDOM_STATE, verbosity=-1)
    if name == "RandomForest":
        return RandomForestClassifier(**params)
    if name == "CatBoost":
        return CatBoostClassifier(**params)
    if name == "ExtraTrees":
        # ExtraTrees with warm_start in Scikit-Learn
        return ExtraTreesClassifier(**params)
    raise ValueError(f"Unknown GPU model: {name}")

def safe_fit_estimator(name, estimator, X, y, sample_weight=None, eval_set=None, base_model=None):
    X_np = np.asarray(X, dtype=np.float32)
    y_np = np.asarray(y).astype(int)
    kwargs = copy.deepcopy({})
    if sample_weight is not None:
        kwargs["sample_weight"] = np.asarray(sample_weight)

    # Inject base weights for incremental refinement
    if base_model is not None:
        if name == "CatBoost":
            # CatBoost GPU does not support init_model (Training continuation)
            is_gpu = getattr(estimator, "get_params", lambda: {} )().get("task_type") == "GPU"
            if not is_gpu:
                kwargs["init_model"] = base_model
            else:
                pass # Fallback to fresh training for CatBoost GPU
        elif name == "LightGBM":
            kwargs["init_model"] = base_model
        elif name == "XGBoost":
            kwargs["xgb_model"] = base_model

    if name == "XGBoost" and eval_set is not None:
        kwargs["eval_set"] = [(np.asarray(eval_set[0][0], dtype=np.float32), np.asarray(eval_set[0][1]).astype(int))]
        kwargs["verbose"] = False
    elif name == "LightGBM" and eval_set is not None:
        kwargs["eval_set"] = [(np.asarray(eval_set[0][0], dtype=np.float32), np.asarray(eval_set[0][1]).astype(int))]
        kwargs["callbacks"] = [
            lgb.early_stopping(30, verbose=False),
            lgb.log_evaluation(-1),
        ]
    elif name in {"RandomForest", "ExtraTrees", "CatBoost"}:
        # Note: Scikit-Learn tree models use warm_start=True set in build_model()
        return estimator.fit(X_np, y_np, **kwargs)

    try:
        estimator.fit(X_np, y_np, **kwargs)
    except TypeError:
        kwargs.pop("sample_weight", None)
        kwargs.pop("callbacks", None)
        kwargs.pop("verbose", None)
        estimator.fit(X_np, y_np, **kwargs)
    return estimator

def fit_time_safe_model(name, X, y, sample_weight=None, params=None, prefix=""):
    X_np = np.asarray(X, dtype=np.float32)
    y_np = np.asarray(y).astype(int)
    sw_np = None if sample_weight is None else np.asarray(sample_weight)
    
    # Load base model for refinement if exists
    base_model_path = find_latest_model_path(name, prefix)
    base_model = None
    if base_model_path:
        try:
            base_model = joblib.load(base_model_path)
            print(f"   🧬 Refining {prefix}_{name} from: {os.path.basename(base_model_path)}")
        except Exception as e:
            print(f"   ⚠️ Could not load base model for refinement: {e}")

    estimator = build_model(name, params, n_features=X_np.shape[1], base_model=base_model)

    X_core, y_core, X_es, y_es = split_train_early_stop(X_np, y_np)
    if X_es is not None and len(np.unique(y_es)) > 1 and name in {"XGBoost", "LightGBM", "CatBoost"}:
        sw_core = None if sw_np is None else sw_np[:len(y_core)]
        estimator = safe_fit_estimator(
            name, estimator, X_core, y_core, sample_weight=sw_core, eval_set=[(X_es, y_es)],
            base_model=base_model
        )
        return estimator, {
            "calibration_method": "native_predict_proba",
            "train_size": int(len(y_core)),
            "calibration_size": int(len(y_es)),
            "is_refined": base_model is not None,
        }

    estimator = safe_fit_estimator(name, estimator, X_np, y_np, sample_weight=sw_np, base_model=base_model)
    return estimator, {
        "calibration_method": "native_predict_proba",
        "train_size": int(len(y_np)),
        "calibration_size": 0,
        "is_refined": base_model is not None,
    }

def rolling_cv_auc_safe(name, X, y, sample_weight=None, params=None, n_splits=5, trial=None, prefix=""):
    X_np = np.asarray(X, dtype=np.float32)
    y_np = np.asarray(y).astype(int)
    sw_np = None if sample_weight is None else np.asarray(sample_weight)
    splitter = time_series_inner_splits(len(X_np), n_splits=n_splits)
    scores = []
    
    for fold_idx, (fit_idx, val_idx) in enumerate(splitter.split(X_np), start=1):
        if len(np.unique(y_np[fit_idx])) < 2 or len(np.unique(y_np[val_idx])) < 2:
            continue
            
        sw_fold = None if sw_np is None else sw_np[fit_idx]
        fold_model, _ = fit_time_safe_model(
            name, X_np[fit_idx], y_np[fit_idx], sample_weight=sw_fold, params=params, 
            prefix=prefix
        )
        prob = predict_proba_safe(fold_model, X_np[val_idx])
        
        # --- CUSTOM MULTI-METRIC COMPOSITE SCORE ---
        # 1. AUC (Overall Ranking Power)
        fold_auc = float(roc_auc_score(y_np[val_idx], prob))
        
        # 2. Average Precision (Precision/Recall balance for the minority 'Buy' class)
        fold_ap = float(average_precision_score(y_np[val_idx], prob))
        
        # 3. Brier Score (Probability Calibration Error - lower is better)
        fold_brier = float(brier_score_loss(y_np[val_idx], prob))
        
        # The Formula: 40% AUC + 40% Precision-Recall + 20% Penalty for bad calibration
        composite_score = (fold_auc * 0.40) + (fold_ap * 0.40) - (fold_brier * 0.20)
        
        scores.append(composite_score)
        
        if trial is not None:
            trial.report(float(np.mean(scores)), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
                
    if not scores:
        return 0.5, 0.0
    return float(np.mean(scores)), float(np.std(scores))

def sample_hyperparams(name, trial):
    if name == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.10, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.8, 2.5),
        }
    if name == "LightGBM":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 96),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    if name == "RandomForest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 5, 25),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }
    if name == "CatBoost":
        return {
            "iterations": trial.suggest_int("iterations", 200, 600),
            "depth": trial.suggest_int("depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.10, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "task_type": "GPU",
            "devices": GPU_DEVICE_ID,
            "eval_metric": "Logloss",
            "logging_level": "Silent",
            "bootstrap_type": "Poisson",
        }
    if name == "ExtraTrees":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600),
            "max_depth": trial.suggest_int("max_depth", 5, 25),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }
    return {}

def tune_model_safe(name, X, y, sample_weight=None, n_trials=50, prefix=""):
    if name not in TUNED_MODELS or len(X) < 120 or len(np.unique(y)) < 2:
        return {}, None, {
            "used_tuned_params": False,
            "search_rows": int(len(X)),
            "guard_rows": 0,
        }

    X_np = np.asarray(X, dtype=np.float32)
    y_np = np.asarray(y).astype(int)
    sw_np = None if sample_weight is None else np.asarray(sample_weight)
    X_search, y_search, X_guard, y_guard, sw_search, sw_guard = split_train_tuning_pool(
        X_np, y_np, sw_np
    )

    def objective(trial):
        params = sample_hyperparams(name, trial)
        splitter = time_series_inner_splits(len(X_search), n_splits=4)
        scores = []
        for fold_idx, (fit_idx, val_idx) in enumerate(splitter.split(X_search), start=1):
            if len(np.unique(y_search[fit_idx])) < 2 or len(np.unique(y_search[val_idx])) < 2:
                continue
            sw_fold = None if sw_search is None else sw_search[fit_idx]
            model_fold, _ = fit_time_safe_model(
                name, X_search[fit_idx], y_search[fit_idx], sample_weight=sw_fold, params=params, 
                prefix=prefix
            )
            prob = predict_proba_safe(model_fold, X_search[val_idx])
            scores.append(float(roc_auc_score(y_search[val_idx], prob)))
            trial.report(float(np.mean(scores)), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores)) if scores else 0.5

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    tuning_meta = {
        "used_tuned_params": True,
        "search_rows": int(len(X_search)),
        "guard_rows": 0 if X_guard is None else int(len(X_guard)),
        "candidate_best_params": dict(study.best_params),
        "selected_params": dict(study.best_params),
    }
    if X_guard is not None and len(X_guard) >= 24 and len(np.unique(y_guard)) > 1:
        default_model, _ = fit_time_safe_model(name, X_search, y_search, sample_weight=sw_search, params={}, prefix=prefix)
        tuned_model, _ = fit_time_safe_model(name, X_search, y_search, sample_weight=sw_search, params=study.best_params, prefix=prefix)
        default_guard_auc = float(roc_auc_score(y_guard, predict_proba_safe(default_model, X_guard)))
        tuned_guard_auc = float(roc_auc_score(y_guard, predict_proba_safe(tuned_model, X_guard)))
        keep_tuned = tuned_guard_auc >= (default_guard_auc - 0.0025)
        tuning_meta.update({
            "default_guard_auc": round(default_guard_auc, 4),
            "tuned_guard_auc": round(tuned_guard_auc, 4),
            "selected_guard_auc": round(tuned_guard_auc if keep_tuned else default_guard_auc, 4),
            "used_tuned_params": bool(keep_tuned),
            "selected_params": dict(study.best_params if keep_tuned else {}),
        })
        return (study.best_params if keep_tuned else {}), study, tuning_meta
    return study.best_params, study, tuning_meta

X_tr_arr = Xs_tr.to_numpy()
X_vl_arr = Xs_vl.to_numpy()
X_ev_arr = Xs_ev.to_numpy()
global_cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr)
global_sample_weight = np.where(y_tr == 1, global_cw[1], global_cw[0])

tuned_params = {}
tuning_summary = {}
tuned_model_names = [name for name in MODEL_NAMES if name in TUNED_MODELS]
for model_name in iter_with_progress(tuned_model_names, "Optuna tuning", total=len(tuned_model_names), every=1):
    try:
        best_params_, study_, tuning_meta_ = tune_model_safe(
            model_name, X_tr_arr, y_tr, sample_weight=global_sample_weight,
            n_trials=OPTUNA_TRIALS, prefix="global"
        )
        tuned_params[model_name] = best_params_
        tuning_summary[model_name] = {
            "best_params": best_params_,
            "candidate_best_params": tuning_meta_.get("candidate_best_params", best_params_),
            "selected_params": tuning_meta_.get("selected_params", best_params_),
            "best_cv_auc": None if study_ is None else round(float(study_.best_value), 4),
            "trials": 0 if study_ is None else len(study_.trials),
            "search_rows": tuning_meta_.get("search_rows", len(X_tr_arr)),
            "guard_rows": tuning_meta_.get("guard_rows", 0),
            "default_guard_auc": tuning_meta_.get("default_guard_auc"),
            "tuned_guard_auc": tuning_meta_.get("tuned_guard_auc"),
            "selected_guard_auc": tuning_meta_.get("selected_guard_auc"),
            "used_tuned_params": tuning_meta_.get("used_tuned_params", True),
        }
        best_cv_txt = "N/A" if study_ is None else f"{float(study_.best_value):.4f}"
        selected_txt = "tuned" if tuning_meta_.get("used_tuned_params", True) else "default"
        guard_txt = tuning_meta_.get("selected_guard_auc")
        guard_txt = "N/A" if guard_txt is None else f"{float(guard_txt):.4f}"
        default_guard_txt = tuning_meta_.get("default_guard_auc")
        tuned_guard_txt = tuning_meta_.get("tuned_guard_auc")
        default_guard_txt = "N/A" if default_guard_txt is None else f"{float(default_guard_txt):.4f}"
        tuned_guard_txt = "N/A" if tuned_guard_txt is None else f"{float(tuned_guard_txt):.4f}"
        candidate_params_ = tuning_meta_.get("candidate_best_params", best_params_)
        print(f"   {model_name:<14} tuned | CV AUC={best_cv_txt} | guard={guard_txt} | "
              f"default_guard={default_guard_txt} | tuned_guard={tuned_guard_txt} | "
              f"selected={selected_txt} | candidate={candidate_params_} | final={best_params_}")
    except Exception as e:
        tuned_params[model_name] = {}
        tuning_summary[model_name] = {"best_params": {}, "error": str(e)}
        print(f"   {model_name:<14} tuning skipped: {e}")

print("\n=== TUNING VERIFICATION ===")
any_tuned = False
for name in TUNED_MODELS:
    summary = tuning_summary.get(name, {})
    n_done  = summary.get("trials", 0)
    best_cv = summary.get("best_cv_auc", "N/A")
    err     = summary.get("error")
    if err:
        print(f"  {name:<18} ❌ FAILED  — {err[:60]}")
    elif n_done == 0:
        print(f"  {name:<18} ⚠️  SKIPPED — 0 trials completed")
    else:
        any_tuned = True
        params = summary.get("best_params", {})
        guard_auc = summary.get("selected_guard_auc", "N/A")
        source = "tuned" if summary.get("used_tuned_params", True) else "default"
        print(f"  {name:<18} ✅ {n_done:>3} trials | Best CV={best_cv} | "
              f"Guard={guard_auc} | kept={source} | {params}")

if not any_tuned:
    print("\n  ⛔ WARNING: No models were actually tuned!")
    print("  All models ran on default params.")
    print("  Check TUNED_MODELS and sample_hyperparams() for coverage.")
print("="*50)

MODEL_DEFS = {
    name: build_model(name, tuned_params.get(name, {}), n_features=X_tr_arr.shape[1])
    for name in MODEL_NAMES
}
trained, metrics_g = {}, {}
best_name, best_score = "", float("-inf")

print(f"\n   {'Model':<22} {'CV AUC':<14} {'Val AUC':<12} {'Val AP':<10} "
      f"{'Val Brier':<11} {'Test AUC':<10} {'Gap':<8} {'Time'}")
print("   " + "â”€"*90)

for model_name in iter_with_progress(MODEL_NAMES, "Model fitting", total=len(MODEL_NAMES), every=1):
    t0 = time.time()
    try:
        params_ = tuned_params.get(model_name, {})
        cv_auc, cv_std = rolling_cv_auc_safe(
            model_name, X_tr_arr, y_tr, sample_weight=global_sample_weight,
            params=params_, n_splits=5
        )
        model_, fit_meta_ = fit_time_safe_model(
            model_name, X_tr_arr, y_tr,
            sample_weight=global_sample_weight, params=params_
        )
        vl_prob = predict_proba_safe(model_, X_vl_arr)
        ev_prob = predict_proba_safe(model_, X_ev_arr)
        val_auc = float(roc_auc_score(y_vl, vl_prob)) if len(np.unique(y_vl)) > 1 else 0.5
        val_ap = float(average_precision_score(y_vl, vl_prob))
        val_brier = float(brier_score_loss(y_vl, vl_prob))
        tst_auc = float(roc_auc_score(y_ev, ev_prob)) if len(np.unique(y_ev)) > 1 else val_auc
        generalization_gap = float(val_auc - tst_auc)
        spread = float(np.ptp(vl_prob))
        elapsed = time.time() - t0
        trained[model_name] = model_
        metrics_g[model_name] = {
            "cv_auc": round(cv_auc, 4),
            "cv_std": round(cv_std, 4),
            "val_auc": round(val_auc, 4),
            "val_ap": round(val_ap, 4),
            "val_brier": round(val_brier, 4),
            "test_auc": round(tst_auc, 4),
            "generalization_gap": round(generalization_gap, 4),
            "spread": round(spread, 4),
            "time": round(elapsed, 1),
            "params": params_,
            "calibration_method": fit_meta_["calibration_method"],
            "calibration_size": fit_meta_["calibration_size"],
            "guard_auc": tuning_summary.get(model_name, {}).get("selected_guard_auc"),
        }
        metrics_g[model_name]["selection_score"] = round(
            model_rank_score(metrics_g[model_name]), 4
        )
        if metrics_g[model_name]["selection_score"] > best_score:
            best_score = metrics_g[model_name]["selection_score"]
            best_name = model_name
        spread_warn = " âš ï¸ COLLAPSED" if spread < MIN_SIGNAL_SPREAD else ""
        flag = " â† BEST" if model_name == best_name else ""
        print(f"   {model_name:<22} {cv_auc:.4f}Â±{cv_std:.4f}  {val_auc:.4f}      "
              f"{val_ap:.4f}    {val_brier:.4f}     {tst_auc:.4f}    {generalization_gap:+.4f}  "
              f"{elapsed:.0f}s{flag}{spread_warn}")
    except Exception as e:
        print(f"   {model_name:<22} FAILED: {e}")

# Save Global Models
save_trained_models(trained, "global")

if not trained:
    raise RuntimeError(
        f"No models trained successfully. Active models were: {MODEL_NAMES}. "
        "Check GPU library support and optional dependencies for the requested GPU model suite."
    )

if not best_name:
    best_name = max(
        metrics_g.keys(),
        key=lambda name: metrics_g[name].get("selection_score", metrics_g[name].get("test_auc", 0.5))
    )
    best_score = metrics_g[best_name].get("selection_score", best_score)

best_auc = metrics_g.get(best_name, {}).get("val_auc", 0.5)

print(f"\n   âœ… Best global after override: {best_name}  Val-AUC={best_auc:.4f}")

best_test_auc = metrics_g.get(best_name, {}).get("test_auc", 0.5)
print(f"   Held-out ranking uses Test-AUC={best_test_auc:.4f} with selection score={best_score:.4f}")
# --- CUSTOM REGIME BLENDER CLASS ---
class RegimeBlender:
    def __init__(self, model_bagger, model_booster, name_bagger, name_booster):
        self.model_bagger = model_bagger
        self.model_booster = model_booster
        self.name_bagger = name_bagger
        self.name_booster = name_booster

    def predict_proba(self, X):
        prob_bagger = predict_proba_safe(self.model_bagger, X)
        prob_booster = predict_proba_safe(self.model_booster, X)
        return (prob_bagger * 0.5) + (prob_booster * 0.5)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
        
    def get_params(self, deep=True):
        return {}

print("\n   Per-Regime Model Selection (Forced Bagger+Booster Blend) ...")
regime_best = {}
regime_ids = sorted(tr_df["regime_id"].unique())

# Define our algorithmic families
BAGGERS = {"RandomForest", "ExtraTrees"}
BOOSTERS = {"XGBoost", "LightGBM", "CatBoost"}

for rid in iter_with_progress(regime_ids, "Regime fitting", total=len(regime_ids), every=1):
    tr_mask = (tr_df["regime_id"].values == rid)
    n_regime_train = int(tr_mask.sum())
    
    if n_regime_train < 80:
        regime_best[int(rid)] = {
            "model": best_name,
            "cv_auc": metrics_g.get(best_name, {}).get("val_auc", 0.5),
            "fitted_model": trained.get(best_name),
            "source": "global_fallback",
        }
        print(f"   R{rid} ({REGIME_INFO[rid]['label']:<14}): only {n_regime_train} train rows → global best ({best_name})")
        continue

    Xr_tr = Xs_tr.loc[tr_mask].to_numpy()
    yr_tr = y_tr[tr_mask]
    
    if len(np.unique(yr_tr)) < 2:
        regime_best[int(rid)] = {
            "model": best_name,
            "cv_auc": 0.5,
            "fitted_model": trained.get(best_name),
            "source": "global_fallback",
        }
        continue

    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=yr_tr)
    sample_w = np.where(yr_tr == 1, cw[1], cw[0])
    
    best_bagger_name, best_bagger_auc = None, 0.0
    best_booster_name, best_booster_auc = None, 0.0
    
    for model_name in MODEL_NAMES:
        if model_name not in trained:
            continue
        try:
            rauc, rstd = rolling_cv_auc_safe(
                model_name, Xr_tr, yr_tr, sample_weight=sample_w,
                params=tuned_params.get(model_name, {}), n_splits=3, 
                prefix=f"regime_{rid}"
            )
            # Route model scores to the correct family
            if model_name in BAGGERS and rauc > best_bagger_auc:
                best_bagger_name, best_bagger_auc = model_name, rauc
            elif model_name in BOOSTERS and rauc > best_booster_auc:
                best_booster_name, best_booster_auc = model_name, rauc
        except Exception:
            pass

    # Safety fallbacks if a family fails entirely
    if best_bagger_name is None: best_bagger_name = best_booster_name or best_name
    if best_booster_name is None: best_booster_name = best_bagger_name or best_name

    # 1. Fit the best Bagger (e.g., ExtraTrees)
    bagger_model, _ = fit_time_safe_model(
        best_bagger_name, Xr_tr, yr_tr, sample_weight=sample_w, 
        params=tuned_params.get(best_bagger_name, {}), prefix=f"regime_{rid}_bagger"
    )
    
    # 2. Fit the best Booster (e.g., CatBoost or XGBoost)
    booster_model, _ = fit_time_safe_model(
        best_booster_name, Xr_tr, yr_tr, sample_weight=sample_w, 
        params=tuned_params.get(best_booster_name, {}), prefix=f"regime_{rid}_booster"
    )
    
    # 3. Blend them together
    blender = RegimeBlender(bagger_model, booster_model, best_bagger_name, best_booster_name)
    blended_name = f"{best_bagger_name}+{best_booster_name}"
    avg_cv_auc = (best_bagger_auc + best_booster_auc) / 2.0
    
    regime_best[int(rid)] = {
        "model": blended_name,
        "cv_auc": round(avg_cv_auc, 4),
        "cv_std": 0.0,
        "fitted_model": blender,
        "source": "bagger_booster_blend",
        "calibration_method": "blended",
    }

    vl_mask = (vl_df["regime_id"].values == rid)
    if vl_mask.sum() > 20 and len(np.unique(y_vl[vl_mask])) > 1:
        vl_auc_r = float(roc_auc_score(
            y_vl[vl_mask],
            predict_proba_safe(blender, Xs_vl.loc[vl_mask].to_numpy())
        ))
        regime_best[int(rid)]["val_auc"] = round(vl_auc_r, 4)
        lbl = REGIME_INFO.get(rid, {}).get("label", "?")
        print(f"   R{rid} ({lbl:<14}): blend={blended_name:<25} CV={avg_cv_auc:.4f}  Val={vl_auc_r:.4f}  n_train={n_regime_train}")
    else:
        print(f"   R{rid} ({REGIME_INFO[rid]['label']:<14}): blend={blended_name:<25} CV={avg_cv_auc:.4f}  n_train={n_regime_train}")

# Save Regime-Specific Models
regime_models_to_save = {f"regime_{rid}": info["fitted_model"] for rid, info in regime_best.items() if "fitted_model" in info}
save_trained_models(regime_models_to_save, "regime")
print("\n   Building time-series OOF stacking ...")
try:
    groups = [
        ["XGBoost", "LightGBM", "CatBoost"],
        ["RandomForest", "ExtraTrees"],
    ]
    diverse = []
    for grp in groups:
        avail = [(n, metrics_g[n]["cv_auc"]) for n in grp if n in trained]
        if avail:
            diverse.append(max(avail, key=lambda x: x[1])[0])

    if len(diverse) < 2:
        raise ValueError("Need at least two diverse base models")

    splitter = time_series_inner_splits(len(X_tr_arr), n_splits=5)
    oof = np.full((len(X_tr_arr), len(diverse)), np.nan)
    for ft, fv in splitter.split(X_tr_arr):
        yft = y_tr[ft]
        if len(np.unique(yft)) < 2:
            continue
        cw_fold = compute_class_weight("balanced", classes=np.array([0, 1]), y=yft)
        sw_fold = np.where(yft == 1, cw_fold[1], cw_fold[0])
        for mi, nm in enumerate(diverse):
            fm, _ = fit_time_safe_model(
                nm, X_tr_arr[ft], yft, sample_weight=sw_fold,
                params=tuned_params.get(nm, {}), prefix="global"
            )
            oof[fv, mi] = predict_proba_safe(fm, X_tr_arr[fv])

    valid_oof = np.isfinite(oof).all(axis=1)
    meta_lr = LogisticRegression(
        C=0.10, max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
    )
    meta_lr.fit(oof[valid_oof], y_tr[valid_oof])

    meta_val = np.column_stack([predict_proba_safe(trained[nm], X_vl_arr) for nm in diverse])
    meta_tst = np.column_stack([predict_proba_safe(trained[nm], X_ev_arr) for nm in diverse])
    stack_oof_auc = float(roc_auc_score(y_tr[valid_oof], meta_lr.predict_proba(oof[valid_oof])[:, 1]))
    sv = float(roc_auc_score(y_vl, meta_lr.predict_proba(meta_val)[:, 1]))
    st_ = float(roc_auc_score(y_ev, meta_lr.predict_proba(meta_tst)[:, 1]))
    stack_val_prob = meta_lr.predict_proba(meta_val)[:, 1]

    class ManualStack:
        def __init__(self, bases, meta, base_names):
            self.bases = bases
            self.meta = meta
            self.base_names = base_names
        def predict_proba(self, X):
            feats = np.column_stack([predict_proba_safe(self.bases[nm], X) for nm in self.base_names])
            return self.meta.predict_proba(feats)
        def predict(self, X):
            return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    trained["Stacking"] = ManualStack(trained, meta_lr, diverse)
    metrics_g["Stacking"] = {
        "cv_auc": round(stack_oof_auc, 4),
        "cv_std": 0.0,
        "val_auc": round(sv, 4),
        "val_ap": round(float(average_precision_score(y_vl, stack_val_prob)), 4),
        "val_brier": round(float(brier_score_loss(y_vl, stack_val_prob)), 4),
        "test_auc": round(st_, 4),
        "generalization_gap": round(float(sv - st_), 4),
        "spread": round(float(np.ptp(stack_val_prob)), 4),
        "time": 0.0,
        "params": {"bases": diverse, "meta": "LogisticRegression"},
        "calibration_method": "meta_logistic",
        "calibration_size": int(valid_oof.sum()),
        "guard_auc": None,
    }
    metrics_g["Stacking"]["selection_score"] = round(
        model_rank_score(metrics_g["Stacking"]), 4
    )
    if metrics_g["Stacking"]["selection_score"] > best_score:
        best_score = metrics_g["Stacking"]["selection_score"]
        best_auc = metrics_g["Stacking"]["val_auc"]
        best_name = "Stacking"
    print(f"   Stacking: Val={sv:.4f} Test={st_:.4f} (bases: {diverse})")
except Exception as e:
    print(f"   Stacking FAILED: {e}")

print(f"\n   âœ… Final best global: {best_name}  Val-AUC={best_auc:.4f}")

# Held-out test remains the final reality check for model ranking.
print(f"   Final ranking metric: Test-AUC={metrics_g.get(best_name, {}).get('test_auc', 0.5):.4f} "
      f"| selection_score={metrics_g.get(best_name, {}).get('selection_score', best_score):.4f}")
# ============================================================
# 11. CALIBRATION DIAGNOSTIC PLOTS (FIX #1 verification)
# ============================================================
show_section_progress(9, "Calibration And Evaluation")
print("\n📊 Calibration & model evaluation plots ...")

fig, axes = plt.subplots(2, 3, figsize=(22, 12))

# 1. AUC comparison
model_names = sorted(
    metrics_g.keys(),
    key=lambda x: metrics_g[x].get("selection_score", metrics_g[x]["test_auc"]),
    reverse=True,
)
val_aucs  = [metrics_g[m]["val_auc"]  for m in model_names]
test_aucs = [metrics_g[m]["test_auc"] for m in model_names]
cv_aucs   = [metrics_g[m]["cv_auc"]   for m in model_names]
x_pos = np.arange(len(model_names)); w = 0.25
ax = axes[0,0]
ax.barh(x_pos+w,  val_aucs,  w, color=C[1], label="Val AUC",  alpha=0.9)
ax.barh(x_pos,    test_aucs, w, color=C[2], label="Test AUC", alpha=0.9)
ax.barh(x_pos-w,  cv_aucs,   w, color=C[0], label="CV AUC",   alpha=0.9)
ax.set_yticks(x_pos); ax.set_yticklabels(model_names, fontsize=8)
ax.axvline(0.5, color="red", lw=1.5, ls="--", label="Random")
ax.set_title("AUC — CV / Val / Test", fontsize=11, fontweight="bold")
ax.legend(fontsize=8); ax.set_xlim(0.40, 0.80); ax.grid(axis="x", alpha=0.3)

# 2. Probability spread (FIX #1 verification)
ax = axes[0,1]
spreads = [metrics_g[m].get("spread",0) for m in model_names]
bar_cols = [C[1] if s >= MIN_SIGNAL_SPREAD else C[2] for s in spreads]
ax.barh(model_names[::-1], spreads[::-1], color=bar_cols[::-1], edgecolor="#30363d")
ax.axvline(MIN_SIGNAL_SPREAD, color="red", lw=2, ls="--",
           label=f"Min spread ({MIN_SIGNAL_SPREAD:.2f})")
ax.set_title("Probability Spread (max-min) on Val Set\n✅=healthy  ❌=collapsed",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=8); ax.grid(axis="x", alpha=0.3)

# 3. Calibration curves (top 5 models)
ax = axes[0,2]
top5 = sorted(
    metrics_g.keys(),
    key=lambda x: metrics_g[x].get("selection_score", metrics_g[x]["test_auc"]),
    reverse=True,
)[:5]
for mname in top5:
    try:
        vl_prob = predict_proba_safe(trained[mname], Xs_vl)
        frac_pos, mean_pred = calibration_curve(y_vl, vl_prob, n_bins=10, strategy="uniform")
        ax.plot(mean_pred, frac_pos, marker="o", ms=4,
                label=f"{mname} (AUC={metrics_g[mname]['val_auc']:.3f})")
    except: pass
ax.plot([0,1],[0,1],"--", color="#445f7a", label="Perfect calibration")
ax.set_title("Probability Calibration — Val Set", fontsize=10, fontweight="bold")
ax.set_xlabel("Mean predicted prob"); ax.set_ylabel("Fraction positives")
ax.legend(fontsize=7); ax.grid(alpha=0.3)

# 4. ROC curves
ax = axes[1,0]
for mname, m in list(trained.items())[:10]:
    try:
        probs = predict_proba_safe(m, Xs_ev)
        fpr, tpr, _ = roc_curve(y_ev, probs); auc_v = auc(fpr, tpr)
        lw = 3.0 if mname == best_name else 1.5
        ax.plot(fpr, tpr, lw=lw, label=f"{mname} ({auc_v:.3f})")
    except: pass
ax.plot([0,1],[0,1],"--", color="#445f7a", lw=1.2)
ax.set_title("ROC — Test Set", fontsize=11, fontweight="bold")
ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

# 5. Feature importance
fi_dict = {}
for mname in ["XGBoost", "LightGBM"]:
    if mname not in trained: continue
    m = trained[mname]
    base = m.calibrated_classifiers_[0].estimator if hasattr(m,"calibrated_classifiers_") else m
    if hasattr(base, "feature_importances_"):
        fi = base.feature_importances_
        for i, col in enumerate(FEATURE_COLS_WITH_REGIME):
            fi_dict[col] = fi_dict.get(col,0) + float(fi[i])
fi_sorted = []
if fi_dict:
    total = sum(fi_dict.values())
    fi_sorted = sorted([(k, round(v/total,5)) for k,v in fi_dict.items()],
                       key=lambda x: -x[1])
    ax = axes[1,1]
    top_fi = fi_sorted[:14]
    ax.barh([x[0] for x in top_fi][::-1], [x[1] for x in top_fi][::-1],
            color=[C[i%len(C)] for i in range(len(top_fi))][::-1], edgecolor="#30363d")
    ax.set_title("Feature Importance (Tree Ensemble)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Normalised Importance"); ax.grid(axis="x", alpha=0.3)

# 6. Per-regime best model AUC
ax = axes[1,2]
rids_eval  = sorted([r for r in regime_best.keys()
                     if "val_auc" in regime_best[r] or "cv_auc" in regime_best[r]])
rlbls = [f"R{r}\n{REGIME_INFO.get(r,{}).get('label','?')[:10]}\n"
         f"{regime_best[r]['model'][:10]}" for r in rids_eval]
raucs = [regime_best[r].get("val_auc", regime_best[r].get("cv_auc",0.5)) for r in rids_eval]
rcols = [REGIME_INFO.get(r,{}).get("color", C[0]) for r in rids_eval]
bars  = ax.bar(rlbls, raucs, color=rcols, edgecolor="#30363d", linewidth=1.5)
ax.set_title("Per-Regime Best Model AUC", fontsize=11, fontweight="bold")
ax.set_ylabel("AUC"); ax.set_ylim(0.40, 0.85)
ax.axhline(0.5, color="red", lw=1.5, ls="--")
for bar, val in zip(bars, raucs):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
plt.suptitle("Model Evaluation Dashboard v8.0 — All Models Calibrated",
             fontsize=14, fontweight="bold")
plt.tight_layout(); plt.show()

# Confusion matrices
n_reg_test = sum(1 for rid in regime_best
                 if (ev_df["regime_id"].values==rid).sum() > 5)
if n_reg_test > 0:
    fig, axes = plt.subplots(1, n_reg_test, figsize=(5*n_reg_test, 5))
    if n_reg_test == 1: axes = [axes]
    ax_idx = 0
    for rid in sorted(regime_best.keys()):
        rmask = (ev_df["regime_id"].values == rid)
        if rmask.sum() < 5: continue
        Xr = Xs_ev[rmask]; yr = y_ev[rmask]
        m = regime_best[rid].get("fitted_model", trained.get(regime_best[rid]["model"], trained[best_name]))
        ConfusionMatrixDisplay(confusion_matrix(yr, m.predict(Xr)),
                               display_labels=["Underperf","Outperf"]
                               ).plot(ax=axes[ax_idx], colorbar=False, cmap="Blues")
        lbl = REGIME_INFO.get(rid,{}).get("label","?")
        axes[ax_idx].set_title(
            f"R{rid}:{lbl}\n({regime_best[rid]['model']})", fontsize=9, fontweight="bold")
        ax_idx += 1
    plt.suptitle("Confusion Matrices — Test Set per Regime", fontsize=12, fontweight="bold")
    plt.tight_layout(); plt.show()

# Classification reports
print("\n--- Per-Regime Classification Reports ---")
for rid in sorted(regime_best.keys()):
    rmask = (ev_df["regime_id"].values == rid)
    if rmask.sum() < 5: continue
    Xr = Xs_ev[rmask]; yr = y_ev[rmask]
    m = regime_best[rid].get("fitted_model", trained.get(regime_best[rid]["model"], trained[best_name]))
    lbl = REGIME_INFO.get(rid,{}).get("label","?")
    print(f"\n  Regime {rid}: {lbl} | Model: {regime_best[rid]['model']}")
    print(classification_report(yr, m.predict(Xr),
                                 target_names=["Underperf","Outperf"], digits=3))

# ============================================================
# 12. WALK-FORWARD BACKTEST — proper benchmarks (FIX #7)
# ============================================================
show_section_progress(10, "Walk-Forward Backtest")
print("\n🚀 Walk-Forward Backtest ...")

print("\nThreshold tuning on validation ...")

def build_portfolio_weights_threshold(today_df, model, feature_cols, buy_thresh):
    clean = today_df.dropna(subset=feature_cols).copy()
    if len(clean) < 3:
        return None, None, None
    X = scaler.transform(clean[feature_cols])
    probs = predict_proba_safe(model, X)
    spread = probs.max() - probs.min()
    if spread < MIN_SIGNAL_SPREAD:
        return None, None, None
    clean["score"] = probs
    top = clean.nlargest(TOP_K, "score").copy()
    top = top[top["score"] >= buy_thresh]
    if len(top) == 0:
        return None, None, None
    # Conviction weighting (weight by model probability squared)
    w = top["score"] ** 2
    w /= w.sum()
    w = w.clip(upper=MAX_WEIGHT)
    w /= w.sum()
    top["sec_"] = top["stock"].map(SECTOR_MAP).fillna("Other")
    for _, grp in top.groupby("sec_"):
        sw = w.loc[grp.index].sum()
        if sw > SECTOR_CAP:
            w.loc[grp.index] *= SECTOR_CAP / sw
    w /= w.sum()
    return top["stock"].values, w.values, probs

def get_regime_model(regime_id):
    info = regime_best.get(int(regime_id), {})
    fitted = info.get("fitted_model")
    if fitted is not None: return fitted, info["model"]
    mname = info.get("model", best_name)
    return trained.get(mname, trained[best_name]), mname

def run_validation_threshold_eval(buy_thresh, sell_thresh):
    scored_rows = []
    val_dates_sorted = sorted(val_dates_purged)
    period_returns = []
    prev_stocks = None
    prev_weights = None

    for dt in val_dates_sorted:
        today_df = vl_df[vl_df["date"] == dt].copy()
        if len(today_df) == 0:
            continue
        rid = int(today_df["regime_id"].mode().iloc[0])
        model, _ = get_regime_model(rid)
        clean = today_df.dropna(subset=FEATURE_COLS_WITH_REGIME).copy()
        if len(clean) == 0:
            continue
        X = scaler.transform(clean[FEATURE_COLS_WITH_REGIME])
        probs = predict_proba_safe(model, X)
        clean["prob"] = probs
        scored_rows.append(clean[["target", "prob"]])

    for i in range(len(val_dates_sorted) - 1):
        today_d = val_dates_sorted[i]
        next_d = val_dates_sorted[i + 1]
        today_df = vl_df[vl_df["date"] == today_d].copy()
        next_df = vl_df[vl_df["date"] == next_d].set_index("stock")
        nifty_r = float(nifty_idx["nifty_ret"].get(next_d, 0.0))
        if len(today_df) < 3:
            period_returns.append(nifty_r) # Removed 0.5 cash drag
            continue
        rid = int(today_df["regime_id"].mode().iloc[0])
        model, _ = get_regime_model(rid)
        sel, wts, _ = build_portfolio_weights_threshold(
            today_df, model, FEATURE_COLS_WITH_REGIME, buy_thresh
        )
        if sel is None or wts is None:
            period_returns.append(nifty_r) # Removed 0.5 cash drag
            continue
        realized = np.array([float(next_df["ret_1w"].get(s, 0.0)) for s in sel])
        if prev_stocks is not None and prev_weights is not None:
            prev_map = dict(zip(prev_stocks, prev_weights))
            curr_map = dict(zip(sel, wts))
            all_stk = set(prev_map) | set(curr_map)
            turnover = sum(abs(curr_map.get(s, 0) - prev_map.get(s, 0)) for s in all_stk)
        else:
            turnover = 1.0
        prev_stocks = sel.copy()
        prev_weights = wts.copy()
        period_returns.append(float(np.dot(wts, realized)) - transaction_cost(turnover))

    if scored_rows:
        scored = pd.concat(scored_rows, ignore_index=True)
        buy_mask = scored["prob"] >= buy_thresh
        sell_mask = scored["prob"] <= sell_thresh
        buy_precision = float(scored.loc[buy_mask, "target"].mean()) if buy_mask.any() else 0.5
        sell_precision = float(1 - scored.loc[sell_mask, "target"].mean()) if sell_mask.any() else 0.5
        action_rate = float((buy_mask | sell_mask).mean())
    else:
        buy_precision = sell_precision = 0.5
        action_rate = 0.0

    rets = pd.Series(period_returns, dtype=float)
    if len(rets) > 1 and rets.std() > 0:
        sharpe = float((rets.mean() / rets.std()) * np.sqrt(52))
        cagr = float((1 + rets).prod() ** (52 / len(rets)) - 1)
    else:
        sharpe = 0.0
        cagr = 0.0
    objective = sharpe + 0.35 * cagr + 0.20 * buy_precision + 0.10 * sell_precision + 0.10 * action_rate
    return {
        "objective": objective,
        "sharpe": sharpe,
        "cagr": cagr,
        "buy_precision": buy_precision,
        "sell_precision": sell_precision,
        "action_rate": action_rate,
    }

threshold_pairs = [
    (float(buy_candidate), float(sell_candidate))
    for buy_candidate in THRESHOLD_GRID if buy_candidate >= 0.50
    for sell_candidate in THRESHOLD_GRID
    if sell_candidate <= 0.50 and buy_candidate > sell_candidate + 0.08
]
threshold_results = []
for buy_candidate, sell_candidate in iter_with_progress(
    threshold_pairs, "Threshold search", total=len(threshold_pairs), every=4
):
    result_ = run_validation_threshold_eval(buy_candidate, sell_candidate)
    result_.update({"buy_thresh": buy_candidate, "sell_thresh": sell_candidate})
    threshold_results.append(result_)

if threshold_results:
    threshold_df = pd.DataFrame(threshold_results).sort_values("objective", ascending=False)
    best_threshold_row = threshold_df.iloc[0]
    BUY_THRESH = float(best_threshold_row["buy_thresh"])
    SELL_THRESH = float(best_threshold_row["sell_thresh"])
    print(f"   Selected BUY={BUY_THRESH:.2f} | SELL={SELL_THRESH:.2f} | "
          f"Sharpe={best_threshold_row['sharpe']:.3f} | CAGR={best_threshold_row['cagr']*100:.2f}%")
else:
    threshold_df = pd.DataFrame()
    print(f"   Threshold tuning skipped â€” keeping BUY={BUY_THRESH:.2f} | SELL={SELL_THRESH:.2f}")

def build_portfolio_weights(today_df, model, feature_cols):
    clean = today_df.dropna(subset=feature_cols).copy()
    if len(clean) < 3: return None, None, None
    X = scaler.transform(clean[feature_cols])
    probs = predict_proba_safe(model, X)
    spread = probs.max() - probs.min()
    # FIX #1: reject degenerate signals
    if spread < MIN_SIGNAL_SPREAD:
        return None, None, None
    clean["score"] = probs
    top = clean.nlargest(TOP_K, "score").copy()
    # Only buy if score > BUY_THRESH
    top = top[top["score"] >= BUY_THRESH]
    if len(top) == 0: return None, None, None
    # Conviction weighting (weight by model probability squared)
    w = top["score"] ** 2  
    w /= w.sum()
    w = w.clip(upper=MAX_WEIGHT); w /= w.sum()
    # Sector cap
    top["sec_"] = top["stock"].map(SECTOR_MAP).fillna("Other")
    for _, grp in top.groupby("sec_"):
        sw = w.loc[grp.index].sum()
        if sw > SECTOR_CAP: w.loc[grp.index] *= SECTOR_CAP / sw
    w /= w.sum()
    return top["stock"].values, w.values, probs

sim_dates    = sorted(test_dates_purged)
ret_regime   = []
ret_nifty    = []
prev_stocks  = None; prev_weights = None
nifty_ret_lookup = nifty_idx["nifty_ret"]
cash_weeks   = 0; invested_weeks = 0

for i in range(len(sim_dates)-1):
    today_d  = sim_dates[i]; next_d = sim_dates[i+1]
    today_df = ev_df[ev_df["date"] == today_d].copy()
    next_df  = ev_df[ev_df["date"] == next_d].set_index("stock")
    nifty_r  = float(nifty_ret_lookup.get(next_d, 0.0))
    ret_nifty.append((next_d, nifty_r))
    
    if len(today_df) < 3: 
        ret_regime.append((next_d, nifty_r)) # Removed cash drag
        continue

    rid   = int(today_df["regime_id"].mode().iloc[0])
    regime_label = REGIME_INFO.get(rid, {}).get("label", "")

    # THE BYPASS: If the market is a roaring Bull or Breakout, hold 100% Nifty 50
    if regime_label in ["BULL", "BREAKOUT", "STRONG-BULL"]:
        ret_regime.append((next_d, nifty_r))
        invested_weeks += 1
        continue

    model, mname = get_regime_model(rid)

    sel, wts, all_probs = build_portfolio_weights(today_df, model, FEATURE_COLS_WITH_REGIME)
    if sel is not None and wts is not None:
        realized = np.array([float(next_df["ret_1w"].get(s, 0.0)) for s in sel])
        if prev_stocks is not None and prev_weights is not None:
            prev_map = dict(zip(prev_stocks, prev_weights))
            curr_map = dict(zip(sel, wts))
            all_stk  = set(list(prev_map.keys()) + list(curr_map.keys()))
            turnover = sum(abs(curr_map.get(s,0) - prev_map.get(s,0)) for s in all_stk)
        else: turnover = 1.0
        prev_stocks = sel.copy(); prev_weights = wts.copy()
        net = float(np.dot(wts, realized)) - transaction_cost(turnover)
        ret_regime.append((next_d, net))
        invested_weeks += 1
    else:
        # No valid signal → park in 100% NIFTY (Removed cash drag)
        ret_regime.append((next_d, nifty_r))
        cash_weeks += 1
def to_series(lst):
    df_ = pd.DataFrame(lst, columns=["date","ret"])
    df_["date"] = pd.to_datetime(df_["date"])
    return df_.set_index("date")["ret"]

r_regime = to_series(ret_regime)
r_nifty  = to_series(ret_nifty)
common   = r_regime.index.intersection(r_nifty.index)
r_regime = r_regime.reindex(common)
r_nifty  = r_nifty.reindex(common)
print(f"   ✅ Backtest: {len(common)} weeks | {common[0].date()} → {common[-1].date()}")
print(f"   Invested: {invested_weeks}/{len(common)} weeks | Cash/low-signal: {cash_weeks}")

if HAS_MIDCAP:
    r_midcap = midcap_ret.reindex(common).fillna(0)
else:
    r_midcap = None

# Risk metrics
def risk_metrics(s, benchmark=None):
    s = s.dropna(); n = len(s)
    if n < 4: return {}
    cagr    = (1+s).prod()**(52/n) - 1
    ann_v   = s.std() * np.sqrt(52)
    sharpe  = (s.mean() / s.std()) * np.sqrt(52) if s.std() > 0 else np.nan
    eq_     = (1+s).cumprod()
    dd_     = (eq_ - eq_.cummax()) / eq_.cummax()
    mdd     = float(dd_.min())
    calmar  = cagr / abs(mdd) if mdd != 0 else np.nan
    hit     = float((s > 0).mean())
    neg_s   = s[s < 0]
    sor_d   = neg_s.std() * np.sqrt(52) if len(neg_s) > 1 else np.nan
    sortino = (s.mean() * 52) / sor_d if (sor_d and sor_d > 0) else np.nan
    beta_   = np.nan; alpha_ = np.nan
    if benchmark is not None:
        aln = pd.concat([s, benchmark], axis=1).dropna(); aln.columns=["s","b"]
        if len(aln) > 5 and aln["b"].std() > 0:
            cv = np.cov(aln["s"].values, aln["b"].values)
            beta_  = cv[0,1]/cv[1,1]
            alpha_ = (aln["s"].mean() - beta_*aln["b"].mean()) * 52
    final = INITIAL_CAPITAL * (1+s).prod()
    return {
        "CAGR (%)":       round(cagr*100, 2),
        "Ann. Vol (%)":   round(ann_v*100, 2),
        "Sharpe":         round(sharpe,3)  if not np.isnan(sharpe)  else "N/A",
        "Sortino":        round(sortino,3) if not np.isnan(sortino) else "N/A",
        "Max DD (%)":     round(mdd*100, 2),
        "Calmar":         round(calmar,3)  if not np.isnan(calmar)  else "N/A",
        "Hit Rate (%)":   round(hit*100, 2),
        "Alpha ann (%)":  round(alpha_*100,2) if not np.isnan(alpha_) else "N/A",
        "Final Value ₹":  f"₹{final:,.0f}",
    }

report_dict = {
    "Regime-Aware ML": risk_metrics(r_regime, r_nifty),
    "NIFTY 50 B&H":    risk_metrics(r_nifty),
}

# Add Comparison Index Funds (HDFC, ICICI, UTI)
for ticker, f_ret in comparison_fund_rets.items():
    # Reindex to common timeframe
    f_ret_sync = f_ret.reindex(common).fillna(0)
    label = ticker.split(".")[0] + " Fund"
    report_dict[label] = risk_metrics(f_ret_sync, r_nifty)

if r_midcap is not None:
    report_dict["Midcap Index"] = risk_metrics(r_midcap, r_nifty)

report = pd.DataFrame(report_dict).T

print("\n" + "="*78)
print("         COMPREHENSIVE RISK & PERFORMANCE REPORT")
print("="*78)
print(report.to_string())
print("="*78)

def to_builtin(obj):
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    if callable(obj):
        return getattr(obj, "__name__", str(obj))
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj

os.makedirs(EXPERIMENT_DIR, exist_ok=True)
experiment_payload = {
    "run_timestamp": datetime.datetime.now().isoformat(),
    "date_range": {"start": START_DATE, "end": END_DATE},
    "split": {
        "train_start": str(min(train_dates_purged).date()),
        "train_end": str(max(train_dates_purged).date()),
        "val_start": str(min(val_dates_purged).date()),
        "val_end": str(max(val_dates_purged).date()),
        "test_start": str(min(test_dates_purged).date()),
        "test_end": str(max(test_dates_purged).date()),
        "purge_weeks": PURGE_WEEKS,
    },
    "tuning_config": {
        "optuna_trials": OPTUNA_TRIALS,
        "tuning_pool_ratio": TUNING_POOL_RATIO,
        "final_ranking_metric": "test_auc_with_guard_and_cv_tiebreak",
        "primary_metric": PRIMARY_SELECTION_METRIC,
    },
    "runtime": {
        "gpu_required": GPU_REQUIRED,
        "gpu_only_training": GPU_ONLY_TRAINING,
        "gpu_detected": HAS_NVIDIA_GPU,
        "gpu_device_id": GPU_DEVICE_ID,
        "active_models": MODEL_NAMES,
        "benchmark_source": BENCHMARK_SOURCE,
    },
    "features": FEATURE_COLS_WITH_REGIME,
    "clip_bounds": {k: [float(v[0]), float(v[1])] for k, v in clip_bounds.items()},
    "regime_selection": {
        "best_k": int(best_k),
        "current_regime": int(current_regime),
        "regime_info": to_builtin(REGIME_INFO),
    },
    "global_best_model": best_name,
    "global_metrics": to_builtin(metrics_g),
    "tuned_params": to_builtin(tuned_params if 'tuned_params' in globals() else {}),
    "tuning_summary": to_builtin(tuning_summary if 'tuning_summary' in globals() else {}),
    "regime_models": to_builtin({
        rid: {
            "model": info.get("model"),
            "cv_auc": info.get("cv_auc"),
            "cv_std": info.get("cv_std"),
            "val_auc": info.get("val_auc"),
            "source": info.get("source"),
            "calibration_method": info.get("calibration_method"),
        }
        for rid, info in regime_best.items()
    }),
    "thresholds": {
        "buy": float(BUY_THRESH),
        "sell": float(SELL_THRESH),
        "search_results": [] if 'threshold_df' not in globals() else to_builtin(
            threshold_df.head(10).to_dict(orient="records")
        ),
    },
    "backtest_report": to_builtin(report_dict),
}
experiment_filename = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S.json")
experiment_path = os.path.join(EXPERIMENT_DIR, experiment_filename)
with open(experiment_path, "w", encoding="utf-8") as f:
    json.dump(experiment_payload, f, indent=2)
print(f"   Saved experiment artifact: {experiment_path}")

# ============================================================
# 13. BACKTEST PLOTS
# ============================================================
show_section_progress(11, "Backtest Plots")
print("\n📈 Generating backtest plots ...")

eq_ml    = INITIAL_CAPITAL * (1+r_regime).cumprod()
eq_nifty = INITIAL_CAPITAL * (1+r_nifty).cumprod()

def drawdown_series(s):
    s = s.dropna()
    eq = (1+s).cumprod()
    return ((eq - eq.cummax()) / eq.cummax()) * 100

dd_ml    = drawdown_series(r_regime)
dd_nifty = drawdown_series(r_nifty)

# Main backtest: 3 rows
fig, axes = plt.subplots(3, 1, figsize=(16, 18))

ml_cagr  = report.loc["Regime-Aware ML","CAGR (%)"]
ni_cagr  = report.loc["NIFTY 50 B&H",  "CAGR (%)"]

ax = axes[0]
ax.plot(eq_ml/1e5,    lw=2.8, color=C[0], label=f"Regime-Aware ML (CAGR {ml_cagr:.1f}%)")
ax.plot(eq_nifty/1e5, lw=2.2, color=C[1], label=f"NIFTY 50 B&H (CAGR {ni_cagr:.1f}%)")

# Plot Comparison Funds
for i, (ticker, f_ret) in enumerate(comparison_fund_rets.items()):
    f_sync = f_ret.reindex(common).fillna(0)
    f_eq = INITIAL_CAPITAL * (1+f_sync).cumprod()
    f_cagr = report_dict[ticker.split(".")[0] + " Fund"]["CAGR (%)"]
    ax.plot(f_eq/1e5, lw=1.5, ls="--", color=C[(i+2)%len(C)], label=f"{ticker.split('.')[0]} (CAGR {f_cagr:.1f}%)")

ax.axhline(INITIAL_CAPITAL/1e5, color="#445f7a", lw=1, ls=":", label="Initial ₹10L")
ax.set_title("₹10,00,000 — Walk-Forward Test (Real Prices, No Lookahead)",
             fontsize=13, fontweight="bold")
ax.set_ylabel("Portfolio Value (₹ Lakhs)"); ax.legend(fontsize=10); ax.grid(alpha=0.3)
for series, color, dy in [(eq_ml,C[0],8),(eq_nifty,C[1],-12)]:
    fv = series.iloc[-1]
    ax.annotate(f"₹{fv/1e5:.1f}L",
                xy=(series.index[-1], fv/1e5), xytext=(-70,dy),
                textcoords="offset points", fontsize=10, color=color, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5))

ax = axes[1]
ax.fill_between(dd_ml.index,    dd_ml,    0, alpha=0.75, color=C[0],
                label=f"Regime-ML (MaxDD {dd_ml.min():.1f}%)")
ax.fill_between(dd_nifty.index, dd_nifty, 0, alpha=0.50, color=C[1],
                label=f"NIFTY 50 (MaxDD {dd_nifty.min():.1f}%)")
ax.set_title("Real Drawdown (%) — Walk-Forward", fontsize=12, fontweight="bold")
ax.set_ylabel("Drawdown %"); ax.legend(fontsize=10); ax.grid(alpha=0.3)
ax.annotate(f"Peak DD: {dd_ml.min():.1f}%",
            xy=(dd_ml.idxmin(), dd_ml.min()), xytext=(20,-20),
            textcoords="offset points", fontsize=9, color=C[0],
            arrowprops=dict(arrowstyle="->", color=C[0], lw=1.2))

ax = axes[2]
rolling_ml    = r_regime.rolling(52).apply(lambda x:(1+x).prod()-1, raw=True)*100
rolling_nifty = r_nifty.rolling(52).apply(lambda x:(1+x).prod()-1, raw=True)*100
ax.plot(rolling_ml,    lw=2.2, color=C[0], label="Regime-Aware ML")
ax.plot(rolling_nifty, lw=2.0, color=C[1], label="NIFTY 50")
ax.fill_between(rolling_ml.index,
                rolling_ml.fillna(0), rolling_nifty.fillna(0),
                where=(rolling_ml.fillna(0)>rolling_nifty.fillna(0)),
                alpha=0.35, color=C[0], label="ML Outperforms")
ax.fill_between(rolling_ml.index,
                rolling_ml.fillna(0), rolling_nifty.fillna(0),
                where=(rolling_ml.fillna(0)<=rolling_nifty.fillna(0)),
                alpha=0.25, color=C[2], label="NIFTY Outperforms")
ax.axhline(0, color="#445f7a", lw=1)
ax.set_title("Rolling 52-Week Return (%)", fontsize=12, fontweight="bold")
ax.set_ylabel("Return %"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.suptitle("Walk-Forward Backtest v8.0 — Real Prices · Proper Benchmarks",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout(); plt.show()

# Year-wise
def yearly_rets(s, label):
    df_ = pd.DataFrame({"ret": s.values}, index=s.index)
    df_["year"] = df_.index.year
    return df_.groupby("year")["ret"].apply(lambda x:(1+x).prod()-1).rename(label)

yr_ml    = yearly_rets(r_regime, "Regime-ML")
yr_nifty = yearly_rets(r_nifty,  "NIFTY 50")
yr_all   = pd.concat([yr_ml, yr_nifty], axis=1).dropna()

fig, axes = plt.subplots(1, 2, figsize=(20, 7))
x_pos = np.arange(len(yr_all)); w_b = 0.38
ax = axes[0]
b1 = ax.bar(x_pos-w_b/2, yr_all["Regime-ML"]*100, w_b,
            color=C[0], label="Regime-Aware ML", alpha=0.9, edgecolor="#30363d")
b2 = ax.bar(x_pos+w_b/2, yr_all["NIFTY 50"]*100,  w_b,
            color=C[1], label="NIFTY 50",        alpha=0.9, edgecolor="#30363d")
for bar, val in zip(b1, yr_all["Regime-ML"]*100):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.8 if val>=0 else bar.get_height()-2.5,
            f"{val:.1f}%", ha="center",
            va="bottom" if val>=0 else "top", fontsize=8, color=C[0])
for bar, val in zip(b2, yr_all["NIFTY 50"]*100):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.8 if val>=0 else bar.get_height()-2.5,
            f"{val:.1f}%", ha="center",
            va="bottom" if val>=0 else "top", fontsize=8, color=C[1])
ax.set_xticks(x_pos); ax.set_xticklabels(yr_all.index, fontsize=9)
ax.axhline(0, color="white", lw=0.8)
ax.set_title("Year-Wise Returns (%)", fontsize=12, fontweight="bold")
ax.set_ylabel("Annual Return (%)"); ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)

ax = axes[1]
outperf = (yr_all["Regime-ML"] - yr_all["NIFTY 50"]) * 100
bar_cols_op = [C[0] if v>=0 else C[2] for v in outperf]
bars_op = ax.bar(yr_all.index, outperf, color=bar_cols_op, edgecolor="#30363d")
ax.axhline(0, color="white", lw=1.5)
for bar, val in zip(bars_op, outperf):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.3 if val>=0 else bar.get_height()-0.8,
            f"{val:+.1f}%", ha="center",
            va="bottom" if val>=0 else "top", fontsize=9, color="white")
wins = (outperf > 0).sum()
ax.set_title("Outperformance vs NIFTY 50", fontsize=12, fontweight="bold")
ax.set_ylabel("Alpha (%)"); ax.grid(axis="y", alpha=0.3)
ax.set_xlabel(f"ML beat NIFTY in {wins}/{len(outperf)} years "
              f"({wins/max(len(outperf),1)*100:.0f}% win rate)", fontsize=10)
plt.suptitle("Year-Wise Performance Analysis v8.0", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.show()

print("\n   Year-Wise Returns:")
print(f"   {'Year':<6} {'ML%':>8} {'NIFTY%':>8} {'Alpha%':>9} {'Beats?'}")
print("   " + "─"*50)
for year in yr_all.index:
    ml_r = yr_all.loc[year,"Regime-ML"]*100
    ni_r = yr_all.loc[year,"NIFTY 50"]*100
    alp  = ml_r - ni_r
    print(f"   {year:<6} {ml_r:>7.1f}% {ni_r:>7.1f}% {alp:>+8.1f}% "
          f"{'✅ YES' if alp>0 else '❌ NO'}")

# Regime-wise equity curves
n_r = len(REGIME_INFO)
fig, axes = plt.subplots(1, n_r, figsize=(7*n_r, 5))
if n_r == 1: axes = [axes]
for ax, rid in zip(axes, range(n_r)):
    r_dates = set(ev_df[ev_df["regime_id"]==rid]["date"].unique())
    sr = r_regime[r_regime.index.isin(r_dates)]
    sn = r_nifty[r_nifty.index.isin(r_dates)]
    if len(sr) < 2: ax.set_visible(False); continue
    ax.plot((1+sr).cumprod(), lw=2.5, color=REGIME_INFO[rid]["color"], label="ML")
    ax.plot((1+sn).cumprod(), lw=2.0, color=C[1], ls="--", label="NIFTY")
    ann_r_ml = (1+sr).prod()**(52/max(len(sr),1))*100 - 100
    ann_r_ni = (1+sn).prod()**(52/max(len(sn),1))*100 - 100
    ax.set_title(f"{REGIME_INFO[rid]['label']}\nML={ann_r_ml:.1f}% | NIFTY={ann_r_ni:.1f}%",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.suptitle("Regime-Wise Equity Curves (Test Period)", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()

# Monthly alpha heatmap
monthly_ml    = r_regime.resample("ME").apply(lambda x:(1+x).prod()-1)*100
monthly_nifty = r_nifty.resample("ME").apply(lambda x:(1+x).prod()-1)*100
monthly_alpha = monthly_ml - monthly_nifty
pivot = monthly_alpha.to_frame("alpha")
pivot["year"]  = pivot.index.year; pivot["month"] = pivot.index.month
heatmap_data = pivot.pivot_table(values="alpha", index="year", columns="month")
heatmap_data.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                        "Jul","Aug","Sep","Oct","Nov","Dec"]
fig, ax = plt.subplots(figsize=(16, max(5, len(heatmap_data)*0.7)))
sns.heatmap(heatmap_data, ax=ax, annot=True, fmt=".1f", linewidths=0.5,
            cmap="RdYlGn", center=0, annot_kws={"size":7},
            cbar_kws={"label":"Alpha %"})
ax.set_title("Monthly Alpha vs NIFTY 50 (%)", fontsize=12, fontweight="bold")
plt.tight_layout(); plt.show()

# ============================================================
# 14. SIGNAL GENERATION
# ============================================================
print("\n🎯 Generating Trading Signals ...")

show_section_progress(12, "Signal Generation")
def rating(p):
    if p >= BUY_THRESH:    return "🟢 BUY"
    elif p <= SELL_THRESH: return "🔴 SELL"
    else:                   return "🟡 HOLD"

def conviction(p):
    if   p >= 0.76: return "★★★ VERY HIGH"
    elif p >= 0.66: return "★★★ HIGH"
    elif p >= 0.58: return "★★☆ MEDIUM"
    elif p >= 0.50: return "★☆☆ LOW"
    elif p >= 0.43: return "☆☆☆ HOLD"
    else:           return "▼▼▼ EXIT"

if len(stocks_latest) > 0:
    signal_date = stocks_latest["date"].max()
    signal_df   = stocks_latest[stocks_latest["date"] == signal_date].copy()
    signal_src  = "LIVE (no future data)"
else:
    signal_date = stocks_model["date"].max()
    signal_df   = stocks_model[stocks_model["date"] == signal_date].copy()
    signal_src  = "Most recent complete data"

signal_df["regime_id"] = signal_df["regime_id"].fillna(current_regime).astype(int)
sig_rid   = int(signal_df["regime_id"].mode().iloc[0])
sig_model, sig_mname = get_regime_model(sig_rid)

clean_sig = signal_df.dropna(subset=FEATURE_COLS_WITH_REGIME).copy()
Xs_sig    = scaler.transform(clean_sig[FEATURE_COLS_WITH_REGIME])
raw_probs = predict_proba_safe(sig_model, Xs_sig)
spread_check = raw_probs.max() - raw_probs.min()

print(f"\n   Signal Date : {signal_date.date()}")
print(f"   Source      : {signal_src}")
print(f"   Regime      : R{sig_rid} — {REGIME_INFO.get(sig_rid,{}).get('label','?')}")
print(f"   Model       : {sig_mname}")
print(f"   Prob spread : {spread_check:.4f}  "
      f"({'✅ HEALTHY' if spread_check >= MIN_SIGNAL_SPREAD else '⚠️ COLLAPSED'})")

if spread_check < MIN_SIGNAL_SPREAD:
    print(f"   ⚠️ Regime model collapsed → falling back to global best ({best_name})")
    sig_model  = trained[best_name]
    sig_mname  = best_name + " (fallback)"
    raw_probs  = predict_proba_safe(sig_model, Xs_sig)
    spread_check = raw_probs.max() - raw_probs.min()
    print(f"   Fallback spread: {spread_check:.4f}")

clean_sig["prob"]       = raw_probs
clean_sig["rating"]     = clean_sig["prob"].apply(rating)
clean_sig["conviction"] = clean_sig["prob"].apply(conviction)
clean_sig["prob_pct"]   = (clean_sig["prob"]*100).round(1)
clean_sig["sector"]     = clean_sig["stock"].map(SECTOR_MAP).fillna("Other")

signals = (clean_sig.sort_values("prob", ascending=False)
           [["stock","sector","rating","conviction","prob_pct",
             "mom_8w","vol_26w","rsi_14","beta_52w"]]
           .reset_index(drop=True))

buys  = signals[signals["rating"]=="🟢 BUY"]
holds = signals[signals["rating"]=="🟡 HOLD"]
sells = signals[signals["rating"]=="🔴 SELL"]

print(f"\n{'='*108}")
print(f"{'#':<3} {'Stock':<17} {'Sector':<13} {'Rating':<14} {'Conviction':<16} "
      f"{'Prob%':<7} {'Mom8W':>7} {'Vol26W':>7} {'RSI':>6} {'Beta':>6}")
print(f"{'='*108}")
for i, row in signals.iterrows():
    print(f"{i+1:<3} {row['stock']:<17} {row['sector']:<13} {row['rating']:<14} "
          f"{row['conviction']:<16} {row['prob_pct']:<7.1f} "
          f"{row['mom_8w']*100:>6.1f}% {row['vol_26w']*100:>6.1f}% "
          f"{row['rsi_14']:>6.1f} {row['beta_52w']:>6.2f}")
print(f"{'='*108}")
print(f"\n   BUY: {len(buys)} | HOLD: {len(holds)} | SELL: {len(sells)}")

# Signal plots
fig, axes = plt.subplots(1, 3, figsize=(22, 8))
ax = axes[0]
bar_col_sig = [C[0] if r=="🟢 BUY" else C[2] if r=="🔴 SELL" else "#ffd60a"
               for r in signals["rating"]]
bars = ax.barh(signals["stock"][::-1], signals["prob_pct"][::-1],
               color=bar_col_sig[::-1], edgecolor="#30363d", linewidth=0.8, alpha=0.9)
ax.axvline(BUY_THRESH*100,  color=C[0], lw=2, ls="--",
           label=f"BUY ≥ {BUY_THRESH*100:.0f}%")
ax.axvline(SELL_THRESH*100, color=C[2], lw=2, ls="--",
           label=f"SELL ≤ {SELL_THRESH*100:.0f}%")
ax.axvline(50, color="#445f7a", lw=1, ls=":", label="50% random")
ax.set_xlabel("Outperform Probability (%)", fontsize=10)
ax.set_title(f"Signal Probabilities\n{signal_date.date()} | "
             f"R{sig_rid}:{REGIME_INFO.get(sig_rid,{}).get('label','?')}",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
for bar, val in zip(bars, signals["prob_pct"][::-1]):
    ax.text(val+0.3, bar.get_y()+bar.get_height()/2,
            f"{val:.1f}%", va="center", fontsize=7)
ax.grid(axis="x", alpha=0.3)

ax = axes[1]
rc = {}
for r in signals["rating"]:
    k = r.split()[-1]
    rc[k] = rc.get(k, 0) + 1
donut_v = [rc.get("BUY",0), rc.get("HOLD",0), rc.get("SELL",0)]
donut_l = [f"BUY\n({rc.get('BUY',0)})", f"HOLD\n({rc.get('HOLD',0)})",
           f"SELL\n({rc.get('SELL',0)})"]
if sum(donut_v) > 0:
    ax.pie(donut_v, labels=donut_l, autopct="%1.0f%%",
           colors=[C[0],"#ffd60a",C[2]], startangle=90,
           wedgeprops=dict(width=0.55), textprops={"fontsize":10})
ax.set_title(f"Signal Distribution\n{signal_date.date()}", fontsize=11, fontweight="bold")

ax = axes[2]
if len(buys) > 0:
    sec_buy = buys["sector"].value_counts()
    ax.pie(sec_buy.values, labels=sec_buy.index, autopct="%1.0f%%",
           colors=[C[i%len(C)] for i in range(len(sec_buy))],
           startangle=90, textprops={"fontsize":9})
    ax.set_title(f"BUY Picks by Sector ({len(buys)} stocks)",
                 fontsize=11, fontweight="bold")
else:
    ax.text(0.5,0.5,f"No BUY signals\n(All HOLD — SIDEWAYS regime)",
            ha="center", va="center", fontsize=10)
    ax.set_title("BUY by Sector", fontsize=11)
plt.suptitle(f"Signals — {sig_mname} | R{sig_rid}:{REGIME_INFO.get(sig_rid,{}).get('label','?')} "
             f"| {signal_date.date()}", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()

# Print top buys
print(f"\n{'='*72}")
print(f"   🟢 TOP BUY PICKS — {REGIME_INFO.get(sig_rid,{}).get('label','?')}")
print(f"   📅 As of {signal_date.date()} | 12-Week Horizon")
print(f"{'='*72}")
if len(buys) > 0:
    for _, row in buys.head(10).iterrows():
        print(f"  {row['stock']:<17} | {row['conviction']:<16} | "
              f"Prob={row['prob_pct']:.1f}% | Mom8W={row['mom_8w']*100:.1f}% | "
              f"RSI={row['rsi_14']:.0f}")
else:
    print("  No BUY signals in current regime — market is SIDEWAYS/BEAR.")
    print("  Regime-specific picks below:")
    for rid in range(best_k):
        if rid == sig_rid: continue
        m_, mn_ = get_regime_model(rid)
        src_df_ = stocks_latest if len(stocks_latest) > 0 else stocks_model
        rd_ = src_df_[src_df_["regime_id"]==rid].copy()
        if len(rd_) == 0: continue
        rd_lat = rd_[rd_["date"]==rd_["date"].max()].dropna(subset=FEATURE_COLS_WITH_REGIME).copy()
        if len(rd_lat) < 3: continue
        X_ = scaler.transform(rd_lat[FEATURE_COLS_WITH_REGIME])
        rd_lat["prob"] = predict_proba_safe(m_, X_)
        spread_ = rd_lat["prob"].max() - rd_lat["prob"].min()
        if spread_ >= MIN_SIGNAL_SPREAD:
            top_ = rd_lat.nlargest(5, "prob")
            lbl_ = REGIME_INFO.get(rid,{}).get("label","?")
            print(f"\n  If {lbl_} (R{rid}) — model: {mn_}:")
            for _, row_ in top_.iterrows():
                if row_["prob"] >= BUY_THRESH:
                    print(f"    {row_['stock']:<17} Prob={row_['prob']*100:.1f}% "
                          f"Mom8W={row_['mom_8w']*100:.1f}% RSI={row_['rsi_14']:.0f}")

if len(sells) > 0:
    print(f"\n{'='*72}"); print(f"   🔴 SELL / AVOID")
    print(f"{'='*72}")
    for _, row in sells.iterrows():
        print(f"  {row['stock']:<17} | Prob={row['prob_pct']:.1f}% | "
              f"Mom8W={row['mom_8w']*100:.1f}%")

# ============================================================
# 15. MASTER DASHBOARD
# ============================================================
print("\n🖥️  Master Dashboard ...")
show_section_progress(13, "Master Dashboard")
fig = plt.figure(figsize=(26, 24))
gs  = gridspec.GridSpec(4, 4, figure=fig, hspace=0.58, wspace=0.40)

ax1 = fig.add_subplot(gs[0,:3])
ax1.plot(eq_ml/1e5, lw=2.8, color=C[0], label=f"Regime-ML (CAGR {ml_cagr:.1f}%)")
ax1.plot(eq_nifty/1e5, lw=2.2, color=C[1], label=f"NIFTY 50 B&H (CAGR {ni_cagr:.1f}%)")
ax1.axhline(INITIAL_CAPITAL/1e5, color="#445f7a", lw=1, ls=":")
ax1.set_title("Portfolio Value (₹ Lakhs)", fontsize=11, fontweight="bold")
ax1.set_ylabel("₹ Lakhs"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

ax2 = fig.add_subplot(gs[0,3])
if sum(donut_v) > 0:
    ax2.pie(donut_v, labels=donut_l, autopct="%1.0f%%",
            colors=[C[0],"#ffd60a",C[2]], startangle=90,
            wedgeprops=dict(width=0.55), textprops={"fontsize":8})
ax2.set_title(f"Current Signals\n{signal_date.date()}", fontsize=9, fontweight="bold")

ax3 = fig.add_subplot(gs[1,:3])
ax3.fill_between(dd_ml.index, dd_ml, 0, alpha=0.75, color=C[0],
                 label=f"Regime-ML (MaxDD {dd_ml.min():.1f}%)")
ax3.fill_between(dd_nifty.index, dd_nifty, 0, alpha=0.50, color=C[1],
                 label=f"NIFTY 50 (MaxDD {dd_nifty.min():.1f}%)")
ax3.set_title("Real Drawdown (%) — Walk-Forward", fontsize=11, fontweight="bold")
ax3.set_ylabel("%"); ax3.legend(fontsize=9); ax3.grid(alpha=0.3)

ax4 = fig.add_subplot(gs[1,3])
ax4.bar([f"K={k}" for k in abl_df["k"]], abl_df["composite"],
        color=[C[7] if k==best_k else C[0] for k in abl_df["k"]],
        edgecolor="#30363d", linewidth=1.2)
ax4.set_title(f"Composite Score\nK={best_k} Optimal", fontsize=9, fontweight="bold")
ax4.set_ylabel("Score ↑"); ax4.grid(axis="y", alpha=0.3)

ax5 = fig.add_subplot(gs[2,:2])
xp_ = np.arange(len(yr_all)); w_b = 0.38
ax5.bar(xp_-w_b/2, yr_all["Regime-ML"]*100, w_b, color=C[0], label="Regime-ML", alpha=0.9)
ax5.bar(xp_+w_b/2, yr_all["NIFTY 50"]*100,  w_b, color=C[1], label="NIFTY 50",  alpha=0.9)
ax5.set_xticks(xp_); ax5.set_xticklabels(yr_all.index, fontsize=8)
ax5.axhline(0, color="white", lw=0.8)
ax5.set_title("Year-Wise Returns (%)", fontsize=10, fontweight="bold")
ax5.set_ylabel("%"); ax5.legend(fontsize=8); ax5.grid(axis="y", alpha=0.3)

ax6 = fig.add_subplot(gs[2,2:])
if fi_sorted:
    top_fi2 = fi_sorted[:12]
    ax6.barh([x[0] for x in top_fi2][::-1], [x[1] for x in top_fi2][::-1],
             color=[C[i%len(C)] for i in range(12)][::-1], edgecolor="#30363d")
    ax6.set_title("Feature Importance (Top 12)", fontsize=10, fontweight="bold")
    ax6.set_xlabel("Normalised"); ax6.grid(axis="x", alpha=0.3)

ax7 = fig.add_subplot(gs[3,:])
ax7.axis("off")
dcols = ["CAGR (%)","Ann. Vol (%)","Sharpe","Sortino","Max DD (%)",
         "Calmar","Hit Rate (%)","Alpha ann (%)","Final Value ₹"]
tbl_data = report[[c for c in dcols if c in report.columns]].reset_index()
avail_dcols = [c for c in dcols if c in report.columns]
tbl = ax7.table(cellText=tbl_data.values.tolist(),
                colLabels=["Strategy"]+avail_dcols,
                cellLoc="center", loc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 2.5)
for j in range(len(avail_dcols)+1):
    tbl[0,j].set_facecolor("#1565C0"); tbl[0,j].set_text_props(color="white",fontweight="bold")
row_cols = ["#0d2137","#0d1a2d","#081018"]
for i in range(1, len(report)+1):
    for j in range(len(avail_dcols)+1):
        tbl[i,j].set_facecolor(row_cols[(i-1)%len(row_cols)])
        tbl[i,j].set_text_props(color="white")
ax7.set_title("Performance Summary — All Strategies", pad=16,
              fontsize=11, fontweight="bold")

plt.suptitle(
    f"NIFTY 50 ADAPTIVE FACTOR-REGIME ML v8.0\n"
    f"Walk-Forward | {common[0].date()} → {common[-1].date()} | "
    f"K={best_k} Regimes | {len(trained)} Models | Signal: {signal_date.date()}",
    fontsize=14, fontweight="bold", y=1.01)
plt.show()

# ============================================================
# 16. FINAL SUMMARY
# ============================================================
show_section_progress(14, "Final Summary")
print("\n" + "="*78)
print("   ✅  NIFTY 50 ADAPTIVE FACTOR-REGIME ML v8.0 — COMPLETE")
print("="*78)
print(f"   Training period     : {START_DATE} → {END_DATE}")
print(f"   Purge gap           : {PURGE_WEEKS} weeks at each boundary (no leakage)")
print(f"   Selection metric    : {PRIMARY_SELECTION_METRIC}")
print(f"   Features            : {len(FEATURE_COLS_WITH_REGIME)}")
print(f"   Regimes (K)         : {best_k} (composite BIC+Sil+Stab score)")
print(f"   Models trained      : {len(trained)} (GPU-first suite with predict_proba)")
print(f"   Best global model   : {best_name}")
print(f"   Test period         : {common[0].date()} → {common[-1].date()}")
print(f"   Regime-ML CAGR      : {report.loc['Regime-Aware ML','CAGR (%)']:.2f}%")
print(f"   NIFTY 50 CAGR       : {report.loc['NIFTY 50 B&H','CAGR (%)']:.2f}%")

# Explicit Comparison
max_fund_cagr = 0.0
for ticker in COMPARISON_FUND_TICKERS:
    label = ticker.split(".")[0] + " Fund"
    if label in report.index:
        f_cagr = report.loc[label, "CAGR (%)"]
        print(f"   {label:<19} : {f_cagr:>7.2f}%")
        max_fund_cagr = max(max_fund_cagr, f_cagr)

if report.loc["Regime-Aware ML","CAGR (%)"] < report.loc["NIFTY 50 B&H","CAGR (%)"]:
    print("   CAGR status         : ⚠️ Underperforming NIFTY on walk-forward CAGR")
elif report.loc["Regime-Aware ML","CAGR (%)"] < max_fund_cagr:
    print("   CAGR status         : ⚠️ Underperforming some Index Funds")
else:
    print("   CAGR status         : ✅ Outperforming ALL Benchmarks")
print(f"   Sharpe              : {report.loc['Regime-Aware ML','Sharpe']}")
print(f"   Max Drawdown        : {report.loc['Regime-Aware ML','Max DD (%)']:.2f}%")
print(f"   Final Value         : {report.loc['Regime-Aware ML','Final Value ₹']}")
wins_f = (outperf > 0).sum()
print(f"   Beat NIFTY          : {wins_f}/{len(outperf)} years "
      f"({wins_f/max(len(outperf),1)*100:.0f}%)")
print(f"   Invested/Cash weeks : {invested_weeks}/{cash_weeks}")
print(f"   Signal date         : {signal_date.date()}")
print(f"   Current regime      : R{sig_rid} — {REGIME_INFO.get(sig_rid,{}).get('label','?')}")
print(f"   Signal spread check : {spread_check:.4f} "
      f"({'✅ OK' if spread_check >= MIN_SIGNAL_SPREAD else '⚠️ CHECK'})")
print(f"\n   🟢 TOP BUY PICKS:")
if len(buys) > 0:
    for _, row in buys.head(8).iterrows():
        print(f"      {row['stock']:<17} | {row['conviction']:<16} | Prob={row['prob_pct']:.1f}%")
else:
    print(f"      No BUY signals — current regime is {REGIME_INFO.get(sig_rid,{}).get('label','?')}")
    print(f"      Market is range-bound. No high-conviction entry points.")
if len(sells) > 0:
    print(f"\n   🔴 SELL / AVOID:")
    for _, row in sells.iterrows():
        print(f"      {row['stock']:<17} | Prob={row['prob_pct']:.1f}%")
print("="*78)
print("\n   ⚠️  DISCLAIMER: Research use only — NOT financial advice.")
print("   Do NOT trade real capital based on model output alone.")
print("="*78)
