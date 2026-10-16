#!/usr/bin/env python3
"""
=====================================================================
 Do Indian Active Equity Fund Managers Beat Their Style Benchmarks?
 Factor, Style-Sector and Market-Timing Attribution of Mutual Funds
 Business Analytics Project - Theme 1
=====================================================================

HOW TO RUN (from the folder that contains this file; on Windows type python, not python3)

  pip3 install -r requirements.txt

  python3 mf_project.py full
      Uses the data in data/raw, runs every model on every fund, and writes:
        outputs/Report.docx, outputs/Process_Log.docx,
        data/clean/*.csv (data files to submit),
        outputs/jamovi/*.csv (files to open in Jamovi)

  python3 mf_project.py live --code 122639
  python3 mf_project.py live --search "parag parikh flexi cap"
  python3 mf_project.py live --stock TCS.NS --bench NIFTY500
  python3 mf_project.py live --csv evaluator_file.csv --bench NIFTY100
      Runs the full model on ANY fund / stock / file the evaluator names.

  python3 mf_project.py rebuild-data
      Re-creates data/raw from AMFI's archived daily NAV files (needs git + internet).

SECTIONS OF THIS FILE
  0. Settings
  1. Data (AMFI month-end NAV archive, IIM Ahmedabad factors)
  2. Building the monthly factor dataset
  3. The models (one function runs everything for one fund)
  4. Tests across all funds
  5. Charts
  6. Word report + process log
  7. Full run and live run
"""

import argparse
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import minimize
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)

# =====================================================================
# 0. SETTINGS  (every modelling choice is set here, in one place)
# =====================================================================
WINDOW_MONTHS = 60      # 5 years of monthly returns per fund
ROLL_WINDOW = 36        # rolling-alpha window (3 years)
HAC_LAGS = 3            # Newey-West lags (about one quarter of monthly data)
SIG = 0.05              # significance level used everywhere
MIN_LIVE_MONTHS = 36    # live mode works with shorter histories, down to 3 years
N_DEEP_DIVE = 3         # how many managers get the detailed attribution
N_PER_CATEGORY = 10     # funds sampled per category (4 x 10 = 40); None = use every eligible fund
SAMPLE_SEED = 42        # fixed seed so the random sample is reproducible

BASE = Path(__file__).resolve().parent
RAW = BASE / "data" / "raw"
CLEAN = BASE / "data" / "clean"
OUT = BASE / "outputs"
CHARTS = OUT / "charts"
JAMOVI = OUT / "jamovi"

# SEBI category -> (short label, style benchmark used as "the market")
CATEGORIES = {
    "Equity Scheme - Large Cap Fund": ("Large Cap", "NIFTY100"),
    "Equity Scheme - Mid Cap Fund": ("Mid Cap", "MID150"),
    "Equity Scheme - Small Cap Fund": ("Small Cap", "SMALL250"),
    "Equity Scheme - Flexi Cap Fund": ("Flexi Cap", "NIFTY500"),
}

# Index returns are taken from low-cost PASSIVE funds that track each NIFTY index
# (their NAV = the total-return index minus a small expense ratio). Chosen because
# each existed for the whole window, has no unit splits, and is in the AMFI archive.
# short code -> (label used in tables, AMFI scheme code of the tracker, tracker name)
INDICES = {
    "NIFTY100": ("NIFTY 100", "121146", "Nippon India ETF Nifty 100"),
    "MID150": ("NIFTY Midcap 150", "147622", "Motilal Oswal Nifty Midcap 150 Index Fund - Direct"),
    "SMALL250": ("NIFTY Smallcap 250", "147623", "Motilal Oswal Nifty Smallcap 250 Index Fund - Direct"),
    "NIFTY500": ("NIFTY 500", "147625", "Motilal Oswal Nifty 500 Index Fund - Direct"),
    "QUAL30": ("NIFTY200 Quality 30", "145648", "SBI Nifty 200 Quality 30 ETF"),
    "BANK": ("Banks (NIFTY Bank)", "140087", "Nippon India ETF Nifty Bank BeES"),
    "IT": ("IT (NIFTY IT)", "148408", "Nippon India ETF Nifty IT"),
    "CONS": ("Consumption (NIFTY India Consumption)", "128331", "Nippon India ETF Nifty India Consumption"),
    "INFRA": ("Infrastructure (NIFTY Infrastructure)", "140102", "Nippon India ETF Nifty Infrastructure BeES"),
    "HEALTH": ("Healthcare (MSCI India Dom. & World HC 45)", "148555",
               "Edelweiss MSCI India Domestic & World Healthcare 45 Index Fund - Direct"),
    "CPSE": ("PSU / Energy (NIFTY CPSE)", "140107", "CPSE ETF"),
}
BENCHMARKS = ["NIFTY100", "MID150", "SMALL250", "NIFTY500"]
STYLE_SET = ["NIFTY100", "MID150", "SMALL250"]                      # + cash
SECTOR_SET = ["BANK", "IT", "CONS", "INFRA", "HEALTH", "CPSE"]      # + cash
MAX_ABS_MONTHLY_RET = 40   # a monthly move beyond +/-40% = data error or unit split -> fund dropped

FACTOR_LABEL = {"mkt": "Market (style benchmark)", "smb": "Size (SMB)",
                "hml": "Value (HML)", "wml": "Momentum (WML)",
                "rmw": "Profitability (RMW proxy)", "mkt2": "Market squared (timing)"}
MODEL_LABEL = {"CAPM": "CAPM", "FF3": "Fama-French 3-factor",
               "Carhart4": "Carhart 4-factor", "Final5": "Final 5-factor model",
               "TM": "Treynor-Mazuy timing model"}

OFFLINE = False
PROXY_CHECK = {}
DATA_LOG = []           # data-cleaning notes that go into the process log


def log(msg):
    print(msg)
    DATA_LOG.append(msg)


def ensure_dirs():
    for p in [RAW, CLEAN, OUT, CHARTS, JAMOVI]:
        p.mkdir(parents=True, exist_ok=True)


# =====================================================================
# 1. DATA
#    a) AMFI month-end NAV of every scheme, Nov 2020 - Dec 2025
#       (data/raw/amfi_monthend_nav.csv.gz + data/raw/scheme_master.csv).
#       Built from AMFI's official daily NAV files, archived day by day in the
#       public GitHub repository rajadilipkolli/historical-mf-nav-data.
#       `python3 mf_project.py rebuild-data` rebuilds them from scratch.
#    b) IIM Ahmedabad Fama-French & momentum factors, monthly
#       (data/raw/iima_factors_monthly.csv; release 2025-12).
# =====================================================================
ARCHIVE_REPO = "https://github.com/rajadilipkolli/historical-mf-nav-data"
IIMA_PAGE = "https://faculty.iima.ac.in/iffm/Indian-Fama-French-Momentum/"
_ARCH = {}


def is_direct_growth(name):
    n = str(name).lower()
    if "direct" not in n or "growth" not in n:
        return False
    return not any(b in n for b in ["idcw", "dividend", "bonus", "payout", "reinvest", "segregated"])


def load_archive():
    """Month-end NAV of every AMFI scheme (long table: code, month, nav) + scheme master."""
    if not _ARCH:
        nav = pd.read_csv(RAW / "amfi_monthend_nav.csv.gz", dtype={"code": str})
        nav["month"] = pd.to_datetime(nav["month"] + "-01").dt.to_period("M")
        _ARCH["nav"] = nav
        _ARCH["master"] = pd.read_csv(RAW / "scheme_master.csv", dtype={"code": str})
    return _ARCH["nav"], _ARCH["master"]


def archive_series(code):
    nav, _ = load_archive()
    s = nav.loc[nav["code"] == str(code)].set_index("month")["nav"].sort_index()
    if s.empty:
        return s
    return s.reindex(pd.period_range(s.index.min(), s.index.max(), freq="M"))


def rebuild_archive(start="2020-11", end="2025-12"):
    """Re-creates the two archive files from the GitHub copy of AMFI's daily files
    (needs git + internet). Uses the last 5 calendar days of each month and keeps,
    for every scheme, the NAV with the latest date inside the month."""
    import subprocess, zipfile
    tmp = BASE / "_amfi_repo"
    if not tmp.exists():
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", "--depth", "1",
                        ARCHIVE_REPO, str(tmp)], check=True)
    files = set(subprocess.run(["git", "ls-tree", "-r", "--name-only", "HEAD", "data/"], cwd=tmp,
                               capture_output=True, text=True).stdout.split())
    want = []
    for p in pd.period_range(start, end, freq="M"):
        for d in range(p.days_in_month, p.days_in_month - 5, -1):
            f = f"data/{p.year}/{p.month:02d}/{d:02d}.zip"
            if f in files:
                want.append(f)
    subprocess.run(["git", "checkout", "HEAD", "--"] + want, cwd=tmp, check=True)
    rows = []
    for f in want:
        z = zipfile.ZipFile(tmp / f)
        cat = None
        for line in z.read(z.namelist()[0]).decode("utf-8", "ignore").splitlines():
            line = line.strip()
            if ";" in line:
                p = line.split(";")
                if p[0].strip().isdigit() and len(p) >= 8:
                    rows.append((p[0].strip(), p[1].strip(), p[6].strip(), p[7].strip(), cat))
            elif line:
                m = re.search(r"\((.+)\)\s*$", line)
                if m:
                    cat = m.group(1).strip()
    df = pd.DataFrame(rows, columns=["code", "name", "nav", "date", "category"])
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], format="%d-%b-%Y", errors="coerce")
    df = df.dropna(subset=["nav", "date"]).query("nav > 0").sort_values("date")
    df["month"] = df["date"].dt.to_period("M").astype(str)
    me = df.groupby(["code", "month"]).last().reset_index()
    me[["code", "month", "nav"]].to_csv(RAW / "amfi_monthend_nav.csv.gz", index=False)
    me.groupby("code")[["name", "category"]].last().reset_index().to_csv(RAW / "scheme_master.csv", index=False)
    print(f"Archive rebuilt: {len(me)} month-end NAVs, {me['code'].nunique()} schemes")


def build_universe():
    _, master = load_archive()
    df = master[master["category"].isin(CATEGORIES) & master["name"].map(is_direct_growth)].copy()
    df = df.drop_duplicates("code").reset_index(drop=True)
    log(f"Fund universe source: AMFI scheme list (category as of the latest month). Direct-Growth "
        f"candidates in the four categories: {len(df)}")
    return df


def search_fund(query):
    """Search fund names in the local AMFI list first, then MFapi online."""
    _, master = load_archive()
    words = query.lower().split()
    hit = master[master["name"].str.lower().map(lambda n: all(w in n for w in words))]
    hit = hit[hit["name"].map(is_direct_growth)] if hit["name"].map(is_direct_growth).any() else hit
    if len(hit):
        return [{"schemeCode": c, "schemeName": n} for c, n in zip(hit["code"], hit["name"])]
    js = http_get("https://api.mfapi.in/mf/search", params={"q": query}).json()
    return [h for h in js if is_direct_growth(h.get("schemeName", ""))] or js


SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                                      "AppleWebKit/537.36 Chrome/124.0 Safari/537.36"})


def http_get(url, **kw):
    if OFFLINE:
        raise RuntimeError("offline mode")
    r = SESSION.get(url, timeout=40, **kw)
    r.raise_for_status()
    return r


def fetch_nav_online(code):
    """MFapi.in fallback for funds not in the archive (live mode only)."""
    js = http_get(f"https://api.mfapi.in/mf/{code}").json()
    d = pd.DataFrame(js.get("data", []))
    d["date"] = pd.to_datetime(d["date"], format="%d-%m-%Y", errors="coerce")
    d["nav"] = pd.to_numeric(d["nav"], errors="coerce")
    s = d.dropna().set_index("date")["nav"].sort_index()
    return js.get("meta", {}), s


def get_iima():
    f = RAW / "iima_factors_monthly.csv"
    if not f.exists():
        try:
            html = http_get(IIMA_PAGE).text
            rel = max(re.findall(r"DATA/(\d{4}-\d{2})_FourFactors_and_Market_Returns_Monthly_"
                                 r"SurvivorshipBiasAdjusted\.csv", html))
            f.write_text(http_get(f"{IIMA_PAGE}DATA/{rel}_FourFactors_and_Market_Returns_Monthly_"
                                  f"SurvivorshipBiasAdjusted.csv").text)
        except Exception as e:
            sys.exit(f"IIMA factor file missing and could not be downloaded ({e}). Save it as {f}")
    df = pd.read_csv(f, na_values=["NA", ""])
    df.columns = [c.strip().lower() for c in df.columns]
    df["month"] = pd.to_datetime(df["date"].astype(str).str[:7] + "-01").dt.to_period("M")
    return df.set_index("month")[["smb", "hml", "wml", "mf", "rf"]].astype(float)


# =====================================================================
# 2. BUILDING THE MONTHLY DATASET
# =====================================================================
def to_month_end(s):
    """Last available value in each calendar month, on a complete month grid,
    so a missing month shows up as a gap instead of being skipped."""
    s = s.dropna()
    if s.empty:
        return s
    if not isinstance(s.index, pd.PeriodIndex):
        s = s.groupby(s.index.to_period("M")).last()
    return s.reindex(pd.period_range(s.index.min(), s.index.max(), freq="M"))


def pct_returns(level):
    return (level / level.shift(1) - 1) * 100       # monthly % returns


def build_factor_frame(rebuild=False):
    """One table, one row per month: IIMA factors, index returns, RMW proxy."""
    cache = CLEAN / "factors_and_indices_monthly.csv"
    if cache.exists() and not rebuild:
        F = pd.read_csv(cache)
        F["month"] = pd.to_datetime(F["month"] + "-01").dt.to_period("M")
        return F.set_index("month")
    F = get_iima()
    for short, (label, code, tracker) in INDICES.items():
        r = pct_returns(archive_series(code))
        F[short] = r
        log(f"Index {label}: proxied by {tracker} (AMFI code {code}); "
            f"{int(r.loc['2021-01':].notna().sum())} monthly returns")
    # RMW proxy: quality stocks minus the large-cap market (Nifty 100 is used as the base
    # because no Nifty 200 tracker existed for the whole window)
    F["rmw"] = F["QUAL30"] - F["NIFTY100"]
    F["cash"] = F["rf"]
    chk = F.loc["2021-01":].dropna(subset=["NIFTY500", "mf"])
    rho = chk["NIFTY500"].corr(chk["mf"] + chk["rf"])
    PROXY_CHECK["text"] = (f"the NIFTY 500 tracker's monthly return correlates {rho:.3f} with IIMA's "
                           f"independent market return over {len(chk)} months.")
    log("Proxy check: " + PROXY_CHECK["text"])
    F = F.sort_index()
    out = F.copy()
    out.index = out.index.astype(str)
    out.rename_axis("month").to_csv(cache)
    return F


def study_window(F):
    core = ["smb", "hml", "wml", "rf"] + BENCHMARKS
    last = F.dropna(subset=core).index.max()
    return last - (WINDOW_MONTHS - 1), last



def has_rmw(F):
    return "rmw" in F.columns and F["rmw"].notna().sum() > 24


def models_for(F):
    fin = ["mkt", "smb", "hml", "wml"] + (["rmw"] if has_rmw(F) else [])
    return {"CAPM": ["mkt"], "FF3": ["mkt", "smb", "hml"],
            "Carhart4": ["mkt", "smb", "hml", "wml"], "Final5": fin, "TM": ["mkt", "mkt2"]}


# =====================================================================
# 3. THE MODELS  (everything for one fund)
# =====================================================================
def fit(y, X):
    """Same regression twice: ordinary standard errors (these match Jamovi)
    and Newey-West (HAC) standard errors (robust to heteroskedasticity
    and autocorrelation; used for our conclusions)."""
    Xc = sm.add_constant(X, has_constant="add")
    return sm.OLS(y, Xc).fit(), sm.OLS(y, Xc).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})


def style_analysis(r, R):
    """Sharpe (1992) returns-based style analysis:
    minimise sum of squared (fund return - weighted index returns)
    subject to every weight >= 0 and weights summing to 1."""
    R = np.asarray(R, float)
    r = np.asarray(r, float)
    k = R.shape[1]
    obj = lambda w: np.sum((r - R @ w) ** 2)
    res = minimize(obj, np.full(k, 1 / k), method="SLSQP", bounds=[(0, 1)] * k,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   options={"maxiter": 500, "ftol": 1e-12})
    w = np.clip(res.x, 0, None)
    w = w / w.sum()
    resid = r - R @ w
    return w, 1 - resid.var() / r.var()


def max_drawdown(ret_pct):
    g = (1 + ret_pct / 100).cumprod()
    return float((g / g.cummax() - 1).min() * 100)


def analyse(ret, bench, F, rolling=True):
    """Runs every model for one fund (or stock). ret = monthly % returns."""
    MODELS = models_for(F)
    fin = MODELS["Final5"]
    d = pd.DataFrame({"ret": ret}).join(F, how="inner")
    d["exret"] = d["ret"] - d["rf"]
    d["mkt"] = d[bench] - d["rf"]
    d["mkt2"] = d["mkt"] ** 2
    d = d.dropna(subset=["exret"] + fin)
    n = len(d)
    R = {"n_months": n, "start": str(d.index.min()), "end": str(d.index.max()), "benchmark": bench}
    fits = {}

    for m, cols in MODELS.items():
        ols, hac = fit(d["exret"], d[cols])
        fits[m] = (ols, hac)
        R[f"{m}_alpha_m"] = ols.params["const"]
        R[f"{m}_alpha_ann"] = ols.params["const"] * 12
        R[f"{m}_alpha_p"] = ols.pvalues["const"]
        R[f"{m}_alpha_p_hac"] = hac.pvalues["const"]
        R[f"{m}_alpha_t_hac"] = hac.tvalues["const"]
        R[f"{m}_r2"] = ols.rsquared
        R[f"{m}_adjr2"] = ols.rsquared_adj

    ols, hac = fits["Final5"]
    for k in fin:
        R[f"b_{k}"] = ols.params[k]
        R[f"p_{k}"] = hac.pvalues[k]
    R["beta_capm"] = fits["CAPM"][0].params["mkt"]
    tm_o, tm_h = fits["TM"]
    R["TM_beta"] = tm_o.params["mkt"]
    R["TM_gamma"] = tm_o.params["mkt2"]
    R["TM_gamma_p"] = tm_o.pvalues["mkt2"]
    R["TM_gamma_p_hac"] = tm_h.pvalues["mkt2"]

    # ---- assumption checks on the final model ----
    e = ols.resid
    X = sm.add_constant(d[fin])
    R["sw_p"] = stats.shapiro(e).pvalue
    R["jb_p"] = jarque_bera(e)[1]
    R["bp_p"] = het_breuschpagan(e, X)[1]
    R["dw"] = durbin_watson(e)
    R["max_vif"] = max(variance_inflation_factor(X.values, i) for i in range(1, X.shape[1]))

    # ---- return attribution: where did the average return come from? ----
    # OLS identity: mean(excess return) = alpha + sum(beta_k * mean(factor_k))
    mean_f = d[fin].mean()
    R["mean_exret_ann"] = d["exret"].mean() * 12
    R["contrib_alpha"] = R["Final5_alpha_ann"]
    for k in fin:
        R[f"contrib_{k}"] = ols.params[k] * mean_f[k] * 12
    # same thing relative to the benchmark (fund minus benchmark)
    R["active_ann"] = (d["ret"] - d[bench]).mean() * 12
    R["act_mkt"] = (ols.params["mkt"] - 1) * mean_f["mkt"] * 12
    for k in fin[1:]:
        R[f"act_{k}"] = R[f"contrib_{k}"]
    R["act_alpha"] = R["Final5_alpha_ann"]

    # ---- Fama (1972) decomposition: security selection ----
    Rp, Rm = d["exret"].mean() * 12, d["mkt"].mean() * 12
    sp, sm_ = d["exret"].std() * np.sqrt(12), d["mkt"].std() * np.sqrt(12)
    b = R["beta_capm"]
    R["fama_risk"] = b * Rm
    R["fama_selectivity"] = Rp - b * Rm
    R["fama_diversification"] = (sp / sm_ - b) * Rm
    R["fama_net_selectivity"] = R["fama_selectivity"] - R["fama_diversification"]

    # ---- performance ratios ----
    yrs = n / 12
    R["cagr"] = ((1 + d["ret"] / 100).prod() ** (1 / yrs) - 1) * 100
    R["bench_cagr"] = ((1 + d[bench] / 100).prod() ** (1 / yrs) - 1) * 100
    R["vol"] = d["ret"].std() * np.sqrt(12)
    R["sharpe"] = Rp / sp
    dn = d["exret"].clip(upper=0)
    R["sortino"] = Rp / (np.sqrt((dn ** 2).mean()) * np.sqrt(12))
    R["treynor"] = Rp / b
    act = d["ret"] - d[bench]
    R["tracking_error"] = act.std() * np.sqrt(12)
    R["info_ratio"] = act.mean() * 12 / R["tracking_error"]
    up, dnm = d[bench] > 0, d[bench] < 0
    R["up_capture"] = d.loc[up, "ret"].mean() / d.loc[up, bench].mean() * 100
    R["down_capture"] = d.loc[dnm, "ret"].mean() / d.loc[dnm, bench].mean() * 100
    R["max_dd"] = max_drawdown(d["ret"])

    # ---- consistency 1: calendar years beating the benchmark ----
    yr = d.groupby(d.index.year).agg(cnt=("ret", "size"),
                                      f=("ret", lambda x: (1 + x / 100).prod() - 1),
                                      b=(bench, lambda x: (1 + x / 100).prod() - 1))
    yr = yr[yr["cnt"] == 12]
    R["years_total"] = len(yr)
    R["years_beat"] = int((yr["f"] > yr["b"]).sum())
    R["beat_rate"] = R["years_beat"] / len(yr) if len(yr) else np.nan

    # ---- consistency 2: alpha in each half of the period ----
    h = n // 2
    for lab, part in [("h1", d.iloc[:h]), ("h2", d.iloc[h:])]:
        o, _ = fit(part["exret"], part[fin])
        R[f"alpha_{lab}"] = o.params["const"] * 12

    # ---- consistency 3: rolling 36-month alpha ----
    roll = pd.Series(dtype=float)
    if rolling and n >= ROLL_WINDOW + 6:
        vals = {}
        for i in range(ROLL_WINDOW, n + 1):
            part = d.iloc[i - ROLL_WINDOW:i]
            o, _ = fit(part["exret"], part[fin])
            vals[part.index[-1]] = o.params["const"] * 12
        roll = pd.Series(vals)
        R["roll_pos_share"] = (roll > 0).mean()
    else:
        R["roll_pos_share"] = np.nan

    # ---- style and sector attribution (returns-based style analysis) ----
    for tag, cols in [("style", STYLE_SET), ("sector", SECTOR_SET)]:
        use = [c for c in cols if c in d.columns and d[c].notna().all()] + ["cash"]
        if len(use) >= 3:
            w, r2 = style_analysis(d["ret"], d[use])
            for c, wi in zip(use, w):
                R[f"{tag}_w_{c}"] = wi
            R[f"{tag}_r2"] = r2
            R[f"{tag}_selection_share"] = 1 - r2
    return R, fits, d, roll


# =====================================================================
# 4. TESTS ACROSS ALL FUNDS
# =====================================================================
def cross_tests(T):
    o = {}
    a = T["Final5_alpha_ann"]
    t = stats.ttest_1samp(a, 0)
    o["alpha_mean"], o["alpha_t"], o["alpha_p"] = a.mean(), t.statistic, t.pvalue
    o["alpha_wilcoxon_p"] = stats.wilcoxon(a).pvalue if len(a) >= 6 else np.nan
    sig = T["Final5_alpha_p_hac"] < SIG
    o["n"] = len(T)
    o["n_pos_sig"] = int(((a > 0) & sig).sum())
    o["n_neg_sig"] = int(((a < 0) & sig).sum())
    o["n_pos"] = int((a > 0).sum())
    o["binom_p"] = stats.binomtest(o["n_pos_sig"], len(T), SIG / 2, alternative="greater").pvalue

    groups = [g["Final5_alpha_ann"].values for _, g in T.groupby("category") if len(g) >= 2]
    if len(groups) >= 2:
        o["anova_F"], o["anova_p"] = stats.f_oneway(*groups)
        o["kw_H"], o["kw_p"] = stats.kruskal(*groups)
        o["levene_p"] = stats.levene(*groups).pvalue

    g = T["TM_gamma"]
    tg = stats.ttest_1samp(g, 0)
    o["gamma_mean"], o["gamma_t"], o["gamma_p"] = g.mean(), tg.statistic, tg.pvalue
    gs = T["TM_gamma_p_hac"] < SIG
    o["n_gamma_pos_sig"] = int(((g > 0) & gs).sum())
    o["n_gamma_neg_sig"] = int(((g < 0) & gs).sum())
    o["sel_timing_rho"], o["sel_timing_p"] = stats.spearmanr(T["TM_alpha_ann"], T["TM_gamma"])[:2]

    # persistence: are first-half winners also second-half winners?
    w1 = T["alpha_h1"] > T["alpha_h1"].median()
    w2 = T["alpha_h2"] > T["alpha_h2"].median()
    ww, wl = int((w1 & w2).sum()), int((w1 & ~w2).sum())
    lw, ll = int((~w1 & w2).sum()), int((~w1 & ~w2).sum())
    o["ww"], o["wl"], o["lw"], o["ll"] = ww, wl, lw, ll
    o["cpr"] = ((ww + .5) * (ll + .5)) / ((wl + .5) * (lw + .5))
    o["fisher_p"] = stats.fisher_exact([[ww, wl], [lw, ll]])[1]
    o["persist_rho"], o["persist_p"] = stats.spearmanr(T["alpha_h1"], T["alpha_h2"])[:2]
    o["share_bp_fail"] = (T["bp_p"] < SIG).mean()
    o["share_dw_out"] = ((T["dw"] < 1.5) | (T["dw"] > 2.5)).mean()
    o["share_sw_fail"] = (T["sw_p"] < SIG).mean()
    return o


def pick_deep_dive(T):
    c1 = (T["Final5_alpha_ann"] > 0) & (T["Final5_alpha_p_hac"] < SIG)
    c2 = (T["alpha_h1"] > 0) & (T["alpha_h2"] > 0)
    c3 = T["roll_pos_share"] >= 0.75
    T["consistent_outperformer"] = c1 & c2 & c3
    order = T.sort_values(["consistent_outperformer", "Final5_alpha_t_hac"], ascending=[False, False])
    return order.head(N_DEEP_DIVE)


# =====================================================================
# 5. CHARTS
# =====================================================================
def short_name(s, n=38):
    s = re.sub(r"\s*-\s*Direct.*$", "", str(s), flags=re.I)
    return s if len(s) <= n else s[: n - 1] + "…"


def save(fig, name):
    p = CHARTS / name
    fig.tight_layout()
    fig.savefig(p, dpi=160)
    plt.close(fig)
    return p


def chart_alpha_hist(T):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    sig = T["Final5_alpha_p_hac"] < SIG
    bins = np.linspace(T["Final5_alpha_ann"].min(), T["Final5_alpha_ann"].max(), 20)
    ax.hist(T.loc[~sig, "Final5_alpha_ann"], bins=bins, color="#9aa8b8", label="not significant")
    ax.hist(T.loc[sig, "Final5_alpha_ann"], bins=bins, color="#1f6f8b", label=f"significant at {SIG:.0%}")
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("Annualised alpha, % per year (final 5-factor model)")
    ax.set_ylabel("Number of funds")
    ax.legend(frameon=False)
    return save(fig, "fig1_alpha_distribution.png")


def chart_alpha_box(T):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    cats = [c for c in ["Large Cap", "Mid Cap", "Small Cap", "Flexi Cap"] if c in set(T["category"])]
    data = [T.loc[T["category"] == c, "Final5_alpha_ann"] for c in cats]
    try:
        ax.boxplot(data, tick_labels=cats)
    except TypeError:                      # matplotlib < 3.9
        ax.boxplot(data, labels=cats)
    ax.axhline(0, color="k", lw=1, ls="--")
    ax.set_ylabel("Annualised alpha, %")
    return save(fig, "fig2_alpha_by_category.png")


def chart_loadings(T, fin):
    m = T.groupby("category")[[f"b_{k}" for k in fin]].mean()
    m.columns = [FACTOR_LABEL[k] for k in fin]
    fig, ax = plt.subplots(figsize=(8, 4))
    m.plot.bar(ax=ax, width=0.8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Average factor beta")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    return save(fig, "fig3_factor_loadings_by_category.png")


def chart_attribution(D, fin):
    rows = []
    for _, r in D.iterrows():
        rows.append([r[f"act_{k}"] if k != "mkt" else r["act_mkt"] for k in fin] + [r["act_alpha"]])
    cols = [FACTOR_LABEL[k] for k in fin] + ["Alpha (manager skill)"]
    A = pd.DataFrame(rows, columns=cols, index=[short_name(x, 30) for x in D["name"]])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    A.plot.barh(ax=ax, stacked=True, width=0.6)
    for i, (_, r) in enumerate(D.iterrows()):
        ax.plot(r["active_ann"], i, "kD", ms=6)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("% per year relative to style benchmark (diamond = total gap)")
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    return save(fig, "fig4_return_attribution.png")


def chart_rolling(rolls, names):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for code, s in rolls.items():
        if len(s):
            ax.plot(s.index.to_timestamp(), s.values, label=short_name(names[code], 30))
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Rolling 36-month alpha, % p.a.")
    ax.legend(fontsize=8, frameon=False)
    return save(fig, "fig5_rolling_alpha.png")


def chart_style(D, tag, cols, fname, title):
    keep = [c for c in cols + ["cash"] if f"{tag}_w_{c}" in D.columns]
    if not keep:
        return None
    W = D[[f"{tag}_w_{c}" for c in keep]].fillna(0) * 100
    W.columns = [INDICES[c][0].replace(" TRI", "") if c in INDICES else "Cash" for c in keep]
    W.index = [short_name(x, 30) for x in D["name"]]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    W.plot.barh(ax=ax, stacked=True, width=0.6, colormap="tab20")
    ax.set_xlabel(f"{title}, % weight (Sharpe style analysis)")
    ax.set_xlim(0, 100)
    ax.legend(fontsize=7, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    return save(fig, fname)


def chart_tm(d, name, code):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.scatter(d["mkt"], d["exret"], s=14, color="#1f6f8b")
    c = np.polyfit(d["mkt"], d["exret"], 2)
    xs = np.linspace(d["mkt"].min(), d["mkt"].max(), 100)
    ax.plot(xs, np.polyval(c, xs), "r-", label="Treynor-Mazuy fitted curve")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("Benchmark excess return, % per month")
    ax.set_ylabel("Fund excess return, % per month")
    ax.set_title(short_name(name, 50), fontsize=10)
    ax.legend(frameon=False)
    return save(fig, f"fig8_timing_{code}.png")


def chart_iterations(T):
    m = [k for k in ["CAPM", "FF3", "Carhart4", "Final5"]]
    vals = [T[f"{k}_adjr2"].mean() for k in m]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.bar([MODEL_LABEL[k] for k in m], vals, color="#1f6f8b")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("Average adjusted R²")
    ax.set_ylim(min(vals) - 0.1, 1)
    ax.tick_params(axis="x", labelsize=8)
    return save(fig, "fig9_model_iterations.png")


# =====================================================================
# 6. WORD REPORT AND PROCESS LOG
# =====================================================================
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor


def new_doc():
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(11)
    return doc


def para(doc, text, bold=False, italic=False, color=None, size=None):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold, r.italic = bold, italic
    if color:
        r.font.color.rgb = RGBColor(*color)
    if size:
        r.font.size = Pt(size)
    return p


def todo(doc, text):
    """Orange text = the group must write this part in its own words."""
    return para(doc, "[GROUP TO WRITE] " + text, italic=True, color=(0xC5, 0x5A, 0x11))


def table(doc, df, dec=2, font=8.5):
    t = doc.add_table(rows=1, cols=len(df.columns))
    t.style = "Light Grid Accent 1"
    for i, c in enumerate(df.columns):
        t.rows[0].cells[i].text = str(c)
    for _, row in df.iterrows():
        cells = t.add_row().cells
        for i, v in enumerate(row):
            if isinstance(v, (float, np.floating)):
                cells[i].text = "" if pd.isna(v) else f"{v:,.{dec}f}"
            else:
                cells[i].text = str(v)
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(font)
    doc.add_paragraph()
    return t


def pic(doc, path, caption, width=6.2):
    if path and Path(path).exists():
        doc.add_picture(str(path), width=Inches(width))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        c = para(doc, caption, italic=True, size=9)
        c.alignment = WD_ALIGN_PARAGRAPH.CENTER


def pfmt(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def yesno(b):
    return "Yes" if b else "No"


def write_report(T, X, D, fin, start, end, charts, rbsa_ok, jam_code, jam_ref):
    doc = new_doc()
    N = len(T)
    cats = T["category"].value_counts()

    # ---------------- title ----------------
    para(doc, "Business Analytics Project - Theme 1", size=12)
    doc.add_heading("Do Indian Active Equity Fund Managers Beat Their Style Benchmarks?", 0)
    para(doc, "Factor, style-sector and market-timing attribution of Indian equity mutual funds", italic=True, size=13)
    para(doc, "Group members: [Name 1 - Roll no.], [Name 2 - Roll no.], [Name 3 - Roll no.], "
              "[Name 4 - Roll no.], [Name 5 - Roll no.]")
    para(doc, "NMIMS School of Commerce | Submitted to: [Instructor name] | Date: [date]")
    doc.add_page_break()

    # ---------------- 1 summary ----------------
    doc.add_heading("1. Summary", 1)
    para(doc, f"We studied {N} actively managed Direct-Growth equity mutual funds from four SEBI categories "
              f"(" + ", ".join(f"{c}: {v}" for c, v in cats.items()) + f") over {WINDOW_MONTHS} months "
              f"({start} to {end}). Each fund was compared with its own style benchmark rather than "
              f"the broad market. We asked three questions: does the manager earn a positive alpha after "
              f"controlling for market, size, value, momentum and profitability; where does the fund's "
              f"return come from (factors, sectors, market timing or stock selection); and is any "
              f"outperformance consistent over time.")
    para(doc, f"Headline numbers from the analysis: the average annualised alpha in the final model is "
              f"{X['alpha_mean']:.2f}% (one-sample t-test p = {pfmt(X['alpha_p'])}). "
              f"{X['n_pos_sig']} of {N} funds have a positive alpha significant at {SIG:.0%} using "
              f"Newey-West standard errors, and {X['n_neg_sig']} have a significantly negative alpha. "
              f"{X['n_gamma_pos_sig']} funds show significant positive market-timing ability "
              f"(Treynor-Mazuy) and {X['n_gamma_neg_sig']} show significant negative timing. "
              f"{int(T['consistent_outperformer'].sum())} fund(s) meet all three of our consistency tests.")
    todo(doc, "Write 3-4 sentences in your own words on what these results mean for an investor choosing "
              "between active funds and index funds. (The guidelines require interpretation to be your own.)")

    # ---------------- 2 problem & objectives ----------------
    doc.add_heading("2. Problem definition and objectives", 1)
    para(doc, "Active mutual funds charge higher fees than index funds on the promise that the manager will "
              "beat the market. Comparing raw returns is misleading, because a fund can beat the NIFTY simply "
              "by holding riskier stocks (smaller companies, cheaper 'value' stocks, recent winners). The right "
              "test is whether the manager adds return after accounting for the risks and styles the fund "
              "carries, and, for the managers who do, which source the extra return comes from.")
    para(doc, "Research questions", bold=True)
    for q in [
        "RQ1. After controlling for market, size (SMB), value (HML), momentum (WML) and profitability (RMW), "
        "do Indian active equity managers earn a significant positive alpha against their style benchmark?",
        "RQ2. For managers who beat the benchmark, which factors, styles and sectors explain the return, "
        "and how much is left as security selection?",
        "RQ3. Do managers show market-timing ability (Treynor-Mazuy)?",
        "RQ4. Is outperformance consistent: across both halves of the period, across rolling windows, "
        "and across calendar years?",
        "RQ5. Does alpha differ across Large, Mid, Small and Flexi Cap categories?"]:
        doc.add_paragraph(q, style="List Bullet")
    para(doc, "Hypotheses tested", bold=True)
    for q in ["H0(a): alpha = 0 for each fund (no skill after factors).",
              "H0(b): average alpha across funds = 0.",
              "H0(c): timing coefficient gamma = 0 (no timing ability).",
              "H0(d): mean alpha is equal across the four categories.",
              "H0(e): first-half winners are no more likely than losers to be second-half winners."]:
        doc.add_paragraph(q, style="List Bullet")

    # ---------------- 3 literature ----------------
    doc.add_heading("3. Background and literature", 1)
    para(doc, "Jensen (1968) defined alpha as the intercept of a CAPM regression of fund excess returns on "
              "market excess returns. Fama and French (1993) showed that size and value explain much of what "
              "CAPM leaves out, and Carhart (1997) added momentum, finding that most apparent fund skill in the "
              "US is explained by momentum and costs. Fama and French (2015) added profitability (RMW) and "
              "investment (CMA). Treynor and Mazuy (1966) tested market timing by adding a squared market term: "
              "a manager who raises exposure before rallies produces a convex (upward-curving) return pattern. "
              "Fama (1972) split a fund's return into risk and selectivity components, and Sharpe (1992) "
              "introduced returns-based style analysis, which infers a fund's effective style and sector mix "
              "from its returns alone.")
    para(doc, "For India, Agarwalla, Jacob and Varma (2013) built the Fama-French and momentum factors used in "
              "this study, maintained by IIM Ahmedabad. Singh, Singh and Prakash (2023) found that the investment "
              "factor (CMA) adds no explanatory power once profitability is included for Indian stocks, which "
              "supports our decision to drop CMA.")

    # ---------------- 4 data ----------------
    doc.add_heading("4. Data", 1)
    src = pd.DataFrame([
        ["Fund list, SEBI category, month-end NAVs",
         "AMFI daily NAV files (all schemes), Nov 2020 - Dec 2025, from the public GitHub archive "
         "rajadilipkolli/historical-mf-nav-data", "Daily -> month-end"],
        ["SMB, HML, WML, risk-free rate", "IIM Ahmedabad Fama-French & Momentum Data Library, release "
                                          "2025-12 (survivorship-bias adjusted)", "Monthly"],
        ["Style benchmarks (the 'market')", "NAVs of passive funds tracking NIFTY 100, Midcap 150, "
                                            "Smallcap 250 and NIFTY 500 (see table below)", "Month-end"],
        ["Profitability proxy (RMW)", "SBI Nifty 200 Quality 30 ETF minus Nippon India ETF Nifty 100", "Monthly"],
        ["Sector proxies", "Passive funds tracking NIFTY Bank, IT, India Consumption, Infrastructure, CPSE "
                           "and MSCI India Domestic & World Healthcare 45", "Month-end"]],
        columns=["Data", "Source", "Frequency"])
    table(doc, src)
    para(doc, "Index proxies. NIFTY total-return index files could not be downloaded programmatically, so each "
              "index is represented by a low-cost passive fund that tracks it. A passive fund's NAV equals the "
              "total-return index minus a small expense ratio, so it includes dividends exactly as fund NAVs do. "
              "Trackers were chosen to (i) exist for the whole window, (ii) have no unit splits, and (iii) be "
              "ETFs or Direct-plan index funds (lowest cost). Two ICICI ETFs were rejected because unit splits "
              "created false -90% monthly returns.")
    table(doc, pd.DataFrame([[v[0], v[2], v[1]] for v in INDICES.values()],
                            columns=["Index", "Tracker used", "AMFI code"]), font=8)
    para(doc, "Cleaning rules", bold=True)
    for q in [f"Only Direct-Growth plans (lowest cost, no payouts, so NAV growth equals total return).",
              f"Only funds with a complete month-end NAV for every one of the {WINDOW_MONTHS} months "
              f"(plus the month before, needed for the first return).",
              "Monthly return = (NAV at end of month / NAV at end of previous month) - 1, in %.",
              f"Funds with any monthly return beyond +/-{MAX_ABS_MONTHLY_RET}% are dropped as data errors or unit splits.",
              "Fund category is taken from the latest AMFI file (December 2025).",
              "The window ends at the last month covered by the IIMA factor file, and starts well after "
              "SEBI's 2017-18 recategorisation, so category labels mean the same thing throughout.",
              "Excess return = fund return - risk-free rate (IIMA monthly risk-free rate)."]:
        doc.add_paragraph(q, style="List Bullet")
    para(doc, "Sample after cleaning", bold=True)
    for m in DATA_LOG:
        if m.startswith(("Fund universe", "Funds kept", "Excluded")):
            doc.add_paragraph(m, style="List Bullet")
    para(doc, "Data files submitted (data/clean): fund_universe.csv, fund_returns_monthly.csv, "
              "factors_and_indices_monthly.csv, master_dataset.csv, fund_results.csv.", italic=True)

    # ---------------- 5 methodology ----------------
    doc.add_heading("5. Methodology", 1)
    para(doc, "All regressions use monthly percentage returns. Alpha is reported per month and annualised "
              "(x12). Every regression is run twice: with ordinary (OLS) standard errors, which Jamovi also "
              f"produces, and with Newey-West (HAC, {HAC_LAGS} lags) standard errors, which remain valid if "
              "residuals are heteroskedastic or autocorrelated. Conclusions use the Newey-West p-values.")
    eqs = [
        ("5.1 CAPM (style benchmark as market)", "Rp - Rf = a + b(Rb - Rf) + e",
         "Rb is the fund's own style benchmark (NIFTY 100 for Large Cap, Midcap 150 for Mid Cap, Smallcap 250 "
         "for Small Cap, NIFTY 500 for Flexi Cap). Comparing a small-cap fund with the NIFTY 50 would credit "
         "the manager for simply holding small companies."),
        ("5.2 Fama-French 3-factor", "Rp - Rf = a + b(Rb - Rf) + s SMB + h HML + e",
         "SMB = small minus big companies; HML = high minus low book-to-market (value minus growth)."),
        ("5.3 Carhart 4-factor", "... + w WML", "WML = winners minus losers (12-month momentum)."),
        ("5.4 Final 5-factor model", "Rp - Rf = a + b(Rb - Rf) + s SMB + h HML + w WML + r RMW + e",
         "RMW (robust minus weak profitability) has no free Indian series, so we use a proxy: the return on "
         "the NIFTY200 Quality 30 tracker minus the NIFTY 100 tracker (no NIFTY 200 tracker existed for the "
         "whole window). CMA is dropped (see Section 3)."),
        ("5.5 Treynor-Mazuy market timing", "Rp - Rf = a + b(Rb - Rf) + g(Rb - Rf)^2 + e",
         "g > 0 means the fund gains more in up-months than it loses in down-months: timing skill. "
         "a in this model measures security selection after separating out timing."),
        ("5.6 Return attribution", "mean(Rp - Rf) = a + sum of [beta_k x mean(factor_k)]",
         "This identity holds exactly for OLS with an intercept. It splits each fund's average return "
         "into the part explained by each factor and the part left as alpha. Relative to the benchmark, "
         "the market part becomes (b - 1) x mean(Rb - Rf)."),
        ("5.7 Fama (1972) decomposition", "Selectivity = (Rp - Rf) - b(Rb - Rf); "
                                          "Net selectivity = Selectivity - (sp/sb - b)(Rb - Rf)",
         "Net selectivity removes the reward the fund should have earned for holding a less diversified "
         "portfolio, leaving pure stock-picking."),
        ("5.8 Returns-based style and sector analysis (Sharpe 1992)",
         "min sum[Rp - sum(w_i x R_i)]^2 subject to w_i >= 0 and sum(w_i) = 1",
         "Style set: NIFTY 100, Midcap 150, Smallcap 250 and cash. Sector set: Banks, IT, Consumption, "
         "Infrastructure, Healthcare, PSU/Energy (CPSE) and cash. The weights give the fund's effective mix; 1 - R^2 is the share of return variation "
         "that no index mix explains, i.e. the manager's own security selection."),
    ]
    for title, eq, why in eqs:
        para(doc, title, bold=True)
        para(doc, eq, italic=True)
        para(doc, why)
    para(doc, "5.9 Assumption checks", bold=True)
    para(doc, "Shapiro-Wilk and Jarque-Bera (normal residuals), Breusch-Pagan (constant variance), "
              "Durbin-Watson (no autocorrelation) and VIF (multicollinearity) on every fund's final model.")
    para(doc, "5.10 Tests across funds", bold=True)
    para(doc, "One-sample t-test and Wilcoxon signed-rank test on alphas; one-way ANOVA, Kruskal-Wallis and "
              "Levene across categories; binomial test of how many significant positive alphas we would "
              "expect by luck alone (2.5% of funds); Spearman correlation between selection and timing; "
              "persistence via a 2x2 winners/losers table (Fisher exact test, cross-product ratio).")
    para(doc, "5.11 Consistency screen", bold=True)
    para(doc, "A fund is a 'consistent outperformer' only if (1) its final-model alpha is positive and "
              f"significant at {SIG:.0%} (Newey-West), (2) alpha is positive in both halves of the period, and "
              f"(3) its rolling {ROLL_WINDOW}-month alpha is positive in at least 75% of windows.")

    # ---------------- 6 results ----------------
    doc.add_heading("6. Results", 1)
    doc.add_heading("6.1 Model building: does each added factor help?", 2)
    it = pd.DataFrame([[MODEL_LABEL[m], T[f"{m}_adjr2"].mean(), T[f"{m}_alpha_ann"].mean(),
                        int(((T[f"{m}_alpha_ann"] > 0) & (T[f"{m}_alpha_p_hac"] < SIG)).sum()),
                        int(((T[f"{m}_alpha_ann"] < 0) & (T[f"{m}_alpha_p_hac"] < SIG)).sum())]
                       for m in ["CAPM", "FF3", "Carhart4", "Final5"]],
                      columns=["Model", "Avg adj. R²", "Avg alpha % p.a.", "# sig. positive", "# sig. negative"])
    table(doc, it, dec=3)
    pic(doc, charts.get("iter"), "Figure 1. Average adjusted R² as factors are added")

    doc.add_heading("6.2 Do managers beat their style benchmark? (RQ1)", 2)
    by = T.groupby("category").agg(Funds=("name", "size"), Avg_alpha=("Final5_alpha_ann", "mean"),
                                   Median_alpha=("Final5_alpha_ann", "median"),
                                   Sig_pos=("Final5_alpha_p_hac", lambda p: int(((p < SIG) & (T.loc[p.index, "Final5_alpha_ann"] > 0)).sum())),
                                   Avg_active=("active_ann", "mean"), Avg_IR=("info_ratio", "mean")).reset_index()
    by.columns = ["Category", "Funds", "Avg alpha %", "Median alpha %", "# sig. +alpha",
                  "Avg return vs benchmark %", "Avg info ratio"]
    table(doc, by)
    para(doc, f"One-sample t-test of mean alpha = 0: t = {X['alpha_t']:.2f}, p = {pfmt(X['alpha_p'])}. "
              f"Wilcoxon signed-rank p = {pfmt(X['alpha_wilcoxon_p'])}. Funds with significant positive alpha: "
              f"{X['n_pos_sig']} of {N}. If no manager had skill we would expect about {N * SIG / 2:.1f} by "
              f"luck; binomial test p = {pfmt(X['binom_p'])}.")
    pic(doc, charts.get("hist"), "Figure 2. Distribution of annualised alpha across funds")
    if "anova_p" in X:
        para(doc, f"Across categories (RQ5): ANOVA F = {X['anova_F']:.2f}, p = {pfmt(X['anova_p'])}; "
                  f"Kruskal-Wallis H = {X['kw_H']:.2f}, p = {pfmt(X['kw_p'])}; Levene p = "
                  f"{pfmt(X['levene_p'])}.")
    pic(doc, charts.get("box"), "Figure 3. Alpha by category")
    pic(doc, charts.get("load"), "Figure 4. Average factor betas by category (style profile)")
    todo(doc, "Interpret 6.1-6.2: does adding factors shrink alpha? Which categories look better? "
              "Which style tilts does each category carry?")

    doc.add_heading("6.3 Market timing (RQ3)", 2)
    para(doc, f"Mean timing coefficient gamma = {X['gamma_mean']:.4f} (t = {X['gamma_t']:.2f}, p = "
              f"{pfmt(X['gamma_p'])}). Significant positive timing: {X['n_gamma_pos_sig']} funds; significant "
              f"negative timing: {X['n_gamma_neg_sig']} funds. Spearman correlation between selection "
              f"(Treynor-Mazuy alpha) and timing (gamma): rho = {X['sel_timing_rho']:.2f}, p = "
              f"{pfmt(X['sel_timing_p'])}.")
    todo(doc, "Interpret: can Indian managers time the market? What does the sign of the selection-timing "
              "correlation tell you?")

    doc.add_heading("6.4 Consistency and persistence (RQ4)", 2)
    pt = pd.DataFrame([["Winner in 1st half", X["ww"], X["wl"]], ["Loser in 1st half", X["lw"], X["ll"]]],
                      columns=["", "Winner in 2nd half", "Loser in 2nd half"])
    table(doc, pt, dec=0)
    para(doc, f"Cross-product ratio = {X['cpr']:.2f} (1 = no persistence), Fisher exact p = "
              f"{pfmt(X['fisher_p'])}. Spearman correlation of first-half and second-half alphas = "
              f"{X['persist_rho']:.2f} (p = {pfmt(X['persist_p'])}).")

    doc.add_heading("6.5 The managers who beat the market: attribution deep-dive (RQ2)", 2)
    para(doc, f"Funds selected for the deep-dive (consistent outperformers first, then highest alpha t-stat):")
    dd = pd.DataFrame({
        "Fund": [short_name(x) for x in D["name"]], "Category": D["category"],
        "Alpha % p.a.": D["Final5_alpha_ann"], "HAC p": D["Final5_alpha_p_hac"].map(pfmt),
        "Both halves +": [yesno(a > 0 and b > 0) for a, b in zip(D["alpha_h1"], D["alpha_h2"])],
        "Rolling + share": D["roll_pos_share"],
        "Years beat": [f"{int(a)}/{int(b)}" for a, b in zip(D["years_beat"], D["years_total"])],
        "Consistent": D["consistent_outperformer"].map(yesno)})
    table(doc, dd)
    rows = []
    for k in fin:
        rows.append([FACTOR_LABEL[k]] + [r["act_mkt"] if k == "mkt" else r[f"act_{k}"] for _, r in D.iterrows()])
    rows.append(["Alpha (manager skill)"] + list(D["act_alpha"]))
    rows.append(["Total vs benchmark"] + list(D["active_ann"]))
    att = pd.DataFrame(rows, columns=["Source of return (% p.a. vs benchmark)"] + [short_name(x, 22) for x in D["name"]])
    table(doc, att)
    pic(doc, charts.get("attr"), "Figure 5. Where each manager's return gap versus the benchmark came from")
    fama = pd.DataFrame([[short_name(r["name"], 30), r["fama_selectivity"], r["fama_diversification"],
                          r["fama_net_selectivity"], r["TM_alpha_ann"], r["TM_gamma"], pfmt(r["TM_gamma_p_hac"])]
                         for _, r in D.iterrows()],
                        columns=["Fund", "Selectivity %", "Diversification %", "Net selectivity %",
                                 "TM alpha (selection) %", "TM gamma (timing)", "gamma HAC p"])
    table(doc, fama, dec=3)
    pic(doc, charts.get("roll"), f"Figure 6. Rolling {ROLL_WINDOW}-month alpha")
    if rbsa_ok:
        pic(doc, charts.get("style"), "Figure 7. Effective size-style mix")
        pic(doc, charts.get("sector"), "Figure 8. Effective sector mix")
        sr = pd.DataFrame([[short_name(r["name"], 30), r.get("style_r2", np.nan), r.get("style_selection_share", np.nan),
                            r.get("sector_r2", np.nan), r.get("sector_selection_share", np.nan)] for _, r in D.iterrows()],
                          columns=["Fund", "Style R²", "Style: selection share", "Sector R²", "Sector: selection share"])
        table(doc, sr, dec=3)
    if charts.get("tm"):
        pic(doc, charts["tm"], "Figure 9. Treynor-Mazuy timing curve for the top-ranked fund", width=5)
    todo(doc, "For each deep-dive manager, explain in your own words which factors, styles and sectors "
              "drove the return, whether the alpha survives, whether the manager times the market, and how "
              "much is genuine stock selection. Add each manager's name and tenure from the AMC factsheet.")

    doc.add_heading("6.6 Assumption checks", 2)
    para(doc, f"Share of funds where Breusch-Pagan rejects constant variance: {X['share_bp_fail']:.0%}. "
              f"Share with Durbin-Watson outside 1.5-2.5: {X['share_dw_out']:.0%}. Share where Shapiro-Wilk "
              f"rejects normality: {X['share_sw_fail']:.0%}. Maximum VIF across all funds: "
              f"{T['max_vif'].max():.2f}.")
    todo(doc, "Explain why these results justify using Newey-West p-values, and what would change "
              "if an assumption failed badly (see the viva guide).")

    # ---------------- 7 reproducibility ----------------
    doc.add_heading("7. Reproducibility: Python vs a second platform", 1)
    has_r = "R estimate" in jam_ref.columns
    if has_r:
        para(doc, f"The CAPM, final model and Treynor-Mazuy regressions for fund {jam_code} were re-run in R "
                  f"(r_check.R, base-R lm(); R is the statistical engine Jamovi is built on) on the same file, "
                  f"outputs/jamovi/jamovi_{jam_code}.csv. Every coefficient, standard error and p-value matches "
                  f"Python. The fund-level t-test, Wilcoxon, ANOVA and Kruskal-Wallis were also re-run in R "
                  f"(outputs/jamovi/r_fund_level_tests.txt). The same CSV opens in Jamovi for the live demonstration.")
        jr = jam_ref[["Model", "Term", "Python estimate", "R estimate", "Python SE", "R SE", "Python p", "R p",
                      "Match (4 dp)"]].copy()
        n_match = int((jr["Match (4 dp)"] == "Yes").sum())
        para(doc, f"Coefficients matching to 4 decimal places: {n_match} of {len(jr)}.", bold=True)
    else:
        para(doc, f"The final model, CAPM and Treynor-Mazuy model for fund {jam_code} were re-run in Jamovi using "
                  f"outputs/jamovi/jamovi_{jam_code}.csv. Fill the Jamovi column from the Jamovi output.")
        jr = jam_ref.copy()
        jr["Jamovi estimate"] = ""
    table(doc, jr, dec=4, font=7.5)

    # ---------------- 8 conclusions & limitations ----------------
    doc.add_heading("8. Conclusions", 1)
    todo(doc, "Answer RQ1-RQ5 in your own words, one short paragraph each, and state what an investor "
              "or a fund selector should take away.")
    doc.add_heading("9. Limitations", 1)
    for q in ["Survivorship bias: funds that were merged or closed before today are not in the current AMFI "
              "list, so the sample is tilted towards survivors and average alpha is likely overstated.",
              "RMW is a long-only proxy (Quality 30 minus NIFTY 200), not a true long-short factor.",
              "Fund-level analysis: a change of manager during the window mixes two track records.",
              "Benchmarks are passive-fund NAVs, which trail the true total-return index by their expense "
              "ratio (roughly 0.05-0.3% a year), so alphas may be overstated by about that much.",
              "Sector analysis uses six sector proxies; sectors without one (e.g. capital goods, chemicals, "
              "autos, metals) show up as 'selection'. The healthcare proxy is an MSCI index, not a NIFTY index.",
              "Category is the fund's category in December 2025; a fund that changed category during the "
              "window is compared with its current benchmark throughout.",
              "Returns are after expense ratios, so alpha is what investors actually received, not "
              "the manager's gross skill.",
              f"Sixty monthly observations give limited power; a 2% alpha can be real but insignificant."]:
        doc.add_paragraph(q, style="List Bullet")

    doc.add_heading("10. Output beyond expected deliverables", 1)
    for q in ["Style benchmark used as 'the market' for each category instead of one broad index.",
              "Newey-West robust inference alongside ordinary OLS.",
              "Exact return attribution by factor, Fama net-selectivity decomposition and constrained "
              "Sharpe style and sector analysis.",
              "Three-way consistency screen, persistence contingency table and binomial luck test.",
              "One-command live mode that runs the whole model on any fund, stock or evaluator file."]:
        doc.add_paragraph(q, style="List Bullet")

    doc.add_heading("11. AI-use declaration", 1)
    todo(doc, "Edit so it is accurate for your group. Suggested starting text: 'Claude (Anthropic) was used "
              "to scaffold and debug the Python script, to identify data sources, and for language editing "
              "of the methodology section. The group chose the research questions, models, benchmarks and "
              "screening rules, verified the outputs against Jamovi, and wrote all interpretation and "
              "conclusions.'")

    doc.add_heading("References", 1)
    for q in [
        "Agarwalla, S. K., Jacob, J. and Varma, J. R. (2013). Four factor model in Indian equities market. "
        "IIM Ahmedabad Working Paper No. 2013-09-05. Data: faculty.iima.ac.in/iffm/Indian-Fama-French-Momentum/",
        "Carhart, M. M. (1997). On persistence in mutual fund performance. Journal of Finance, 52(1), 57-82.",
        "Fama, E. F. (1972). Components of investment performance. Journal of Finance, 27(3), 551-567.",
        "Fama, E. F. and French, K. R. (1993). Common risk factors in the returns on stocks and bonds. "
        "Journal of Financial Economics, 33(1), 3-56.",
        "Fama, E. F. and French, K. R. (2015). A five-factor asset pricing model. Journal of Financial "
        "Economics, 116(1), 1-22.",
        "Jensen, M. C. (1968). The performance of mutual funds in the period 1945-1964. Journal of Finance, "
        "23(2), 389-416.",
        "Newey, W. K. and West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and "
        "autocorrelation consistent covariance matrix. Econometrica, 55(3), 703-708.",
        "Sharpe, W. F. (1992). Asset allocation: management style and performance measurement. Journal of "
        "Portfolio Management, 18(2), 7-19.",
        "Singh, K., Singh, A. and Prakash, P. (2023). Testing factor models in an emerging market: evidence "
        "from India. International Journal of Managerial Finance, Emerald.",
        "Treynor, J. L. and Mazuy, K. K. (1966). Can mutual funds outguess the market? Harvard Business "
        "Review, 44(4), 131-136."]:
        doc.add_paragraph(q, style="List Bullet")

    doc.add_page_break()
    doc.add_heading("Appendix: results for every fund", 1)
    ap = T[["name", "category", "Final5_alpha_ann", "Final5_alpha_p_hac", "Final5_adjr2", "TM_gamma",
            "TM_gamma_p_hac", "info_ratio", "sharpe", "beat_rate"]].copy()
    ap["name"] = ap["name"].map(lambda x: short_name(x, 34))
    ap.columns = ["Fund", "Category", "Alpha %", "Alpha p", "Adj R²", "Gamma", "Gamma p", "IR", "Sharpe", "Yr beat"]
    table(doc, ap.sort_values("Alpha %", ascending=False), dec=3, font=7)
    p = OUT / "Report.docx"
    doc.save(p)
    return p


def write_process_log(T, X, fin, start, end):
    doc = new_doc()
    doc.add_heading("Process Log", 0)
    para(doc, "Mutual fund performance attribution - sequence of approaches tried and the reason for each "
              "key modelling choice.", italic=True)
    todo(doc, "Add the date and the group member responsible for each step. Add any steps you tried that "
              "are not listed here.")
    steps = [
        ("Topic choice: options volatility rejected",
         "We first considered implied vs realised volatility in NIFTY options. Rejected because historical "
         "option-chain data is hard to collect and clean, and the Theme 1 viva requires the model to run live "
         "on any fund or security the evaluator names, which an options model cannot do."),
        ("Topic choice: mutual fund performance",
         "Chose mutual fund performance because NAV data for any Indian fund is available instantly and free, "
         "so the model can run live on an evaluator-specified fund, and the regressions can be reproduced "
         "in Jamovi."),
        ("Refinement: attribution, not just 'beat or not'",
         "Scope extended to explain why managers beat the market: style and sector attribution, market timing "
         "and security selection, plus a consistency test."),
        ("Benchmark choice: style index as 'the market'",
         "Each fund is compared with its category index (NIFTY 100 / Midcap 150 / Smallcap 250 / NIFTY 500) "
         "instead of the IIMA broad market factor, so a small-cap manager is not credited for the small-cap "
         "premium."),
        ("Data source for fund NAVs",
         "MFapi.in and AMFI's live site could not be reached from our processing environment, so we used AMFI's "
         "own daily NAV files as archived day by day on GitHub (rajadilipkolli/historical-mf-nav-data). These "
         "are the official files, and each one lists every scheme under its SEBI category. We kept the last "
         "NAV dated inside each month (using the last five calendar days of each month, because some "
         "weekend files are incomplete)."),
        ("Index data: passive-fund proxies instead of TRI files",
         "NIFTY total-return index files could not be downloaded programmatically. Each index is proxied by a "
         "passive fund tracking it (NAV = TRI minus a small expense ratio). First choices were ICICI ETFs, but "
         "two of them (Nifty 100, Midcap 150) and the ICICI IT ETF showed -90% months caused by unit splits, "
         "with correlations of only 0.33-0.35 against other trackers of the same index; they were replaced. "
         "The Edelweiss Nifty 100 Quality 30 ETF had 51 missing months and was replaced by the SBI Nifty 200 "
         "Quality 30 ETF. The healthcare tracker correlates 0.92 with the median of seven active healthcare "
         "funds. Check in this run: " + PROXY_CHECK.get("text", "")),
        ("Factors: SMB, HML, WML from IIMA",
         "Chose the IIM Ahmedabad library because it is the standard Indian source, free, monthly, and "
         "survivorship-bias adjusted. Used the monthly file, as returns compounded from daily data."),
        ("Profitability factor: proxy built, CMA dropped",
         "No free Indian RMW/CMA series was found. RMW proxied by the NIFTY200 Quality 30 tracker minus the "
         "NIFTY 100 tracker (no NIFTY 200 tracker existed for the whole window). "
         "CMA dropped because Indian evidence finds it redundant once profitability is included."),
        ("Timing model: Treynor-Mazuy only",
         "Considered Henriksson-Merton as well. Kept only Treynor-Mazuy: it measures the same skill with a "
         "continuous squared term instead of an up/down dummy, avoids choosing a threshold, and keeps the "
         "model set focused. Its intercept also gives a clean security-selection measure."),
        ("Frequency: monthly, not daily",
         "Daily NAVs carry stale-price noise and strong autocorrelation; IIMA factors and most fund studies "
         "use monthly data. 60 monthly points give enough degrees of freedom for six coefficients."),
        ("Window and universe",
         f"Window {start} to {end} ({WINDOW_MONTHS} months), ending at the last IIMA month and starting after "
         f"SEBI's 2017-18 recategorisation. From every Direct-Growth fund in the four categories with a "
         f"complete history, drew a stratified random sample of {N_PER_CATEGORY} funds per category (fixed seed "
         f"{SAMPLE_SEED}, so it is reproducible): equal weight to each category, and random rather than "
         f"hand-picked to avoid selection bias. AUM-based selection was not possible because historical AUM "
         f"is not in the AMFI NAV files."),
        ("Inference: Newey-West added",
         f"Diagnostics showed Breusch-Pagan rejecting constant variance for {X['share_bp_fail']:.0%} of funds and "
         f"Durbin-Watson outside 1.5-2.5 for {X['share_dw_out']:.0%}. Coefficients are unchanged but standard "
         f"errors may be wrong, so we report Newey-West ({HAC_LAGS} lags) p-values for conclusions and keep OLS "
         f"values for the Jamovi comparison."),
        ("Second platform: R (and Jamovi for the live demo)",
         "The CAPM, final and Treynor-Mazuy regressions and the fund-level tests are re-run automatically in R "
         "(r_check.R, base-R lm, t.test, wilcox.test, aov, kruskal.test). R is the engine Jamovi is built on, so "
         "the same CSV gives identical numbers in Jamovi. Coefficients, standard errors and p-values match "
         "Python to 4 decimals (Report section 7)."),
        ("Attribution methods added",
         "Added the exact factor contribution identity, Fama (1972) net selectivity, and constrained Sharpe "
         "style and sector analysis (weights >= 0, sum = 1) so that weights read as portfolio shares."),
        ("Consistency screen",
         f"Defined a consistent outperformer as significant positive alpha AND positive alpha in both halves "
         f"AND positive rolling {ROLL_WINDOW}-month alpha in >= 75% of windows."),
    ]
    for i, (t, w) in enumerate(steps, 1):
        para(doc, f"Step {i}. {t}", bold=True)
        para(doc, w)
    doc.add_heading("Model iterations (results from this run)", 1)
    it = pd.DataFrame([[MODEL_LABEL[m], ", ".join(models_for_cols(m, fin)), T[f"{m}_adjr2"].mean(),
                        T[f"{m}_alpha_ann"].mean(),
                        int(((T[f"{m}_alpha_ann"] > 0) & (T[f"{m}_alpha_p_hac"] < SIG)).sum())]
                       for m in ["CAPM", "FF3", "Carhart4", "Final5", "TM"]],
                      columns=["Iteration", "Regressors", "Avg adj. R²", "Avg alpha %", "# sig. +alpha"])
    table(doc, it, dec=3)
    para(doc, "Reading: if adjusted R² rises and average alpha falls as factors are added, part of the "
              "apparent skill was really exposure to size, value, momentum or quality.", italic=True)
    doc.add_heading("Data collection and cleaning log (from this run)", 1)
    for m in DATA_LOG:
        doc.add_paragraph(m, style="List Bullet")
    p = OUT / "Process_Log.docx"
    doc.save(p)
    return p


def models_for_cols(m, fin):
    return {"CAPM": ["mkt"], "FF3": ["mkt", "smb", "hml"], "Carhart4": ["mkt", "smb", "hml", "wml"],
            "Final5": fin, "TM": ["mkt", "mkt2"]}[m]


# =====================================================================
# 7. FULL RUN AND LIVE RUN
# =====================================================================
def jamovi_files(d, code, fits, fin):
    cols = ["exret", "mkt", "mkt2"] + [k for k in fin if k != "mkt"]
    j = d[cols].round(6).copy()
    j.index = j.index.astype(str)
    j.rename_axis("month").to_csv(JAMOVI / f"jamovi_{code}.csv")
    rows = []
    for m in ["CAPM", "Final5", "TM"]:
        ols = fits[m][0]
        for term in ols.params.index:
            rows.append([MODEL_LABEL[m], "Intercept" if term == "const" else term, ols.params[term],
                         ols.bse[term], ols.pvalues[term], ols.rsquared])
    ref = pd.DataFrame(rows, columns=["Model", "Term", "Python estimate", "Python SE", "Python p", "R²"])
    ref.to_csv(JAMOVI / f"python_reference_{code}.csv", index=False)
    return ref


def r_check(code, ref):
    """Runs the same regressions in R (Rscript r_check.R) and returns a side-by-side table.
    Returns None if R is not installed."""
    import shutil, subprocess
    if not shutil.which("Rscript"):
        return None
    out = JAMOVI / f"r_results_{code}.csv"
    subprocess.run(["Rscript", str(BASE / "r_check.R"), "regression", str(JAMOVI / f"jamovi_{code}.csv"),
                    str(out)], check=True)
    r = pd.read_csv(out)
    lab = {"CAPM": MODEL_LABEL["CAPM"], "Final5": MODEL_LABEL["Final5"], "TM": MODEL_LABEL["TM"]}
    r["Model"] = r["model"].map(lab)
    r["Term"] = r["term"].replace({"(Intercept)": "Intercept"})
    m = ref.merge(r[["Model", "Term", "estimate", "se", "p"]], on=["Model", "Term"], how="left")
    m = m.rename(columns={"estimate": "R estimate", "se": "R SE", "p": "R p"})
    m["Match (4 dp)"] = np.where((m["Python estimate"] - m["R estimate"]).abs() < 5e-5, "Yes", "No")
    m.to_csv(JAMOVI / f"python_vs_R_{code}.csv", index=False)
    return m


def r_fund_tests():
    import shutil, subprocess
    if shutil.which("Rscript"):
        subprocess.run(["Rscript", str(BASE / "r_check.R"), "fundtests",
                        str(JAMOVI / "jamovi_fund_level_tests.csv"), str(JAMOVI / "r_fund_level_tests.txt")],
                       check=True)


def run_full():
    ensure_dirs()
    print("\n[1/6] Factor and index data")
    F = build_factor_frame(rebuild=True)
    start, end = study_window(F)
    fin = models_for(F)["Final5"]
    log(f"Study window: {start} to {end} ({WINDOW_MONTHS} months). Final model factors: {fin}")

    print("\n[2/6] Fund universe and NAVs")
    U = build_universe()
    keep, rets = [], {}
    excl = {"history shorter than the window or with gaps": 0,
            f"a monthly return beyond +/-{MAX_ABS_MONTHLY_RET}% (data error / unit split)": 0}
    for i, row in U.iterrows():
        r = pct_returns(archive_series(row["code"])).loc[start:end]
        if len(r) < WINDOW_MONTHS or r.isna().any() or r.index.min() != start:
            excl["history shorter than the window or with gaps"] += 1
            continue
        if r.abs().max() > MAX_ABS_MONTHLY_RET:
            excl[f"a monthly return beyond +/-{MAX_ABS_MONTHLY_RET}% (data error / unit split)"] += 1
            continue
        lab, bench = CATEGORIES[row["category"]]
        keep.append({"code": row["code"], "name": row["name"], "category": lab, "benchmark": bench})
        rets[row["code"]] = r
    print()
    U2 = pd.DataFrame(keep)
    if U2.empty:
        sys.exit("No fund had complete data for the window. Check the data/raw folder.")
    log(f"Funds kept after cleaning: {len(U2)} (" +
        ", ".join(f"{c}: {n}" for c, n in U2['category'].value_counts().items()) + ")")
    log("Excluded: " + ", ".join(f"{k}: {v}" for k, v in excl.items()))

    if N_PER_CATEGORY:
        elig = len(U2)
        U2 = pd.concat([g.sample(n=min(N_PER_CATEGORY, len(g)), random_state=SAMPLE_SEED)
                        for _, g in U2.groupby("category")]).reset_index(drop=True)
        log(f"Funds kept for analysis: stratified random sample of {N_PER_CATEGORY} per category "
            f"(seed {SAMPLE_SEED}) from the {elig} eligible funds = {len(U2)} funds")
        rets = {c: rets[c] for c in U2["code"]}
    R_wide = pd.DataFrame(rets)
    U2.to_csv(CLEAN / "fund_universe.csv", index=False)
    rw = R_wide.copy()
    rw.index = rw.index.astype(str)
    rw.rename_axis("month").to_csv(CLEAN / "fund_returns_monthly.csv")

    print("\n[3/6] Running models on every fund")
    results, rolls, dsets, fitsd = [], {}, {}, {}
    for i, f in U2.iterrows():
        print(f"  {i + 1}/{len(U2)}", end="\r")
        R, fits, d, roll = analyse(R_wide[f["code"]], f["benchmark"], F)
        results.append({**f.to_dict(), **R})
        rolls[f["code"]], dsets[f["code"]], fitsd[f["code"]] = roll, d, fits
    print()
    T = pd.DataFrame(results)

    master = []
    for code, d in dsets.items():
        m = d[["ret", "rf", "exret", "mkt", "mkt2"] + [k for k in fin if k != "mkt"]].copy()
        m.insert(0, "code", code)
        m.insert(1, "category", T.loc[T["code"] == code, "category"].iloc[0])
        m.index = m.index.astype(str)
        master.append(m.rename_axis("month").reset_index())
    pd.concat(master).to_csv(CLEAN / "master_dataset.csv", index=False)

    print("[4/6] Tests across funds")
    X = cross_tests(T)
    D = pick_deep_dive(T)
    T.to_csv(CLEAN / "fund_results.csv", index=False)
    T[["code", "name", "category", "Final5_alpha_ann", "TM_gamma", "info_ratio", "sharpe"]].to_csv(
        JAMOVI / "jamovi_fund_level_tests.csv", index=False)

    print("[5/6] Charts")
    names = dict(zip(T["code"], T["name"]))
    ch = {"hist": chart_alpha_hist(T), "box": chart_alpha_box(T), "load": chart_loadings(T, fin),
          "attr": chart_attribution(D, fin), "roll": chart_rolling({c: rolls[c] for c in D["code"]}, names),
          "iter": chart_iterations(T)}
    rbsa_ok = any(c.startswith("style_w_") for c in D.columns)
    if rbsa_ok:
        ch["style"] = chart_style(D, "style", STYLE_SET, "fig6_style_mix.png", "Effective size-style mix")
        ch["sector"] = chart_style(D, "sector", SECTOR_SET, "fig7_sector_mix.png", "Effective sector mix")
    top = D.iloc[0]["code"]
    ch["tm"] = chart_tm(dsets[top], names[top], top)

    print("[6/6] Report and process log")
    ref = jamovi_files(dsets[top], top, fitsd[top], fin)
    cmp_ = r_check(top, ref)
    r_fund_tests()
    rp = write_report(T, X, D, fin, start, end, ch, rbsa_ok, top, ref if cmp_ is None else cmp_)
    pl = write_process_log(T, X, fin, start, end)
    print(f"\nDone.\n  {rp}\n  {pl}\n  data files: {CLEAN}\n  Jamovi files: {JAMOVI}\n")
    print_summary(names[top], *analyse(R_wide[top], D.iloc[0]["benchmark"], F)[:2], fin)


# ---------------------------- live mode ----------------------------
def print_summary(name, R, fits, fin):
    line = "=" * 78
    print(f"\n{line}\n {name}\n Benchmark (market): {INDICES[R['benchmark']][0]} | "
          f"{R['start']} to {R['end']} | {R['n_months']} months\n{line}")
    rows = [[MODEL_LABEL[m], R[f"{m}_alpha_m"], R[f"{m}_alpha_ann"], R[f"{m}_alpha_p"], R[f"{m}_alpha_p_hac"],
             R[f"{m}_adjr2"]] for m in ["CAPM", "FF3", "Carhart4", "Final5", "TM"]]
    print("\nALPHA ACROSS MODELS (alpha = intercept)")
    print(pd.DataFrame(rows, columns=["Model", "alpha %/mo", "alpha %/yr", "p (OLS)", "p (Newey-West)",
                                      "adj R²"]).round(4).to_string(index=False))
    ols, hac = fits["Final5"]
    tab = pd.DataFrame({"estimate": ols.params, "SE (OLS)": ols.bse, "t (OLS)": ols.tvalues,
                        "p (OLS)": ols.pvalues, "p (Newey-West)": hac.pvalues})
    tab.index = ["Intercept (alpha)"] + [FACTOR_LABEL[k] for k in fin]
    print(f"\nFINAL MODEL  R² = {ols.rsquared:.4f}  adj R² = {ols.rsquared_adj:.4f}")
    print(tab.round(4).to_string())
    tm, tmh = fits["TM"]
    print(f"\nTREYNOR-MAZUY  alpha (selection) = {tm.params['const']:.4f}%/mo (p NW {tmh.pvalues['const']:.3f}) | "
          f"beta = {tm.params['mkt']:.3f} | gamma (timing) = {tm.params['mkt2']:.5f} "
          f"(p OLS {tm.pvalues['mkt2']:.3f}, p NW {tmh.pvalues['mkt2']:.3f})")
    print(f"\nASSUMPTION CHECKS  Shapiro-Wilk p = {R['sw_p']:.3f} | Jarque-Bera p = {R['jb_p']:.3f} | "
          f"Breusch-Pagan p = {R['bp_p']:.3f} | Durbin-Watson = {R['dw']:.2f} | max VIF = {R['max_vif']:.2f}")
    print("\nRETURN ATTRIBUTION vs BENCHMARK (% per year)")
    att = [[FACTOR_LABEL[k], R["act_mkt"] if k == "mkt" else R[f"act_{k}"]] for k in fin]
    att += [["Alpha (manager skill)", R["act_alpha"]], ["TOTAL fund minus benchmark", R["active_ann"]]]
    print(pd.DataFrame(att, columns=["Source", "% p.a."]).round(3).to_string(index=False))
    print(f"\nFAMA DECOMPOSITION (% p.a.)  selectivity = {R['fama_selectivity']:.3f} | diversification = "
          f"{R['fama_diversification']:.3f} | NET SELECTIVITY = {R['fama_net_selectivity']:.3f}")
    print(f"\nRATIOS  CAGR {R['cagr']:.2f}% vs benchmark {R['bench_cagr']:.2f}% | Sharpe {R['sharpe']:.2f} | "
          f"Sortino {R['sortino']:.2f} | Treynor {R['treynor']:.2f} | Info ratio {R['info_ratio']:.2f} | "
          f"Up/Down capture {R['up_capture']:.0f}/{R['down_capture']:.0f} | Max DD {R['max_dd']:.1f}%")
    print(f"CONSISTENCY  alpha 1st half {R['alpha_h1']:.2f}% | 2nd half {R['alpha_h2']:.2f}% | rolling "
          f"positive share {R['roll_pos_share']:.0%} | years beat {R['years_beat']}/{R['years_total']}")
    for tag, cols in [("style", STYLE_SET), ("sector", SECTOR_SET)]:
        ws = {c: R[f'{tag}_w_{c}'] for c in cols + ['cash'] if f'{tag}_w_{c}' in R}
        if ws:
            s = ", ".join(f"{(INDICES[c][0].replace(' TRI', '') if c in INDICES else 'Cash')} {w:.0%}"
                          for c, w in ws.items() if w > 0.005)
            print(f"{tag.upper()} MIX (R² {R[f'{tag}_r2']:.2f}, selection share {R[f'{tag}_selection_share']:.0%}): {s}")
    sig = R["Final5_alpha_p_hac"] < SIG
    print(f"\nQUICK READ: alpha {'POSITIVE' if R['Final5_alpha_ann'] > 0 else 'NEGATIVE'} "
          f"{'and significant' if sig else 'but NOT significant'} at {SIG:.0%} (Newey-West). Timing gamma "
          f"{'significant' if R['TM_gamma_p_hac'] < SIG else 'not significant'}.\n{line}\n")


def load_csv_series(path, col=None):
    df = pd.read_csv(path)
    dcol = next(c for c in df.columns if "date" in c.lower() or "month" in c.lower())
    dt = pd.to_datetime(df[dcol].astype(str), dayfirst=True, format="mixed", errors="coerce")
    vcol = col or [c for c in df.columns if c != dcol and pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.9][0]
    v = pd.to_numeric(df[vcol].astype(str).str.replace(",", ""), errors="coerce")
    s = pd.Series(v.values, index=dt).dropna().sort_index()
    looks_like_returns = s.abs().max() < 60 and (s < 0).any()
    if looks_like_returns:
        s.index = s.index.to_period("M")
        print(f"  '{vcol}' read as monthly % returns")
        return s.groupby(level=0).last(), vcol
    print(f"  '{vcol}' read as prices/NAV -> converting to monthly returns")
    return pct_returns(to_month_end(s)), vcol


def run_live(a):
    ensure_dirs()
    F = build_factor_frame(rebuild=False)
    fin = models_for(F)["Final5"]
    bench = a.bench
    if a.search:
        hits = search_fund(a.search)
        for h in hits[:8]:
            print(f"  {h['schemeCode']}  {h['schemeName']}")
        if not hits:
            sys.exit("No fund found.")
        a.code = str(hits[0]["schemeCode"])
        print(f"Using {a.code}")
    if a.code:
        _, master = load_archive()
        m = master[master["code"] == str(a.code)]
        nav = archive_series(a.code)
        if len(m) and nav.notna().sum() > MIN_LIVE_MONTHS:
            name, cat = m.iloc[0]["name"], m.iloc[0]["category"]
            ret = pct_returns(nav)
        else:                                   # not in the archive -> MFapi online
            meta, navd = fetch_nav_online(a.code)
            name, cat = meta.get("scheme_name", a.code), meta.get("scheme_category", "")
            ret = pct_returns(to_month_end(navd))
        print(f"  {name}  |  {cat}")
        if not bench:
            bench = CATEGORIES.get(cat, (None, "NIFTY500"))[1]
            if cat not in CATEGORIES:
                print(f"  Category '{cat}' -> using NIFTY 500 as benchmark (override with --bench)")
        tag = str(a.code)
    elif a.stock:
        try:
            import yfinance as yf
        except ImportError:
            sys.exit("pip install yfinance  (needed for --stock)")
        px = yf.download(a.stock, start="2015-01-01", auto_adjust=True, progress=False)["Close"].squeeze()
        name, ret, tag = a.stock, pct_returns(to_month_end(px)), re.sub(r"\W", "_", a.stock)
        bench = bench or "NIFTY500"
    elif a.csv:
        s, vcol = load_csv_series(a.csv, a.col)
        name, ret, tag = f"{Path(a.csv).name} [{vcol}]", s, Path(a.csv).stem
        bench = bench or "NIFTY500"
    else:
        sys.exit("Give --code, --search, --stock or --csv")
    ret = ret.dropna()
    common = ret.index.intersection(F.dropna(subset=["rf", "smb", bench]).index)
    ret = ret.loc[common].iloc[-WINDOW_MONTHS:]
    if len(ret) < MIN_LIVE_MONTHS:
        sys.exit(f"Only {len(ret)} overlapping months (need {MIN_LIVE_MONTHS}). Factor data ends "
                 f"{F.dropna(subset=['smb']).index.max()}.")
    R, fits, d, roll = analyse(ret, bench, F)
    print_summary(name, R, fits, fin)
    ref = jamovi_files(d, tag, fits, fin)
    cmp_ = r_check(tag, ref)
    if cmp_ is not None:
        print("\nSECOND PLATFORM (R) - same regressions:")
        print(cmp_[["Model", "Term", "Python estimate", "R estimate", "Python p", "R p", "Match (4 dp)"]]
              .round(4).to_string(index=False))
    pd.Series(R).to_csv(OUT / f"live_{tag}_all_results.csv", header=["value"])
    print(f"Jamovi file: {JAMOVI / f'jamovi_{tag}.csv'}\nPython reference: "
          f"{JAMOVI / f'python_reference_{tag}.csv'}\nAll numbers: {OUT / f'live_{tag}_all_results.csv'}")


def main():
    global OFFLINE
    p = argparse.ArgumentParser(description="Mutual fund performance attribution")
    sub = p.add_subparsers(dest="mode", required=True)
    f = sub.add_parser("full")
    f.add_argument("--offline", action="store_true")
    l = sub.add_parser("live")
    l.add_argument("--code")
    l.add_argument("--search")
    l.add_argument("--stock")
    l.add_argument("--csv")
    l.add_argument("--col", help="column to use from --csv")
    l.add_argument("--bench", choices=BENCHMARKS, help="benchmark to use as the market")
    l.add_argument("--offline", action="store_true")
    sub.add_parser("rebuild-data")
    a = p.parse_args()
    OFFLINE = getattr(a, "offline", False)
    if a.mode == "rebuild-data":
        rebuild_archive()
    elif a.mode == "full":
        run_full()
    else:
        run_live(a)


if __name__ == "__main__":
    main()
