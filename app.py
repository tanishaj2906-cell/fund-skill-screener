"""
Fund Skill Screener - user interface for the BA project
Run:  python3 -m streamlit run app.py      (or double-click 3_Run_App.command)

What it does
  1. Rates every eligible Large / Mid / Small / Flexi Cap Direct-Growth fund using the
     same models as the report (5-factor alpha with Newey-West p-values, consistency
     tests, Treynor-Mazuy timing, down-capture) plus its most recent 12-month return.
  2. Gives an investor a shortlist based on risk appetite and horizon.
  3. "Update data live" downloads the latest NAVs (MFapi.in, mirrors AMFI) and the
     latest IIM Ahmedabad factor release, then re-rates everything.
  4. Lets the user upload their own NAV / price file, or a newer factor file.

Educational tool built on past data. Not investment advice.
"""
import io
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import mf_project as mp

LIVE = mp.BASE / "data" / "live"
LIVE_NAV = LIVE / "nav"
RISK_CATS = {
    "Conservative": ["Large Cap", "Flexi Cap"],
    "Moderate": ["Large Cap", "Flexi Cap", "Mid Cap"],
    "Aggressive": ["Large Cap", "Flexi Cap", "Mid Cap", "Small Cap"],
}
RATING_ORDER = ["Strong", "Good", "Average", "Weak"]
RATING_ICON = {"Strong": "🟢", "Good": "🟡", "Average": "⚪", "Weak": "🔴"}

st.set_page_config(page_title="Fund Skill Screener", page_icon="📊", layout="wide")


# =====================================================================
# DATA LAYER: bundled AMFI archive (offline) or live MFapi + IIMA
# =====================================================================
def live_nav_monthly(code):
    """Month-end NAVs from the live download cache (complete months only)."""
    f = LIVE_NAV / f"{code}.json"
    if not f.exists():
        return pd.Series(dtype=float)
    js = json.loads(f.read_text())
    d = pd.DataFrame(js.get("data", []))
    if d.empty:
        return pd.Series(dtype=float)
    d["date"] = pd.to_datetime(d["date"], format="%d-%m-%Y", errors="coerce")
    d["nav"] = pd.to_numeric(d["nav"], errors="coerce")
    s = d.dropna().set_index("date")["nav"].sort_index()
    m = mp.to_month_end(s)
    return m[m.index < pd.Period(pd.Timestamp.today(), "M")]       # drop the running month


def nav_monthly(code, source):
    if source == "live":
        s = live_nav_monthly(code)
        if s.notna().sum() > 12:
            return s
    return mp.archive_series(code)


def latest_nav(code, source):
    """Latest NAV and its date (for display)."""
    f = LIVE_NAV / f"{code}.json"
    if source == "live" and f.exists():
        d = json.loads(f.read_text()).get("data", [])
        if d:
            return float(d[0]["nav"]), d[0]["date"]
    s = mp.archive_series(code).dropna()
    return (float(s.iloc[-1]), str(s.index[-1])) if len(s) else (np.nan, "")


def load_factors(source, uploaded=None):
    if uploaded is not None:
        df = uploaded.copy()
    else:
        f = LIVE / "iima_factors_monthly.csv"
        f = f if (source == "live" and f.exists()) else mp.RAW / "iima_factors_monthly.csv"
        df = pd.read_csv(f, na_values=["NA", ""])
    df.columns = [c.strip().lower() for c in df.columns]
    df["month"] = pd.to_datetime(df["date"].astype(str).str[:7] + "-01").dt.to_period("M")
    return df.set_index("month")[["smb", "hml", "wml", "mf", "rf"]].astype(float)


@st.cache_data(show_spinner=False)
def build_frame(source, factor_bytes, stamp):
    up = pd.read_csv(io.BytesIO(factor_bytes), na_values=["NA", ""]) if factor_bytes else None
    F = load_factors(source, up)
    for short, (_, code, _) in mp.INDICES.items():
        F[short] = mp.pct_returns(nav_monthly(code, source))
    F["rmw"] = F["QUAL30"] - F["NIFTY100"]
    F["cash"] = F["rf"]
    return F.sort_index()


def index_returns(source):
    """Benchmark returns beyond the factor file (for the recent-performance check)."""
    return pd.DataFrame({b: mp.pct_returns(nav_monthly(mp.INDICES[b][1], source)) for b in mp.BENCHMARKS})


def update_live(progress):
    """Download latest NAVs for every candidate fund and tracker, and the latest IIMA file."""
    LIVE_NAV.mkdir(parents=True, exist_ok=True)
    mp.OFFLINE = False
    msgs = []
    try:
        html = mp.http_get(mp.IIMA_PAGE).text
        rel = max(re.findall(r"DATA/(\d{4}-\d{2})_FourFactors_and_Market_Returns_Monthly_"
                             r"SurvivorshipBiasAdjusted\.csv", html))
        txt = mp.http_get(f"{mp.IIMA_PAGE}DATA/{rel}_FourFactors_and_Market_Returns_Monthly_"
                          f"SurvivorshipBiasAdjusted.csv").text
        (LIVE / "iima_factors_monthly.csv").write_text(txt)
        msgs.append(f"IIMA factors: release {rel}")
    except Exception as e:
        msgs.append(f"IIMA factors: download failed ({e}); using bundled file")
    codes = list(candidates()["code"]) + [v[1] for v in mp.INDICES.values()]
    ok = 0
    for i, c in enumerate(codes):
        progress.progress((i + 1) / len(codes), text=f"Downloading NAVs {i + 1}/{len(codes)}")
        try:
            js = mp.http_get(f"https://api.mfapi.in/mf/{c}").json()
            if js.get("data"):
                (LIVE_NAV / f"{c}.json").write_text(json.dumps(js))
                ok += 1
        except Exception:
            pass
        time.sleep(0.15)
    msgs.append(f"NAVs downloaded for {ok} of {len(codes)} schemes")
    (LIVE / "last_update.txt").write_text(pd.Timestamp.now().strftime("%d %b %Y %H:%M"))
    return msgs


@st.cache_data(show_spinner=False)
def candidates():
    _, master = mp.load_archive()
    df = master[master["category"].isin(mp.CATEGORIES) & master["name"].map(mp.is_direct_growth)]
    return df.drop_duplicates("code").reset_index(drop=True)


# =====================================================================
# RATING: same models as the report, turned into a transparent score
# =====================================================================
def rate(R, recent_excess):
    pts, why = 0, []
    a, p = R["Final5_alpha_ann"], R["Final5_alpha_p_hac"]
    if a > 0 and p < mp.SIG:
        pts += 2; why.append(f"✅ Skill: alpha {a:.1f}% a year after all 5 factors, statistically significant")
    elif a > 0:
        pts += 1; why.append(f"➖ Positive alpha ({a:.1f}% a year) but could be luck (p = {p:.2f})")
    elif p < mp.SIG:
        pts -= 2; why.append(f"❌ Significantly negative alpha ({a:.1f}% a year)")
    else:
        why.append(f"❌ No evidence of skill (alpha {a:.1f}% a year)")
    if R["alpha_h1"] > 0 and R["alpha_h2"] > 0:
        pts += 1; why.append("✅ Alpha positive in both halves of the period")
    else:
        why.append("➖ Alpha not positive in both halves")
    if R["roll_pos_share"] >= 0.75:
        pts += 1; why.append(f"✅ Positive in {R['roll_pos_share']:.0%} of rolling 3-year windows")
    if R["beat_rate"] >= 0.8:
        pts += 1; why.append(f"✅ Beat its benchmark in {R['years_beat']} of {R['years_total']} calendar years")
    if R["down_capture"] < 90:
        pts += 1; why.append(f"✅ Falls less than the market (down-capture {R['down_capture']:.0f}%)")
    if pd.notna(recent_excess):
        if recent_excess > 0:
            pts += 1; why.append(f"✅ Ahead of its benchmark over the last 12 months (+{recent_excess:.1f}%)")
        elif recent_excess < -5:
            pts -= 1; why.append(f"⚠️ Behind its benchmark over the last 12 months ({recent_excess:.1f}%)")
        else:
            why.append(f"➖ Last 12 months vs benchmark: {recent_excess:.1f}%")
    if R["TM_gamma_p_hac"] < mp.SIG:
        why.append(f"ℹ️ Market timing {'positive' if R['TM_gamma'] > 0 else 'negative'} and significant")
    label = "Strong" if pts >= 6 else "Good" if pts >= 4 else "Average" if pts >= 2 else "Weak"
    return pts, label, why


def recent_12m(ret_full, bench_full):
    j = pd.concat([ret_full, bench_full], axis=1).dropna().iloc[-12:]
    if len(j) < 12:
        return np.nan, np.nan, ""
    f = ((1 + j.iloc[:, 0] / 100).prod() - 1) * 100
    b = ((1 + j.iloc[:, 1] / 100).prod() - 1) * 100
    return f, f - b, f"{j.index[0]} to {j.index[-1]}"


@st.cache_data(show_spinner=False)
def screen_all(source, factor_bytes, stamp):
    F = build_frame(source, factor_bytes, stamp)
    start, end = mp.study_window(F)
    IR = index_returns(source)
    rows, details = [], {}
    for _, c in candidates().iterrows():
        full = mp.pct_returns(nav_monthly(c["code"], source))
        r = full.loc[start:end]
        if len(r) < mp.WINDOW_MONTHS or r.isna().any() or r.abs().max() > mp.MAX_ABS_MONTHLY_RET:
            continue
        lab, bench = mp.CATEGORIES[c["category"]]
        R, fits, d, roll = mp.analyse(r, bench, F)
        f12, x12, per = recent_12m(full, IR[bench])
        pts, label, why = rate(R, x12)
        rows.append({"code": c["code"], "Fund": re.sub(r"\s*-\s*Direct.*$", "", c["name"], flags=re.I),
                     "Category": lab, "Rating": label, "Score (0-7)": pts,
                     "Alpha % p.a.": R["Final5_alpha_ann"], "Alpha p (NW)": R["Final5_alpha_p_hac"],
                     "Return % p.a.": R["cagr"], "Benchmark % p.a.": R["bench_cagr"],
                     "Last 12m %": f12, "Last 12m vs bench %": x12, "Sharpe": R["sharpe"],
                     "Down-capture %": R["down_capture"], "Max drawdown %": R["max_dd"],
                     "Years beat": f"{R['years_beat']}/{R['years_total']}", "Benchmark": mp.INDICES[bench][0]})
        details[c["code"]] = (R, why, roll, per)
    T = pd.DataFrame(rows).sort_values(["Score (0-7)", "Alpha % p.a."], ascending=False).reset_index(drop=True)
    return T, details, str(start), str(end)


def show_fund(R, why, roll, name, per=""):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Alpha (5-factor)", f"{R['Final5_alpha_ann']:.2f}% p.a.", f"p = {R['Final5_alpha_p_hac']:.3f}",
              delta_color="off")
    c2.metric("Return vs benchmark", f"{R['cagr']:.1f}% p.a.", f"{R['cagr'] - R['bench_cagr']:+.1f}% vs index")
    c3.metric("Sharpe ratio", f"{R['sharpe']:.2f}")
    c4.metric("Down-capture", f"{R['down_capture']:.0f}%", help="Below 100% = falls less than the market")
    st.markdown("**Why this rating**")
    st.markdown("\n".join(f"- {w}" for w in why))
    if per:
        st.caption(f"Last-12-month check uses {per}.")
    a, b = st.columns(2)
    fin = [k for k in ["mkt", "smb", "hml", "wml", "rmw"] if f"b_{k}" in R]
    att = pd.DataFrame({"% per year vs benchmark":
                        [R["act_mkt"]] + [R[f"act_{k}"] for k in fin[1:]] + [R["act_alpha"]]},
                       index=[mp.FACTOR_LABEL[k] for k in fin] + ["Alpha (manager skill)"])
    a.markdown("**Where the return gap came from**")
    a.bar_chart(att, horizontal=True)
    if len(roll):
        b.markdown("**Rolling 3-year alpha (% p.a.)**")
        rs = roll.copy()
        rs.index = rs.index.to_timestamp()
        b.line_chart(rs.rename("alpha"))
    with st.expander("All model results"):
        tab = pd.DataFrame({
            "Model": [mp.MODEL_LABEL[m] for m in ["CAPM", "FF3", "Carhart4", "Final5", "TM"]],
            "Alpha % p.a.": [R[f"{m}_alpha_ann"] for m in ["CAPM", "FF3", "Carhart4", "Final5", "TM"]],
            "p (Newey-West)": [R[f"{m}_alpha_p_hac"] for m in ["CAPM", "FF3", "Carhart4", "Final5", "TM"]],
            "Adj R²": [R[f"{m}_adjr2"] for m in ["CAPM", "FF3", "Carhart4", "Final5", "TM"]]})
        st.dataframe(tab.round(3), hide_index=True, width="stretch")
        betas = pd.DataFrame({"Beta": [R[f"b_{k}"] for k in fin], "p (NW)": [R[f"p_{k}"] for k in fin]},
                             index=[mp.FACTOR_LABEL[k] for k in fin])
        st.dataframe(betas.round(3), width="stretch")
        st.write(f"Treynor-Mazuy timing gamma = {R['TM_gamma']:.4f} (p = {R['TM_gamma_p_hac']:.3f}). "
                 f"Fama net selectivity = {R['fama_net_selectivity']:.2f}% p.a.")
        sty = {k.replace("style_w_", ""): v for k, v in R.items() if k.startswith("style_w_") and v > 0.01}
        sec = {k.replace("sector_w_", ""): v for k, v in R.items() if k.startswith("sector_w_") and v > 0.01}
        lab = lambda c: mp.INDICES[c][0] if c in mp.INDICES else "Cash"
        if sty:
            st.write("Effective style mix: " + ", ".join(f"{lab(k)} {v:.0%}" for k, v in sty.items()))
        if sec:
            st.write("Effective sector mix: " + ", ".join(f"{lab(k)} {v:.0%}" for k, v in sec.items()))


# =====================================================================
# PAGE
# =====================================================================
st.title("Fund Skill Screener")
st.write("Which Indian equity mutual funds have shown real manager skill, after adjusting for the risks "
         "they take, and are still delivering now?")

with st.sidebar:
    st.header("Data")
    has_live = (LIVE / "last_update.txt").exists()
    source = st.radio("Use", ["Bundled data (to Dec 2025)", "Latest downloaded data"],
                      index=1 if has_live else 0, disabled=not has_live,
                      help="Download the latest data first to enable the second option.")
    source = "live" if source.startswith("Latest") else "archive"
    if has_live:
        st.caption(f"Last live update: {(LIVE / 'last_update.txt').read_text()}")
    if st.button("🔄 Update data live", help="Downloads the latest NAVs and factor file (about 1 minute)"):
        bar = st.progress(0.0)
        for m in update_live(bar):
            st.write(m)
        st.cache_data.clear()
        st.rerun()
    st.header("Investor profile")
    risk = st.selectbox("Risk appetite", list(RISK_CATS), index=1)
    horizon = st.selectbox("Investment horizon", ["Under 3 years", "3-5 years", "5+ years"], index=2)
    n_show = st.slider("Funds to show per category", 1, 5, 3)

factor_bytes = st.session_state.get("factor_upload")
stamp = (LIVE / "last_update.txt").read_text() if (source == "live" and has_live) else "archive"
with st.spinner("Rating funds with the 5-factor, timing and consistency models..."):
    T, details, w0, w1 = screen_all(source, factor_bytes, stamp)

tab1, tab2, tab3, tab4 = st.tabs(["Recommendations", "Check a fund", "Upload your data", "Method and data"])

with tab1:
    st.caption(f"{len(T)} funds rated. Skill measured over {w0} to {w1} (last month of factor data); "
               f"recent performance uses the latest 12 complete months available.")
    if horizon == "Under 3 years":
        st.warning("Equity funds can fall 20-30% in a bad year. For money needed within 3 years, debt or "
                   "hybrid funds are usually more suitable. The list below is for information only.")
    cats = RISK_CATS[risk]
    st.subheader(f"Shortlist for a {risk.lower()} investor")
    st.write(f"Categories that fit this risk level: {', '.join(cats)}.")
    cols = st.columns(len(cats))
    for col, cat in zip(cols, cats):
        top = T[(T["Category"] == cat) & T["Rating"].isin(["Strong", "Good"])].head(n_show)
        col.markdown(f"**{cat}**")
        if top.empty:
            col.info("No fund rated Good or Strong. An index fund is the sensible default here.")
        for _, r in top.iterrows():
            col.markdown(f"{RATING_ICON[r['Rating']]} **{r['Fund']}**  \n"
                         f"{r['Rating']} · alpha {r['Alpha % p.a.']:.1f}% · last 12m "
                         f"{r['Last 12m vs bench %']:+.1f}% vs index")
    weak = T[T["Category"].isin(cats) & (T["Rating"] == "Weak")]
    if len(weak):
        with st.expander(f"🔴 Funds rated Weak in these categories ({len(weak)})"):
            st.dataframe(weak[["Fund", "Category", "Score (0-7)", "Alpha % p.a.", "Last 12m vs bench %"]]
                         .round(2), hide_index=True, width="stretch")
    st.subheader("All rated funds")
    f1, f2 = st.columns(2)
    pick_c = f1.multiselect("Category", list(T["Category"].unique()), default=list(T["Category"].unique()))
    pick_r = f2.multiselect("Rating", RATING_ORDER, default=RATING_ORDER)
    view = T[T["Category"].isin(pick_c) & T["Rating"].isin(pick_r)].drop(columns=["code"])
    st.dataframe(view.round(2), hide_index=True, width="stretch")
    st.download_button("Download ratings (CSV)", view.to_csv(index=False), "fund_ratings.csv")
    counts = T.groupby(["Category", "Rating"]).size().unstack(fill_value=0).reindex(columns=RATING_ORDER,
                                                                                    fill_value=0)
    st.bar_chart(counts, color=["#2e7d32", "#f9a825", "#9e9e9e", "#c62828"])

with tab2:
    st.write("Pick any rated fund, or type an AMFI scheme code for any other fund.")
    choice = st.selectbox("Fund", [""] + [f"{r.Fund} ({r.code})" for r in T.itertuples()])
    code_in = st.text_input("…or AMFI scheme code")
    code = code_in.strip() or (re.search(r"\((\d+)\)$", choice).group(1) if choice else "")
    if code:
        if code in details:
            R, why, roll, per = details[code]
            row = T[T["code"] == code].iloc[0]
            st.subheader(f"{RATING_ICON[row['Rating']]} {row['Fund']} — {row['Rating']} "
                         f"({row['Score (0-7)']}/7)")
            st.caption(f"{row['Category']} · benchmark {row['Benchmark']} · latest NAV "
                       f"{latest_nav(code, source)[0]:.2f} on {latest_nav(code, source)[1]}")
            show_fund(R, why, roll, row["Fund"], per)
        else:
            F = build_frame(source, factor_bytes, stamp)
            s = nav_monthly(code, source)
            if s.notna().sum() < mp.MIN_LIVE_MONTHS:
                try:
                    _, daily = mp.fetch_nav_online(code)
                    s = mp.to_month_end(daily)
                except Exception:
                    pass
            ret = mp.pct_returns(s).dropna()
            ok = ret.index.intersection(F.dropna(subset=["smb", "rf"]).index)
            if len(ok) < mp.MIN_LIVE_MONTHS:
                st.error("Not enough monthly history overlapping the factor data (need 36 months), "
                         "or the code was not found. Use 'Update data live' or check the code.")
            else:
                bench = st.selectbox("Benchmark", mp.BENCHMARKS, index=3,
                                     format_func=lambda b: mp.INDICES[b][0])
                R, fits, d, roll = mp.analyse(ret.loc[ok].iloc[-mp.WINDOW_MONTHS:], bench, F)
                f12, x12, per = recent_12m(mp.pct_returns(s), index_returns(source)[bench])
                pts, label, why = rate(R, x12)
                st.subheader(f"{RATING_ICON[label]} Scheme {code} — {label} ({pts}/7)")
                show_fund(R, why, roll, code, per)

with tab3:
    st.markdown("**Analyse your own data.** Upload a CSV with a date column and either NAVs/prices or "
                "monthly % returns, e.g. a fund factsheet export or a stock's price history.")
    up = st.file_uploader("NAV / price file (CSV)", type="csv", key="navfile")
    bench_u = st.selectbox("Compare with", mp.BENCHMARKS, index=3, format_func=lambda b: mp.INDICES[b][0],
                           key="bu")
    if up is not None:
        try:
            s, col = mp.load_csv_series(up)
            F = build_frame(source, factor_bytes, stamp)
            ok = s.dropna().index.intersection(F.dropna(subset=["smb", "rf", bench_u]).index)
            if len(ok) < mp.MIN_LIVE_MONTHS:
                st.error(f"Only {len(ok)} months overlap with the factor data; 36 are needed.")
            else:
                R, fits, d, roll = mp.analyse(s.loc[ok].iloc[-mp.WINDOW_MONTHS:], bench_u, F)
                pts, label, why = rate(R, np.nan)
                st.subheader(f"{RATING_ICON[label]} {up.name} [{col}] — {label} ({pts}/6, no recent-return check)")
                show_fund(R, why, roll, up.name)
        except Exception as e:
            st.error(f"Could not read this file: {e}")
    st.divider()
    st.markdown("**Newer factor data.** When IIM Ahmedabad publishes a new release, upload the monthly "
                "'Fama French and Momentum Factors' CSV here. Every rating is recomputed with it.")
    fup = st.file_uploader("IIMA factor file (CSV)", type="csv", key="facfile")
    if fup is not None and st.button("Use this factor file"):
        st.session_state["factor_upload"] = fup.getvalue()
        st.rerun()
    if factor_bytes and st.button("Go back to the default factor file"):
        st.session_state.pop("factor_upload")
        st.rerun()

with tab4:
    st.markdown(f"""
**How a fund is rated (0-7 points)**

| Check | Points |
|---|---|
| 5-factor alpha positive and significant (Newey-West p < 0.05) | +2 (positive but not significant +1, significantly negative −2) |
| Alpha positive in both halves of the 5-year period | +1 |
| Rolling 3-year alpha positive in at least 75% of windows | +1 |
| Beat its benchmark in at least 80% of calendar years | +1 |
| Down-capture below 90% (falls less than the market) | +1 |
| Last 12 months ahead of benchmark (+1); more than 5% behind (−1) | ±1 |

**Strong** 6-7 · **Good** 4-5 · **Average** 2-3 · **Weak** 0-1 or below.

**Risk profiles.** Conservative: Large and Flexi Cap. Moderate: adds Mid Cap. Aggressive: adds Small Cap.

**Models.** Each fund is compared with its own style benchmark (NIFTY 100, Midcap 150, Smallcap 250 or
NIFTY 500) using the CAPM, Fama-French, Carhart and final 5-factor models and the Treynor-Mazuy timing model,
exactly as in the project report.

**Data.** Bundled: AMFI month-end NAVs (Nov 2020 - Dec 2025) and IIM Ahmedabad factors (release 2025-12).
Live update: latest NAVs from MFapi.in (which mirrors AMFI) and the newest IIMA release. Indices are
represented by passive funds that track them. Current skill window: {w0} to {w1}.

**Limitations.** Ratings use past performance, which does not guarantee future returns. Only funds that
still exist are rated (survivorship bias). Skill is measured up to the last month of factor data, which
lags the market by a few months.
""")
    st.warning("Educational tool for a college project. It is not investment advice and the authors are not "
               "SEBI-registered advisers. Consult a registered adviser before investing.")
