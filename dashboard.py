#!/usr/bin/env python3
"""
NEPSE Forecast Dashboard
Run with: streamlit run dashboard.py
"""

import json
import sqlite3
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

ROOT = Path(__file__).parent
FORECAST_DIR = ROOT / "xlstm_forecasts"
MODEL_DIR    = ROOT / "xlstm_models"

import config

st.set_page_config(
    page_title="NEPSE xLSTM Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    section[data-testid="stSidebar"] { background: #0f1117; }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] label { color: #9ca3af; font-size: 12px; letter-spacing: 0.05em; text-transform: uppercase; }
    section[data-testid="stSidebar"] h1 { font-size: 14px !important; font-weight: 600; color: #e5e7eb; letter-spacing: 0.08em; text-transform: uppercase; padding-bottom: 4px; border-bottom: 1px solid #1f2937; margin-bottom: 16px; }
    span[data-baseweb="tag"] { background-color: #1f2937 !important; border: 1px solid #374151 !important; border-radius: 4px !important; }
    span[data-baseweb="tag"] span { color: #e5e7eb !important; font-size: 11px !important; }
    section[data-testid="stSidebar"] .stButton button {
        width: 100%; background: transparent; border: 1px solid #374151;
        color: #9ca3af; font-size: 12px; letter-spacing: 0.05em;
        text-transform: uppercase; padding: 6px 12px;
    }
    section[data-testid="stSidebar"] .stButton button:hover { border-color: #6b7280; color: #e5e7eb; }
    .metric-card { background: #111827; border: 1px solid #1f2937; border-radius: 8px; padding: 16px 20px; text-align: center; }
    .metric-label { color: #6b7280; font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 6px; }
    .metric-value { color: #f9fafb; font-size: 24px; font-weight: 700; }
    .metric-sub   { color: #6b7280; font-size: 11px; margin-top: 4px; }
    .status-running { color: #22c55e !important; }
    .status-done    { color: #3b82f6 !important; }
    .status-idle    { color: #6b7280 !important; }
    .up   { color: #22c55e !important; }
    .down { color: #ef4444 !important; }
    .stDataFrame { font-size: 13px; }
    .page-header { border-bottom: 1px solid #d1d5db; padding-bottom: 12px; margin-bottom: 20px; }
    .page-header h2 { font-size: 22px; font-weight: 700; color: #111827; margin: 0; letter-spacing: -0.01em; }
    .page-header span { font-size: 13px; color: #4b5563; }
    .acc-badge-good { background:#166534; color:#bbf7d0; padding:3px 10px; border-radius:4px; font-size:13px; font-weight:600; }
    .acc-badge-ok   { background:#854d0e; color:#fef08a; padding:3px 10px; border-radius:4px; font-size:13px; font-weight:600; }
    .acc-badge-poor { background:#7f1d1d; color:#fca5a5; padding:3px 10px; border-radius:4px; font-size:13px; font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_latest_predictions():
    files = sorted(FORECAST_DIR.glob("predictions_*.json"), reverse=True)
    if not files:
        return None, None
    latest = files[0]
    with open(latest) as f:
        data = json.load(f)
    return latest.stem.replace("predictions_", ""), data


@st.cache_data(ttl=60)
def load_all_prediction_files():
    files = sorted(FORECAST_DIR.glob("predictions_*.json"), reverse=True)
    result = []
    for f in files:
        with open(f) as fh:
            result.append((f.stem.replace("predictions_", ""), json.load(fh)))
    return result


@st.cache_data(ttl=30)
def get_training_status():
    logs = sorted(ROOT.glob("logs/train_all_xlstm_*.log"), reverse=True)
    if not logs:
        return None, 0, 0, 0, False
    log   = logs[0]
    lines = log.read_text(errors="ignore").splitlines()
    trained = skipped = failed = 0
    last_symbol = ""
    is_running  = True

    for line in reversed(lines):
        if "COMPLETED" in line.strip():
            is_running = False
            break
    for line in reversed(lines):
        stripped = line.strip()
        if "Trained:" in stripped and "Skipped:" in stripped:
            try:
                parts    = stripped.split()
                trained  = int(parts[parts.index("Trained:") + 1])
                skipped  = int(parts[parts.index("Skipped:") + 1])
                failed   = int(parts[parts.index("Failed:") + 1])
            except Exception:
                pass
            break
    for line in reversed(lines):
        if line.startswith("🚀 xLSTM MARKET-AWARE"):
            last_symbol = line.split(":")[-1].strip()
            break
    return last_symbol, trained, skipped, failed, is_running


@st.cache_data(ttl=60)
def count_models():
    return len([p for p in MODEL_DIR.glob("*_xlstm.pt")
                if "_eval" not in p.name and "PFL" not in p.stem[:3]])


@st.cache_data(ttl=120)
def get_actual_prices(symbols: tuple, dates: tuple) -> dict:
    if not symbols or not dates:
        return {}
    conn = sqlite3.connect(config.DB_PATH)
    cur  = conn.cursor()
    sp   = ",".join("?" * len(symbols))
    dp   = ",".join("?" * len(dates))
    cur.execute(
        f"SELECT symbol, date, close FROM price_history "
        f"WHERE symbol IN ({sp}) AND date IN ({dp})",
        list(symbols) + list(dates),
    )
    rows = cur.fetchall()
    conn.close()
    result = {}
    for sym, d, close in rows:
        result.setdefault(sym, {})[d] = close
    return result


def compute_accuracy(pred_date: str, preds: dict, horizons=(1, 3, 5)):
    symbols    = list(preds.keys())
    all_dates  = set()
    for r in preds.values():
        for h in horizons:
            p = r["predictions"].get(str(h)) or r["predictions"].get(h)
            if p:
                all_dates.add(p["target_date"])

    actuals = get_actual_prices(tuple(symbols), tuple(sorted(all_dates)))

    agg  = {h: {"dir": [], "abs_err": [], "pct_err": [], "rows": []} for h in horizons}

    for sym, r in preds.items():
        cur_price = r["current_price"]
        for h in horizons:
            p = r["predictions"].get(str(h)) or r["predictions"].get(h)
            if not p:
                continue
            target_date  = p["target_date"]
            pred_price   = p["predicted_price"]
            actual_price = actuals.get(sym, {}).get(target_date)
            if actual_price is None:
                continue

            abs_err = abs(pred_price - actual_price)
            pct_err = abs_err / actual_price * 100
            dir_ok  = int((pred_price > cur_price) == (actual_price > cur_price))

            agg[h]["dir"].append(dir_ok)
            agg[h]["abs_err"].append(abs_err)
            agg[h]["pct_err"].append(pct_err)
            agg[h]["rows"].append({
                "Symbol":      sym,
                "Sector":      r.get("sector") or "—",
                "Cur Price":   round(cur_price, 1),
                "Predicted":   round(pred_price, 1),
                "Actual":      round(actual_price, 1),
                "Error":       round(pred_price - actual_price, 1),
                "Err %":       round(pct_err, 2),
                "Dir":         "OK" if dir_ok else "MISS",
                "Target Date": target_date,
            })

    summary = {}
    for h in horizons:
        d = agg[h]
        n = len(d["dir"])
        if n == 0:
            summary[h] = None
            continue
        summary[h] = {
            "n":       n,
            "dir_acc": round(100 * sum(d["dir"]) / n, 1),
            "mae":     round(float(np.mean(d["abs_err"])), 2),
            "mape":    round(float(np.mean(d["pct_err"])), 2),
            "medape":  round(float(np.median(d["pct_err"])), 2),
            "rows":    sorted(agg[h]["rows"], key=lambda x: x["Err %"]),
        }
    return summary


def build_df(data: dict) -> pd.DataFrame:
    rows = []
    for sym, r in data.items():
        preds = r.get("predictions", {})
        def g(h):
            return preds.get(h) or preds.get(str(h))
        p1, p3, p5, p10, p15, p21 = g(1), g(3), g(5), g(10), g(15), g(21)
        if not p1:
            continue
        row = {
            "Symbol":  sym,
            "Sector":  r.get("sector") or "—",
            "Price":   r.get("current_price", 0),
            "1d %":    round(p1["pct_change"], 2),
            "5d %":    round(p5["pct_change"], 2) if p5 else None,
            "10d %":   round(p10["pct_change"], 2) if p10 else None,
            "15d %":   round(p15["pct_change"], 2) if p15 else None,
            "21d %":   round(p21["pct_change"], 2) if p21 else None,
            "Conf":    p1["confidence"],
        }
        # Per-horizon predicted price + 95% CI bounds
        for label, p in [("1d", p1), ("5d", p5), ("10d", p10), ("21d", p21)]:
            if p:
                row[f"{label} Pred"] = round(p["predicted_price"], 1)
                row[f"{label} Low"]  = round(p["lower_bound"], 1)
                row[f"{label} High"] = round(p["upper_bound"], 1)
            else:
                row[f"{label} Pred"] = None
                row[f"{label} Low"]  = None
                row[f"{label} High"] = None
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values("1d %", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.index += 1
    return df


def color_pct(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return "color: #22c55e" if val > 0 else "color: #ef4444" if val < 0 else ""


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Filters")

    date, data = load_latest_predictions()
    if data:
        df_full = build_df(data)
        sectors = sorted(s for s in df_full["Sector"].dropna().unique() if s != "—")
        sel_sectors = st.multiselect("Sector", sectors, default=sectors)
        conf_filter = st.multiselect("Confidence", ["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM"])
        sort_by = st.selectbox("Sort by", ["1d %", "5d %", "10d %", "21d %"], index=0)
        show_n  = st.slider("Show top N", 10, 200, 50)

    st.divider()
    st.caption(f"Data: {date or '—'}  ·  auto-refreshes 60s")
    if st.button("Refresh"):
        st.cache_data.clear()
        st.rerun()


# ── Page header ───────────────────────────────────────────────────────────────

st.markdown("""
<div class="page-header">
  <h2>NEPSE Forecast</h2>
  <span>Machine learning price forecasts for NEPSE equities</span>
</div>
""", unsafe_allow_html=True)

last_sym, trained, skipped, failed_n, is_training = get_training_status()
n_models = count_models()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Models Saved</div>
        <div class="metric-value">{n_models}</div>
        <div class="metric-sub">production checkpoints</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Trained Today</div>
        <div class="metric-value">{trained}</div>
        <div class="metric-sub">skipped {skipped} · failed {failed_n}</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Forecast Date</div>
        <div class="metric-value" style="font-size:20px">{date or '—'}</div>
        <div class="metric-sub">latest predictions file</div>
    </div>""", unsafe_allow_html=True)
with col4:
    if is_training and last_sym:
        status_cls, status_text, status_sub = "status-running", f"Running — {last_sym}", "training in progress"
    elif last_sym:
        status_cls, status_text, status_sub = "status-done", "Complete", f"last: {last_sym}"
    else:
        status_cls, status_text, status_sub = "status-idle", "Idle", "no active run"
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Training Status</div>
        <div class="metric-value {status_cls}" style="font-size:16px">{status_text}</div>
        <div class="metric-sub">{status_sub}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

if not data:
    st.warning("No predictions file found. Run `python3 train_all_xlstm.py` first.")
    st.stop()


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_forecast, tab_accuracy = st.tabs(["Forecasts", "Accuracy"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Forecasts
# ══════════════════════════════════════════════════════════════════════════════

with tab_forecast:
    df = df_full.copy()
    if sel_sectors:
        df = df[df["Sector"].isin(sel_sectors)]
    if conf_filter:
        df = df[df["Conf"].isin(conf_filter)]
    df.sort_values(sort_by, ascending=False, inplace=True, na_position="last")
    df.reset_index(drop=True, inplace=True)
    df.index += 1

    top_df = df.head(show_n)

    bullish = (df["1d %"] > 0).sum()
    bearish = (df["1d %"] < 0).sum()
    avg_1d  = df["1d %"].mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("Bullish signals", bullish)
    c2.metric("Bearish signals", bearish)
    c3.metric("Avg 1d forecast", f"{avg_1d:+.2f}%")

    st.markdown("---")

    # 1d Low / 1d High hidden — current models are single-ensemble (N_MODELS=1)
    # so std=0 and bounds collapse to predicted price. Will be re-enabled after
    # next retrain with N_MODELS=3.
    all_display_cols = ["Symbol", "Sector", "Price", "1d %", "5d %", "10d %", "21d %", "Conf"]
    display_cols = [c for c in all_display_cols if c in top_df.columns]
    pct_cols     = [c for c in display_cols if c.endswith(" %")]

    fmt_pct = lambda v: f"{v:+.2f}%" if v is not None and not pd.isna(v) else "—"
    fmt_map = {"Price": "{:.1f}", "1d Low": "{:.1f}", "1d High": "{:.1f}"}
    for c in pct_cols:
        fmt_map[c] = fmt_pct

    styled = (
        top_df[display_cols]
        .style
        .applymap(color_pct, subset=pct_cols)
        .format(fmt_map, na_rep="—")
    )
    st.dataframe(styled, use_container_width=True, height=600)

    # ── Forecast price bands ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Forecast Price Bands")
    st.caption(
        "Predicted price with 95% confidence interval (Low / High) per horizon. "
        "Bands collapse to the prediction when the model is a single-ensemble "
        "(current state — bounds become meaningful after the next 3-model retrain)."
    )

    band_cols = ["Symbol", "Sector", "Price"]
    for h in ("1d", "5d", "10d", "21d"):
        band_cols += [f"{h} Low", f"{h} Pred", f"{h} High"]
    band_cols = [c for c in band_cols if c in top_df.columns]

    band_df = top_df[band_cols].copy()

    # Format all numeric price columns to 1 decimal place
    band_fmt = {}
    for c in band_df.columns:
        if c not in ("Symbol", "Sector"):
            band_fmt[c] = lambda v: f"{v:.1f}" if v is not None and not pd.isna(v) else "—"

    st.dataframe(
        band_df.style.format(band_fmt, na_rep="—"),
        use_container_width=True, height=420,
    )

    st.markdown("---")
    st.subheader("Sector — Avg 1-Day Forecast")

    sector_avg = (
        df.groupby("Sector")["1d %"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"1d %": "Avg 1d %"})
    )
    chart = (
        alt.Chart(sector_avg)
        .mark_bar()
        .encode(
            x=alt.X("Avg 1d %:Q", title="Average 1-day forecast %"),
            y=alt.Y("Sector:N", sort="-x"),
            color=alt.condition(
                alt.datum["Avg 1d %"] > 0,
                alt.value("#22c55e"),
                alt.value("#ef4444"),
            ),
            tooltip=["Sector", alt.Tooltip("Avg 1d %:Q", format="+.2f")],
        )
        .properties(height=max(200, len(sector_avg) * 28))
    )
    st.altair_chart(chart, use_container_width=True)

    st.markdown("---")
    col_top, col_bot = st.columns(2)

    with col_top:
        st.subheader("Top 10 — Buy")
        top10 = df.nlargest(10, "1d %")[["Symbol", "Sector", "Price", "1d %", "5d %", "Conf"]]
        st.dataframe(
            top10.style.applymap(color_pct, subset=["1d %", "5d %"])
                       .format({"Price": "{:.1f}", "1d %": "{:+.2f}%",
                                "5d %": lambda v: f"{v:+.2f}%" if v is not None and not pd.isna(v) else "—"}),
            use_container_width=True, hide_index=True,
        )

    with col_bot:
        st.subheader("Top 10 — Sell")
        bot10 = df.nsmallest(10, "1d %")[["Symbol", "Sector", "Price", "1d %", "5d %", "Conf"]]
        st.dataframe(
            bot10.style.applymap(color_pct, subset=["1d %", "5d %"])
                       .format({"Price": "{:.1f}", "1d %": "{:+.2f}%",
                                "5d %": lambda v: f"{v:+.2f}%" if v is not None and not pd.isna(v) else "—"}),
            use_container_width=True, hide_index=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Accuracy
# ══════════════════════════════════════════════════════════════════════════════

with tab_accuracy:
    st.subheader("Prediction Accuracy — Real Forward-Looking Test")
    st.caption("Compares each predictions file against what actually happened in the database.")

    all_files = load_all_prediction_files()
    if not all_files:
        st.warning("No predictions files found.")
        st.stop()

    file_labels = [d for d, _ in all_files]
    sel_date = st.selectbox("Predictions file", file_labels, index=0)
    sel_preds = dict(all_files)[sel_date]

    HORIZONS = [1, 3, 5]
    with st.spinner("Computing accuracy..."):
        summary = compute_accuracy(sel_date, sel_preds, horizons=HORIZONS)

    # ── Summary metric row ────────────────────────────────────────────────────
    has_data = [h for h in HORIZONS if summary.get(h)]
    if not has_data:
        st.info("No actual prices in DB yet for this predictions file's target dates.")
        st.stop()

    horizon_cols = st.columns(len(has_data))
    for col, h in zip(horizon_cols, has_data):
        s = summary[h]
        da = s["dir_acc"]
        badge_cls = "acc-badge-good" if da >= 70 else "acc-badge-ok" if da >= 55 else "acc-badge-poor"
        col.markdown(f"""
<div class="metric-card">
  <div class="metric-label">{h}-Day Directional Accuracy</div>
  <div class="metric-value"><span class="{badge_cls}">{da:.1f}%</span></div>
  <div class="metric-sub">MAE {s['mae']:.1f} · MedAPE {s['medape']:.2f}% · N={s['n']}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Horizon selector for detail table ────────────────────────────────────
    available_labels = {f"{h}d (target: {summary[h]['rows'][0]['Target Date']})": h
                        for h in has_data if summary[h] and summary[h]["rows"]}
    sel_horizon_label = st.radio("Show detail for", list(available_labels.keys()),
                                 horizontal=True)
    sel_h = available_labels[sel_horizon_label]
    rows  = summary[sel_h]["rows"]

    if rows:
        df_acc = pd.DataFrame(rows).drop(columns=["Target Date"])

        # Directional breakdown
        n_ok   = sum(1 for r in rows if r["Dir"] == "OK")
        n_miss = len(rows) - n_ok
        up_pred   = [r for r in rows if r["Predicted"] > r["Cur Price"]]
        down_pred = [r for r in rows if r["Predicted"] <= r["Cur Price"]]
        up_ok   = sum(1 for r in up_pred   if r["Dir"] == "OK")
        down_ok = sum(1 for r in down_pred if r["Dir"] == "OK")

        da1, da2, da3 = st.columns(3)
        da1.metric("Correct direction", n_ok, delta=f"{n_miss} missed")
        da2.metric("Predicted UP", len(up_pred),
                   delta=f"{up_ok} correct ({100*up_ok/max(len(up_pred),1):.0f}%)" if up_pred else "—")
        da3.metric("Predicted DOWN", len(down_pred),
                   delta=f"{down_ok} correct ({100*down_ok/max(len(down_pred),1):.0f}%)" if down_pred else "—")

        st.markdown("---")

        # Error distribution chart
        err_df = pd.DataFrame({"Err %": [r["Err %"] for r in rows]})
        hist = (
            alt.Chart(err_df)
            .mark_bar(color="#3b82f6", opacity=0.8)
            .encode(
                x=alt.X("Err %:Q", bin=alt.Bin(maxbins=30), title="Absolute Error %"),
                y=alt.Y("count()", title="Stocks"),
                tooltip=[alt.Tooltip("Err %:Q", bin=True), "count()"],
            )
            .properties(height=180, title="Error distribution")
        )
        st.altair_chart(hist, use_container_width=True)

        # Full detail table
        st.markdown("---")
        col_best, col_worst = st.columns(2)

        with col_best:
            st.caption("Most accurate (lowest error %)")
            best5 = pd.DataFrame(rows[:10])
            st.dataframe(
                best5[["Symbol", "Sector", "Cur Price", "Predicted", "Actual", "Err %", "Dir"]]
                .style
                .applymap(lambda v: "color: #22c55e" if v == "OK" else "color: #ef4444",
                          subset=["Dir"])
                .format({"Cur Price": "{:.1f}", "Predicted": "{:.1f}",
                         "Actual": "{:.1f}", "Err %": "{:.2f}%"}),
                use_container_width=True, hide_index=True,
            )

        with col_worst:
            st.caption("Least accurate (highest error %)")
            worst5 = pd.DataFrame(list(reversed(rows[-10:])))
            st.dataframe(
                worst5[["Symbol", "Sector", "Cur Price", "Predicted", "Actual", "Err %", "Dir"]]
                .style
                .applymap(lambda v: "color: #22c55e" if v == "OK" else "color: #ef4444",
                          subset=["Dir"])
                .format({"Cur Price": "{:.1f}", "Predicted": "{:.1f}",
                         "Actual": "{:.1f}", "Err %": "{:.2f}%"}),
                use_container_width=True, hide_index=True,
            )

        st.markdown("---")
        st.caption("Full results — all symbols")
        df_full_acc = pd.DataFrame(rows)
        st.dataframe(
            df_full_acc[["Symbol", "Sector", "Cur Price", "Predicted", "Actual", "Error", "Err %", "Dir"]]
            .style
            .applymap(lambda v: "color: #22c55e" if v == "OK" else "color: #ef4444",
                      subset=["Dir"])
            .applymap(lambda v: "color: #ef4444" if isinstance(v, float) and v < 0
                      else ("color: #22c55e" if isinstance(v, float) and v > 0 else ""),
                      subset=["Error"])
            .format({"Cur Price": "{:.1f}", "Predicted": "{:.1f}",
                     "Actual": "{:.1f}", "Error": "{:+.1f}", "Err %": "{:.2f}%"}),
            use_container_width=True, hide_index=True, height=500,
        )

# ── Footer ────────────────────────────────────────────────────────────────────
time.sleep(0.1)
st.caption(f"Last loaded: {datetime.now().strftime('%H:%M:%S')}  ·  Data from {date}")
