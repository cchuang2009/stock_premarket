"""
盤前策略回測工具 - Streamlit + yfinance 版
策略：盤前買 → 一拉即賣 → 跌即補回 → 拉高再賣

安裝：
    pip install streamlit yfinance pandas plotly

執行：
    streamlit run premarket_backtest.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ── 頁面設定 ──
st.set_page_config(
    page_title="盤前策略回測",
    page_icon="📡",
    layout="wide",
)

st.title("📡 盤前策略回測｜Premarket Scalp Simulator")
st.caption("策略：盤前買入 → 一拉即賣 → 跌即補回 → 拉高再賣")

# ── 側邊欄：參數 ──
with st.sidebar:
    st.header("⚙️ 策略參數")

    tickers_input = st.text_input("股票代碼（逗號分隔）", value="AAOI,AXTI,COHR")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    days = st.slider("回測天數", min_value=5, max_value=30, value=5, step=1)

    st.divider()

    # 依第一檔 ticker 自動載入最優預設參數
    OPTIMAL = {
        "AAOI": dict(sell=9.0,  sl=5.5, drop=5.5, resell=6.0),
        "AXTI": dict(sell=11.0, sl=6.0, drop=6.5, resell=7.0),
        "COHR": dict(sell=7.0,  sl=4.0, drop=4.0, resell=5.0),
        "LITE": dict(sell=7.5,  sl=4.0, drop=4.0, resell=5.0),
        "CIEN": dict(sell=5.0,  sl=3.0, drop=3.0, resell=4.0),
    }
    first = tickers[0] if tickers else "AAOI"
    d = OPTIMAL.get(first, OPTIMAL["AAOI"])

    st.subheader("第一波：盤前買 → 拉高賣")
    st.caption(f"📌 預設值依 {first} 近期最佳化")
    sell_target_pct = st.slider("止盈目標 (%)", 1.0, 20.0, d["sell"], 0.5)
    stop_loss_pct   = st.slider("停損幅度 (%)",  1.0, 12.0, d["sl"],   0.5)

    st.divider()
    st.subheader("第二波：跌補 → 再賣")
    rebuy_drop_pct    = st.slider("跌補觸發幅度 (%)",  1.0, 15.0, d["drop"],   0.5)
    resell_target_pct = st.slider("第二波止盈目標 (%)", 1.0, 15.0, d["resell"], 0.5)

    st.divider()
    st.info(
        "**各股最優參數速查**\n\n"
        "🔴 **AAOI**：+9% / SL-5.5% / 補-5.5% / 再+6%\n\n"
        "🟠 **AXTI**：+11% / SL-6% / 補-6.5% / 再+7%\n\n"
        "🔵 **COHR**：+7% / SL-4% / 補-4% / 再+5%\n\n"
        "🟢 **LITE**：+7.5% / SL-4% / 補-4% / 再+5%\n\n"
        "⚡ 開盤後前 **30分鐘** 是黃金執行窗口"
    )

    st.divider()
    run_btn = st.button("🚀 執行回測", use_container_width=True, type="primary")

# ── 抓數據 ──
@st.cache_data(ttl=300)
def fetch_data(ticker: str, days: int) -> pd.DataFrame:
    """
    yfinance 的盤前數據需要 interval='1m' + prepost=True。
    這裡用 1d OHLC 估算盤前行為：
      - premarket 估算 = 開盤前 gap（用前日收盤 + 開盤價差推算）
      - 若有60m數據則更精確
    """
    end = datetime.today()
    start = end - timedelta(days=days + 10)  # 多抓幾天避開假日

    # 日線：取 OHLC + 前收
    df_day = yf.download(ticker, start=start, end=end, interval="1d",
                         auto_adjust=True, progress=False)
    if df_day.empty:
        return pd.DataFrame()

    # 只取最近 N 個交易日
    df_day = df_day.tail(days).copy()
    df_day.index = pd.to_datetime(df_day.index)

    # 嘗試抓 30m 盤前數據 (prepost=True)
    df_pre = yf.download(ticker, start=start, end=end, interval="30m",
                         prepost=True, auto_adjust=True, progress=False)

    rows = []
    prev_close = None

    for date, row in df_day.iterrows():
        # 修正 MultiIndex columns（yfinance 新版）
        def g(col):
            try:
                return float(row[(col, ticker)])
            except (KeyError, TypeError):
                try:
                    return float(row[col])
                except:
                    return None

        open_p  = g("Open")
        high_p  = g("High")
        low_p   = g("Low")
        close_p = g("Close")

        if None in (open_p, high_p, low_p, close_p):
            prev_close = close_p
            continue

        # 盤前估算：用 30m prepost 取當天 04:00-09:30 的最後一根
        premarket_est = None
        if not df_pre.empty:
            day_str = date.strftime("%Y-%m-%d")
            mask = (df_pre.index.date == date.date()) & (df_pre.index.hour < 9)
            pre_rows = df_pre[mask]
            if not pre_rows.empty:
                last_pre = pre_rows.iloc[-1]
                try:
                    premarket_est = float(last_pre[("Close", ticker)])
                except (KeyError, TypeError):
                    try:
                        premarket_est = float(last_pre["Close"])
                    except:
                        pass

        # 若抓不到盤前，用開盤價代替（最保守估計）
        if premarket_est is None or premarket_est <= 0:
            premarket_est = open_p

        rows.append({
            "date":       date.strftime("%m/%d"),
            "prev_close": prev_close if prev_close else open_p,
            "premarket":  premarket_est,
            "open":       open_p,
            "high":       high_p,
            "low":        low_p,
            "close":      close_p,
        })
        prev_close = close_p

    return pd.DataFrame(rows)


def simulate(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    sell_target  = params["sell_target_pct"] / 100
    stop_loss    = params["stop_loss_pct"] / 100
    rebuy_drop   = params["rebuy_drop_pct"] / 100
    resell_target = params["resell_target_pct"] / 100

    results = []
    for _, r in df.iterrows():
        buy   = r["premarket"]
        high  = r["high"]
        low   = r["low"]
        close = r["close"]

        # 第一波
        w1_target = buy * (1 + sell_target)
        w1_sl     = buy * (1 - stop_loss)
        w1_hit    = high >= w1_target
        if w1_hit:
            w1_exit = w1_target
        elif low <= w1_sl:
            w1_exit = w1_sl
        else:
            w1_exit = close
        w1_pnl = (w1_exit - buy) / buy * 100

        # 第二波
        rebuy_price = buy * (1 - rebuy_drop)
        w2_triggered = low <= rebuy_price
        w2_pnl = None
        w2_exit = None
        if w2_triggered:
            w2_target_price = rebuy_price * (1 + resell_target)
            w2_hit = high >= w2_target_price
            w2_exit = w2_target_price if w2_hit else close
            w2_pnl = (w2_exit - rebuy_price) / rebuy_price * 100

        combined = w1_pnl + (w2_pnl if w2_pnl is not None else 0)

        results.append({
            "日期":       r["date"],
            "盤前價":     round(buy, 2),
            "盤前Gap%":   round((buy - r["prev_close"]) / r["prev_close"] * 100, 2),
            "開盤":       round(r["open"], 2),
            "最高":       round(high, 2),
            "最低":       round(low, 2),
            "收盤":       round(close, 2),
            "日內振幅%":  round((high - low) / low * 100, 2),
            "W1出場價":   round(w1_exit, 2),
            "W1損益%":    round(w1_pnl, 2),
            "W1成功":     w1_hit,
            "跌補觸發":   w2_triggered,
            "W2損益%":    round(w2_pnl, 2) if w2_pnl is not None else None,
            "綜合損益%":  round(combined, 2),
        })

    return pd.DataFrame(results)


def color_pnl(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return "color: #22c55e" if val > 0 else ("color: #ef4444" if val < 0 else "")


def render_ticker(ticker: str, params: dict):
    st.subheader(f"🔵 {ticker}")

    with st.spinner(f"抓取 {ticker} 數據中..."):
        df_raw = fetch_data(ticker, days)

    if df_raw.empty:
        st.error(f"無法取得 {ticker} 數據，請確認代碼正確且有網路連線。")
        return

    df = simulate(df_raw, params)

    # ── 摘要指標 ──
    n = len(df)
    win1   = (df["W1損益%"] > 0).sum()
    w2_rows = df[df["跌補觸發"]]
    win2   = (w2_rows["W2損益%"] > 0).sum() if not w2_rows.empty else 0
    w2_total = len(w2_rows)
    avg_combined = df["綜合損益%"].mean()
    total_win = (df["綜合損益%"] > 0).sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("第一波勝率", f"{win1}/{n} ({win1/n*100:.0f}%)")
    c2.metric("第二波勝率", f"{win2}/{w2_total}" if w2_total else "未觸發")
    c3.metric("平均綜合報酬", f"{avg_combined:+.2f}%",
              delta_color="normal" if avg_combined >= 0 else "inverse")
    c4.metric("獲利天數", f"{total_win}/{n}")

    # ── 圖表 ──
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["每日損益 (%)", "日內振幅 & 盤前Gap (%)"],
        vertical_spacing=0.15,
        row_heights=[0.6, 0.4],
    )

    colors_w1 = ["#22c55e" if v > 0 else "#ef4444" for v in df["W1損益%"]]
    colors_combined = ["#3b82f6" if v > 0 else "#f97316" for v in df["綜合損益%"]]

    fig.add_trace(go.Bar(
        x=df["日期"], y=df["W1損益%"],
        name="第一波損益%", marker_color=colors_w1, opacity=0.75,
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=df["日期"], y=df["綜合損益%"],
        name="綜合損益%", marker_color=colors_combined, opacity=0.85,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df["日期"], y=df["日內振幅%"],
        name="日內振幅%", mode="lines+markers",
        line=dict(color="#f59e0b", width=2),
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=df["日期"], y=df["盤前Gap%"],
        name="盤前Gap%", mode="lines+markers",
        line=dict(color="#a78bfa", width=2, dash="dot"),
    ), row=2, col=1)

    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    fig.update_layout(
        height=500, template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── 明細表 ──
    st.caption("逐日明細")
    display_cols = ["日期", "盤前價", "盤前Gap%", "最高", "最低", "日內振幅%",
                    "W1出場價", "W1損益%", "跌補觸發", "W2損益%", "綜合損益%"]
    df_show = df[display_cols].copy()

    def fmt_bool(val):
        return "✅" if val else "—"

    df_show["跌補觸發"] = df["跌補觸發"].apply(fmt_bool)
    df_show["W2損益%"]  = df["W2損益%"].apply(lambda v: f"{v:+.2f}%" if v is not None and not pd.isna(v) else "—")

    styled = (
        df_show.style
        .applymap(lambda v: color_pnl(v) if isinstance(v, float) else "",
                  subset=["W1損益%", "綜合損益%", "盤前Gap%"])
        .format({
            "盤前價": "${:.2f}",
            "最高":   "${:.2f}",
            "最低":   "${:.2f}",
            "W1出場價": "${:.2f}",
            "盤前Gap%": "{:+.2f}%",
            "日內振幅%": "{:.2f}%",
            "W1損益%": "{:+.2f}%",
            "綜合損益%": "{:+.2f}%",
        })
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.divider()


# ── 新聞抓取 ──
@st.cache_data(ttl=600)
def fetch_news(ticker: str, max_items: int = 6) -> list:
    """用 yfinance 抓最新新聞"""
    try:
        t = yf.Ticker(ticker)
        news = t.news or []
        results = []
        for item in news[:max_items]:
            content   = item.get("content", {})
            title     = content.get("title", item.get("title", "N/A"))
            publisher = content.get("provider", {}).get("displayName", item.get("publisher", ""))
            link      = content.get("canonicalUrl", {}).get("url", item.get("link", ""))
            pub_ts    = content.get("pubDate", "")
            if pub_ts:
                try:
                    dt = datetime.fromisoformat(pub_ts.replace("Z", "+00:00"))
                    pub_str = dt.strftime("%m/%d %H:%M")
                except Exception:
                    pub_str = pub_ts[:10]
            else:
                pub_ts2 = item.get("providerPublishTime", 0)
                pub_str = datetime.fromtimestamp(pub_ts2).strftime("%m/%d %H:%M") if pub_ts2 else ""
            results.append({"title": title, "link": link, "publisher": publisher, "published": pub_str})
        return results
    except Exception:
        return []


def sentiment_badge(title: str) -> str:
    """簡易關鍵字情緒判斷"""
    bull_kw = ["surge", "soar", "jump", "rally", "beat", "upgrade", "buy", "bullish",
               "order", "contract", "record", "high", "gain", "rise", "strong", "outperform"]
    bear_kw = ["drop", "fall", "miss", "downgrade", "sell", "bearish", "cut", "loss",
               "decline", "warn", "disappoint", "weak", "below", "concern", "risk", "insider sell"]
    t = title.lower()
    b_score = sum(1 for w in bull_kw if w in t)
    r_score = sum(1 for w in bear_kw if w in t)
    if b_score > r_score:
        return "🟢 Bullish"
    elif r_score > b_score:
        return "🔴 Bearish"
    return "⚪ Neutral"


def render_news_section(tickers: list):
    st.header("📰 Latest News & Sentiment")
    st.caption("Source: Yahoo Finance  |  Auto-sentiment via keyword scoring")

    cols = st.columns(len(tickers))
    for col, ticker in zip(cols, tickers):
        with col:
            st.subheader(f"🔵 {ticker}")
            news = fetch_news(ticker)
            if not news:
                st.warning("No news found.")
                continue

            bull_count = 0
            bear_count = 0
            for item in news:
                sentiment = sentiment_badge(item["title"])
                if "Bullish" in sentiment:
                    bull_count += 1
                elif "Bearish" in sentiment:
                    bear_count += 1
                border_color = "#22c55e" if "Bullish" in sentiment else "#ef4444" if "Bearish" in sentiment else "#6b7280"
                st.markdown(
                    f"""<div style='background:#1e293b;border-radius:8px;padding:10px;
                    margin-bottom:8px;border-left:3px solid {border_color}'>
                    <div style='font-size:11px;color:#94a3b8;margin-bottom:4px'>
                    {sentiment} &nbsp;·&nbsp; {item["publisher"]} &nbsp;·&nbsp; {item["published"]}</div>
                    <div style='font-size:13px;color:#e2e8f0;line-height:1.4'>
                    <a href='{item["link"]}' target='_blank'
                    style='color:#e2e8f0;text-decoration:none'>{item["title"]}</a>
                    </div></div>""",
                    unsafe_allow_html=True,
                )

            # 情緒儀表
            total = len(news)
            bull_pct = int(bull_count / total * 100) if total else 0
            bear_pct = int(bear_count / total * 100) if total else 0
            neu_pct  = 100 - bull_pct - bear_pct
            st.markdown("**Sentiment Summary**")
            c1, c2, c3 = st.columns(3)
            c1.metric("🟢 Bull",    f"{bull_pct}%")
            c2.metric("🔴 Bear",    f"{bear_pct}%")
            c3.metric("⚪ Neutral", f"{neu_pct}%")

            # 爆發訊號評分
            score = bull_pct - bear_pct
            if score >= 40:
                st.success(f"⚡ Explosion: STRONG ({score:+d})")
            elif score >= 10:
                st.info(f"📈 Explosion: MODERATE ({score:+d})")
            elif score <= -20:
                st.error(f"⚠️ Explosion: WEAK ({score:+d})")
            else:
                st.warning(f"➡️ Explosion: NEUTRAL ({score:+d})")


# ── 主流程 ──
if run_btn or True:  # 預設自動執行
    params = dict(
        sell_target_pct=sell_target_pct,
        stop_loss_pct=stop_loss_pct,
        rebuy_drop_pct=rebuy_drop_pct,
        resell_target_pct=resell_target_pct,
    )

    # 新聞區塊（置頂）
    render_news_section(tickers)
    st.divider()
    st.header("📊 Strategy Backtest")

    for ticker in tickers:
        render_ticker(ticker, params)

st.caption("⚠️ 本工具僅供量化研究，不構成任何投資建議。數據來源：Yahoo Finance (yfinance)")
