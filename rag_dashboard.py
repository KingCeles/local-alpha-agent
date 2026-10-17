import urllib.parse
import pandas as pd
import streamlit as st
import yfinance as yf
import feedparser
import chromadb
import ollama
from bs4 import BeautifulSoup

# ==========================================
# 1. PAGE CONFIG & HELPER FUNCTIONS
# ==========================================
st.set_page_config(page_title="Local RTX 3060 Financial RAG Terminal", layout="wide")

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:7b"

@st.cache_resource
def get_chroma_client():
    return chromadb.Client()

chroma_client = get_chroma_client()

def fmt_num(val, decimals=2, prefix="", suffix=""):
    if val is None or val == "N/A":
        return "N/A"
    try:
        num = float(val)
        formatted = format(num, ",." + str(decimals) + "f")
        return prefix + formatted + suffix
    except Exception:
        return str(val)

def clean_html(raw_html: str) -> str:
    return BeautifulSoup(raw_html, "html.parser").get_text(separator=" ", strip=True)

def clean_df_for_display(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df_clean = df.dropna(how="all").iloc[:, :4].copy()
    df_clean.columns = [
        col.strftime("%Y-%m-%d") if hasattr(col, "strftime") else str(col)
        for col in df_clean.columns
    ]
    return df_clean

def format_statement_csv(df: pd.DataFrame, title: str) -> str:
    df_clean = clean_df_for_display(df)
    if df_clean.empty:
        return ""
    return "=== " + title + " ===\n" + df_clean.to_csv() + "\n"

def split_into_safe_chunks(text: str, max_chars: int = 1800) -> list:
    lines = text.splitlines()
    if not lines:
        return []
    header = lines[0] if lines[0].startswith("===") else ""
    chunks, current_chunk, current_len = [], [], 0

    for line in lines:
        if current_len + len(line) > max_chars and current_chunk:
            chunks.append("\n".join(current_chunk))
            current_chunk = [header + " (Continued)"] if header else []
            current_len = len(current_chunk[0]) if current_chunk else 0
        current_chunk.append(line)
        current_len += len(line) + 1

    if current_chunk:
        chunks.append("\n".join(current_chunk))
    return chunks

# ==========================================
# 2. DATA INGESTION & CHROMADB INDEXING
# ==========================================
def load_and_index_ticker(ticker_symbol: str):
    ticker_symbol = ticker_symbol.upper().strip()
    stock = yf.Ticker(ticker_symbol)
    info = stock.info or {}

    if "shortName" not in info and "currentPrice" not in info:
        return None

    pinned_financials = []
    searchable_docs = []
    news_items = []

    inc_df = clean_df_for_display(stock.income_stmt)
    bal_df = clean_df_for_display(stock.balance_sheet)
    cf_df = clean_df_for_display(stock.cashflow)

    fcf = info.get("freeCashflow") or 0
    ocf = info.get("operatingCashflow") or 0
    total_cash = info.get("totalCash", 0) or 0
    total_debt = info.get("totalDebt", 0) or 0
    shares = info.get("sharesOutstanding") or 1
    beta = float(info.get("beta", 1.2) or 1.2)
    rev_growth = float(info.get("revenueGrowth", 0.10) or 0.10)
    est_wacc = round(0.042 + beta * 0.05, 4)

    dcf_summary = "DCF Baseline: Free Cash Flow is negative or unavailable."
    if fcf and fcf > 0 and shares and shares > 0:
        growth_rate = min(max(rev_growth, 0.05), 0.25)
        term_growth = 0.03
        pv_fcf = sum([(fcf * ((1 + growth_rate) ** yr)) / ((1 + est_wacc) ** yr) for yr in range(1, 6)])
        yr5_fcf = fcf * ((1 + growth_rate) ** 5)
        terminal_val = (yr5_fcf * (1 + term_growth)) / (est_wacc - term_growth)
        pv_terminal = terminal_val / ((1 + est_wacc) ** 5)
        enterprise_value = pv_fcf + pv_terminal
        equity_value = enterprise_value + total_cash - total_debt
        intrinsic_per_share = equity_value / shares

        dcf_summary = (
            "Baseline 5-Year DCF Model: "
            + "Trailing FCF = " + fmt_num(fcf, 0, "$")
            + " | Operating Cash Flow = " + fmt_num(ocf, 0, "$")
            + " | Total Cash = " + fmt_num(total_cash, 0, "$")
            + " | Total Debt = " + fmt_num(total_debt, 0, "$")
            + " | Shares = " + fmt_num(shares, 0)
            + " | Beta = " + fmt_num(beta, 2)
            + " | WACC = " + fmt_num(est_wacc * 100, 2, "", "%")
            + " | 5-Yr FCF Growth = " + fmt_num(growth_rate * 100, 1, "", "%")
            + " | Terminal Growth = 3.0%"
            + " | Enterprise Value = " + fmt_num(enterprise_value, 0, "$")
            + " | Equity Value = " + fmt_num(equity_value, 0, "$")
            + " | DCF Intrinsic Value Per Share = " + fmt_num(intrinsic_per_share, 2, "$")
        )

    overview = (
        "=== CORE VALUATION & DCF SUMMARY (" + ticker_symbol + ") ===\n"
        + "Company: " + str(info.get("shortName", ticker_symbol))
        + " | Sector: " + str(info.get("sector", "N/A"))
        + " | Current Price: $" + str(info.get("currentPrice", "N/A"))
        + " | Market Cap: " + fmt_num(info.get("marketCap"), 0, "$")
        + " | Forward P/E: " + str(info.get("forwardPE", "N/A"))
        + " | Trailing P/E: " + str(info.get("trailingPE", "N/A"))
        + " | Gross Margin: " + str(info.get("grossMargins", "N/A"))
        + " | Operating Margin: " + str(info.get("operatingMargins", "N/A"))
        + " | Analyst Target: $" + str(info.get("targetMeanPrice", "N/A"))
        + " (" + str(info.get("recommendationKey", "N/A")) + ")\n"
        + dcf_summary
    )
    pinned_financials.append(overview)

    for df_obj, label in [
        (stock.income_stmt, "ANNUAL INCOME STATEMENT (" + ticker_symbol + ")"),
        (stock.balance_sheet, "ANNUAL BALANCE SHEET (" + ticker_symbol + ")"),
        (stock.cashflow, "ANNUAL CASH FLOW STATEMENT (" + ticker_symbol + ")"),
    ]:
        csv_text = format_statement_csv(df_obj, label)
        if csv_text:
            pinned_financials.append(csv_text)

    for df_obj, label in [
        (stock.quarterly_income_stmt, "QUARTERLY INCOME STATEMENT (" + ticker_symbol + ")"),
        (stock.quarterly_balance_sheet, "QUARTERLY BALANCE SHEET (" + ticker_symbol + ")"),
        (stock.quarterly_cashflow, "QUARTERLY CASH FLOW STATEMENT (" + ticker_symbol + ")"),
    ]:
        q_csv = format_statement_csv(df_obj, label)
        if q_csv:
            searchable_docs.extend(split_into_safe_chunks(q_csv, max_chars=1800))

    try:
        for item in (stock.news or [])[:8]:
            content = item.get("content", {})
            title = content.get("title") or item.get("title", "")
            summary = content.get("summary") or item.get("summary", "")
            if title:
                doc_text = "[Yahoo Finance News] " + title + " - " + summary[:800]
                searchable_docs.append(doc_text)
                news_items.append({"Source": "Yahoo Finance", "Headline": title, "Summary": summary[:300]})
    except Exception:
        pass

    query = urllib.parse.quote(ticker_symbol + " stock (site:bloomberg.com OR site:reuters.com OR site:cnbc.com)")
    rss_urls = [
        "https://news.google.com/rss/search?q=" + query + "&hl=en-US&gl=US&ceid=US:en",
        "https://feeds.bloomberg.com/markets/news.rss"
    ]
    for url in rss_urls:
        feed = feedparser.parse(url)
        for entry in feed.entries[:10]:
            title = entry.get("title", "")
            summary = clean_html(entry.get("summary", ""))
            if "bloomberg.com/markets" in url and ticker_symbol.lower() not in (title + summary).lower():
                continue
            searchable_docs.append("[Market News] " + title + " - " + summary[:400])
            news_items.append({"Source": "Bloomberg / Media RSS", "Headline": title, "Summary": summary[:300]})

    collection_name = "rag_" + ticker_symbol.lower()
    try:
        chroma_client.delete_collection(collection_name)
    except Exception:
        pass

    collection = chroma_client.create_collection(name=collection_name)
    for idx, doc in enumerate(searchable_docs):
        safe_doc = doc[:2000]
        embed_res = ollama.embeddings(model=EMBED_MODEL, prompt=safe_doc)
        collection.add(ids=["doc_" + str(idx)], embeddings=[embed_res["embedding"]], documents=[safe_doc])

    return {
        "ticker": ticker_symbol,
        "info": info,
        "inc_df": inc_df,
        "bal_df": bal_df,
        "cf_df": cf_df,
        "news": pd.DataFrame(news_items),
        "pinned_financials": pinned_financials,
        "collection_name": collection_name,
        "chunk_count": len(searchable_docs)
    }

# ==========================================
# 3. SIDEBAR
# ==========================================
with st.sidebar:
    st.header("1. Select Stock Ticker")
    ticker_input = st.text_input("Ticker Symbol", value="MU", placeholder="e.g., MU, NVDA, AAPL, TSLA").upper().strip()
    load_btn = st.button("Load & Index Financial RAG", use_container_width=True)

    if load_btn and ticker_input:
        with st.spinner("Pulling statements & indexing " + ticker_input + " into ChromaDB on RTX 3060..."):
            data = load_and_index_ticker(ticker_input)
            if data is None:
                st.error("Could not find ticker: " + ticker_input)
            else:
                st.session_state["rag_data"] = data
                st.session_state["messages"] = []
                st.success("Indexed " + data["ticker"] + " successfully!")

    if "rag_data" in st.session_state:
        d = st.session_state["rag_data"]
        st.divider()
        st.caption("Active RAG Index: " + str(d["ticker"]))
        st.caption("Pinned Statements: " + str(len(d["pinned_financials"])))
        st.caption("Vector Chunks: " + str(d["chunk_count"]))
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()

# ==========================================
# 4. MAIN WINDOW
# ==========================================
st.title("Local RTX 3060 Financial Statements & RAG Terminal")

if "rag_data" not in st.session_state:
    st.info("Enter a stock ticker (e.g., MU, NVDA, AAPL) in the sidebar and click Load & Index Financial RAG to begin.")
else:
    data = st.session_state["rag_data"]
    info = data["info"]
    ticker = data["ticker"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Ticker", str(info.get("shortName", ticker)) + " (" + ticker + ")")
    c2.metric("Current Price", "$" + str(info.get("currentPrice", "N/A")))
    c3.metric("Forward P/E", fmt_num(info.get("forwardPE"), 2))
    mcap = info.get("marketCap", 0) or 0
    c4.metric("Market Cap", fmt_num(mcap / 1e9, 2, "$", "B") if mcap else "N/A")
    c5.metric("Analyst Target", "$" + str(info.get("targetMeanPrice", "N/A")))

    tab_dcf, tab_inc, tab_bal, tab_cf, tab_news = st.tabs([
        "Interactive DCF Model",
        "Income Statement",
        "Balance Sheet",
        "Cash Flow Statement",
        "Indexed News (Yahoo & Bloomberg)"
    ])

    with tab_dcf:
        st.subheader("Interactive 5-Year DCF Valuation (" + ticker + ")")
        fcf = info.get("freeCashflow") or 0
        total_cash = info.get("totalCash") or 0
        total_debt = info.get("totalDebt") or 0
        shares = info.get("sharesOutstanding") or 1
        beta = float(info.get("beta", 1.2) or 1.2)
        default_wacc = round((0.042 + beta * 0.05) * 100, 1)
        default_growth = round(min(max(float(info.get("revenueGrowth", 0.12) or 0.12), 0.05), 0.30) * 100, 1)

        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        fcf_input = col_s1.number_input("Base Annual Free Cash Flow ($)", value=float(fcf), step=1e8, format="%.0f")
        g_rate = col_s2.slider("5-Year FCF Growth Rate (%)", -10.0, 50.0, float(default_growth), 0.5) / 100.0
        wacc_rate = col_s3.slider("Discount Rate / WACC (%)", 5.0, 20.0, float(default_wacc), 0.2) / 100.0
        t_rate = col_s4.slider("Terminal Growth Rate (%)", 1.0, 5.0, 3.0, 0.1) / 100.0

        if wacc_rate <= t_rate:
            st.error("WACC must be strictly greater than the Terminal Growth Rate.")
        elif fcf_input <= 0:
            st.warning("Trailing Free Cash Flow is negative or zero. Enter a positive normalized FCF above to model intrinsic value.")
        else:
            pv_fcf = sum([(fcf_input * ((1 + g_rate) ** yr)) / ((1 + wacc_rate) ** yr) for yr in range(1, 6)])
            yr5_fcf = fcf_input * ((1 + g_rate) ** 5)
            term_val = (yr5_fcf * (1 + t_rate)) / (wacc_rate - t_rate)
            pv_term = term_val / ((1 + wacc_rate) ** 5)
            ev = pv_fcf + pv_term
            eq_val = ev + total_cash - total_debt
            dcf_price = eq_val / shares
            curr_p = float(info.get("currentPrice", 0) or 0)
            upside = ((dcf_price - curr_p) / curr_p * 100) if curr_p > 0 else 0

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Implied Enterprise Value", fmt_num(ev / 1e9, 2, "$", "B"))
            m2.metric("Implied Equity Value", fmt_num(eq_val / 1e9, 2, "$", "B"))
            m3.metric("DCF Intrinsic Value / Share", fmt_num(dcf_price, 2, "$"), fmt_num(upside, 1, "", "% vs Current"))
            m4.metric("Net Cash / (Debt)", fmt_num((total_cash - total_debt) / 1e9, 2, "$", "B"))

    with tab_inc:
        st.dataframe(data["inc_df"], use_container_width=True)

    with tab_bal:
        st.dataframe(data["bal_df"], use_container_width=True)

    with tab_cf:
        st.dataframe(data["cf_df"], use_container_width=True)

    with tab_news:
        st.dataframe(data["news"], use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("Ask Your Local RTX 3060 Analyst About " + ticker)

    for msg in st.session_state.get("messages", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("Ask about " + ticker + " balance sheet, cash flows, DCF, margins, or recent news...")
    if user_q:
        st.session_state["messages"].append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        with st.chat_message("assistant"):
            collection = chroma_client.get_collection(data["collection_name"])
            q_embed = ollama.embeddings(model=EMBED_MODEL, prompt=user_q[:1500])["embedding"]
            results = collection.query(query_embeddings=[q_embed], n_results=min(5, collection.count()))
            retrieved_chunks = results["documents"][0]

            with st.expander("View Retrieved ChromaDB Chunks + Pinned Financial Statements"):
                st.markdown("**Pinned Statements:** Core Valuation + Annual Income, Balance Sheet & Cash Flow CSVs")
                for i, ch in enumerate(retrieved_chunks, 1):
                    st.text("--- Vector Match #" + str(i) + " ---\n" + ch[:500] + "...")

            combined_context = "\n\n".join(list(data["pinned_financials"]) + retrieved_chunks)
            prompt = (
                "You are an institutional equity research analyst. Answer the user's question about " + ticker
                + " using the complete financial statements (Income Statement, Balance Sheet, Cash Flow Statement), "
                + "DCF model, and news context provided below. Always cite exact dollar figures, percentages, and dates.\n\n"
                + "=== RETRIEVED FINANCIAL STATEMENTS & LIVE CONTEXT ===\n"
                + combined_context
                + "\n\n=== USER QUESTION ===\n"
                + user_q
            )

            response_box = st.empty()
            full_reply = ""
            stream = ollama.chat(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1, "num_ctx": 12288},
                stream=True
            )
            for chunk in stream:
                full_reply += chunk["message"]["content"]
                response_box.markdown(full_reply + "▌")
            response_box.markdown(full_reply)

            st.session_state["messages"].append({"role": "assistant", "content": full_reply})
