import urllib.parse
import pandas as pd
import yfinance as yf
import feedparser
import chromadb
import ollama
from bs4 import BeautifulSoup

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:7b"

chroma_client = chromadb.Client()

def clean_html(raw_html: str) -> str:
    return BeautifulSoup(raw_html, "html.parser").get_text(separator=" ", strip=True)

def format_statement(df: pd.DataFrame, title: str) -> str:
    """Formats a yfinance DataFrame as compact CSV text (65% fewer tokens than to_string)."""
    if df is None or df.empty:
        return ""
    df_clean = df.dropna(how="all").iloc[:, :4]
    df_clean.columns = [
        col.strftime("%Y-%m-%d") if hasattr(col, "strftime") else str(col)
        for col in df_clean.columns
    ]
    return f"=== {title} ===\n{df_clean.to_csv()}\n"

def split_into_safe_chunks(text: str, max_chars: int = 1800) -> list[str]:
    """Splits long tables or articles into smaller pieces so nomic-embed-text never overflows."""
    lines = text.splitlines()
    if not lines:
        return []
    header = lines[0] if lines[0].startswith("===") else ""
    chunks, current_chunk, current_len = [], [], 0

    for line in lines:
        if current_len + len(line) > max_chars and current_chunk:
            chunks.append("\n".join(current_chunk))
            # Repeat the table title header on split chunks so context isn't lost
            current_chunk = [f"{header} (Continued)"] if header else []
            current_len = len(current_chunk[0]) if current_chunk else 0
        current_chunk.append(line)
        current_len += len(line) + 1

    if current_chunk:
        chunks.append("\n".join(current_chunk))
    return chunks

# ==========================================
# 1. INGEST ALL 3 FINANCIAL STATEMENTS + DCF + NEWS
# ==========================================
def fetch_ticker_documents(ticker_symbol: str) -> tuple[list[str], list[str]]:
    ticker_symbol = ticker_symbol.upper().strip()
    print(f"\n[*] Fetching Income Statement, Balance Sheet, Cash Flow & News for {ticker_symbol}...")
    
    pinned_financials = []  # Core Annual Statements + DCF (Always passed directly to Qwen)
    searchable_docs = []    # Quarterly Statements + News (Embedded in ChromaDB)

    stock = yf.Ticker(ticker_symbol)
    info = stock.info or {}

    # A. Core Valuation & Pre-Computed 5-Year DCF Model
    if "shortName" in info or "currentPrice" in info:
        fcf = info.get("freeCashflow")
        ocf = info.get("operatingCashflow")
        total_cash = info.get("totalCash", 0) or 0
        total_debt = info.get("totalDebt", 0) or 0
        shares = info.get("sharesOutstanding")
        beta = info.get("beta", 1.2) or 1.2
        rev_growth = info.get("revenueGrowth", 0.10) or 0.10

        est_wacc = round(0.042 + float(beta) * 0.05, 4)

        dcf_summary = "DCF Baseline: Free Cash Flow is negative or unavailable in summary."
        if fcf and fcf > 0 and shares and shares > 0:
            growth_rate = min(max(float(rev_growth), 0.05), 0.25)
            term_growth = 0.03
            pv_fcf = sum([
                (fcf * ((1 + growth_rate) ** yr)) / ((1 + est_wacc) ** yr)
                for yr in range(1, 6)
            ])
            yr5_fcf = fcf * ((1 + growth_rate) ** 5)
            terminal_val = (yr5_fcf * (1 + term_growth)) / (est_wacc - term_growth)
            pv_terminal = terminal_val / ((1 + est_wacc) ** 5)
            enterprise_value = pv_fcf + pv_terminal
            equity_value = enterprise_value + total_cash - total_debt
            intrinsic_per_share = equity_value / shares

            dcf_summary = (
                f"Baseline 5-Year DCF Model: "
                f"Trailing FCF = ${fcf:,.0f} | Operating Cash Flow = ${ocf or 0:,.0f} | "
                f"Total Cash = ${total_cash:,.0f} | Total Debt = ${total_debt:,.0f} | "
                f"Shares Outstanding = {shares:,.0f} | Beta = {beta} | "
                f"Estimated WACC = {est_wacc*100:.2f}% | Assumed 5-Yr Growth = {growth_rate*100:.1f}% | "
                f"Terminal Growth = 3.0% | Implied Enterprise Value = ${enterprise_value:,.0f} | "
                f"Implied Equity Value = ${equity_value:,.0f} | "
                f"Calculated DCF Intrinsic Value Per Share = ${intrinsic_per_share:.2f}"
            )

        overview = (
            f"=== CORE VALUATION & DCF SUMMARY ({ticker_symbol}) ===\n"
            f"Company: {info.get('shortName')} | Sector: {info.get('sector', 'N/A')} | "
            f"Current Price: ${info.get('currentPrice', 'N/A')} | "
            f"Market Cap: ${info.get('marketCap', 0):,} | Enterprise Value: ${info.get('enterpriseValue', 0):,} | "
            f"Forward P/E: {info.get('forwardPE', 'N/A')} | Trailing P/E: {info.get('trailingPE', 'N/A')} | "
            f"Gross Margin: {info.get('grossMargins', 'N/A')} | Operating Margin: {info.get('operatingMargins', 'N/A')} | "
            f"Return on Equity (ROE): {info.get('returnOnEquity', 'N/A')} | "
            f"Analyst Target: ${info.get('targetMeanPrice', 'N/A')} ({info.get('recommendationKey', 'N/A')})\n"
            f"{dcf_summary}"
        )
        pinned_financials.append(overview)

    # B. Pull All 3 Annual Financial Statements (Pinned directly to LLM context)
    for attr, label in [
        ("income_stmt", "ANNUAL INCOME STATEMENT"),
        ("balance_sheet", "ANNUAL BALANCE SHEET"),
        ("cashflow", "ANNUAL CASH FLOW STATEMENT"),
    ]:
        try:
            stmt_csv = format_statement(getattr(stock, attr), f"{label} ({ticker_symbol})")
            if stmt_csv:
                pinned_financials.append(stmt_csv)
                print(f"  [+] Pulled {label.title()}")
        except Exception as e:
            print(f"  [!] {label} warning: {e}")

    # C. Pull Quarterly Statements (Split into safe chunks for ChromaDB embedding)
    for attr, label in [
        ("quarterly_income_stmt", "QUARTERLY INCOME STATEMENT"),
        ("quarterly_balance_sheet", "QUARTERLY BALANCE SHEET"),
        ("quarterly_cashflow", "QUARTERLY CASH FLOW STATEMENT"),
    ]:
        try:
            q_csv = format_statement(getattr(stock, attr), f"{label} ({ticker_symbol})")
            if q_csv:
                searchable_docs.extend(split_into_safe_chunks(q_csv, max_chars=1800))
        except Exception:
            pass
    print("  [+] Pulled & chunked Quarterly Financial Statements")

    # D. Pull Yahoo Finance & Bloomberg/Reuters/CNBC News
    try:
        for item in (stock.news or [])[:8]:
            content = item.get("content", {})
            title = content.get("title") or item.get("title", "")
            summary = content.get("summary") or item.get("summary", "")
            if title:
                searchable_docs.append(f"[Yahoo News] {title} - {summary[:800]}")
    except Exception:
        pass

    query = urllib.parse.quote(f"{ticker_symbol} stock (site:bloomberg.com OR site:reuters.com OR site:cnbc.com)")
    rss_urls = [
        f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en",
        "https://feeds.bloomberg.com/markets/news.rss"
    ]
    for url in rss_urls:
        feed = feedparser.parse(url)
        for entry in feed.entries[:10]:
            title = entry.get("title", "")
            summary = clean_html(entry.get("summary", ""))
            if "bloomberg.com/markets" in url and ticker_symbol.lower() not in (title + summary).lower():
                continue
            searchable_docs.append(f"[Market News] {title} - {summary[:400]}")

    print(f"[+] Loaded {len(pinned_financials)} pinned annual tables & {len(searchable_docs)} safe RAG chunks.")
    return pinned_financials, searchable_docs

# ==========================================
# 2. EMBED & INDEX INTO VECTOR DATABASE
# ==========================================
def build_vector_index(ticker_symbol: str, documents: list[str]):
    collection_name = f"rag_{ticker_symbol.lower()}"
    try:
        chroma_client.delete_collection(collection_name)
    except Exception:
        pass

    collection = chroma_client.create_collection(name=collection_name)
    print(f"[*] Embedding {len(documents)} chunks into ChromaDB using {EMBED_MODEL}...")

    for idx, doc in enumerate(documents):
        # Hard safety cap at 2000 chars so nomic-embed-text never throws HTTP 500
        safe_doc = doc[:2000]
        embed_res = ollama.embeddings(model=EMBED_MODEL, prompt=safe_doc)
        collection.add(
            ids=[f"doc_{idx}"],
            embeddings=[embed_res["embedding"]],
            documents=[safe_doc]
        )
    return collection

# ==========================================
# 3. RETRIEVE & GENERATE (WITH ALL 3 STATEMENTS PINNED)
# ==========================================
def query_rag(collection, pinned_financials: list[str], ticker_symbol: str, user_question: str):
    q_embed = ollama.embeddings(model=EMBED_MODEL, prompt=user_question[:1500])["embedding"]
    results = collection.query(query_embeddings=[q_embed], n_results=min(5, collection.count()))
    retrieved_chunks = results["documents"][0]

    combined_chunks = list(pinned_financials) + retrieved_chunks
    context_block = "\n\n".join(combined_chunks)

    prompt = f"""
    You are an institutional equity research analyst. Answer the user's question about {ticker_symbol} 
    using the complete financial statements (Income Statement, Balance Sheet, Cash Flow Statement), 
    DCF model, and news context provided below. Always cite exact figures and dates from the tables.

    === RETRIEVED FINANCIAL STATEMENTS & LIVE CONTEXT ===
    {context_block}

    === USER QUESTION ===
    {user_question}
    """

    print("\n[AI Analyst Response]:")
    stream = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0.1,
            "num_ctx": 12288  # 12k context window holds all 3 CSV statements + retrieved chunks
        },
        stream=True
    )
    for chunk in stream:
        print(chunk["message"]["content"], end="", flush=True)
    print("\n")

# ==========================================
# 4. INTERACTIVE CLI LOOP
# ==========================================
if __name__ == "__main__":
    print("===============================================================")
    print(" ⚡ Local RTX 3060 Full-Statement Financial RAG Terminal")
    print("===============================================================")
    while True:
        ticker = input("\nEnter a stock ticker to research (e.g., MU, NVDA, AAPL) or 'q' to quit: ").strip()
        if ticker.lower() == 'q':
            break
        if not ticker:
            continue

        pinned_fin, search_docs = fetch_ticker_documents(ticker)
        if not pinned_fin and not search_docs:
            print("[!] No data found for that ticker. Check the symbol and try again.")
            continue

        collection = build_vector_index(ticker, search_docs)

        while True:
            question = input(f"\nAsk anything about {ticker.upper()} (or type 'new' to switch ticker): ").strip()
            if question.lower() == 'new':
                break
            if not question:
                continue
            query_rag(collection, pinned_fin, ticker.upper(), question)
