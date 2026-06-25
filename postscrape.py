#actually this sucks.
import json
import sys
import os
import re
from pathlib import Path
import feedparser
from urllib.parse import quote
from llama_cpp import Llama

COMPANY_QUERIES = { #for searching the company name or CEO name
    "NVDA": ["NVIDIA", "Jensen Huang"],
    "TSLA": ["Tesla", "Elon Musk"],
    "AMD": ["AMD","Lisa Su"],
    "AAPL": ["Apple", "Tim Cook"],
    "GOOG": ["Google", "Sundar Pichai"],
    "AMZN": ["Amazon", "Andy Jassy"],
    "WMT": ["Walmart", "John Furner"],
    "MSFT": ["Microsoft", "Satya Nadella"],
    "META": ["Meta", "Mark Zuckerberg"],
    "NFLX": ["Netflix", "Ted Sarandos & Greg Peters"],
    "BRK.A": ["Berkshire Hathaway", "Warren Buffett"],
    "JPM": ["JPMorgan Chase", "Jamie Dimon"],
    "LLY": ["Eli Lilly", "David Ricks"]
}

FETCH_NEWS = 30 #number of headlines analyzing
THESIS_HEADLINES = 10 #number of headlines provided to local model to generate thesis

MODEL_PATH = r"Qwen3-8B-Q4_K_M.gguf"

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_gpu_layers=-1,
    n_batch=512,
    n_threads=8,
    verbose=False
)


def safe_filename(text, max_len=60):

    text = re.sub(r'[<>:"/\\|?*]', "_", text)
    text = re.sub(r"\s+", "_", text)

    return text[:max_len]


def get_news(ticker_symbol, limit=50):

    stories = []
    seen_titles = set()

    queries = COMPANY_QUERIES.get(
        ticker_symbol.upper(),
        [ticker_symbol]
    )

    for query in queries:

        rss_url = (
            "https://news.google.com/rss/search?" #searching in google news
            f"q={quote(query)}"
            "&hl=en-US"
            "&gl=US"
            "&ceid=US:en"
        )

        feed = feedparser.parse(rss_url)

        print(
            f"{query}: "
            f"{len(feed.entries)} entries"
        )

        for entry in feed.entries:

            title = entry.get("title", "").strip()

            if not title:
                continue

            if title in seen_titles:
                continue

            seen_titles.add(title)

            stories.append({
                "title": title,
                "published":
                    entry.get("published", ""),
                "summary":
                    entry.get("summary", "")
            })

    return stories[:limit]


def parse_output(text):

    imp = re.search(
        r"IMPORTANCE\s*=\s*(-?\d+)",
        text,
        re.IGNORECASE
    )
    rel = re.search(
        r"RELEVANCE\s*=\s*(-?\d+)",
        text,
        re.IGNORECASE
    )
    sent = re.search(
        r"SENTIMENT\s*=\s*(-?\d+)",
        text,
        re.IGNORECASE
    )
    return {
        "importance":
            int(imp.group(1)) if imp else 1,
        "relevance":
            int(rel.group(1)) if rel else 1,
        "sentiment":
            int(sent.group(1)) if sent else 0
    }


def evaluate_headline(title, ticker):

    prompt = f"""
        You are a professional stock analyst.
        Analyze those headlines critically.

        Company:
        {ticker}

        Headline:
        {title}

        Determine Value:

        IMPORTANCE
        Major Specific Event, Meaningful Announces = 2
        Meaningful but Secondary News = 1
        Investor's Opinions, Speculation, Clickbait, Ads = 0
        Repeated or Simillar Headlines = 0
        

        RELEVANCE
        Relevant = 2
        Uncertain = 1
        Irrelevant = 0

        SENTIMENT
        Very Bullish = 2 : Likely materially increases future earnings.
        Bullish = 1 : Moderately positive.
        Neutral = 0 : No direct earnings impact or uncertain.
        Bearish = -1 : Moderately negative.
        Very Bearish = -2 : Likely materially decreases future earnings.

        Output ONLY:

        IMPORTANCE=<number>
        RELEVANCE=<number>
        SENTIMENT=<number>
    """

    result = llm(
        prompt,
        max_tokens=30,
        temperature=0
    )

    return result["choices"][0]["text"]

def load_headlines(ticker):

    path = (
        f"news_runs/{ticker}/"
        "headlines.json"
    )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)

def clean_output(text, ticker):
    text = re.split(r"\n\s*\nOkay[,.:]?", text, maxsplit=1)[0].strip()
    return f"{ticker} Analysis:\n\n{text}"

def select_thesis_headlines(ticker,count):

    headlines = load_headlines(
        ticker
    )

    headlines.sort(
        key=lambda x: (
            x["importance"] * 2 +
            x["relevance"] +
            abs(x["sentiment"])
        ),
        reverse=True
    )
    return headlines[:count]

def build_thesis_context(ticker):

    selected = (
        select_thesis_headlines(ticker, THESIS_HEADLINES)
    )

    lines = []

    for item in selected:

        lines.append(
            f"""
            Headline:
            {item['title']}

            Importance:
            {item['importance']}

            Relevance:
            {item['relevance']}

            Sentiment:
            {item['sentiment']}
            """
        )

    return "\n".join(lines)

def generate_market_thesis(ticker):

    context = (
        build_thesis_context(ticker)
    )

    #print(context)

    prompt = f"""
    You are an equity analyst.

        Return ONLY bullet points. No sentences outside bullets. No titles except numbers.

        COMPANY: {ticker}

        HEADLINES:
        {context}

        Format:

        1. Bullish factors:
        - bullet
        - bullet

        2. Bearish factors:
        - bullet
        - bullet

        3. Key risks:
        - bullet
        - bullet

        4. Short-term outlook:
        - bullet
        - bullet

        5. Long-term outlook:
        - bullet
        - bullet
    """

    #print(len(prompt))
    #print(prompt[:1000])
    #print(prompt[-1000:])

    result = llm( #to force meaningful output
        prompt,
        max_tokens=500,
        # temperature=0.3,
        # top_p=0.9
        temperature=0.2,
        top_p=0.85,
        repeat_penalty=1.1
    )
    #print(result)

    return result["choices"][0]["text"]

def headline_stats(ticker): #test function

    headlines = load_headlines(ticker)

    return sum(
        item["relevance"]
        * item["sentiment"]
        * item["importance"]
        for item in headlines
    )

def analyze_ticker(ticker):

    os.makedirs(
        f"news_runs/{ticker}",
        exist_ok=True
    )

    news = get_news(
        ticker,
        FETCH_NEWS
    )

    print(
        f"\nCollected and Analyzing "
        f"{len(news)} headlines\n"
    )

    analyzed = []

    for idx, item in enumerate(
        news,
        start=1
    ):

        title = item["title"]
        raw = evaluate_headline(
            title,
            ticker
        )
        scores = parse_output(raw)
        item_result = {
            "title": title,
            "importance": scores["importance"],
            "relevance": scores["relevance"],
            "sentiment": scores["sentiment"]
        }
        analyzed.append(item_result)

        print(
            f"[{idx}/{len(news)}] "
            f"{title}"
        )

    with open(
        f"news_runs/{ticker}/headlines.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            analyzed,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"\nSaved "
        f"{len(news)} headlines"
    )

if __name__ == "__main__":
    # sys.argv = ["script.py", "NVDA", "1"]
    # e.g. python AnalyzeTicker.py NVDA 1

    if len(sys.argv) < 3:
        print("Usage: python script.py <TICKER> <RUN_FLAG>")
        sys.exit(1)

    ticker = sys.argv[1]
    run_flag = sys.argv[2]

    if run_flag == "1":
        analyze_ticker(ticker)
    else:
        print(f"Skipping analysis for {ticker}")

    tickerscore = headline_stats(ticker)
    with open(f"scores.txt", "a") as f:
        f.write(f"{ticker}:{tickerscore}\n")
    print("score sum saved")
    # print("writing thesis...")
    # thesis = generate_market_thesis(ticker)
    # thesis = clean_output(thesis, ticker)

    # with open(
    #     f"news_runs/{ticker}/market_thesis.txt",
    #     "w",
    #     encoding="utf-8"
    # ) as f:

    #     f.write(thesis)
    print("done.")