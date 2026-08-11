import argparse
from pathlib import Path


def download_hourly_close(
    output: str | Path,
    ticker: str = "ETH-USD",
    period: str = "60d",
) -> int:
    """Download hourly Close data from Yahoo Finance into pipeline CSV format.

    Yahoo Finance limits intraday history; keep the default period conservative
    and collect longer experiments in rolling downloads when needed.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Install dependencies first: pip install -r requirements.txt") from exc

    frame = yf.download(
        tickers=ticker,
        period=period,
        interval="1h",
        auto_adjust=False,
        progress=False,
        group_by="column",
        multi_level_index=False,
    )
    if frame is None or frame.empty or "Close" not in frame:
        raise RuntimeError(f"No hourly Close data returned for {ticker}")

    close = frame["Close"]
    if hasattr(close, "columns"):
        close = close.iloc[:, 0]
    result = close.rename("close").dropna().to_frame()
    result.index.name = "timestamp"
    result = result.reset_index()
    result["timestamp"] = result["timestamp"].astype(str)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    return len(result)


def main():
    parser = argparse.ArgumentParser(description="Download ETH/USD hourly Close data")
    parser.add_argument("--output", default="data/eth_usd_1h.csv")
    parser.add_argument("--ticker", default="ETH-USD")
    parser.add_argument("--period", default="60d", help="Yahoo Finance period, e.g. 60d or 730d")
    args = parser.parse_args()
    count = download_hourly_close(args.output, args.ticker, args.period)
    print(f"Saved {count} rows to {args.output}")


if __name__ == "__main__":
    main()
