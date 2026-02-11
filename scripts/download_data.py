import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf


def main() -> None:
    parser = argparse.ArgumentParser()

    project_root = Path(__file__).resolve().parent.parent
    default_path = project_root / "data" / "raw"

    parser.add_argument("--tickers", nargs="+", required=True)
    parser.add_argument("--period", default="max")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--path", type=Path, default=default_path)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    args.path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading data for: {args.tickers}")
    data = yf.download(
        tickers=args.tickers,
        period=args.period,
        interval=args.interval,
        start=args.start,
        end=args.end,
        group_by="ticker",
        auto_adjust=True,
    )

    if data.empty:
        print("No data downloaded.")
        return

    for ticker in args.tickers:
        ticker_upper = ticker.upper()

        try:
            if isinstance(data.columns, pd.MultiIndex):
                if ticker_upper in data.columns.get_level_values(0):
                    ticker_data = data[ticker_upper]
                else:
                    print(f"Warning: No data found for {ticker}")
                    continue
            else:
                ticker_data = data
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue

        if not ticker_data.empty:
            file_path = args.path / f"{ticker_upper}.csv"
            ticker_data.to_csv(file_path)
            print(f"Saved {ticker} to {file_path}")
        else:
            print(f"Warning: Data for {ticker} is empty")


if __name__ == "__main__":
    main()
