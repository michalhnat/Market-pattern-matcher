import argparse
import pandas as pd
import numpy as np
from pathlib import Path

def process_file(file_path: Path, window_size: int, method_name: str, output_base: Path):
    if not file_path.exists():
        print(f"Error: File {file_path} does not exist.")
        return

    try:
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    if 'Close' not in df.columns:
        print(f"Error: 'Close' column not found in {file_path}. Columns: {df.columns}")
        return
    
    prices = df[["Open","High","Low","Close"]]
    volume = df["Volume"]

    dates = df.index
    
    ticker = file_path.stem
    
    all_windows = []
    all_metadata = []
    normalization_params = []
    
    print(f"Processing {ticker} with window size {window_size}...")
    
    for i in range(len(prices) - window_size + 1):
        window = prices[i : i + window_size]
        volume_window = volume[i : i + window_size]
        window_dates = dates[i : i + window_size]
        
        pmin = window.values.min()
        pmax = window.values.max()
        
        if pmax - pmin == 0:
            continue
            
        normalized_window = (window - pmin) / (pmax - pmin)
        vol = np.log1p(volume_window.values.astype(np.float64))
        vmin, vmax = vol.min(), vol.max()

        normalized_vol = (vol - vmin) / (vmax - vmin) if vmax > vmin else np.zeros(window_size)
        
        result = np.stack([
            normalized_window["Open"].values,
            normalized_window["High"].values,
            normalized_window["Low"].values,
            normalized_window["Close"].values,
            normalized_vol,
        ])  # (5, window_size)
        
        all_windows.append(result)
        
        start_date = window_dates[0]
        end_date = window_dates[-1]
        
        if hasattr(start_date, 'date'):
            start_date_str = str(start_date.date())
        else:
            start_date_str = str(start_date).split()[0]
            
        if hasattr(end_date, 'date'):
            end_date_str = str(end_date.date())
        else:
            end_date_str = str(end_date).split()[0]

        all_metadata.append({
            "ticker": ticker,
            "start_date": start_date_str,
            "end_date": end_date_str
        })

        normalization_params.append({
            "ticker": ticker,
            "start_date": start_date_str,
            "end_date": end_date_str,
            "pmin": pmin,
            "pmax": pmax,
            "vmin": vmin,
            "vmax": vmax
        })
        
    if not all_windows:
        print(f"No valid windows found for {ticker} (data length: {len(prices)}).")
        return

    X = np.array(all_windows, dtype=np.float32)
    df_meta = pd.DataFrame(all_metadata)
    df_norm = pd.DataFrame(normalization_params)
    
    folder_name = f"{ticker}_{method_name}_{window_size}"
    output_dir = output_base / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data_path = output_dir / "data.npy"
    meta_path = output_dir / "metadata.csv"
    norm_path = output_dir / "normalization_params.csv"
    
    np.save(data_path, X)
    df_meta.to_csv(meta_path, index=False)
    df_norm.to_csv(norm_path, index=False)
    
    print(f"Data saved to: {data_path}")
    print(f"Metadata saved to: {meta_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("-m", "--method", default="minmax", choices=["minmax"])
    
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parent.parent
    output_base = project_root / "data" / "preprocessed"
    
    process_file(args.input, args.window_size, args.method, output_base)

if __name__ == "__main__":
    main()
