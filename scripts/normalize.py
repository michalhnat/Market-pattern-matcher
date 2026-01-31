import argparse
import pandas as pd
from pathlib import Path

def z_score(df: pd.DataFrame):
    df = df.copy()
    cols = df.select_dtypes(include=['number']).columns
    for c in cols:
        if df[c].std() != 0:
            df[c] = (df[c] - df[c].mean()) / df[c].std()
    return df

def min_max(df: pd.DataFrame):
    df = df.copy()
    cols = df.select_dtypes(include=['number']).columns
    for c in cols:
        d = df[c].max() - df[c].min()
        if d != 0:
            df[c] = (df[c] - df[c].min()) / d
    return df

METHODS = {'z_score': z_score, 'min_max': min_max}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input")
    parser.add_argument("-o", "--output")
    parser.add_argument("-m", "--method", choices=METHODS.keys(), default="z_score")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    norm_df = METHODS[args.method](df)

    out = args.output
    if not out:
        p = Path(args.input)
        output_dir = Path(__file__).resolve().parent.parent / "data" / "normalized"
        output_dir.mkdir(parents=True, exist_ok=True)
        out = output_dir / f"{p.stem}_{args.method}{p.suffix}"

    norm_df.to_csv(out, index=False)

if __name__ == "__main__":
    main()
