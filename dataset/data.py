from pathlib import Path
import pandas as pd

def load_dataset(path):
    df = pd.read_csv(Path(path))
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("CSV must have text,label columns.")
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df = df[df.label.isin(["ham", "spam"])]
    df = df[df.text.str.len() > 0].drop_duplicates("text")
    if df.label.nunique() != 2:
        raise ValueError("Dataset must contain both ham and spam labels.")
    return df.reset_index(drop=True)
