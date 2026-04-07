import pandas as pd
from . import phthalicacid as pa
from . import rdkit_calc as rc
from . import dft_calc as dc
from pathlib import Path


def make_csv(name=[]):
    df = pd.read_csv(
        rf"C:\Users\kkyom\OneDrive\デスクトップ\pka_activity\ver_4\out\csvfile\unknown\unknown.csv"
    )
    filepath = Path(
        r"C:\Users\kkyom\OneDrive\デスクトップ\pka_activity\ver_4\out\csvfile\unknown\unknown_data.csv"
    )
    df = pa.detect_phthalic_acid(df, benzene=True)
    for n in name:
        n_start, n_end = n.split("-")
        n_start, n_end = int(n_start) - 1, int(n_end)
        df_sub = df.iloc[n_start:n_end].copy().reset_index(drop=True)

        df_sub = dc.get_data(df_sub, path=f"unknown/{n}/logfile")
        df_sub = rc.sub_idx(df_sub)
        df_sub = rc.get_descriptors(df_sub)
        df_sub.drop(columns=["cooh", "benzene"], inplace=True)

        if not filepath.exists():
            df_sub.to_csv(
                r"C:\Users\kkyom\OneDrive\デスクトップ\pka_activity\ver_4\out\csvfile\unknown\unknown_data.csv",
                index=False,
            )
        else:
            df_basis = pd.read_csv(
                rf"C:\Users\kkyom\OneDrive\デスクトップ\pka_activity\ver_4\out\csvfile\unknown\unknown_data.csv"
            )
            df_concat = pd.concat([df_basis, df_sub], ignore_index=True)
            df_concat.to_csv(
                r"C:\Users\kkyom\OneDrive\デスクトップ\pka_activity\ver_4\out\csvfile\unknown\unknown_data.csv",
                index=False,
            )
