from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


def screening(df: pd.DataFrame, name: str, known: bool = True) -> pd.DataFrame:
    """Screening the SMILES string for validity and suitability for further processing."""
    # df has information about the substances each row.
    # If known is True, the 'columns' parameter in df contains 'cas', 'smiles', 'yield', 'conversion', and 'selectivity'.
    # If known is False, the 'columns' parameter in df contains 'cas' and 'name' and 'smiles', 'isomeric smiles' and so on.
    for i in df.index:
        tf = True
        smiles = df.loc[i, "smiles"]
        # Extract the molecule from the SMILES using RDKit.
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            df.drop(index=i, inplace=True)
            continue
        # Add hydrogens to the molecule because SMILES don't contain hydrogen information by default.
        mol = Chem.AddHs(mol)

        # Exclude substances that form ions or salts.
        ejects = [".", "-", "+"]
        for eject in ejects:
            if eject in smiles:
                df.drop(index=i, inplace=True)
                tf = False
                break
        if not tf:
            continue

        # Exclude substances that can't be embedded in 3D space.
        if AllChem.EmbedMolecule(mol) == -1:
            df.drop(i, inplace=True)
            continue

    if known:
        df = df[
            [
                "cas",
                "smiles",
                "yield",
                "conversion",
                "selectivity",
                "oligomers",
                "fructose_mannose",
                "5-hmf",
                "levoglucosan",
                "furfural",
                "others",
            ]
        ]
        directory = Path(__file__).parent.parent / f"data/known/proceed/{name}.csv"
    elif not known:
        df = df[["cas", "smiles"]]
        directory = Path(__file__).parent.parent / f"data/unknown/proceed/{name}.csv"

    df.to_csv(directory, index=False)
    return df


def unknown_to_csv(file_path: str, limit: int = 10000):
    paths = [
        "501-1000",
        "1001-1500",
        "1501-2000",
        "2001-2500",
        "2501-3000",
        "3001-3500",
        "3501-4000",
        "4001-4500",
        "4501-5000",
        "5001-5500",
        "5501-6000",
        "6001-6500",
        "6501-7000",
        "7001-7500",
        "7501-8000",
        "8001-8500",
        "8501-9000",
        "9001-9500",
        "9501-10000",
    ]
    limit = limit / 500 - 1
    paths = paths[: int(limit)]
    directory = Path(__file__).parent.parent.parent / f"data/in/unknown/{file_path}"
    df = pd.read_excel(directory / "Substance_1-500.xlsx", header=4)
    for path in paths:
        df1 = pd.read_excel(directory / f"Substance_{path}.xlsx", header=4)
        df = pd.concat([df, df1], ignore_index=True)
    df = df.dropna(subset=[df.columns[1]]).reset_index(drop=True)
    df = df.rename(columns={df.columns[0]: "cas"})
    df = df.rename(columns={df.columns[1]: "smiles"})
    df.to_csv(
        Path(__file__).parent.parent / f"data/unknown/raw/{file_path}.csv", index=False
    )
