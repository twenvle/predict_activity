from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


def screening(
    df: pd.DataFrame, name: str, known: bool = True, isomer: bool = False
) -> pd.DataFrame:
    """Screening the SMILES string for validity and suitability for further processing."""
    for i in df.index:
        tf = True
        smiles = df.loc[i, "smiles"]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            df.drop(index=i, inplace=True)
            continue
        mol = Chem.AddHs(mol)

        # イオンや塩を形成しているものは除外
        ejects = [".", "-", "+"]
        for eject in ejects:
            if eject in smiles:
                df.drop(index=i, inplace=True)
                tf = False
                break
        if not tf:
            continue

        # 金属を含むものは除外
        for atom in mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            if (
                (3 <= atomic_num <= 4)
                or (11 < atomic_num <= 13)
                or (19 <= atomic_num <= 32)
                or (37 <= atomic_num <= 51)
                or (55 <= atomic_num <= 84)
                or (87 <= atomic_num <= 118)
            ):
                df.drop(i, inplace=True)
                tf = False
                break
        if not tf:
            continue

        # 同位体を含むものは除外
        if isomer:
            if type(df.loc[i, "iso"]) == str:
                iso = df.loc[i, "iso"]
                if "2H" in iso or "3H" in iso or "13C" in iso or "14C" in iso:
                    df.drop(i, inplace=True)
                    continue

        phthalic_pattern = Chem.MolFromSmarts("c1ccc(C(=O)[OH])c(C(=O)[OH])c1")
        matches = mol.GetSubstructMatches(phthalic_pattern)
        if not matches:
            df.drop(i, inplace=True)
            continue

        if AllChem.EmbedMolecule(mol) == -1:
            df.drop(i, inplace=True)
            continue
    if known:
        df = df[["cas", "smiles", "yield", "conversion", "selectivity"]]
        directory = Path(__file__).parent.parent / f"data/known/proceed/{name}.csv"
    elif not known:
        df = df[["cas", "smiles"]]
        directory = Path(__file__).parent.parent / f"data/unknown/proceed/{name}.csv"

    df.to_csv(directory, index=False)
    return df
