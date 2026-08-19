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

        # Exclude substances that contain metals.
        for atom in mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            if (
                (3 <= atomic_num <= 4)
                or (11 <= atomic_num <= 13)
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

        # Exclude substances that contain isotopes.
        if not known:
            if type(df.loc[i, "iso"]) == str:
                iso = df.loc[i, "iso"]
                if "2H" in iso or "3H" in iso or "13C" in iso or "14C" in iso:
                    df.drop(i, inplace=True)
                    continue

        # Exclude substances that don't contain phthalic acid.
        phthalic_pattern = Chem.MolFromSmarts("c1ccc(C(=O)[OH])c(C(=O)[OH])c1")
        matches = mol.GetSubstructMatches(phthalic_pattern)
        if not matches:
            df.drop(i, inplace=True)
            continue

        # Exclude substances that can't be embedded in 3D space.
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
