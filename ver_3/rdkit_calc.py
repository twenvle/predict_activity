import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import phthalicacid as pa


def sub_idx(df):
    df = pa.detect_phthalic_acid(df, benzene=True)
    df["3and6"] = 0
    df["4and5"] = 0
    for i in range(len(df)):
        smiles = df.loc[i, "smiles"]
        c_acid1 = df.loc[i, "cooh"][0][0]
        c_acid2 = df.loc[i, "cooh"][1][0]
        c_benzene1 = df.loc[i, "cooh"][0][1]
        c_benzene2 = df.loc[i, "cooh"][1][1]
        benzene_tuple = df.loc[i, "benzene"]
        benzene_list = [c_acid2, c_acid1, c_benzene2, c_benzene1]

        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        while len(benzene_list) != 8:
            atom = mol.GetAtomWithIdx(benzene_list[-1])
            neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
            for neighbor in neighbors:
                if neighbor not in benzene_list and neighbor in benzene_tuple:
                    benzene_list.append(neighbor)

        idx = ["3and6", "4and5", "4and5", "3and6"]
        for j in range(4, 8):
            atom = mol.GetAtomWithIdx(benzene_list[j])
            neighbors = [n.GetSymbol() for n in atom.GetNeighbors()]
            if "H" in neighbors:
                df.loc[i, f"{idx[j-4]}"] += 0
            else:
                df.loc[i, f"{idx[j-4]}"] += 1
        """
        if df.loc[i, "3rd"] < df.loc[i, "6th"]:
            df.loc[i, "3rd"], df.loc[i, "6th"] = df.loc[i, "6th"], df.loc[i, "3th"]
        if df.loc[i, "4th"] < df.loc[i, "5th"]:
            df.loc[i, "4th"], df.loc[i, "5th"] = df.loc[i, "5th"], df.loc[i, "4th"]
        """
    df.drop(columns=["cooh", "benzene"], inplace=True)
    return df
