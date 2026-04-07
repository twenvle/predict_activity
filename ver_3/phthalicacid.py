import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
import warnings
from rdkit import RDLogger
import os
import cclib
import glob
from pathlib import Path


RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")


names = [["basis1", "basis2"], ["basis_ion1", "basis_ion2"]]

df0 = pd.read_csv(
    f"../sample/basis_data/basis.txt", sep="\\s+", names=["atom", "x", "y", "z"]
)

df0 = df0[["x", "y", "z"]].to_numpy()

df0_basis = [
    df0[0],
    df0[6],
    df0[10],
    df0[11],
    df0[17],
    df0[7],
    df0[1],
    df0[9],
    df0[8],
    df0[16],
]


def samples_data():
    paths = [
        "101-200",
        "201-300",
        "301-400",
        "401-500",
        "501-600",
        "601-700",
        "701-800",
        "801-900",
        "901-1000",
        "1001-1100",
        "1101-1200",
        "1201-1300",
        "1301-1400",
        "1401-1500",
        "1501-1600",
        "1601-1700",
        "1701-1800",
        "1801-1900",
        "1901-2000",
        "2001-2100",
        "2101-2200",
        "2201-2300",
        "2301-2400",
        "2401-2500",
        "2501-2600",
        "2601-2700",
        "2701-2800",
        "2801-2900",
        "2901-3000",
        "3001-3100",
        "3101-3200",
        "3201-3300",
        "3301-3400",
        "3401-3500",
        "3501-3600",
        "3601-3700",
    ]
    df = pd.read_excel("../sample/data/Substance_1-100.xlsx", header=4)
    for path in paths:
        df1 = pd.read_excel(f"../sample/data/Substance_{path}.xlsx", header=4)
        df = pd.concat([df, df1], ignore_index=True)
    df = df.dropna(subset=[df.columns[2]]).reset_index(drop=True)
    return pd.concat(
        [df[df.columns[0]], df[df.columns[2]], df[df.columns[3]]],
        axis=1,
        keys=["cas", "smiles", "iso"],
    )


def select_smiles(df):
    for i in range(len(df)):
        tf = False
        smiles = df.loc[i, "smiles"]
        if "iso" in df.columns:
            iso = df.loc[i, "iso"]
        mol = Chem.MolFromSmiles(smiles)

        # イオンを形成しているものは除外
        ejects = [".", "-", "+"]
        for eject in ejects:
            if eject in smiles:
                df.drop(i, inplace=True)
                tf = True
                break

        if tf:
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
                tf = True
                break

        if tf:
            continue

        # 炭素や水素の同位体を含むものは除外
        if "iso" in df.columns:
            if type(iso) == str:
                if "2H" in iso or "3H" in iso or "13C" in iso or "14C" in iso:
                    df.drop(i, inplace=True)
                    continue

        # カルボキシ基が二つ以上あるかを確認
        carboxylic_acid = Chem.MolFromSmarts("C(=O)[OH]")
        acid_num = len(mol.GetSubstructMatches(carboxylic_acid))
        if acid_num < 2:
            df.drop(i, inplace=True)

    return df.reset_index(drop=True)


def detect_phthalic_acid(df, benzene=False):
    # いきなりdf.atは使えないので一旦Noneで初期化
    df["cooh"] = None
    if benzene:
        df["benzene"] = None
    for i in range(len(df)):
        smiles = df.loc[i, "smiles"]
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)  # 水素を追加
        phthalic_pattern = Chem.MolFromSmarts("c1ccc(C(=O)O)c(C(=O)O)c1")

        # フタル酸構造を持つかどうか
        matches_data = mol.GetSubstructMatches(phthalic_pattern)

        oh_list = []
        cooh_list = []

        if matches_data:
            num = 0
            while True:
                # 2つ以上フタル酸構造を含まれていることを考慮して一つだけ検出
                matches = matches_data[num]
                for match in matches:
                    atom = mol.GetAtomWithIdx(match)  # その番号の原子の情報
                    symbol = atom.GetSymbol()  # その番号の原子

                    # まだこの時点では水素の場所は検出されていないのでカルボン酸の水素を検出
                    if symbol == "O":
                        neighbors = atom.GetNeighbors()  # その原子に隣接する原子の情報
                        elements = [n.GetSymbol() for n in neighbors]
                        if "H" in elements and "C" in elements:
                            for neighbor in neighbors:
                                if neighbor.GetSymbol() == "H":
                                    oh_list.append((atom.GetIdx(), neighbor.GetIdx()))

                # 片側もしくは両側がエステルになっている場合を考慮
                if len(oh_list) < 2:
                    num += 1
                    oh_list = []
                    continue

                idx = matches

                for match in matches:
                    atom = mol.GetAtomWithIdx(match)
                    symbol = atom.GetSymbol()
                    # カルボン酸の炭素を起点に他の原子を検出
                    if symbol == "C":
                        neighbors = atom.GetNeighbors()
                        elements = [n.GetSymbol() for n in neighbors]
                        if elements.count("O") >= 2:
                            for neighbor in neighbors:
                                if neighbor.GetSymbol() == "C":
                                    c = neighbor.GetIdx()
                                elif neighbor.GetSymbol() == "O":
                                    if len(neighbor.GetNeighbors()) == 1:
                                        o_double = neighbor.GetIdx()
                                    elif len(neighbor.GetNeighbors()) == 2:
                                        for oh in oh_list:
                                            if oh[0] == neighbor.GetIdx():
                                                o_single = oh[0]
                                                h = oh[1]
                            # 自身の炭素原子、それに結合しているベンゼン環内の炭素原子、
                            # 炭素と2重結合している酸素原子、炭素と単結合している酸素原子、水素の順
                            cooh_list.append([atom.GetIdx(), c, o_double, o_single, h])
                break
            df.at[i, "cooh"] = cooh_list
            if benzene:
                df.at[i, "benzene"] = idx
        else:
            df.drop(i, inplace=True)
    return df.reset_index(drop=True)


def generate_coord(df, divide=100):
    df_length = len(df)
    div = divide
    if div > df_length:
        div = df_length
    num = df_length // div
    rem = df_length % div
    if rem != 0:
        num += 1
    for n in range(num):
        all_sh = ""
        if n == num - 1 and rem != 0:
            div = rem
        os.makedirs(f"out/samples_{n*divide+1}-{n*divide+div}/logfile", exist_ok=True)
        for i in range(div):
            geometory = ""
            smiles = df.loc[i + n * divide, "smiles"]
            cooh = df.loc[i + n * divide, "cooh"]
            cas = df.loc[i + n * divide, "cas"]
            c_cooh = cooh[0][0]
            c_benzene1 = cooh[0][1]
            c_benzene2 = cooh[1][1]

            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)  # 水素を追加

            AllChem.EmbedMolecule(mol)  # 3次元座標を自動で生成
            # 3D構造が作れない場合は除外
            if AllChem.EmbedMolecule(mol) == -1:
                continue
            AllChem.UFFOptimizeMolecule(mol)  # エネルギー最小化

            conf = mol.GetConformer()
            coordinates = []
            for atom in mol.GetAtoms():
                pos = conf.GetAtomPosition(atom.GetIdx())
                coordinates.append([pos.x, pos.y, pos.z])
            coordinates = np.array(coordinates)

            # 目的化合物のジカルボン酸の座標を基準値と一致させる
            r = coordinates[c_benzene1]

            # カルボン酸と結合しているベンゼン環の炭素を原点とする
            t = -r
            coords = np.array([coordinates[k] + t for k in range(mol.GetNumAtoms())])
            # カルボン酸の炭素をx軸上に置く
            x_axis = coords[c_cooh].copy()
            x_axis /= np.linalg.norm(x_axis)

            x = np.array([1, 0, 0])
            axis = np.cross(x_axis, x)
            axis_norm = np.linalg.norm(axis)

            if axis_norm > 1e-8:
                axis /= axis_norm
                theta = np.arccos(np.dot(x_axis, x))
                c, s = np.cos(theta), np.sin(theta)
                K = np.array(
                    [
                        [0, -axis[2], axis[1]],
                        [axis[2], 0, -axis[0]],
                        [-axis[1], axis[0], 0],
                    ]
                )
                R = np.eye(3) + s * K + (1 - c) * (K @ K)
                coords = coords @ R.T

            # 隣のカルボン酸と結合しているベンゼン環の炭素をxy平面上に置く
            r3 = coords[c_benzene2].copy()
            angle_z = np.arctan2(r3[2], r3[1])
            c, s = np.cos(-angle_z), np.sin(-angle_z)
            R2 = np.array(
                [
                    [1, 0, 0],
                    [0, c, -s],
                    [0, s, c],
                ]
            )
            coords = coords @ R2.T

            # x軸上で逆向きになっていた場合回転させる
            a = df0_basis[1].copy()
            a /= np.linalg.norm(a)
            b = coords[c_cooh].copy()
            b = b / np.linalg.norm(b)

            cos_theta1 = np.dot(a, b)
            if not np.isclose(cos_theta1, 1.0, atol=1e-1):
                Ry = np.array(
                    [
                        [-1, 0, 0],
                        [0, -1, 0],
                        [0, 0, 1],
                    ]
                )
                coords = coords @ Ry.T

            # xy平面上で逆向きになっていた場合回転させる
            c = df0_basis[6].copy()
            c /= np.linalg.norm(c)
            d = coords[c_benzene2].copy()
            d = d / np.linalg.norm(d)

            cos_theta2 = np.dot(c, d)
            if not np.isclose(cos_theta2, 1.0, atol=1e-1):
                Rx = np.array(
                    [
                        [1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1],
                    ]
                )
                coords = coords @ Rx.T

            fix_idx = [x for row in cooh for x in row]
            # ベンゼン環の炭素とカルボン酸の炭素の結合の長さを基準値に揃える
            coords[c_cooh] = df0_basis[1]
            for j in range(2, 10):
                coords[fix_idx[j]] = df0_basis[j]

            for k in range(mol.GetNumAtoms()):
                conf.SetAtomPosition(k, Point3D(*coords[k]))

            for atom in mol.GetAtoms():
                pos = conf.GetAtomPosition(atom.GetIdx())
                geometory += f"{atom.GetSymbol()} {pos.x} {pos.y} {pos.z}\n"
            with open(
                f"out/samples_{n*divide+1}-{n*divide+div}/sub_{cas}.gjf",
                "w",
                newline="\n",
            ) as f:
                f.write("%mem=4GB\n")
                f.write(f"%chk=sub_{cas}.chk\n")
                f.write(
                    "#p opt freq rcam-b3lyp/6-311+g(d,p) scrf=(smd,solvent=water) pop=nboread\n"
                )
                f.write("\n")
                f.write("no comm\n")
                f.write("\n")
                f.write("0 1\n")
                f.write(geometory)
                f.write("\n")
                f.write("$nbo bndidx $end")
                f.write("\n")

            with open(
                f"out/samples_{n*divide+1}-{n*divide+div}/sub_{cas}.sh",
                "w",
                newline="\n",
            ) as f1:
                f1.write("#!/bin/bash\n")
                f1.write("#$ -cwd\n")
                f1.write("#$ -l cpu_4=1\n")
                f1.write("#$ -l h_rt=23:00:00\n")
                f1.write("#$ -V\n")
                f1.write("\n")
                f1.write(". /etc/profile.d/modules.sh\n")
                f1.write("module load gaussian\n")
                f1.write("\n")
                f1.write(f"g16 sub_{cas}.gjf && formchk sub_{cas}.chk sub_{cas}.fchk\n")
                f1.write("\n")
                f1.write(f"count=$(grep -c 'Normal termination' sub_{cas}.log)\n")
                f1.write("\n")
                f1.write("if [ $count -eq 2 ]; then\n")
                f1.write("\n")
                f1.write(
                    f"    if grep 'Frequencies -- ' sub_{cas}.log | grep -q '\\-[0-9]'; then\n"
                )
                f1.write(f"        mv sub_{cas}.log sub_{cas}_imag.log\n")
                f1.write(f"        mv sub_{cas}.fchk sub_{cas}_imag.fchk\n")
                f1.write("    fi\n")
                f1.write("\n")
                f1.write(f"    mv sub_{cas}.log logfile/\n")
                f1.write("\n")
                f1.write("fi")

            # shファイルは最後に空行があってはいけない
            if i != div - 1:
                all_sh += f"qsub -g tga-ynabae sub_{cas}.sh\n"
            else:
                all_sh += f"qsub -g tga-ynabae sub_{cas}.sh"

        with open(
            f"out/samples_{n*divide+1}-{n*divide+div}/all.sh", "w", newline="\n"
        ) as f:
            f.write(all_sh)


def save_data(df, name):
    os.makedirs("out/save", exist_ok=True)
    df["h1"] = None
    df["h2"] = None
    for i in range(len(df)):
        df.loc[i, "h1"] = df.loc[i, "cooh"][1][4]
        df.loc[i, "h2"] = df.loc[i, "cooh"][0][4]
    df = pd.concat([df["cas"], df["smiles"], df["h1"], df["h2"]], axis=1)
    df.to_csv(f"out/save/{name}.csv", index=False)


atom = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    15: "P",
    16: "S",
    17: "Cl",
    35: "Br",
    53: "I",
}


def make_ion_file(df, path):
    path_in = f"in/{path}"
    path_out = f"out/{path}"
    os.makedirs(f"{path_out}_ion", exist_ok=True)
    os.makedirs(f"{path_out}_ion/logfile", exist_ok=True)
    all_sh = ""
    for i in range(len(df)):
        geometory = ""
        cas = df.loc[i, "cas"]
        cooh = df.loc[i, "h1"]
        path_data = Path(path_in)
        logfile = list(path_data.glob(f"**/sub_{cas}.log"))
        if not logfile:
            continue
        logfile = logfile[0]
        logfile = str(logfile)
        data = cclib.io.ccread(logfile)
        coords = data.atomcoords[-1]
        atom_nums = data.atomnos
        for j, atom_num, (x, y, z) in zip(range(len(atom_nums)), atom_nums, coords):
            if j == cooh:
                continue
            geometory += f"{atom[atom_num]} {x} {y} {z}\n"
        with open(f"{path_out}_ion/sub_{cas}_ion.gjf", "w", newline="\n") as f:
            f.write("%mem=4GB\n")
            f.write(f"%chk=sub_{cas}_ion.chk\n")
            f.write(
                "#p opt freq rcam-b3lyp/6-311+g(d,p) scrf=(smd,solvent=water) pop=nboread\n"
            )
            f.write("\n")
            f.write("no comm\n")
            f.write("\n")
            f.write("-1 1\n")
            f.write(geometory)
            f.write("\n")
            f.write("$nbo bndidx $end")
            f.write("\n")
        with open(f"{path_out}_ion/sub_{cas}_ion.sh", "w", newline="\n") as f1:
            f1.write("#!/bin/bash\n")
            f1.write("#$ -cwd\n")
            f1.write("#$ -l cpu_4=1\n")
            f1.write("#$ -l h_rt=23:00:00\n")
            f1.write("#$ -V\n")
            f1.write("\n")
            f1.write(". /etc/profile.d/modules.sh\n")
            f1.write("module load gaussian\n")
            f1.write("\n")
            f1.write(
                f"g16 sub_{cas}_ion.gjf && formchk sub_{cas}_ion.chk sub_{cas}_ion.fchk\n"
            )
            f1.write("\n")
            f1.write(f"count=$(grep -c 'Normal termination' sub_{cas}_ion.log)\n")
            f1.write("\n")
            f1.write("if [ $count -eq 2 ]; then\n")
            f1.write("\n")
            f1.write(
                f"    if grep 'Frequencies -- ' sub_{cas}_ion.log | grep -q '\\-[0-9]'; then\n"
            )
            f1.write(f"        mv sub_{cas}_ion.log sub_{cas}_ion_imag.log\n")
            f1.write(f"        mv sub_{cas}_ion.fchk sub_{cas}_ion_imag.fchk\n")
            f1.write("    fi\n")
            f1.write("\n")
            f1.write(f"    mv sub_{cas}_ion.log logfile/\n")
            f1.write("\n")
            f1.write("fi")
        if i != len(df) - 1:
            all_sh += f"qsub -g tga-ynabae sub_{cas}_ion.sh\n"
        else:
            all_sh += f"qsub -g tga-ynabae sub_{cas}_ion.sh"
    with open(f"{path_out}_ion/all.sh", "w", newline="\n") as f:
        f.write(all_sh)
