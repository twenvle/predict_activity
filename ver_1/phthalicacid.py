import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
import warnings
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")


names = [["basis1", "basis2"], ["basis_ion1", "basis_ion2"]]

df = []
water = []
for i in range(2):
    df1 = pd.read_csv(
        f"../sample/past/{names[i][0]}.txt", sep="\\s+", names=["atom", "x", "y", "z"]
    )
    df2 = pd.read_csv(
        f"../sample/past/{names[i][1]}.txt", sep="\\s+", names=["atom", "x", "y", "z"]
    )

    df1 = df1[["x", "y", "z"]].to_numpy()
    df2 = df2[["x", "y", "z"]].to_numpy()

    df1_basis = [
        df1[0],
        df1[6],
        df1[1],
        df1[10] - df1[6],
        df1[11] - df1[6],
        df1[17 - i] - df1[6],
    ]
    if i == 0:
        df2_basis = [
            df2[1],
            df2[7],
            df2[0],
            df2[9] - df2[7],
            df2[8] - df2[7],
            df2[16] - df2[7],
        ]
    elif i == 1:
        df2_basis = [
            df2[1],
            df2[7],
            df2[0],
            df2[9] - df2[7],
            df2[8] - df2[7],
        ]

    water_coords = [
        df2[18 - i],
        df2[19 - i],
        df2[20 - i],
        df2[21 - i],
        df2[22 - i],
        df2[23 - i],
    ]

    df.append([df1_basis, df2_basis])
    water.append(water_coords)


def samples_data(type="all"):
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
    df = pd.read_excel("sample/Substance_1-100.xlsx", header=4)
    for path in paths:
        df1 = pd.read_excel(f"sample/Substance_{path}.xlsx", header=4)
        df = pd.concat([df, df1], ignore_index=True)
    df = df.dropna(subset=[df.columns[2]]).reset_index(drop=True)
    if type == "smiles":
        return df[df.columns[2]]
    elif type == "isomer":
        return df[df.columns[3]]
    elif type == "cas":
        return df[df.columns[0]]
    elif type == "all":
        return df[df.columns[2]], df[df.columns[3]], df[df.columns[0]]


def select_smiles(smiles_data, iso_data=None, cas_data=None):
    mol = Chem.MolFromSmiles(smiles_data)

    # イオンを形成しているものは除外
    ejects = [".", "-", "+"]
    for eject in ejects:
        if eject in smiles_data:
            return None

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
            return None

    # 炭素や水素の同位体を含むものは除外
    if type(iso_data) == str:
        if (
            "2H" in iso_data
            or "3H" in iso_data
            or "13C" in iso_data
            or "14C" in iso_data
        ):
            return None

    # カルボキシ基が二つ以上あるかを確認
    carboxylic_acid = Chem.MolFromSmarts("C(=O)[OH]")
    acid_num = len(mol.GetSubstructMatches(carboxylic_acid))
    if acid_num >= 2:
        return smiles_data


def detect_phthalic_acid(smiles_data):
    mol = Chem.MolFromSmiles(smiles_data)
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
        return cooh_list
    else:
        return None


def generate_coord(smiles_data, cooh_list):
    c_cooh1 = cooh_list[0][0]
    c_cooh2 = cooh_list[1][0]
    c_benzene1 = cooh_list[0][1]
    c_benzene2 = cooh_list[1][1]

    cooh_idx = [c_cooh1, c_cooh2]
    benzene_idx = [c_benzene2, c_benzene1]

    # 目的の化合物とそのイオン化した二つの構造の座標を構築
    fix = []
    geom = []
    for i in range(2):
        mol = Chem.MolFromSmiles(smiles_data)
        mol = Chem.AddHs(mol)  # 水素を追加

        AllChem.EmbedMolecule(mol)  # 3次元座標を自動で生成
        # 3D構造が作れない場合は除外
        if AllChem.EmbedMolecule(mol) == -1:
            return None
        AllChem.UFFOptimizeMolecule(mol)  # エネルギー最小化

        if i == 0:
            conf = mol.GetConformer()
            coordinates = []
            for atom in mol.GetAtoms():
                pos = conf.GetAtomPosition(atom.GetIdx())
                coordinates.append([pos.x, pos.y, pos.z])
            coordinates = np.array(coordinates)

        # イオンの場合はカルボン酸の水素を一つ除く
        elif i == 1:
            mol_new = Chem.RWMol(mol)
            mol_new.RemoveAtom(cooh_list[1][4])
            mol = mol_new.GetMol()
            conf = mol.GetConformer()

            if cooh_list[0][4] > cooh_list[1][4]:
                cooh_list[0][4] -= 1

            coordinates = []
            for atom in mol.GetAtoms():
                pos = conf.GetAtomPosition(atom.GetIdx())
                coordinates.append([pos.x, pos.y, pos.z])
            coordinates = np.array(coordinates)

        # 目的化合物のジカルボン酸の座標を基準値と一致させる
        for j in range(2):
            r = [coordinates[c_benzene1], coordinates[c_benzene2]]

            # カルボン酸と結合しているベンゼン環の炭素を原点とする
            t = -r[j]
            coords = np.array([coordinates[k] + t for k in range(mol.GetNumAtoms())])
            # カルボン酸の炭素をx軸上に置く
            x_axis = coords[cooh_idx[j]].copy()
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
            r3 = coords[benzene_idx[j]].copy()
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
            a = df[i][j][1].copy()
            a /= np.linalg.norm(a)
            b = coords[cooh_idx[j]].copy()
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
            c = df[i][j][2].copy()
            c /= np.linalg.norm(c)
            d = coords[benzene_idx[j]]
            d = d / np.linalg.norm(d).copy()

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

            df_ij = df[i][j].copy()
            # ベンゼン環の炭素とカルボン酸の炭素の結合の長さを基準値に揃える
            coords[cooh_list[j][0]] = df_ij[1]

            # OOHについては原点からのベクトルを表すようにする
            df_ij += df_ij[1]

            if i == 1 and j == 1:
                for k in range(2, 4):
                    coords[cooh_list[j][k]] = df_ij[k + 1]

            else:
                for k in range(2, 5):
                    coords[cooh_list[j][k]] = df_ij[k + 1]

            if j == 1:
                diff = df[i][j][2] - coords[cooh_list[0][1]]
                for k in cooh_list[0]:
                    coords[k] += diff

            for k in range(mol.GetNumAtoms()):
                conf.SetAtomPosition(k, Point3D(*coords[k]))

            if j == 1:
                mol = Chem.RWMol(mol)
                for k in range(2):
                    o = mol.AddAtom(Chem.Atom("O"))
                    h1 = mol.AddAtom(Chem.Atom("H"))
                    h2 = mol.AddAtom(Chem.Atom("H"))
                    conf = mol.GetConformer()
                    conf.SetAtomPosition(o, water[i][k * 3])
                    conf.SetAtomPosition(h1, water[i][k * 3 + 1])
                    conf.SetAtomPosition(h2, water[i][k * 3 + 2])

            coordinates = coords
        fix_idx = [x + 1 for row in cooh_list for x in row]
        if i == 1:
            fix_idx.pop()

        fix_text = "\nconstraints {\n"
        for idx in fix_idx:
            fix_text += f"    fix atom {idx}\n"
        fix_text += "}\n"
        fix.append(fix_text)

        geom_text = "0 1\n"
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            geom_text += f"{atom.GetSymbol()} {pos.x:.4f} {pos.y:.4f} {pos.z:.4f}\n"
        geom.append(geom_text)

    return geom, fix
