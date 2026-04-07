import numpy as np
import pandas as pd
import rdkit.Chem as Chem
import rdkit.Chem as AllChem
from rdkit.Geometry import Point3D

df1 = pd.read_csv("sample/basis1.txt", sep="\\s+", names=["atom", "x", "y", "z"])
df2 = pd.read_csv("sample/basis2.txt", sep="\\s+", names=["atom", "x", "y", "z"])
df1 = df1[["x", "y", "z"]].to_numpy()
df2 = df2[["x", "y", "z"]].to_numpy()


smiles = "CCC"  # 例としてプロパン
mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol, randomSeed=42)

conf = mol.GetConformer()


# --- 2. 原子の座標を取得 ---
def get_pos(i):
    p = conf.GetAtomPosition(i)
    return np.array([p.x, p.y, p.z])


r1 = get_pos(0)  # C1
r2 = get_pos(1)  # C2
r3 = get_pos(2)  # C3

# --- 3. 平行移動: C1を原点に ---
R = np.eye(3)
t = -r1
coords = np.array([get_pos(i) + t for i in range(mol.GetNumAtoms())])

# --- 4. 回転: C2をx軸上に ---
x_vec = coords[1]  # C2のベクトル
x_axis = x_vec / np.linalg.norm(x_vec)

# z軸との外積で回転軸を求める
z = np.array([0, 0, 1])
axis = np.cross(x_axis, np.array([1, 0, 0]))
axis_norm = np.linalg.norm(axis)

if axis_norm > 1e-8:
    axis /= axis_norm
    # 角度を求める
    theta = np.arccos(np.dot(x_axis, [1, 0, 0]))
    # 回転行列（ロドリゲスの回転公式）
    c, s = np.cos(theta), np.sin(theta)
    K = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    R = np.eye(3) + s * K + (1 - c) * (K @ K)
    coords = coords @ R.T

# --- 5. 回転: C3をxy平面上に（z成分を0にする） ---
r3_new = coords[2]
# C3 の z 成分を 0 にするための回転軸は x 軸方向
angle_z = np.arctan2(r3_new[2], r3_new[1])
c, s = np.cos(-angle_z), np.sin(-angle_z)
R2 = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
coords = coords @ R2.T

# --- 6. 新しい座標を反映 ---
for i in range(mol.GetNumAtoms()):
    conf.SetAtomPosition(i, Point3D(*coords[i]))

# --- 7. 結果確認 ---
for i in range(3):  # C1, C2, C3 のみ表示
    p = conf.GetAtomPosition(i)
    print(f"C{i+1}: ({p.x:.3f}, {p.y:.3f}, {p.z:.3f})")
