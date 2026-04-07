# マニュアル
## 仮想環境の作り方

1. ターミナルを開く(Ctrl+Shift+@)

2. minicondaを開く
```
"C:\Users\kkyom\anaconda3\Scripts\activate"
"C:\Users\kkyom\anaconda3\envs"
C:\Users\kkyom\OneDrive\デスクトップ\pka_activity\venv312\Scripts\Activate.ps1
```

3. 仮想環境を作る
```
conda create -n <name> python=3.11 ←pythonのバージョンを指定
```

4. 仮想環境を有効化
```
conda activate <name>
```

5. 今回インストールするパッケージ

例1: kkyom(python3.12.12)の場合
```
conda install numpy
conda install pandas
conda install rdkit
conda install openpyxl
conda install ipykernel
conda isntall jupyter
```

例2: nnml(python3.11.14)の場合
```
conda install numpy
conda install pandas
conda install rdkit
conda install psi4 python=3.11 -c conda-forge 
```

※__注意__\
絶対にpsi4とipykernelは同時にインストールしないこと(何故か仮想環境が壊れて使えなくなる)(※1)

※1(2025_1031時点)\
何故かわからないが以下の通りにインストールするとjupyternotebookも使えるようになる\
jkl(python3.11.14)の場合をinstallした順番に示す
```
conda install notebook
conda install psi4 python=3.11 -c conda-forge/label/libint_dev -c conda-forge 
conda install pandas
conda install rdkit
conda install openpyxl
```

---
## phthalicacid.pyについて
```
import phthalicacid as pa

smiles_data, iso_data, cas_number = pa.samples_data()
# これでSMILES構造、その同位体、CASがリストで得られる

select = pa.select_smiles(smiles_data[0],iso[0])
# これでそのSMILES構造が有効かを判別し、SMILES構造を返す(塩を形成していたり、金属が含まれていたり、同位体がある場合はNoneが返される)

detect = pa.detect_phthalic_acid(select)
# これによりフタル酸のジカルボン酸構造のインデックスをリストで返す
(detect = [[1,3,0,2,18], [5,4,6,7,19]]
だった場合、detect[0],detect[1]がそれぞれのインデックスを表す。なお、それぞれ対応する原子としては、detect[0]であれば
1がC-"C"OOH,
3が"C"-COOH,
0がC-C"O"OH,
2がC-CO"O"H,
18がC-COO"H")

coord = pa.generate_coord(smiles_data[0],detect)
# これによりジカルボン酸構造が基準と同じで水も導入したxyz座標を返す


