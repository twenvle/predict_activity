# 操作マニュアル
## 仮想環境の作り方
1. ターミナルを開く(Ctrl+Shiht+@)

2. 仮想環境を作りたいフォルダに移動

3. 仮想環境を作る

```
py -3.12 -m venv venv312
```
4. 仮想環境を有効化する
```
venv312\Scripts\Activate.ps1
```
---
## 新しくGitHubへ保存
1. GitHubで新しいリポジトリを作成
2. VSCodeでターミナルを開く(Ctrl+Shiht+@)
3. 初期化する 
```
git init
```
4. 追加する
```
git add .
```
5. GitHubのQuick setupに表示されている\
`https://github.com/ユーザー名/リポジトリ名.git`\
をコピーしてターミナルに貼り付ける
```
git remote add origin https://github.com/ユーザー名/リポジトリ名.git
```
※gitignoreでファイルが除外されない場合
```
git rm -r --cached .
```
---
## 今回インストールするライブラリ
```
pip3 install torch torchvision
pandas
matplotlib
```
※pipをアップグレードする場合は\
`python -m pip install --upgrade pip`

## ubuntu操作マニュアル
```
ssh -Y -2 -C -X uy06578@login.t4.gsic.titech.ac.jp # SSHログイン
bash *.sh # シェルスクリプトの実行
qrsh -g tga-ynabae -l cpu_4=1 -l h_rt=1:00:00 # gaussviewを使うためのジョブを取得
module load gaussian gaussview # gaussviewを使えるようにする
gview -soft # gaussviewの起動
qsub -g tga-ynabae *.sh # ジョブを投げる
qstat -u uy06578 # ジョブの確認
qdel -u uy06578　# ジョブを全て削除
qdel ジョブID # 指定したジョブを削除
dos2unix *.gjf *.sh # Windowsで作成したファイルをLinux形式に変換
```