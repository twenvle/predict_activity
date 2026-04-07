import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot(name, descriptors, x=10, y=8):
    # データの読み込み
    df = pd.read_csv(
        rf"C:\Users\kkyom\OneDrive\デスクトップ\pka_activity\ver_4\out\csvfile\known\{name}.csv"
    )

    df = df.dropna(how="any").reset_index(drop=True)

    X = df[descriptors]  # 記述子のみにする

    # 1. 相関係数の計算
    corr_matrix = X.corr()

    # 重複を除いてループ
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            val = corr_matrix.iloc[i, j]

    # 2. ヒートマップの描画
    plt.figure(figsize=(x, y))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Heatmap")
    plt.show()
