import numpy as np
import pandas as pd
from scipy.stats import norm
import joblib
from sklearn.neighbors import NearestNeighbors


def calculate_acquisition_functions(mu, sigma, current_best, xi=0.1, beta=2.0):
    """
    ベイズ最適化の獲得関数 (EI, PI, UCB) を計算する
    """
    # ゼロ除算や無効な計算（0/0など）の警告を一時的に無視
    with np.errstate(divide="ignore", invalid="ignore"):
        imp = mu - current_best - xi
        Z = imp / sigma

        # Expected Improvement (EI)
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        # Probability of Improvement (PI)
        pi = norm.cdf(Z)
        # Upper Confidence Bound (UCB)
        ucb = mu + beta * sigma

        # sigmaが0（不確実性が全くない）場合は、EIとPIを0にする
        ei[sigma <= 1e-9] = 0.0
        pi[sigma <= 1e-9] = 0.0

    return ei, pi, ucb


def make_csv(train_name, pkl_name, descriptors=None, xi=0.1, beta=2.0):
    # デフォルト引数のミュータブルオブジェクト問題を回避
    if descriptors is None:
        descriptors = []

    # データの読み込み
    df = pd.read_csv(
        r"C:\Users\kkyom\OneDrive\デスクトップ\pka_activity\ver_4\out\csvfile\unknown\unknown_data.csv"
    )
    df_train = pd.read_csv(
        rf"C:\Users\kkyom\OneDrive\デスクトップ\pka_activity\ver_4\out\csvfile\known\{train_name}.csv"
    )
    content = joblib.load(
        rf"C:\Users\kkyom\OneDrive\デスクトップ\pka_activity\ver_4\out\pklfile\{pkl_name}.pkl"
    )

    model = content["model"]
    scaler = content["scaler"]

    # 欠損値の削除（元のインデックスを保持しておくことを推奨します）
    df = df.dropna(how="any").reset_index(drop=True)

    # 記述子の取得
    X_unknown = df[descriptors]
    X_train = df_train[descriptors]
    y_train = df_train["yield"]

    # スケーリング
    X_unknown_scaled = scaler.transform(X_unknown)
    X_train_scaled = scaler.transform(X_train)

    # 1. 予測値 (mu) と 未校正の不確実性 (sigma_raw) の算出
    mu, sigma_raw = model.predict(X_unknown_scaled, return_std=True)

    # 2. 不確実性の校正 (sigma_cal)
    # ※ 本来はバリデーションデータから算出したスケーリング係数(calibration_factor)を使用します。
    # ここでは例として係数1.0（補正なし）とするか、pklから読み込む設計にしています。
    calibration_factor = content.get("calibration_factor", 1.0)
    sigma_cal = sigma_raw * calibration_factor

    # 3. 信頼区間 (ci_lower, ci_upper) - 95%信頼区間 (1.96 * sigma)
    ci_lower = mu - 1.96 * sigma_cal
    ci_upper = mu + 1.96 * sigma_cal

    # 4. 適用領域 (Applicability Domain: AD) の計算
    # 学習データ内での距離分布からADの閾値を設定 (平均距離 + 3 * 標準偏差)
    nn_train = NearestNeighbors(n_neighbors=2, metric="euclidean")
    nn_train.fit(X_train_scaled)
    distances_train, _ = nn_train.kneighbors(X_train_scaled)
    # 自身(距離0)を除外した最短距離
    nearest_distances_train = distances_train[:, 1]
    ad_threshold = np.mean(nearest_distances_train) + 3 * np.std(
        nearest_distances_train
    )

    if ad_threshold == 0.0:
        ad_threshold = 1e-9  # ゼロ除算回避

    # 未知データの学習データに対する最短距離を計算
    nn_unknown = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn_unknown.fit(X_train_scaled)
    distances_unknown, _ = nn_unknown.kneighbors(X_unknown_scaled)

    ad_distance = distances_unknown.flatten()
    ad_ratio = ad_distance / ad_threshold
    in_ad = ad_ratio <= 1.0

    # 距離が極めて0に近い場合は既知データと判定
    known_any = ad_distance < 1e-6

    # 5. 獲得関数 (ei, pi, ucb) の計算
    current_best_y = np.max(y_train)
    ei, pi, ucb = calculate_acquisition_functions(
        mu, sigma_cal, current_best_y, xi=xi, beta=beta
    )

    # 6. 結果の格納
    df["mu"] = mu
    df["sigma_raw"] = sigma_raw
    df["sigma_cal"] = sigma_cal
    df["ci_lower"] = ci_lower
    df["ci_upper"] = ci_upper
    df["ad_distance"] = ad_distance
    df["ad_ratio"] = ad_ratio
    df["in_ad"] = in_ad
    df["ei"] = ei
    df["pi"] = pi
    df["ucb"] = ucb
    df["known_any"] = known_any

    # CSV出力
    df.to_csv(
        rf"C:\Users\kkyom\OneDrive\デスクトップ\pka_activity\ver_4\out\csvfile\predict\{pkl_name}.csv",
        index=False,
    )
