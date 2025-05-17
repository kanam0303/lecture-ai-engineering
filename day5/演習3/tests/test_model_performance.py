import os
import pytest
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# テスト用データとモデルパスを定義
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/Titanic.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")
MODEL_PATH = os.path.join(MODEL_DIR, "titanic_model.pkl")
PERFORMANCE_HISTORY_PATH = os.path.join(MODEL_DIR, "performance_history.json")


@pytest.fixture
def sample_data():
    """テスト用データセットを読み込む"""
    return pd.read_csv(DATA_PATH)


@pytest.fixture
def load_model():
    """モデルを読み込む"""
    if not os.path.exists(MODEL_PATH):
        pytest.skip("モデルファイルが存在しないためスキップします")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    return model


@pytest.fixture
def prepare_test_data(sample_data):
    """テストデータを準備する"""
    X = sample_data.drop("Survived", axis=1)
    y = sample_data["Survived"].astype(int)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_test, y_test


def test_model_metrics(load_model, prepare_test_data):
    """モデルの複数の評価指標を検証する"""
    model = load_model
    X_test, y_test = prepare_test_data

    # 予測
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # 陽性クラスの確率

    # 複数の評価指標を計算
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    # 各指標の閾値を設定
    thresholds = {
        "accuracy": 0.75,
        "precision": 0.70,
        "recall": 0.65,
        "f1": 0.70,
        "roc_auc": 0.80,
    }

    # 結果を出力
    for metric, value in metrics.items():
        threshold = thresholds[metric]
        print(f"{metric}: {value:.4f} (閾値: {threshold})")
        assert (
            value >= threshold
        ), f"{metric}が閾値を下回っています: {value:.4f} < {threshold}"

    # パフォーマンス履歴を保存
    save_performance_history(metrics)

    return metrics


def save_performance_history(metrics):
    """モデルのパフォーマンス履歴を保存する"""
    # タイムスタンプを含めた結果
    import datetime

    result = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
    }

    # 既存の履歴を読み込む
    history = []
    if os.path.exists(PERFORMANCE_HISTORY_PATH):
        try:
            with open(PERFORMANCE_HISTORY_PATH, "r") as f:
                history = json.load(f)
        except json.JSONDecodeError:
            history = []

    # 履歴に追加
    history.append(result)

    # 保存
    os.makedirs(os.path.dirname(PERFORMANCE_HISTORY_PATH), exist_ok=True)
    with open(PERFORMANCE_HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)


def test_model_performance_stability():
    """モデルのパフォーマンス安定性を検証する"""
    if not os.path.exists(PERFORMANCE_HISTORY_PATH):
        pytest.skip("パフォーマンス履歴が存在しないためスキップします")

    with open(PERFORMANCE_HISTORY_PATH, "r") as f:
        history = json.load(f)

    # 履歴が2つ以上ある場合のみ比較
    if len(history) < 2:
        pytest.skip("比較するための十分なパフォーマンス履歴がありません")

    # 最新と1つ前の結果を比較
    latest = history[-1]["metrics"]
    previous = history[-2]["metrics"]

    # 性能が著しく低下していないか確認（5%以上の低下を許容しない）
    for metric, value in latest.items():
        prev_value = previous[metric]
        max_degradation = 0.05  # 5%の低下まで許容
        degradation = prev_value - value

        print(
            f"{metric}: 現在値 {value:.4f}, 前回値 {prev_value:.4f}, 変化量 {-degradation:.4f}"
        )
        assert (
            degradation <= max_degradation
        ), f"{metric}が前回より著しく低下しています: {value:.4f} < {prev_value:.4f} (差: {degradation:.4f})"


def test_feature_importance(load_model):
    """特徴量の重要度を検証する"""
    model = load_model

    # モデルパイプラインから特徴量名とRandomForestClassifierを取得
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    # One-hot encodingされた特徴量名を取得
    cat_features = preprocessor.transformers_[1][2]  # カテゴリカル特徴
    cat_encoder = preprocessor.transformers_[1][1].named_steps["onehot"]

    num_features = preprocessor.transformers_[0][2]  # 数値特徴

    # One-hot encodingされた特徴量名
    encoded_cat_features = []
    for i, feature in enumerate(cat_features):
        if hasattr(cat_encoder, "get_feature_names_out"):
            feature_names = cat_encoder.get_feature_names_out([feature])
        else:
            feature_names = [f"{feature}_{j}" for j in cat_encoder.categories_[i]]
        encoded_cat_features.extend(feature_names)

    # 全ての特徴量名
    all_features = list(num_features) + encoded_cat_features

    # 特徴量の重要度を取得
    feature_importances = classifier.feature_importances_

    # 重要度が高い上位特徴量を表示
    indices = np.argsort(feature_importances)[::-1]
    top_n = min(10, len(all_features))

    print("Top features by importance:")
    for i in range(top_n):
        if i < len(indices):
            idx = indices[i]
            if idx < len(all_features):
                print(f"{i+1}. {all_features[idx]}: {feature_importances[idx]:.4f}")

    # 重要度の合計が0より大きいことを確認
    assert sum(feature_importances) > 0, "特徴量の重要度の合計が0以下です"

    # 最も重要な特徴量が一定の閾値以上であることを確認
    assert (
        feature_importances[indices[0]] >= 0.1
    ), f"最も重要な特徴量の重要度が低すぎます: {feature_importances[indices[0]]:.4f}"


def test_prediction_distribution(load_model, prepare_test_data):
    """予測分布を検証する"""
    model = load_model
    X_test, y_test = prepare_test_data

    # 予測確率
    y_prob = model.predict_proba(X_test)[:, 1]

    # 予測分布の統計
    mean_prob = np.mean(y_prob)
    std_prob = np.std(y_prob)
    min_prob = np.min(y_prob)
    max_prob = np.max(y_prob)

    print(
        f"予測確率の統計: 平均={mean_prob:.4f}, 標準偏差={std_prob:.4f}, 最小={min_prob:.4f}, 最大={max_prob:.4f}"
    )

    # 予測分布が極端に偏っていないことを確認
    assert 0.2 <= mean_prob <= 0.8, f"予測確率の平均が極端です: {mean_prob:.4f}"
    assert std_prob >= 0.1, f"予測確率の標準偏差が小さすぎます: {std_prob:.4f}"
    assert (
        max_prob - min_prob >= 0.5
    ), f"予測確率の範囲が狭すぎます: {max_prob - min_prob:.4f}"
