"""
app.py - v6.1
Mega Sena ML focado em jogos fortes (terno / quadra / quina), usando:

- estrutura original por concurso
- 1 modelo por dezena
- LightGBM, XGBoost ou ensemble_gbm
- ensemble por ranking entre LightGBM + XGBoost
- geração de 3 jogos concentrados por concurso
- Monte Carlo como recurso adicional
- pool-size configurável
- prints reorganizados:
  1) jogos com match primeiro
  2) métricas depois

Observação:
- O número de jogos mostrados no relatório NÃO é o total gerado.
- O total gerado = concursos de teste × jogos por concurso.
"""

import argparse
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    brier_score_loss,
)

DATA_PATH = Path(__file__).resolve().parent / "data" / "raw" / "mega_sena_2000_2026.csv"
CUTOFF_DATE = pd.Timestamp("2026-01-01")

TOTAL_NUMBERS = 60
NUMBERS_PER_DRAW = 6

DEZENA_COLS = [
    "dezena_1",
    "dezena_2",
    "dezena_3",
    "dezena_4",
    "dezena_5",
    "dezena_6",
]

PRIZE_SCORE = {
    0: 0,
    1: 0,
    2: 0,
    3: 1,
    4: 10,
    5: 50,
    6: 500,
}


def load_mega_sena(filepath: Path) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df["data_parsed"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df = df.sort_values("concurso").reset_index(drop=True)
    return df


def build_frequency_features(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    all_draws = df[DEZENA_COLS].values.astype(int)
    features_list = []

    for i in range(window, len(df)):
        recent_draws = all_draws[max(0, i - window):i]
        recent_numbers = recent_draws.flatten()

        freq_features = {}
        delay_features = {}
        streak_features = {}
        multi_window_features = {}

        for n in range(1, TOTAL_NUMBERS + 1):
            freq_features[f"freq_{n}"] = float(np.sum(recent_numbers == n) / len(recent_draws))

            last_seen = -1
            for j in range(len(recent_draws) - 1, -1, -1):
                if n in recent_draws[j]:
                    last_seen = j
                    break
            delay_features[f"atraso_{n}"] = float(
                (len(recent_draws) - last_seen) if last_seen >= 0 else (window + 1)
            )

            streak = 0
            for j in range(len(recent_draws) - 1, -1, -1):
                if n in recent_draws[j]:
                    streak += 1
                else:
                    break
            streak_features[f"streak_{n}"] = float(streak)

            for w in (5, 10, 20, 30, 60):
                subset = all_draws[max(0, i - w):i]
                if len(subset) == 0:
                    freq = 0.0
                    delay = float(w + 1)
                    streak_w = 0.0
                else:
                    flat = subset.flatten()
                    freq = float(np.sum(flat == n) / len(subset))

                    last_seen = -1
                    for j in range(len(subset) - 1, -1, -1):
                        if n in subset[j]:
                            last_seen = j
                            break
                    delay = float((len(subset) - last_seen) if last_seen >= 0 else (w + 1))

                    streak_local = 0
                    for j in range(len(subset) - 1, -1, -1):
                        if n in subset[j]:
                            streak_local += 1
                        else:
                            break
                    streak_w = float(streak_local)

                multi_window_features[f"freq_w{w}_{n}"] = freq
                multi_window_features[f"delay_w{w}_{n}"] = delay
                multi_window_features[f"streak_w{w}_{n}"] = streak_w

            multi_window_features[f"trend_10_60_{n}"] = (
                multi_window_features[f"freq_w10_{n}"] - multi_window_features[f"freq_w60_{n}"]
            )
            multi_window_features[f"trend_5_30_{n}"] = (
                multi_window_features[f"freq_w5_{n}"] - multi_window_features[f"freq_w30_{n}"]
            )

        last_draw = sorted(all_draws[i - 1].tolist())

        soma_ultimo = int(np.sum(last_draw))
        pares_ultimo = int(np.sum(np.array(last_draw) % 2 == 0))
        impares_ultimo = 6 - pares_ultimo

        somas = [int(np.sum(d)) for d in recent_draws]
        soma_media = float(np.mean(somas))
        soma_std = float(np.std(somas))

        consecutivas = sum(
            1 for k in range(len(last_draw) - 1)
            if last_draw[k + 1] - last_draw[k] == 1
        )

        q1 = sum(1 for n in last_draw if 1 <= n <= 15)
        q2 = sum(1 for n in last_draw if 16 <= n <= 30)
        q3 = sum(1 for n in last_draw if 31 <= n <= 45)
        q4 = sum(1 for n in last_draw if 46 <= n <= 60)

        gaps = [last_draw[k + 1] - last_draw[k] for k in range(len(last_draw) - 1)]
        gap_medio = float(np.mean(gaps)) if gaps else 0.0
        gap_max = float(max(gaps)) if gaps else 0.0
        gap_min = float(min(gaps)) if gaps else 0.0
        gap_std = float(np.std(gaps)) if len(gaps) > 1 else 0.0

        decadas = {}
        for dec in range(6):
            low = dec * 10 + 1
            high = (dec + 1) * 10
            decadas[f"decada_{low}_{high}"] = sum(1 for n in last_draw if low <= n <= high)

        hot_count = sum(
            1 for n in range(1, TOTAL_NUMBERS + 1)
            if freq_features[f"freq_{n}"] > 0.12
        )
        cold_count = sum(
            1 for n in range(1, TOTAL_NUMBERS + 1)
            if freq_features[f"freq_{n}"] < 0.08
        )

        amplitude = float(last_draw[-1] - last_draw[0])
        mediana = float(np.median(last_draw))

        row = {
            "concurso": int(df.iloc[i]["concurso"]),
            "idx": i,
            **freq_features,
            **delay_features,
            **streak_features,
            **multi_window_features,
            **decadas,
            "soma_ultimo": soma_ultimo,
            "pares_ultimo": pares_ultimo,
            "impares_ultimo": impares_ultimo,
            "soma_media_janela": soma_media,
            "soma_std_janela": soma_std,
            "consecutivas": consecutivas,
            "quadrante_1_15": q1,
            "quadrante_16_30": q2,
            "quadrante_31_45": q3,
            "quadrante_46_60": q4,
            "gap_medio": gap_medio,
            "gap_max": gap_max,
            "gap_min": gap_min,
            "gap_std": gap_std,
            "hot_count": hot_count,
            "cold_count": cold_count,
            "amplitude": amplitude,
            "mediana": mediana,
        }
        features_list.append(row)

    return pd.DataFrame(features_list)


def build_target(df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    for n in range(1, TOTAL_NUMBERS + 1):
        targets = []
        for _, row in features_df.iterrows():
            idx = int(row["idx"])
            drawn = df.iloc[idx][DEZENA_COLS].astype(int).values
            targets.append(1 if n in drawn else 0)
        features_df[f"target_{n}"] = targets
    return features_df


def split_by_date(df_original: pd.DataFrame, features_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    features_df = features_df.merge(
        df_original[["concurso", "data_parsed"]],
        on="concurso",
        how="left",
    )
    train = features_df[features_df["data_parsed"] < CUTOFF_DATE].copy()
    test = features_df[features_df["data_parsed"] >= CUTOFF_DATE].copy()
    return train, test


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {"concurso", "idx", "data_parsed"}
    target_cols = {c for c in df.columns if c.startswith("target_")}
    return [c for c in df.columns if c not in exclude and c not in target_cols]


def build_lightgbm_aggressive() -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        n_estimators=700,
        learning_rate=0.02,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=8,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.05,
        reg_lambda=0.05,
        class_weight={0: 1.0, 1: 12.0},
        random_state=42,
        verbosity=-1,
    )


def build_xgboost_aggressive() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=3,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=9.0,
        random_state=42,
        n_jobs=-1,
    )


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(brier_score_loss(y_true, y_prob))
    except Exception:
        return float("nan")


def train_and_evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str = "lightgbm",
    rank_weights: tuple[float, float] = (0.5, 0.5),
) -> dict:
    feature_cols = get_feature_columns(train_df)
    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values

    results_per_number = {}
    all_probabilities = {}
    per_concurso_probas = {}
    per_concurso_ranks = {}

    print(f"\nTreinando modelos por dezena: {model_name}")

    for n in range(1, TOTAL_NUMBERS + 1):
        target_col = f"target_{n}"
        y_train = train_df[target_col].values
        y_test = test_df[target_col].values

        if model_name == "lightgbm":
            model = build_lightgbm_aggressive()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            proba_col = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba[:, 0]
            pred_col = y_pred
            model_store = model

        elif model_name == "xgboost":
            model = build_xgboost_aggressive()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            proba_col = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba[:, 0]
            pred_col = y_pred
            model_store = model

        elif model_name == "ensemble_gbm":
            model_lgbm = build_lightgbm_aggressive()
            model_xgb = build_xgboost_aggressive()

            model_lgbm.fit(X_train, y_train)
            model_xgb.fit(X_train, y_train)

            proba_lgbm = model_lgbm.predict_proba(X_test)
            proba_xgb = model_xgb.predict_proba(X_test)

            proba_lgbm_col = proba_lgbm[:, 1] if proba_lgbm.shape[1] > 1 else proba_lgbm[:, 0]
            proba_xgb_col = proba_xgb[:, 1] if proba_xgb.shape[1] > 1 else proba_xgb[:, 0]

            proba_col = (proba_lgbm_col + proba_xgb_col) / 2.0
            pred_col = (proba_col >= 0.5).astype(int)
            model_store = {"lightgbm": model_lgbm, "xgboost": model_xgb}

            per_concurso_probas[f"lightgbm_{n}"] = proba_lgbm_col
            per_concurso_probas[f"xgboost_{n}"] = proba_xgb_col

        else:
            raise ValueError(f"Modelo inválido: {model_name}")

        acc = float(accuracy_score(y_test, pred_col))
        precision = float(precision_score(y_test, pred_col, zero_division=0))
        recall = float(recall_score(y_test, pred_col, zero_division=0))
        auc = _safe_auc(y_test, proba_col)
        brier = _safe_brier(y_test, proba_col)

        results_per_number[n] = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "auc": auc,
            "brier": brier,
            "avg_probability": float(np.mean(proba_col)),
            "actual_frequency": float(np.mean(y_test)),
            "predicted_frequency": float(np.mean(pred_col)),
            "model": model_store,
        }

        if model_name != "ensemble_gbm":
            all_probabilities[n] = float(np.mean(proba_col))
            per_concurso_probas[n] = proba_col

    if model_name == "ensemble_gbm":
        num_test = len(test_df)
        final_mean_scores = {}

        for n in range(1, TOTAL_NUMBERS + 1):
            lgbm_col = per_concurso_probas[f"lightgbm_{n}"]
            xgb_col = per_concurso_probas[f"xgboost_{n}"]
            final_mean_scores[n] = float(np.mean((lgbm_col + xgb_col) / 2.0))

        all_probabilities = final_mean_scores

        for i in range(num_test):
            lgbm_scores = {n: float(per_concurso_probas[f"lightgbm_{n}"][i]) for n in range(1, TOTAL_NUMBERS + 1)}
            xgb_scores = {n: float(per_concurso_probas[f"xgboost_{n}"][i]) for n in range(1, TOTAL_NUMBERS + 1)}

            lgbm_sorted = sorted(lgbm_scores.items(), key=lambda x: x[1], reverse=True)
            xgb_sorted = sorted(xgb_scores.items(), key=lambda x: x[1], reverse=True)

            lgbm_rank = {n: rank + 1 for rank, (n, _) in enumerate(lgbm_sorted)}
            xgb_rank = {n: rank + 1 for rank, (n, _) in enumerate(xgb_sorted)}

            final_rank_scores = {}
            for n in range(1, TOTAL_NUMBERS + 1):
                final_rank_scores[n] = (
                    rank_weights[0] * lgbm_rank[n] +
                    rank_weights[1] * xgb_rank[n]
                )

            per_concurso_ranks[i] = final_rank_scores

    ranking = sorted(all_probabilities.items(), key=lambda x: x[1], reverse=True)

    return {
        "results_per_number": results_per_number,
        "ranking": ranking,
        "feature_columns": feature_cols,
        "per_concurso_probas": per_concurso_probas,
        "per_concurso_ranks": per_concurso_ranks,
        "model_name": model_name,
        "rank_weights": rank_weights,
    }


def analyze_last_year(df: pd.DataFrame, test_df: pd.DataFrame, results: dict, top_n: int = 10) -> dict:
    ranking = results["ranking"]
    top_numbers = [n for n, _ in ranking[:top_n]]

    test_concursos = test_df["concurso"].values
    test_original = df[df["concurso"].isin(test_concursos)]
    all_drawn = test_original[DEZENA_COLS].values.astype(int).flatten()
    actual_freq = Counter(all_drawn)

    most_common_actual = actual_freq.most_common(top_n)
    actual_top = {n for n, _ in most_common_actual}
    predicted_top = set(top_numbers)
    overlap = actual_top & predicted_top

    return {
        "top_predicted": top_numbers,
        "top_actual": most_common_actual,
        "overlap": overlap,
        "overlap_count": len(overlap),
    }


def _build_candidate_games_from_pool(sorted_numbers: list[int]) -> list[list[int]]:
    if len(sorted_numbers) < 7:
        return [sorted(sorted_numbers[:6])]

    r = sorted_numbers
    games = [
        sorted([r[0], r[1], r[2], r[3], r[4], r[5]]),
        sorted([r[0], r[1], r[2], r[3], r[4], r[6]]),
        sorted([r[0], r[1], r[2], r[3], r[5], r[6]]),
    ]

    unique_games = []
    seen = set()
    for g in games:
        key = tuple(g)
        if key not in seen and len(g) == 6:
            unique_games.append(g)
            seen.add(key)

    return unique_games


def _weighted_sample_game(pool_numbers: list[int], pool_weights: list[float], rng: np.random.Generator) -> list[int]:
    weights = np.array(pool_weights, dtype=float)
    weights = weights / weights.sum()
    chosen = rng.choice(pool_numbers, size=6, replace=False, p=weights)
    return sorted(int(x) for x in chosen.tolist())


def _score_game_internal(game: list[int], ranked_pool: list[int]) -> float:
    rank_pos = {n: i + 1 for i, n in enumerate(ranked_pool)}
    score = 0.0
    for n in game:
        pos = rank_pos.get(n, len(ranked_pool) + 5)
        score += 1.0 / pos

    score += len(set(game[:4]).intersection(set(ranked_pool[:4]))) * 0.15
    return score


def predict_multi_games(
    df: pd.DataFrame,
    test_df: pd.DataFrame,
    results: dict,
    pool_size: int = 10,
) -> dict:
    per_concurso_probas = results["per_concurso_probas"]
    per_concurso_ranks = results["per_concurso_ranks"]
    model_name = results["model_name"]

    test_concursos = test_df["concurso"].values
    num_test = len(test_concursos)

    concursos_data = []
    all_generated_games = []

    for i in range(num_test):
        concurso = int(test_concursos[i])
        original_row = df[df["concurso"] == concurso].iloc[0]
        actual_numbers = sorted(original_row[DEZENA_COLS].astype(int).values.tolist())
        data_jogo = original_row["data"]

        if model_name == "ensemble_gbm":
            final_rank_scores = per_concurso_ranks[i]
            ranked_numbers = [n for n, _ in sorted(final_rank_scores.items(), key=lambda x: x[1])]
        else:
            probas = {n: float(per_concurso_probas[n][i]) for n in range(1, TOTAL_NUMBERS + 1)}
            ranked_numbers = [n for n, _ in sorted(probas.items(), key=lambda x: x[1], reverse=True)]

        top_pool = ranked_numbers[:pool_size]
        games = _build_candidate_games_from_pool(top_pool)

        game_results = []
        best_hits = 0
        best_game = None

        for idx_game, predicted_numbers in enumerate(games, start=1):
            actual_set = set(actual_numbers)
            predicted_set = set(predicted_numbers)
            matched = sorted(actual_set & predicted_set)
            hits = len(matched)
            prize_score = PRIZE_SCORE.get(hits, 0)

            row_game = {
                "concurso": concurso,
                "data": data_jogo,
                "game_id": idx_game,
                "predicted": sorted(predicted_numbers),
                "actual": actual_numbers,
                "matched": matched,
                "hits": hits,
                "prize_score": prize_score,
                "top_pool": top_pool,
            }
            game_results.append(row_game)
            all_generated_games.append(row_game)

            if hits > best_hits:
                best_hits = hits
                best_game = row_game
            elif hits == best_hits and best_game is not None and prize_score > best_game["prize_score"]:
                best_game = row_game

        concursos_data.append({
            "concurso": concurso,
            "data": data_jogo,
            "actual": actual_numbers,
            "top_pool": top_pool,
            "games": game_results,
            "best_game": best_game,
            "best_hits": best_hits,
        })

    return summarize_generated_games(concursos_data, all_generated_games, games_per_concurso=3, pool_size=pool_size, label="PADRÃO")


def predict_multi_games_monte_carlo(
    df: pd.DataFrame,
    test_df: pd.DataFrame,
    results: dict,
    pool_size: int = 10,
    mc_samples: int = 400,
    mc_keep_games: int = 3,
    seed: int = 42,
) -> dict:
    per_concurso_probas = results["per_concurso_probas"]
    per_concurso_ranks = results["per_concurso_ranks"]
    model_name = results["model_name"]

    test_concursos = test_df["concurso"].values
    num_test = len(test_concursos)
    rng = np.random.default_rng(seed)

    concursos_data = []
    all_generated_games = []

    for i in range(num_test):
        concurso = int(test_concursos[i])
        original_row = df[df["concurso"] == concurso].iloc[0]
        actual_numbers = sorted(original_row[DEZENA_COLS].astype(int).values.tolist())
        data_jogo = original_row["data"]

        if model_name == "ensemble_gbm":
            final_rank_scores = per_concurso_ranks[i]
            ranked_numbers = [n for n, _ in sorted(final_rank_scores.items(), key=lambda x: x[1])]
        else:
            probas = {n: float(per_concurso_probas[n][i]) for n in range(1, TOTAL_NUMBERS + 1)}
            ranked_numbers = [n for n, _ in sorted(probas.items(), key=lambda x: x[1], reverse=True)]

        top_pool = ranked_numbers[:pool_size]

        # pesos maiores para ranks mais altos
        base_weights = [float(pool_size - idx) for idx in range(pool_size)]
        candidate_games = {}
        for _ in range(mc_samples):
            game = _weighted_sample_game(top_pool, base_weights, rng)
            key = tuple(game)
            score = _score_game_internal(game, top_pool)
            if key not in candidate_games or score > candidate_games[key]:
                candidate_games[key] = score

        selected_games = [
            list(k) for k, _ in sorted(candidate_games.items(), key=lambda x: x[1], reverse=True)[:mc_keep_games]
        ]

        game_results = []
        best_hits = 0
        best_game = None

        for idx_game, predicted_numbers in enumerate(selected_games, start=1):
            actual_set = set(actual_numbers)
            predicted_set = set(predicted_numbers)
            matched = sorted(actual_set & predicted_set)
            hits = len(matched)
            prize_score = PRIZE_SCORE.get(hits, 0)

            row_game = {
                "concurso": concurso,
                "data": data_jogo,
                "game_id": idx_game,
                "predicted": sorted(predicted_numbers),
                "actual": actual_numbers,
                "matched": matched,
                "hits": hits,
                "prize_score": prize_score,
                "top_pool": top_pool,
            }
            game_results.append(row_game)
            all_generated_games.append(row_game)

            if hits > best_hits:
                best_hits = hits
                best_game = row_game
            elif hits == best_hits and best_game is not None and prize_score > best_game["prize_score"]:
                best_game = row_game

        concursos_data.append({
            "concurso": concurso,
            "data": data_jogo,
            "actual": actual_numbers,
            "top_pool": top_pool,
            "games": game_results,
            "best_game": best_game,
            "best_hits": best_hits,
        })

    return summarize_generated_games(
        concursos_data,
        all_generated_games,
        games_per_concurso=mc_keep_games,
        pool_size=pool_size,
        label="MONTE CARLO",
        extra_info={"mc_samples": mc_samples}
    )


def summarize_generated_games(
    concursos_data: list[dict],
    all_generated_games: list[dict],
    games_per_concurso: int,
    pool_size: int,
    label: str,
    extra_info: dict | None = None,
) -> dict:
    hits_all_games = [g["hits"] for g in all_generated_games]
    hits_distribution_all = Counter(hits_all_games)

    total_hits_all = sum(hits_all_games)
    total_numbers_all = len(all_generated_games) * 6 if all_generated_games else 0
    hit_rate_all_games = (total_hits_all / total_numbers_all) if total_numbers_all > 0 else 0.0
    prize_score_total = sum(g["prize_score"] for g in all_generated_games)

    best_games = [c["best_game"] for c in concursos_data if c["best_game"] is not None]
    best_hits_list = [g["hits"] for g in best_games]
    best_hits_distribution = Counter(best_hits_list)

    total_best_hits = sum(best_hits_list)
    total_best_numbers = len(best_games) * 6 if best_games else 0
    hit_rate_best = (total_best_hits / total_best_numbers) if total_best_numbers > 0 else 0.0
    best_prize_score_total = sum(g["prize_score"] for g in best_games)

    summary = {
        "label": label,
        "concursos": concursos_data,
        "all_games": all_generated_games,
        "best_games": best_games,
        "total_concursos": len(concursos_data),
        "games_per_concurso": games_per_concurso,
        "total_games_generated": len(all_generated_games),
        "pool_size": pool_size,
        "hit_rate_all_games": hit_rate_all_games,
        "hit_rate_best_games": hit_rate_best,
        "hits_distribution_all_games": dict(sorted(hits_distribution_all.items())),
        "hits_distribution_best_games": dict(sorted(best_hits_distribution.items())),
        "prize_score_total": prize_score_total,
        "best_prize_score_total": best_prize_score_total,
        "num_ternos_all": sum(1 for g in all_generated_games if g["hits"] == 3),
        "num_quadras_all": sum(1 for g in all_generated_games if g["hits"] == 4),
        "num_quinas_all": sum(1 for g in all_generated_games if g["hits"] == 5),
        "num_senas_all": sum(1 for g in all_generated_games if g["hits"] == 6),
        "num_ternos_best": sum(1 for g in best_games if g["hits"] == 3),
        "num_quadras_best": sum(1 for g in best_games if g["hits"] == 4),
        "num_quinas_best": sum(1 for g in best_games if g["hits"] == 5),
        "num_senas_best": sum(1 for g in best_games if g["hits"] == 6),
        "max_hits_all": max(hits_all_games) if hits_all_games else 0,
        "max_hits_best": max(best_hits_list) if best_hits_list else 0,
        "min_hits_all": min(hits_all_games) if hits_all_games else 0,
        "min_hits_best": min(best_hits_list) if best_hits_list else 0,
        "avg_hits_all_games": float(np.mean(hits_all_games)) if hits_all_games else 0.0,
        "avg_hits_best_games": float(np.mean(best_hits_list)) if best_hits_list else 0.0,
    }
    if extra_info:
        summary.update(extra_info)
    return summary


def print_model_metrics(results: dict) -> None:
    rows = []
    for n, r in results["results_per_number"].items():
        rows.append({
            "dezena": n,
            "accuracy": r["accuracy"],
            "precision": r["precision"],
            "recall": r["recall"],
            "auc": r["auc"],
            "brier": r["brier"],
            "avg_probability": r["avg_probability"],
            "actual_frequency": r["actual_frequency"],
            "predicted_frequency": r["predicted_frequency"],
        })

    metrics_df = pd.DataFrame(rows).sort_values("avg_probability", ascending=False)

    print("\n" + "=" * 90)
    print("MÉTRICAS AGREGADAS DOS MODELOS POR DEZENA")
    print("=" * 90)
    print(f"Accuracy média:   {metrics_df['accuracy'].mean():.4f}")
    print(f"Precision média:  {metrics_df['precision'].mean():.4f}")
    print(f"Recall médio:     {metrics_df['recall'].mean():.4f}")
    print(f"AUC média:        {metrics_df['auc'].dropna().mean():.4f}")
    print(f"Brier médio:      {metrics_df['brier'].dropna().mean():.4f}")

    print("\nTop 10 dezenas por probabilidade média:")
    print("-" * 90)
    for _, row in metrics_df.head(10).iterrows():
        auc_str = f"{row['auc']:.4f}" if not pd.isna(row["auc"]) else "nan"
        print(
            f"Dezena {int(row['dezena']):02d} | "
            f"proba={row['avg_probability']:.4f} | "
            f"acc={row['accuracy']:.4f} | "
            f"prec={row['precision']:.4f} | "
            f"rec={row['recall']:.4f} | "
            f"auc={auc_str}"
        )


def print_top10_analysis(analysis: dict) -> None:
    print("\n" + "=" * 90)
    print("TOP 10 DEZENAS MAIS PROVÁVEIS vs MAIS SORTEADAS")
    print("=" * 90)
    print(f"Top previsto: {sorted(analysis['top_predicted'])}")
    print(f"Top real:     {sorted([n for n, _ in analysis['top_actual']])}")
    print(f"Overlap:      {sorted(analysis['overlap'])} ({analysis['overlap_count']}/10)")


def print_games_with_matches(summary: dict, top_n: int = 20) -> None:
    matched_games = [g for g in summary["all_games"] if g["hits"] > 0]
    matched_games = sorted(matched_games, key=lambda g: (g["hits"], g["prize_score"]), reverse=True)

    print("\n" + "=" * 90)
    print(f"JOGOS COM MATCH - {summary['label']}")
    print("=" * 90)
    print(f"Concursos no teste:         {summary['total_concursos']}")
    print(f"Jogos por concurso:         {summary['games_per_concurso']}")
    print(f"Total de jogos gerados:     {summary['total_games_generated']}")
    print(f"Top report mostrado:        {min(top_n, len(matched_games))}")
    if "mc_samples" in summary:
        print(f"Amostras Monte Carlo:       {summary['mc_samples']}")

    if not matched_games:
        print("Nenhum jogo com match foi encontrado.")
        return

    for g in matched_games[:top_n]:
        print(
            f"Concurso {g['concurso']} | Data {g['data']} | "
            f"Jogo #{g['game_id']} | "
            f"Acertos {g['hits']}/6 | "
            f"Prize {g['prize_score']} | "
            f"Previsto {g['predicted']} | "
            f"Real {g['actual']} | "
            f"Match {g['matched']}"
        )


def print_summary_metrics(summary: dict) -> None:
    print("\n" + "=" * 90)
    print(f"MÉTRICAS RESUMIDAS - {summary['label']}")
    print("=" * 90)
    print(f"Concursos analisados:                 {summary['total_concursos']}")
    print(f"Jogos por concurso:                   {summary['games_per_concurso']}")
    print(f"Total de jogos gerados:               {summary['total_games_generated']}")
    print(f"Pool de dezenas por concurso:         top-{summary['pool_size']}")
    if "mc_samples" in summary:
        print(f"Amostras Monte Carlo por concurso:    {summary['mc_samples']}")

    print("\n--- Todos os jogos gerados ---")
    print(f"Hit rate geral:                       {summary['hit_rate_all_games']:.4f} ({summary['hit_rate_all_games'] * 100:.2f}%)")
    print(f"Média de acertos por jogo:            {summary['avg_hits_all_games']:.4f}")
    print(f"Maior acerto:                         {summary['max_hits_all']}")
    print(f"Menor acerto:                         {summary['min_hits_all']}")
    print(f"Ternos:                               {summary['num_ternos_all']}")
    print(f"Quadras:                              {summary['num_quadras_all']}")
    print(f"Quinas:                               {summary['num_quinas_all']}")
    print(f"Senas:                                {summary['num_senas_all']}")
    print(f"Prize score total:                    {summary['prize_score_total']}")

    print("\nDistribuição de acertos (todos os jogos):")
    for hits, count in summary["hits_distribution_all_games"].items():
        print(f"{hits} acertos: {count}")

    print("\n--- Melhor jogo de cada concurso ---")
    print(f"Hit rate melhor jogo:                 {summary['hit_rate_best_games']:.4f} ({summary['hit_rate_best_games'] * 100:.2f}%)")
    print(f"Média de acertos do melhor jogo:      {summary['avg_hits_best_games']:.4f}")
    print(f"Maior acerto (melhor jogo):           {summary['max_hits_best']}")
    print(f"Menor acerto (melhor jogo):           {summary['min_hits_best']}")
    print(f"Ternos (melhor jogo):                 {summary['num_ternos_best']}")
    print(f"Quadras (melhor jogo):                {summary['num_quadras_best']}")
    print(f"Quinas (melhor jogo):                 {summary['num_quinas_best']}")
    print(f"Senas (melhor jogo):                  {summary['num_senas_best']}")
    print(f"Prize score total (melhor jogo):      {summary['best_prize_score_total']}")

    print("\nDistribuição de acertos (melhor jogo por concurso):")
    for hits, count in summary["hits_distribution_best_games"].items():
        print(f"{hits} acertos: {count}")


def run_mega_sena_analysis(
    model_name: str = "ensemble_gbm",
    window: int = 30,
    pool_size: int = 10,
    top_report: int = 20,
    rank_lgbm_weight: float = 0.5,
    rank_xgb_weight: float = 0.5,
    mc_samples: int = 400,
    mc_keep_games: int = 3,
) -> dict:
    print("=" * 90)
    print("1. CARREGANDO DADOS")
    print("=" * 90)

    if not DATA_PATH.exists():
        print(f"Erro: Arquivo não encontrado: {DATA_PATH}")
        print("Execute primeiro a coleta de dados.")
        sys.exit(1)

    df = load_mega_sena(DATA_PATH)
    print(f"Concursos carregados: {len(df)}")
    print(f"Período: {df['data'].iloc[0]} até {df['data'].iloc[-1]}")

    print("\n" + "=" * 90)
    print("2. ENGENHARIA DE FEATURES")
    print("=" * 90)
    print(f"Janela base: {window}")
    features_df = build_frequency_features(df, window=window)
    print(f"Concursos com features: {len(features_df)}")
    print(f"Features criadas: {len(get_feature_columns(features_df))}")

    print("\nConstruindo targets...")
    features_df = build_target(df, features_df)
    print(f"Targets criados para dezenas 1-{TOTAL_NUMBERS}")

    print("\n" + "=" * 90)
    print("3. DIVISÃO TEMPORAL")
    print("=" * 90)
    train_df, test_df = split_by_date(df, features_df)
    print(f"Treino: {len(train_df)} concursos")
    print(f"Teste:  {len(test_df)} concursos")
    print(f"Corte temporal: {CUTOFF_DATE.strftime('%d/%m/%Y')}")

    print("\n" + "=" * 90)
    print("4. TREINAMENTO")
    print("=" * 90)
    results = train_and_evaluate(
        train_df=train_df,
        test_df=test_df,
        model_name=model_name,
        rank_weights=(rank_lgbm_weight, rank_xgb_weight),
    )

    print("\n" + "=" * 90)
    print("5. ANÁLISE")
    print("=" * 90)
    analysis = analyze_last_year(df, test_df, results, top_n=10)

    standard_summary = predict_multi_games(
        df=df,
        test_df=test_df,
        results=results,
        pool_size=pool_size,
    )

    monte_carlo_summary = predict_multi_games_monte_carlo(
        df=df,
        test_df=test_df,
        results=results,
        pool_size=pool_size,
        mc_samples=mc_samples,
        mc_keep_games=mc_keep_games,
    )

    print_model_metrics(results)
    print_top10_analysis(analysis)

    # Primeiro os jogos com match
    print_games_with_matches(standard_summary, top_n=top_report)
    print_games_with_matches(monte_carlo_summary, top_n=top_report)

    # Depois as métricas
    print_summary_metrics(standard_summary)
    print_summary_metrics(monte_carlo_summary)

    return {
        "results": results,
        "analysis": analysis,
        "standard_summary": standard_summary,
        "monte_carlo_summary": monte_carlo_summary,
        "train_size": len(train_df),
        "test_size": len(test_df),
    }


def main():
    parser = argparse.ArgumentParser(
        description="ML Mega Sena v6.1 - GBMs + jogos concentrados + Monte Carlo"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ensemble_gbm",
        choices=["lightgbm", "xgboost", "ensemble_gbm"],
        help="Modelo a usar",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=30,
        help="Janela base para features (default: 30)",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=10,
        help="Tamanho do pool de dezenas fortes por concurso (default: 10)",
    )
    parser.add_argument(
        "--top-report",
        type=int,
        default=20,
        help="Quantidade de jogos com match a mostrar no relatório (default: 20)",
    )
    parser.add_argument(
        "--rank-lgbm-weight",
        type=float,
        default=0.5,
        help="Peso do ranking do LightGBM no ensemble_gbm (default: 0.5)",
    )
    parser.add_argument(
        "--rank-xgb-weight",
        type=float,
        default=0.5,
        help="Peso do ranking do XGBoost no ensemble_gbm (default: 0.5)",
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=400,
        help="Quantidade de amostras Monte Carlo por concurso (default: 400)",
    )
    parser.add_argument(
        "--mc-keep-games",
        type=int,
        default=3,
        help="Quantidade de jogos Monte Carlo mantidos por concurso (default: 3)",
    )

    args = parser.parse_args()

    total_weight = args.rank_lgbm_weight + args.rank_xgb_weight
    if total_weight <= 0:
        print("Erro: a soma dos pesos do ranking deve ser maior que zero.")
        sys.exit(1)

    run_mega_sena_analysis(
        model_name=args.model,
        window=args.window,
        pool_size=args.pool_size,
        top_report=args.top_report,
        rank_lgbm_weight=args.rank_lgbm_weight / total_weight,
        rank_xgb_weight=args.rank_xgb_weight / total_weight,
        mc_samples=args.mc_samples,
        mc_keep_games=args.mc_keep_games,
    )


if __name__ == "__main__":
    main()