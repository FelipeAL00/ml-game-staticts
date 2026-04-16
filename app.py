"""
app.py - v8
Base vencedora: v6.5 Rotating Core
Objetivo:
- subir consistência média
- manter / aumentar quadras
- comparar contra acaso de forma robusta
- escolher melhor estratégia entre rotating_core e rotating_balance

Requisitos:
pip install xgboost scikit-learn pandas numpy
"""

import argparse
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
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


# ======================================================================================
# LOAD / PREP
# ======================================================================================

def load_mega_sena(filepath: Path) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df["data_parsed"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df = df.sort_values("concurso").reset_index(drop=True)
    return df


def get_draw_location(row: pd.Series) -> str:
    if "local" in row.index and pd.notna(row["local"]):
        return str(row["local"])
    return "Local não disponível"


# ======================================================================================
# FEATURE ENGINEERING (base v6.5)
# ======================================================================================

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


# ======================================================================================
# MODEL
# ======================================================================================

def build_xgboost_model() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=600,
        learning_rate=0.025,
        max_depth=4,
        min_child_weight=4,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.15,
        reg_lambda=1.5,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=10.0,
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


def train_and_evaluate(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    feature_cols = get_feature_columns(train_df)
    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values

    results_per_number = {}
    per_concurso_probas = {}

    for n in range(1, TOTAL_NUMBERS + 1):
        target_col = f"target_{n}"
        y_train = train_df[target_col].values
        y_test = test_df[target_col].values

        model = build_xgboost_model()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        proba_col = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba[:, 0]

        results_per_number[n] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "auc": _safe_auc(y_test, proba_col),
            "brier": _safe_brier(y_test, proba_col),
            "avg_probability": float(np.mean(proba_col)),
            "actual_frequency": float(np.mean(y_test)),
            "predicted_frequency": float(np.mean(y_pred)),
            "model": model,
        }
        per_concurso_probas[n] = proba_col

    return {
        "results_per_number": results_per_number,
        "feature_columns": feature_cols,
        "per_concurso_probas": per_concurso_probas,
    }


# ======================================================================================
# GAME BUILDING
# ======================================================================================

def _game_distance(g1: list[int], g2: list[int]) -> int:
    return len(set(g1) ^ set(g2))


def _dedupe_games(games: list[list[int]]) -> list[list[int]]:
    out = []
    seen = set()
    for g in games:
        key = tuple(sorted(g))
        if key not in seen:
            seen.add(key)
            out.append(sorted(g))
    return out


def build_rotating_core_games(ranked_pool: list[int]) -> list[list[int]]:
    """
    Estratégia vencedora da v6.5:
    fixa top-4 e gira 2 dezenas entre ranks 5..10
    """
    rp = ranked_pool[:10]
    if len(rp) < 8:
        return [sorted(rp[:6])]

    core = rp[:4]
    tail = rp[4:]

    pairs = [
        (tail[0], tail[1]),
        (tail[0], tail[2]) if len(tail) > 2 else None,
        (tail[1], tail[2]) if len(tail) > 2 else None,
        (tail[0], tail[3]) if len(tail) > 3 else None,
        (tail[1], tail[3]) if len(tail) > 3 else None,
    ]

    games = []
    for pair in pairs:
        if pair is None:
            continue
        game = sorted(core + [pair[0], pair[1]])
        if len(set(game)) == 6:
            games.append(game)

    return _dedupe_games(games)[:5]


def build_rotating_balance_games(ranked_pool: list[int]) -> list[list[int]]:
    """
    Estratégia nova da v8:
    menos concentrada para tentar subir consistência média.
    Mistura top-3 fixo + rotações entre ranks 4..10.
    """
    rp = ranked_pool[:10]
    if len(rp) < 10:
        rp = ranked_pool

    if len(rp) < 8:
        return [sorted(rp[:6])]

    top3 = rp[:3]
    r4, r5, r6, r7, r8, r9, r10 = rp[3:10]

    candidates = [
        sorted(top3 + [r4, r5, r6]),
        sorted(top3 + [r4, r5, r7]),
        sorted(top3 + [r4, r6, r8]),
        sorted(top3 + [r5, r7, r9]),
        sorted(top3 + [r6, r8, r10]),
    ]

    # reforço de diversidade mínima
    selected = []
    for g in _dedupe_games(candidates):
        if not selected:
            selected.append(g)
        else:
            if all(_game_distance(g, prev) >= 4 for prev in selected):
                selected.append(g)

    if len(selected) < 5:
        for g in _dedupe_games(candidates):
            if g not in selected:
                selected.append(g)
            if len(selected) == 5:
                break

    return selected[:5]


def _weighted_sample_game(pool_numbers: list[int], pool_weights: list[float], rng: np.random.Generator) -> list[int]:
    weights = np.array(pool_weights, dtype=float)
    weights = weights / weights.sum()
    chosen = rng.choice(pool_numbers, size=6, replace=False, p=weights)
    return sorted(int(x) for x in chosen.tolist())


def _score_game_internal(game: list[int], ranked_pool: list[int], mode: str = "core") -> float:
    rank_pos = {n: i + 1 for i, n in enumerate(ranked_pool)}
    score = 0.0

    for n in game:
        pos = rank_pos.get(n, len(ranked_pool) + 5)
        score += 1.0 / pos

    if mode == "core":
        score += len(set(game).intersection(set(ranked_pool[:4]))) * 0.20
    elif mode == "spread":
        top4 = len(set(game).intersection(set(ranked_pool[:4])))
        mid = len(set(game).intersection(set(ranked_pool[4:7])))
        tail = len(set(game).intersection(set(ranked_pool[7:10])))
        score += top4 * 0.10 + mid * 0.12 + tail * 0.14

    return score


def _select_diverse_games(candidate_games: dict[tuple, float], keep_games: int, min_distance: int = 6) -> list[list[int]]:
    ordered = sorted(candidate_games.items(), key=lambda x: x[1], reverse=True)
    selected = []

    for game_tuple, _ in ordered:
        game = list(game_tuple)
        if not selected:
            selected.append(game)
            if len(selected) == keep_games:
                break
            continue

        if all(_game_distance(game, prev) >= min_distance for prev in selected):
            selected.append(game)
            if len(selected) == keep_games:
                break

    if len(selected) < keep_games:
        for game_tuple, _ in ordered:
            game = list(game_tuple)
            if game not in selected:
                selected.append(game)
                if len(selected) == keep_games:
                    break

    return _dedupe_games(selected)[:keep_games]


# ======================================================================================
# EVALUATION
# ======================================================================================

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


def _evaluate_games_for_concursos(
    df: pd.DataFrame,
    test_df: pd.DataFrame,
    games_by_concurso: dict,
    label: str,
    pool_size: int,
    extra_info: dict | None = None,
) -> dict:
    test_concursos = test_df["concurso"].values
    concursos_data = []
    all_generated_games = []

    for concurso in test_concursos:
        concurso = int(concurso)
        original_row = df[df["concurso"] == concurso].iloc[0]
        actual_numbers = sorted(original_row[DEZENA_COLS].astype(int).values.tolist())
        data_jogo = original_row["data"]
        local_jogo = get_draw_location(original_row)

        games = games_by_concurso[concurso]

        game_results = []
        best_game = None
        best_hits = -1
        best_prize = -1

        for idx_game, predicted_numbers in enumerate(games, start=1):
            matched = sorted(set(predicted_numbers) & set(actual_numbers))
            hits = len(matched)
            prize_score = PRIZE_SCORE.get(hits, 0)

            game = {
                "concurso": concurso,
                "data": data_jogo,
                "local": local_jogo,
                "game_id": idx_game,
                "predicted": sorted(predicted_numbers),
                "actual": actual_numbers,
                "matched": matched,
                "hits": hits,
                "prize_score": prize_score,
            }
            game_results.append(game)
            all_generated_games.append(game)

            if hits > best_hits or (hits == best_hits and prize_score > best_prize):
                best_game = game
                best_hits = hits
                best_prize = prize_score

        concursos_data.append({
            "concurso": concurso,
            "data": data_jogo,
            "local": local_jogo,
            "actual": actual_numbers,
            "games": game_results,
            "best_game": best_game,
        })

    return summarize_generated_games(
        concursos_data=concursos_data,
        all_generated_games=all_generated_games,
        games_per_concurso=len(next(iter(games_by_concurso.values()))),
        pool_size=pool_size,
        label=label,
        extra_info=extra_info,
    )


# ======================================================================================
# PREDICTION MODES
# ======================================================================================

def _build_ranked_pool_for_concurso(results: dict, i: int, pool_size: int) -> list[int]:
    per_concurso_probas = results["per_concurso_probas"]
    probas = {n: float(per_concurso_probas[n][i]) for n in range(1, TOTAL_NUMBERS + 1)}
    ranked_numbers = [n for n, _ in sorted(probas.items(), key=lambda x: x[1], reverse=True)]
    return ranked_numbers[:pool_size]


def predict_rotating_core_games(
    df: pd.DataFrame,
    test_df: pd.DataFrame,
    results: dict,
    pool_size: int = 10,
) -> dict:
    games_by_concurso = {}

    for i, concurso in enumerate(test_df["concurso"].values):
        top_pool = _build_ranked_pool_for_concurso(results, i, pool_size)
        games = build_rotating_core_games(top_pool)

        fallback = sorted(top_pool[:6])
        while len(games) < 5:
            if fallback not in games:
                games.append(fallback)
            else:
                break

        games_by_concurso[int(concurso)] = _dedupe_games(games)[:5]

    return _evaluate_games_for_concursos(
        df=df,
        test_df=test_df,
        games_by_concurso=games_by_concurso,
        label="ROTATING CORE",
        pool_size=pool_size,
    )


def predict_rotating_balance_games(
    df: pd.DataFrame,
    test_df: pd.DataFrame,
    results: dict,
    pool_size: int = 10,
) -> dict:
    games_by_concurso = {}

    for i, concurso in enumerate(test_df["concurso"].values):
        top_pool = _build_ranked_pool_for_concurso(results, i, pool_size)
        games = build_rotating_balance_games(top_pool)

        fallback = sorted(top_pool[:6])
        while len(games) < 5:
            if fallback not in games:
                games.append(fallback)
            else:
                break

        games_by_concurso[int(concurso)] = _dedupe_games(games)[:5]

    return _evaluate_games_for_concursos(
        df=df,
        test_df=test_df,
        games_by_concurso=games_by_concurso,
        label="ROTATING BALANCE",
        pool_size=pool_size,
    )


def predict_random_games(
    df: pd.DataFrame,
    test_df: pd.DataFrame,
    games_per_concurso: int = 5,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    games_by_concurso = {}

    for concurso in test_df["concurso"].values:
        games = []
        seen = set()
        while len(games) < games_per_concurso:
            game = sorted(rng.choice(np.arange(1, 61), size=6, replace=False).tolist())
            key = tuple(game)
            if key not in seen:
                seen.add(key)
                games.append(game)
        games_by_concurso[int(concurso)] = games

    return _evaluate_games_for_concursos(
        df=df,
        test_df=test_df,
        games_by_concurso=games_by_concurso,
        label="ALEATÓRIO",
        pool_size=60,
        extra_info={"random_seed": seed},
    )


def run_random_baseline_average(
    df: pd.DataFrame,
    test_df: pd.DataFrame,
    games_per_concurso: int = 5,
    random_trials: int = 50,
    seed_base: int = 1000,
) -> dict:
    summaries = []
    for k in range(random_trials):
        summaries.append(
            predict_random_games(
                df=df,
                test_df=test_df,
                games_per_concurso=games_per_concurso,
                seed=seed_base + k,
            )
        )

    keys_mean = [
        "hit_rate_all_games",
        "hit_rate_best_games",
        "avg_hits_all_games",
        "avg_hits_best_games",
        "prize_score_total",
        "best_prize_score_total",
        "num_ternos_all",
        "num_quadras_all",
        "num_quinas_all",
        "num_ternos_best",
        "num_quadras_best",
        "num_quinas_best",
        "max_hits_all",
        "max_hits_best",
    ]

    avg = {}
    for key in keys_mean:
        avg[key] = float(np.mean([s[key] for s in summaries]))

    avg["label"] = f"ALEATÓRIO MÉDIO ({random_trials} trials)"
    avg["total_concursos"] = summaries[0]["total_concursos"]
    avg["games_per_concurso"] = summaries[0]["games_per_concurso"]
    avg["total_games_generated"] = summaries[0]["total_games_generated"]
    avg["pool_size"] = 60
    avg["random_trials"] = random_trials
    return avg


# ======================================================================================
# REPORTS
# ======================================================================================

def print_games_with_matches(summary: dict, top_n: int = 20) -> None:
    matched_games = [g for g in summary["all_games"] if g["hits"] > 0]
    matched_games = sorted(matched_games, key=lambda g: (g["hits"], g["prize_score"]), reverse=True)

    print("\n" + "=" * 100)
    print(f"JOGOS COM MATCH - {summary['label']}")
    print("=" * 100)
    print(f"Concursos no teste:         {summary['total_concursos']}")
    print(f"Jogos por concurso:         {summary['games_per_concurso']}")
    print(f"Total de jogos gerados:     {summary['total_games_generated']}")
    print(f"Top report mostrado:        {min(top_n, len(matched_games))}")

    if not matched_games:
        print("Nenhum jogo com match foi encontrado.")
        return

    for g in matched_games[:top_n]:
        extra_local = f" | Local {g['local']}" if g["hits"] >= 4 else ""
        print(
            f"Concurso {g['concurso']} | Data {g['data']}{extra_local} | "
            f"Jogo #{g['game_id']} | "
            f"Acertos {g['hits']}/6 | Prize {g['prize_score']} | "
            f"Previsto {g['predicted']} | Real {g['actual']} | Match {g['matched']}"
        )


def print_summary_metrics(summary: dict) -> None:
    print("\n" + "=" * 100)
    print(f"MÉTRICAS RESUMIDAS - {summary['label']}")
    print("=" * 100)
    print(f"Concursos analisados:                 {summary['total_concursos']}")
    print(f"Jogos por concurso:                   {summary['games_per_concurso']}")
    print(f"Total de jogos gerados:               {summary['total_games_generated']}")
    print(f"Pool de dezenas por concurso:         top-{summary['pool_size']}")

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


def print_vs_random(model_summary: dict, random_avg: dict) -> None:
    print("\n" + "=" * 100)
    print("COMPARATIVO: MODELO VS ALEATÓRIO MÉDIO")
    print("=" * 100)

    def safe_div(a, b):
        return (a / b) if b not in (0, 0.0) else float("inf") if a > 0 else 0.0

    print(f"Modelo:        {model_summary['label']}")
    print(f"Aleatório avg: {random_avg['label']}")

    print("\n--- Todos os jogos ---")
    print(f"Hit rate modelo:                  {model_summary['hit_rate_all_games']:.4f}")
    print(f"Hit rate aleatório médio:         {random_avg['hit_rate_all_games']:.4f}")
    print(f"Lift hit rate vs acaso:           {safe_div(model_summary['hit_rate_all_games'], random_avg['hit_rate_all_games']):.4f}x")

    print(f"Média acertos modelo:             {model_summary['avg_hits_all_games']:.4f}")
    print(f"Média acertos aleatório:          {random_avg['avg_hits_all_games']:.4f}")
    print(f"Lift média acertos:               {safe_div(model_summary['avg_hits_all_games'], random_avg['avg_hits_all_games']):.4f}x")

    print(f"Prize score modelo:               {model_summary['prize_score_total']:.2f}")
    print(f"Prize score aleatório médio:      {random_avg['prize_score_total']:.2f}")
    print(f"Lift prize score vs acaso:        {safe_div(model_summary['prize_score_total'], random_avg['prize_score_total']):.4f}x")

    print(f"Quadras modelo:                   {model_summary['num_quadras_all']}")
    print(f"Quadras aleatório médio:          {random_avg['num_quadras_all']:.2f}")
    print(f"Quinas modelo:                    {model_summary['num_quinas_all']}")
    print(f"Quinas aleatório médio:           {random_avg['num_quinas_all']:.2f}")

    print("\n--- Melhor jogo por concurso ---")
    print(f"Hit rate best modelo:             {model_summary['hit_rate_best_games']:.4f}")
    print(f"Hit rate best aleatório médio:    {random_avg['hit_rate_best_games']:.4f}")
    print(f"Lift best vs acaso:               {safe_div(model_summary['hit_rate_best_games'], random_avg['hit_rate_best_games']):.4f}x")

    print(f"Prize score best modelo:          {model_summary['best_prize_score_total']:.2f}")
    print(f"Prize score best aleatório médio: {random_avg['best_prize_score_total']:.2f}")
    print(f"Lift best prize vs acaso:         {safe_div(model_summary['best_prize_score_total'], random_avg['best_prize_score_total']):.4f}x")

    print(f"Max hits best modelo:             {model_summary['max_hits_best']}")
    print(f"Max hits best aleatório médio:    {random_avg['max_hits_best']:.2f}")


def composite_score(summary: dict) -> float:
    """
    Score composto da v8:
    prioriza consistência, mas sem perder prêmio.
    """
    score = 0.0
    score += summary["avg_hits_all_games"] * 20.0
    score += summary["hit_rate_all_games"] * 100.0
    score += summary["num_ternos_all"] * 1.5
    score += summary["num_quadras_all"] * 8.0
    score += summary["num_quinas_all"] * 30.0
    score += summary["prize_score_total"] * 1.0
    score += summary["max_hits_all"] * 2.0
    return score


# ======================================================================================
# MAIN
# ======================================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ML Mega Sena v8 - evolução conservadora da v6.5"
    )
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--pool-size", type=int, default=10)
    parser.add_argument("--top-report", type=int, default=20)
    parser.add_argument("--random-trials", type=int, default=50)
    parser.add_argument("--random-seed", type=int, default=42)

    args = parser.parse_args()

    if not DATA_PATH.exists():
        print(f"Erro: Arquivo não encontrado: {DATA_PATH}")
        sys.exit(1)

    df = load_mega_sena(DATA_PATH)

    print("=" * 100)
    print("MEGA SENA ML v8 - BASE v6.5 + CONSISTÊNCIA + ACASO ROBUSTO")
    print("=" * 100)
    print(f"Concursos carregados: {len(df)}")
    print(f"Período: {df['data'].iloc[0]} até {df['data'].iloc[-1]}")
    print(f"Window:              {args.window}")
    print(f"Pool:                {args.pool_size}")
    print(f"Random trials:       {args.random_trials}")
    print(f"Random seed:         {args.random_seed}")

    features_df = build_frequency_features(df, window=args.window)
    features_df = build_target(df, features_df)
    train_df, test_df = split_by_date(df, features_df)

    print(f"Treino: {len(train_df)} concursos")
    print(f"Teste:  {len(test_df)} concursos")

    results = train_and_evaluate(train_df=train_df, test_df=test_df)

    rotating_core_summary = predict_rotating_core_games(
        df=df,
        test_df=test_df,
        results=results,
        pool_size=args.pool_size,
    )

    rotating_balance_summary = predict_rotating_balance_games(
        df=df,
        test_df=test_df,
        results=results,
        pool_size=args.pool_size,
    )

    random_avg = run_random_baseline_average(
        df=df,
        test_df=test_df,
        games_per_concurso=5,
        random_trials=args.random_trials,
        seed_base=1000,
    )

    candidates = [rotating_core_summary, rotating_balance_summary]
    best_model_summary = sorted(
        candidates,
        key=lambda s: (
            composite_score(s),
            s["num_quinas_all"],
            s["num_quadras_all"],
            s["prize_score_total"],
            s["avg_hits_all_games"],
        ),
        reverse=True
    )[0]

    print_games_with_matches(best_model_summary, top_n=args.top_report)

    print_summary_metrics(rotating_core_summary)
    print_summary_metrics(rotating_balance_summary)

    print("\n" + "=" * 100)
    print("COMPARATIVO INTERNO V8")
    print("=" * 100)
    print(f"Composite score - ROTATING CORE:     {composite_score(rotating_core_summary):.4f}")
    print(f"Composite score - ROTATING BALANCE:  {composite_score(rotating_balance_summary):.4f}")
    print(f"Melhor cenário v8:                   {best_model_summary['label']}")

    print_vs_random(best_model_summary, random_avg)


if __name__ == "__main__":
    main()