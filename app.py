"""
app.py - v6.3
Mega Sena ML focado em jogos fortes, com:

- XGBoost selective como preset principal
- benchmark reduzido para os cenários mais promissores
- correção da seleção do melhor cenário
- Monte Carlo com diversidade mínima entre jogos
- exibição do local de sorteio quando houver 4+ acertos
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


def load_mega_sena(filepath: Path) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df["data_parsed"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df = df.sort_values("concurso").reset_index(drop=True)
    return df


def get_draw_location(row: pd.Series) -> str:
    return str(row["local"]) if pd.notna(row["local"]) else "Local não disponível"


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
    all_probabilities = {}
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
        all_probabilities[n] = float(np.mean(proba_col))
        per_concurso_probas[n] = proba_col

    ranking = sorted(all_probabilities.items(), key=lambda x: x[1], reverse=True)

    return {
        "results_per_number": results_per_number,
        "ranking": ranking,
        "feature_columns": feature_cols,
        "per_concurso_probas": per_concurso_probas,
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
    score += len(set(game).intersection(set(ranked_pool[:4]))) * 0.15
    return score


def _game_distance(g1: list[int], g2: list[int]) -> int:
    return len(set(g1) ^ set(g2))


def _select_diverse_games(candidate_games: dict[tuple, float], keep_games: int, min_distance: int = 4) -> list[list[int]]:
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

    return [sorted(g) for g in selected[:keep_games]]


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


def predict_standard_games(
    df: pd.DataFrame,
    test_df: pd.DataFrame,
    results: dict,
    pool_size: int = 8,
) -> dict:
    per_concurso_probas = results["per_concurso_probas"]
    test_concursos = test_df["concurso"].values
    num_test = len(test_concursos)

    concursos_data = []
    all_generated_games = []

    for i in range(num_test):
        concurso = int(test_concursos[i])
        original_row = df[df["concurso"] == concurso].iloc[0]
        actual_numbers = sorted(original_row[DEZENA_COLS].astype(int).values.tolist())
        data_jogo = original_row["data"]
        local_jogo = get_draw_location(original_row)

        probas = {n: float(per_concurso_probas[n][i]) for n in range(1, TOTAL_NUMBERS + 1)}
        ranked_numbers = [n for n, _ in sorted(probas.items(), key=lambda x: x[1], reverse=True)]

        top_pool = ranked_numbers[:pool_size]
        games = _build_candidate_games_from_pool(top_pool)

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
                "top_pool": top_pool,
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
            "top_pool": top_pool,
            "games": game_results,
            "best_game": best_game,
        })

    return summarize_generated_games(
        concursos_data=concursos_data,
        all_generated_games=all_generated_games,
        games_per_concurso=3,
        pool_size=pool_size,
        label="PADRÃO",
    )


def predict_monte_carlo_games(
    df: pd.DataFrame,
    test_df: pd.DataFrame,
    results: dict,
    pool_size: int = 8,
    mc_samples: int = 1000,
    mc_keep_games: int = 3,
    seed: int = 42,
) -> dict:
    per_concurso_probas = results["per_concurso_probas"]
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
        local_jogo = get_draw_location(original_row)

        probas = {n: float(per_concurso_probas[n][i]) for n in range(1, TOTAL_NUMBERS + 1)}
        ranked_numbers = [n for n, _ in sorted(probas.items(), key=lambda x: x[1], reverse=True)]

        top_pool = ranked_numbers[:pool_size]
        weights = [float(pool_size - idx) for idx in range(pool_size)]

        candidate_games = {}
        for _ in range(mc_samples):
            game = _weighted_sample_game(top_pool, weights, rng)
            key = tuple(game)
            score = _score_game_internal(game, top_pool)
            if key not in candidate_games or score > candidate_games[key]:
                candidate_games[key] = score

        selected_games = _select_diverse_games(candidate_games, keep_games=mc_keep_games, min_distance=4)

        game_results = []
        best_game = None
        best_hits = -1
        best_prize = -1

        for idx_game, predicted_numbers in enumerate(selected_games, start=1):
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
                "top_pool": top_pool,
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
            "top_pool": top_pool,
            "games": game_results,
            "best_game": best_game,
        })

    return summarize_generated_games(
        concursos_data=concursos_data,
        all_generated_games=all_generated_games,
        games_per_concurso=mc_keep_games,
        pool_size=pool_size,
        label="MONTE CARLO",
        extra_info={"mc_samples": mc_samples},
    )


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
        extra_local = f" | Local {g['local']}" if g["hits"] >= 4 else ""
        print(
            f"Concurso {g['concurso']} | Data {g['data']}{extra_local} | "
            f"Jogo #{g['game_id']} | "
            f"Acertos {g['hits']}/6 | Prize {g['prize_score']} | "
            f"Previsto {g['predicted']} | Real {g['actual']} | Match {g['matched']}"
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


def print_benchmark_table(benchmark_df: pd.DataFrame, top_n: int = 12) -> None:
    print("\n" + "=" * 120)
    print("BENCHMARK AUTOMÁTICO DOS CENÁRIOS")
    print("=" * 120)

    if benchmark_df.empty:
        print("Nenhum cenário foi gerado.")
        return

    cols = [
        "window", "pool_size", "mode",
        "num_quadras_best", "num_quinas_best",
        "max_hits_best", "best_prize_score_total",
        "avg_hits_best_games", "hit_rate_best_games"
    ]

    show_df = benchmark_df[cols].sort_values(
        by=["num_quinas_best", "num_quadras_best", "max_hits_best", "best_prize_score_total", "avg_hits_best_games"],
        ascending=False
    ).head(top_n)

    print(show_df.to_string(index=False))


def run_single_scenario(
    df: pd.DataFrame,
    window: int,
    pool_size: int,
    mc_samples: int,
    mc_keep_games: int,
) -> dict:
    print(f"\nExecutando cenário -> window={window} | pool={pool_size}")

    features_df = build_frequency_features(df, window=window)
    features_df = build_target(df, features_df)
    train_df, test_df = split_by_date(df, features_df)

    results = train_and_evaluate(train_df=train_df, test_df=test_df)

    standard_summary = predict_standard_games(
        df=df,
        test_df=test_df,
        results=results,
        pool_size=pool_size,
    )

    monte_carlo_summary = predict_monte_carlo_games(
        df=df,
        test_df=test_df,
        results=results,
        pool_size=pool_size,
        mc_samples=mc_samples,
        mc_keep_games=mc_keep_games,
    )

    return {
        "results": results,
        "standard_summary": standard_summary,
        "monte_carlo_summary": monte_carlo_summary,
        "train_size": len(train_df),
        "test_size": len(test_df),
    }


def run_benchmark(
    df: pd.DataFrame,
    windows: list[int],
    pools: list[int],
    mc_samples: int,
    mc_keep_games: int,
) -> tuple[pd.DataFrame, dict]:
    rows = []
    detailed_results = {}

    for window in windows:
        for pool_size in pools:
            scenario_key = f"w={window}|p={pool_size}"
            scenario_result = run_single_scenario(
                df=df,
                window=window,
                pool_size=pool_size,
                mc_samples=mc_samples,
                mc_keep_games=mc_keep_games,
            )
            detailed_results[scenario_key] = scenario_result

            for mode_name, summary in [
                ("standard", scenario_result["standard_summary"]),
                ("monte_carlo", scenario_result["monte_carlo_summary"]),
            ]:
                rows.append({
                    "scenario_key": scenario_key,
                    "window": window,
                    "pool_size": pool_size,
                    "mode": mode_name,
                    "num_quadras_best": summary["num_quadras_best"],
                    "num_quinas_best": summary["num_quinas_best"],
                    "num_senas_best": summary["num_senas_best"],
                    "max_hits_best": summary["max_hits_best"],
                    "best_prize_score_total": summary["best_prize_score_total"],
                    "avg_hits_best_games": summary["avg_hits_best_games"],
                    "hit_rate_best_games": summary["hit_rate_best_games"],
                    "num_ternos_best": summary["num_ternos_best"],
                })

    benchmark_df = pd.DataFrame(rows)
    return benchmark_df, detailed_results


def main():
    parser = argparse.ArgumentParser(
        description="ML Mega Sena v6.3 - XGBoost selective + benchmark reduzido + local de sorteio"
    )
    parser.add_argument(
        "--windows",
        type=str,
        default="15,30",
        help='Lista de windows separadas por vírgula. Ex: "15,30"',
    )
    parser.add_argument(
        "--pools",
        type=str,
        default="8,9",
        help='Lista de pool-sizes separadas por vírgula. Ex: "8,9"',
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=1000,
        help="Quantidade de amostras Monte Carlo por concurso (default: 1000)",
    )
    parser.add_argument(
        "--mc-keep-games",
        type=int,
        default=3,
        help="Quantidade de jogos Monte Carlo mantidos por concurso (default: 3)",
    )
    parser.add_argument(
        "--top-report",
        type=int,
        default=20,
        help="Quantidade de jogos com match a mostrar no relatório detalhado final (default: 20)",
    )
    parser.add_argument(
        "--show-best-details",
        action="store_true",
        help="Mostra o relatório detalhado do melhor cenário ao final",
    )

    args = parser.parse_args()

    if not DATA_PATH.exists():
        print(f"Erro: Arquivo não encontrado: {DATA_PATH}")
        sys.exit(1)

    df = load_mega_sena(DATA_PATH)

    windows = [int(x.strip()) for x in args.windows.split(",") if x.strip()]
    pools = [int(x.strip()) for x in args.pools.split(",") if x.strip()]

    print("=" * 100)
    print("MEGA SENA ML v6.3 - BENCHMARK REDUZIDO")
    print("=" * 100)
    print(f"Concursos carregados: {len(df)}")
    print(f"Período: {df['data'].iloc[0]} até {df['data'].iloc[-1]}")
    print(f"Windows testadas: {windows}")
    print(f"Pools testados:   {pools}")
    print(f"Preset:           selective")
    print(f"Monte Carlo:      {args.mc_samples} amostras / {args.mc_keep_games} jogos mantidos")

    benchmark_df, detailed_results = run_benchmark(
        df=df,
        windows=windows,
        pools=pools,
        mc_samples=args.mc_samples,
        mc_keep_games=args.mc_keep_games,
    )

    print_benchmark_table(benchmark_df, top_n=10)

    if args.show_best_details and not benchmark_df.empty:
        best_row = benchmark_df.sort_values(
            by=["num_quinas_best", "num_quadras_best", "max_hits_best", "best_prize_score_total", "avg_hits_best_games"],
            ascending=False
        ).iloc[0]

        best_key = best_row["scenario_key"]
        best_mode = best_row["mode"]
        best_result = detailed_results[best_key]

        print("\n" + "=" * 100)
        print(f"RELATÓRIO DETALHADO DO MELHOR CENÁRIO: {best_key} | mode={best_mode}")
        print("=" * 100)

        summary = best_result["monte_carlo_summary"] if best_mode == "monte_carlo" else best_result["standard_summary"]
        print_games_with_matches(summary, top_n=args.top_report)
        print_summary_metrics(summary)


if __name__ == "__main__":
    main()