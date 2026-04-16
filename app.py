"""
app.py - v7
Mega Sena ML com:

- dataset longo
- XGBRanker
- ranking por concurso
- walk-forward validation
- comparação contra múltiplos aleatórios
- geração de jogos por ranking
- métricas de consistência + prêmio

Objetivo:
- melhorar acerto médio
- manter/buscar quadras e quinas
- medir se estamos acima do acaso de forma robusta
"""

import argparse
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from xgboost import XGBRanker

DATA_PATH = Path(__file__).resolve().parent / "data" / "raw" / "mega_sena_2000_2026.csv"
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
    if "local" in row.index and pd.notna(row["local"]):
        return str(row["local"])
    return "Local não disponível"


def _get_recent_draws(all_draws: np.ndarray, i: int, window: int) -> np.ndarray:
    start = max(0, i - window)
    return all_draws[start:i]


def _calc_number_features(recent_draws: np.ndarray, n: int, window: int) -> dict:
    if len(recent_draws) == 0:
        return {"freq": 0.0, "delay": float(window + 1), "streak": 0.0}

    flat = recent_draws.flatten()
    freq = float(np.sum(flat == n) / len(recent_draws))

    last_seen = -1
    for j in range(len(recent_draws) - 1, -1, -1):
        if n in recent_draws[j]:
            last_seen = j
            break
    delay = float((len(recent_draws) - last_seen) if last_seen >= 0 else (len(recent_draws) + 1))

    streak = 0
    for j in range(len(recent_draws) - 1, -1, -1):
        if n in recent_draws[j]:
            streak += 1
        else:
            break

    return {"freq": freq, "delay": delay, "streak": float(streak)}


def _build_context_features(last_draw: list[int], recent_draws: np.ndarray) -> dict:
    soma_ultimo = int(np.sum(last_draw))
    pares_ultimo = int(np.sum(np.array(last_draw) % 2 == 0))
    impares_ultimo = 6 - pares_ultimo

    somas = [int(np.sum(d)) for d in recent_draws] if len(recent_draws) > 0 else [0]
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

    decadas = {}
    for dec in range(6):
        low = dec * 10 + 1
        high = (dec + 1) * 10
        decadas[f"decada_{low}_{high}"] = sum(1 for n in last_draw if low <= n <= high)

    return {
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
        "gap_medio": float(np.mean(gaps)) if gaps else 0.0,
        "gap_max": float(max(gaps)) if gaps else 0.0,
        "gap_min": float(min(gaps)) if gaps else 0.0,
        "gap_std": float(np.std(gaps)) if len(gaps) > 1 else 0.0,
        "amplitude": float(last_draw[-1] - last_draw[0]),
        "mediana": float(np.median(last_draw)),
        **decadas,
    }


def build_long_dataset(
    df: pd.DataFrame,
    min_window: int = 30,
    windows: tuple[int, ...] = (5, 10, 20, 30, 60),
) -> pd.DataFrame:
    all_draws = df[DEZENA_COLS].values.astype(int)
    rows = []

    for i in range(min_window, len(df)):
        actual_draw = set(df.iloc[i][DEZENA_COLS].astype(int).tolist())
        last_draw = sorted(all_draws[i - 1].tolist())
        recent_base = _get_recent_draws(all_draws, i, min_window)
        context = _build_context_features(last_draw, recent_base)

        number_rows = []
        for n in range(1, TOTAL_NUMBERS + 1):
            row = {
                "concurso": int(df.iloc[i]["concurso"]),
                "data": df.iloc[i]["data"],
                "data_parsed": df.iloc[i]["data_parsed"],
                "idx": i,
                "local": get_draw_location(df.iloc[i]),
                "dezena": n,
                "target": 1 if n in actual_draw else 0,
                "is_even": 1 if n % 2 == 0 else 0,
                "bucket_decade": int((n - 1) // 10) + 1,
                "quadrant_num": (
                    1 if 1 <= n <= 15 else
                    2 if 16 <= n <= 30 else
                    3 if 31 <= n <= 45 else
                    4
                ),
            }

            for w in windows:
                recent_draws = _get_recent_draws(all_draws, i, w)
                feats = _calc_number_features(recent_draws, n, w)
                row[f"freq_w{w}"] = feats["freq"]
                row[f"delay_w{w}"] = feats["delay"]
                row[f"streak_w{w}"] = feats["streak"]

            row["trend_10_60"] = row.get("freq_w10", 0.0) - row.get("freq_w60", 0.0)
            row["trend_5_30"] = row.get("freq_w5", 0.0) - row.get("freq_w30", 0.0)

            row.update(context)
            number_rows.append(row)

        tmp = pd.DataFrame(number_rows)

        relative_cols = [
            "freq_w10", "freq_w30", "freq_w60",
            "delay_w10", "delay_w30", "delay_w60",
            "streak_w10", "streak_w30",
            "trend_10_60", "trend_5_30",
        ]

        for col in relative_cols:
            tmp[f"rank_desc_{col}"] = tmp[col].rank(method="average", ascending=False)
            tmp[f"rank_asc_{col}"] = tmp[col].rank(method="average", ascending=True)
            tmp[f"pct_{col}"] = tmp[col].rank(method="average", pct=True)

            std = tmp[col].std()
            mean = tmp[col].mean()
            if std == 0 or pd.isna(std):
                tmp[f"z_{col}"] = 0.0
            else:
                tmp[f"z_{col}"] = (tmp[col] - mean) / std

        rows.append(tmp)

    return pd.concat(rows, ignore_index=True)


def build_ranker() -> XGBRanker:
    return XGBRanker(
        objective="rank:pairwise",
        n_estimators=500,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=3,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )


def get_feature_columns(long_df: pd.DataFrame) -> list[str]:
    exclude = {
        "concurso", "data", "data_parsed", "idx", "local",
        "target"
    }
    return [c for c in long_df.columns if c not in exclude]


def fit_ranker(train_df: pd.DataFrame, feature_cols: list[str]) -> XGBRanker:
    X_train = train_df[feature_cols].values
    y_train = train_df["target"].values
    group_train = train_df.groupby("concurso").size().to_list()

    model = build_ranker()
    model.fit(X_train, y_train, group=group_train)
    return model


def score_concursos(model: XGBRanker, df_part: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = df_part.copy()
    X = out[feature_cols].values
    out["score"] = model.predict(X)
    return out


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


def _weighted_sample_game(pool_numbers: list[int], pool_weights: list[float], rng: np.random.Generator) -> list[int]:
    weights = np.array(pool_weights, dtype=float)
    weights = weights / weights.sum()
    chosen = rng.choice(pool_numbers, size=6, replace=False, p=weights)
    return sorted(int(x) for x in chosen.tolist())


def _game_distance(g1: list[int], g2: list[int]) -> int:
    return len(set(g1) ^ set(g2))


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

    final = []
    seen = set()
    for g in selected:
        key = tuple(sorted(g))
        if key not in seen:
            final.append(sorted(g))
            seen.add(key)

    return final[:keep_games]


def _build_rotating_core_games(ranked_pool: list[int]) -> list[list[int]]:
    rp = ranked_pool[:10]
    if len(rp) < 10:
        rp = ranked_pool

    if len(rp) < 8:
        return [sorted(rp[:6])]

    core = rp[:4]
    tail = rp[4:]

    candidates = []
    pairs = [
        (tail[0], tail[1]),
        (tail[0], tail[2]) if len(tail) > 2 else None,
        (tail[1], tail[2]) if len(tail) > 2 else None,
        (tail[0], tail[3]) if len(tail) > 3 else None,
        (tail[1], tail[3]) if len(tail) > 3 else None,
    ]

    for pair in pairs:
        if pair is None:
            continue
        game = sorted(core + [pair[0], pair[1]])
        if len(set(game)) == 6:
            candidates.append(game)

    unique = []
    seen = set()
    for g in candidates:
        key = tuple(g)
        if key not in seen:
            unique.append(g)
            seen.add(key)

    return unique[:5]


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


def _evaluate_games_for_concursos(scored_test_df: pd.DataFrame, games_by_concurso: dict, label: str, pool_size: int, extra_info: dict | None = None) -> dict:
    concursos_data = []
    all_generated_games = []

    for concurso, concurso_df in scored_test_df.groupby("concurso"):
        concurso = int(concurso)
        row0 = concurso_df.iloc[0]
        actual_numbers = sorted(concurso_df.loc[concurso_df["target"] == 1, "dezena"].astype(int).tolist())
        data_jogo = row0["data"]
        local_jogo = row0["local"]

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


def predict_rotating_core_games(scored_test_df: pd.DataFrame, pool_size: int = 10) -> dict:
    games_by_concurso = {}

    for concurso, concurso_df in scored_test_df.groupby("concurso"):
        ranked_numbers = concurso_df.sort_values("score", ascending=False)["dezena"].astype(int).tolist()
        top_pool = ranked_numbers[:pool_size]
        games = _build_rotating_core_games(top_pool)

        if len(games) < 5:
            fallback = sorted(top_pool[:6])
            while len(games) < 5:
                if fallback not in games:
                    games.append(fallback)
                else:
                    break

        games_by_concurso[int(concurso)] = games[:5]

    return _evaluate_games_for_concursos(
        scored_test_df=scored_test_df,
        games_by_concurso=games_by_concurso,
        label="ROTATING CORE",
        pool_size=pool_size,
    )


def predict_monte_carlo_games(scored_test_df: pd.DataFrame, pool_size: int = 10, mc_samples: int = 3000, mc_keep_games: int = 5, mc_mode: str = "core", seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    games_by_concurso = {}

    for concurso, concurso_df in scored_test_df.groupby("concurso"):
        ranked_numbers = concurso_df.sort_values("score", ascending=False)["dezena"].astype(int).tolist()
        top_pool = ranked_numbers[:pool_size]
        weights = [float(pool_size - idx) for idx in range(pool_size)]

        candidate_games = {}
        for _ in range(mc_samples):
            game = _weighted_sample_game(top_pool, weights, rng)
            key = tuple(game)
            score = _score_game_internal(game, top_pool, mode=mc_mode)
            if key not in candidate_games or score > candidate_games[key]:
                candidate_games[key] = score

        selected_games = _select_diverse_games(candidate_games, keep_games=mc_keep_games, min_distance=6)
        games_by_concurso[int(concurso)] = selected_games

    return _evaluate_games_for_concursos(
        scored_test_df=scored_test_df,
        games_by_concurso=games_by_concurso,
        label=f"MONTE CARLO ({mc_mode.upper()})",
        pool_size=pool_size,
        extra_info={"mc_samples": mc_samples, "mc_mode": mc_mode},
    )


def predict_random_games(scored_test_df: pd.DataFrame, games_per_concurso: int = 5, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    games_by_concurso = {}

    for concurso in scored_test_df["concurso"].unique().tolist():
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
        scored_test_df=scored_test_df,
        games_by_concurso=games_by_concurso,
        label="ALEATÓRIO",
        pool_size=60,
        extra_info={"random_seed": seed},
    )


def print_summary_metrics(summary: dict) -> None:
    print("\n" + "=" * 100)
    print(f"MÉTRICAS RESUMIDAS - {summary['label']}")
    print("=" * 100)
    print(f"Concursos analisados:                 {summary['total_concursos']}")
    print(f"Jogos por concurso:                   {summary['games_per_concurso']}")
    print(f"Total de jogos gerados:               {summary['total_games_generated']}")
    print(f"Pool de dezenas por concurso:         top-{summary['pool_size']}")
    if "mc_samples" in summary:
        print(f"Amostras Monte Carlo por concurso:    {summary['mc_samples']}")
    if "mc_mode" in summary:
        print(f"Modo Monte Carlo:                     {summary['mc_mode']}")

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


def print_vs_random(model_summary: dict, random_summary: dict) -> None:
    print("\n" + "=" * 100)
    print("COMPARATIVO: MODELO VS ALEATÓRIO")
    print("=" * 100)

    def safe_div(a, b):
        return (a / b) if b not in (0, 0.0) else float("inf") if a > 0 else 0.0

    print(f"Modelo:   {model_summary['label']}")
    print(f"Aleatório:{random_summary['label']}")

    print("\n--- Todos os jogos ---")
    print(f"Hit rate modelo:                  {model_summary['hit_rate_all_games']:.4f}")
    print(f"Hit rate aleatório:               {random_summary['hit_rate_all_games']:.4f}")
    print(f"Lift hit rate vs acaso:           {safe_div(model_summary['hit_rate_all_games'], random_summary['hit_rate_all_games']):.4f}x")
    print(f"Prize score modelo:               {model_summary['prize_score_total']}")
    print(f"Prize score aleatório:            {random_summary['prize_score_total']}")
    print(f"Lift prize score vs acaso:        {safe_div(model_summary['prize_score_total'], random_summary['prize_score_total']):.4f}x")

    print("\n--- Melhor jogo por concurso ---")
    print(f"Hit rate best modelo:             {model_summary['hit_rate_best_games']:.4f}")
    print(f"Hit rate best aleatório:          {random_summary['hit_rate_best_games']:.4f}")
    print(f"Lift best vs acaso:               {safe_div(model_summary['hit_rate_best_games'], random_summary['hit_rate_best_games']):.4f}x")
    print(f"Prize score best modelo:          {model_summary['best_prize_score_total']}")
    print(f"Prize score best aleatório:       {random_summary['best_prize_score_total']}")
    print(f"Lift best prize vs acaso:         {safe_div(model_summary['best_prize_score_total'], random_summary['best_prize_score_total']):.4f}x")
    print(f"Max hits best modelo:             {model_summary['max_hits_best']}")
    print(f"Max hits best aleatório:          {random_summary['max_hits_best']}")


def build_walk_forward_folds(concursos: list[int], train_concursos: int, test_concursos: int, step_concursos: int) -> list[tuple[list[int], list[int]]]:
    folds = []
    total = len(concursos)
    start = train_concursos

    while start + test_concursos <= total:
        train_ids = concursos[:start]
        test_ids = concursos[start:start + test_concursos]
        folds.append((train_ids, test_ids))
        start += step_concursos

    return folds


def run_walk_forward(long_df: pd.DataFrame, feature_cols: list[str], pool_size: int, mc_samples: int, mc_keep_games: int, mc_mode: str, random_trials: int, train_concursos: int, test_concursos: int, step_concursos: int) -> pd.DataFrame:
    concursos = sorted(long_df["concurso"].unique().tolist())
    folds = build_walk_forward_folds(concursos, train_concursos, test_concursos, step_concursos)

    rows = []

    for fold_idx, (train_ids, test_ids) in enumerate(folds, start=1):
        train_df = long_df[long_df["concurso"].isin(train_ids)].copy()
        test_df = long_df[long_df["concurso"].isin(test_ids)].copy()

        model = fit_ranker(train_df, feature_cols)
        scored_test = score_concursos(model, test_df, feature_cols)

        rotating_core = predict_rotating_core_games(scored_test, pool_size=pool_size)
        monte_carlo = predict_monte_carlo_games(
            scored_test,
            pool_size=pool_size,
            mc_samples=mc_samples,
            mc_keep_games=mc_keep_games,
            mc_mode=mc_mode,
            seed=42 + fold_idx,
        )

        random_summaries = []
        for trial in range(random_trials):
            random_summaries.append(
                predict_random_games(scored_test, games_per_concurso=mc_keep_games, seed=1000 + fold_idx * 100 + trial)
            )

        def avg_metric(summaries, key):
            vals = [s[key] for s in summaries]
            return float(np.mean(vals))

        random_avg = {
            "hit_rate_all_games": avg_metric(random_summaries, "hit_rate_all_games"),
            "hit_rate_best_games": avg_metric(random_summaries, "hit_rate_best_games"),
            "prize_score_total": avg_metric(random_summaries, "prize_score_total"),
            "best_prize_score_total": avg_metric(random_summaries, "best_prize_score_total"),
            "max_hits_best": avg_metric(random_summaries, "max_hits_best"),
            "num_quadras_best": avg_metric(random_summaries, "num_quadras_best"),
            "num_quinas_best": avg_metric(random_summaries, "num_quinas_best"),
        }

        for label, summary in [("rotating_core", rotating_core), ("monte_carlo", monte_carlo)]:
            rows.append({
                "fold": fold_idx,
                "mode": label,
                "model_hit_rate": summary["hit_rate_all_games"],
                "model_best_hit_rate": summary["hit_rate_best_games"],
                "model_prize_score": summary["prize_score_total"],
                "model_best_prize_score": summary["best_prize_score_total"],
                "model_max_hits_best": summary["max_hits_best"],
                "model_quadras_best": summary["num_quadras_best"],
                "model_quinas_best": summary["num_quinas_best"],
                "random_hit_rate": random_avg["hit_rate_all_games"],
                "random_best_hit_rate": random_avg["hit_rate_best_games"],
                "random_prize_score": random_avg["prize_score_total"],
                "random_best_prize_score": random_avg["best_prize_score_total"],
                "random_max_hits_best": random_avg["max_hits_best"],
                "random_quadras_best": random_avg["num_quadras_best"],
                "random_quinas_best": random_avg["num_quinas_best"],
            })

    return pd.DataFrame(rows)


def print_walk_forward_summary(wf_df: pd.DataFrame) -> None:
    print("\n" + "=" * 100)
    print("WALK-FORWARD SUMMARY")
    print("=" * 100)

    if wf_df.empty:
        print("Nenhum fold gerado.")
        return

    grouped = wf_df.groupby("mode").mean(numeric_only=True)

    for mode, row in grouped.iterrows():
        print(f"\nModo: {mode}")
        print(f"Hit rate modelo:              {row['model_hit_rate']:.4f}")
        print(f"Hit rate aleatório:           {row['random_hit_rate']:.4f}")
        print(f"Lift hit rate:                {(row['model_hit_rate'] / row['random_hit_rate']) if row['random_hit_rate'] > 0 else 0.0:.4f}x")
        print(f"Best hit rate modelo:         {row['model_best_hit_rate']:.4f}")
        print(f"Best hit rate aleatório:      {row['random_best_hit_rate']:.4f}")
        print(f"Lift best hit rate:           {(row['model_best_hit_rate'] / row['random_best_hit_rate']) if row['random_best_hit_rate'] > 0 else 0.0:.4f}x")
        print(f"Prize score modelo:           {row['model_prize_score']:.2f}")
        print(f"Prize score aleatório:        {row['random_prize_score']:.2f}")
        print(f"Best prize modelo:            {row['model_best_prize_score']:.2f}")
        print(f"Best prize aleatório:         {row['random_best_prize_score']:.2f}")
        print(f"Max hits best modelo:         {row['model_max_hits_best']:.2f}")
        print(f"Max hits best aleatório:      {row['random_max_hits_best']:.2f}")
        print(f"Quadras best modelo:          {row['model_quadras_best']:.2f}")
        print(f"Quadras best aleatório:       {row['random_quadras_best']:.2f}")
        print(f"Quinas best modelo:           {row['model_quinas_best']:.2f}")
        print(f"Quinas best aleatório:        {row['random_quinas_best']:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="ML Mega Sena v7 - XGBRanker + walk-forward + random benchmark"
    )
    parser.add_argument("--min-window", type=int, default=30)
    parser.add_argument("--pool-size", type=int, default=10)
    parser.add_argument("--mc-mode", type=str, default="core", choices=["core", "spread"])
    parser.add_argument("--mc-samples", type=int, default=3000)
    parser.add_argument("--mc-keep-games", type=int, default=5)
    parser.add_argument("--random-trials", type=int, default=100)
    parser.add_argument("--train-concursos", type=int, default=2200)
    parser.add_argument("--test-concursos", type=int, default=43)
    parser.add_argument("--step-concursos", type=int, default=43)

    args = parser.parse_args()

    if not DATA_PATH.exists():
        print(f"Erro: Arquivo não encontrado: {DATA_PATH}")
        sys.exit(1)

    df = load_mega_sena(DATA_PATH)

    print("=" * 100)
    print("MEGA SENA ML v7 - XGBRANKER + WALK-FORWARD + ACASO")
    print("=" * 100)
    print(f"Concursos carregados: {len(df)}")
    print(f"Período: {df['data'].iloc[0]} até {df['data'].iloc[-1]}")
    print(f"min_window:         {args.min_window}")
    print(f"pool_size:          {args.pool_size}")
    print(f"mc_mode:            {args.mc_mode}")
    print(f"mc_samples:         {args.mc_samples}")
    print(f"mc_keep_games:      {args.mc_keep_games}")
    print(f"random_trials:      {args.random_trials}")
    print(f"train_concursos:    {args.train_concursos}")
    print(f"test_concursos:     {args.test_concursos}")
    print(f"step_concursos:     {args.step_concursos}")

    long_df = build_long_dataset(df, min_window=args.min_window)
    feature_cols = get_feature_columns(long_df)

    print(f"Linhas do dataset longo: {len(long_df)}")
    print(f"Concursos com features:  {long_df['concurso'].nunique()}")
    print(f"Features usadas:         {len(feature_cols)}")

    # último bloco para inspeção direta
    concursos = sorted(long_df["concurso"].unique().tolist())
    train_ids = concursos[:-args.test_concursos]
    test_ids = concursos[-args.test_concursos:]

    train_df = long_df[long_df["concurso"].isin(train_ids)].copy()
    test_df = long_df[long_df["concurso"].isin(test_ids)].copy()

    print(f"Treino final: {len(train_ids)} concursos")
    print(f"Teste final:  {len(test_ids)} concursos")

    model = fit_ranker(train_df, feature_cols)
    scored_test = score_concursos(model, test_df, feature_cols)

    rotating_core = predict_rotating_core_games(scored_test, pool_size=args.pool_size)
    monte_carlo = predict_monte_carlo_games(
        scored_test,
        pool_size=args.pool_size,
        mc_samples=args.mc_samples,
        mc_keep_games=args.mc_keep_games,
        mc_mode=args.mc_mode,
        seed=42,
    )
    random_summary = predict_random_games(
        scored_test,
        games_per_concurso=args.mc_keep_games,
        seed=42,
    )

    candidates = [rotating_core, monte_carlo]
    best_model_summary = sorted(
        candidates,
        key=lambda s: (
            s["num_quinas_best"],
            s["num_quadras_best"],
            s["max_hits_best"],
            s["best_prize_score_total"],
            s["avg_hits_best_games"],
        ),
        reverse=True
    )[0]

    print_summary_metrics(rotating_core)
    print_summary_metrics(monte_carlo)
    print_summary_metrics(random_summary)
    print_vs_random(best_model_summary, random_summary)

    wf_df = run_walk_forward(
        long_df=long_df,
        feature_cols=feature_cols,
        pool_size=args.pool_size,
        mc_samples=args.mc_samples,
        mc_keep_games=args.mc_keep_games,
        mc_mode=args.mc_mode,
        random_trials=args.random_trials,
        train_concursos=args.train_concursos,
        test_concursos=args.test_concursos,
        step_concursos=args.step_concursos,
    )
    print_walk_forward_summary(wf_df)


if __name__ == "__main__":
    main()