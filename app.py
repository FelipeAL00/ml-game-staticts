"""
app.py - v8.5 SAFE MIX

Objetivo:
- manter a linha saudável da v8.4 SAFE
- adicionar mistura forte + média + rara (3,2,1)
- reduzir vício de concentração
- buscar quadras mais distribuídas
- abrir caminho para quina

Modos comparados:
- safe_balance_5
- safe_hybrid_5
- safe_321_mix_5
- safe_321_pure_5
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
# LOAD
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
# FEATURES
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
# GAME UTILS
# ======================================================================================

def _build_ranked_pool_for_concurso(results: dict, i: int, pool_size: int) -> list[int]:
    per_concurso_probas = results["per_concurso_probas"]
    probas = {n: float(per_concurso_probas[n][i]) for n in range(1, TOTAL_NUMBERS + 1)}
    ranked_numbers = [n for n, _ in sorted(probas.items(), key=lambda x: x[1], reverse=True)]
    return ranked_numbers[:pool_size]


def _game_distance(g1: list[int], g2: list[int]) -> int:
    return len(set(g1) ^ set(g2))


def _overlap_size(g1: list[int], g2: list[int]) -> int:
    return len(set(g1) & set(g2))


def _dedupe_games(games: list[list[int]]) -> list[list[int]]:
    out = []
    seen = set()
    for g in games:
        key = tuple(sorted(g))
        if key not in seen:
            seen.add(key)
            out.append(sorted(g))
    return out


def _pairwise_redundancy_ok(candidate: list[int], selected: list[list[int]], max_overlap: int = 4) -> bool:
    return all(_overlap_size(candidate, prev) <= max_overlap for prev in selected)


# ======================================================================================
# STRATEGIES
# ======================================================================================

def build_rotating_balance_games(ranked_pool: list[int]) -> list[list[int]]:
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

    selected = []
    for g in _dedupe_games(candidates):
        if not selected or _pairwise_redundancy_ok(g, selected, max_overlap=4):
            selected.append(g)

    if len(selected) < 5:
        for g in _dedupe_games(candidates):
            if g not in selected:
                selected.append(g)
            if len(selected) == 5:
                break

    return selected[:5]


def build_safe_hybrid_games(ranked_pool: list[int]) -> list[list[int]]:
    balance_games = build_rotating_balance_games(ranked_pool)
    rp = ranked_pool[:10]
    if len(rp) < 8:
        return balance_games[:5]

    core = rp[:4]
    tail = rp[4:]
    extra = []
    pairs = [
        (tail[0], tail[1]),
        (tail[0], tail[2]) if len(tail) > 2 else None,
        (tail[1], tail[2]) if len(tail) > 2 else None,
    ]
    for pair in pairs:
        if pair is None:
            continue
        extra.append(sorted(core + [pair[0], pair[1]]))

    selected = []
    for g in balance_games[:3] + extra[:3]:
        if not selected or _pairwise_redundancy_ok(g, selected, max_overlap=4):
            selected.append(g)
        if len(selected) == 5:
            break

    if len(selected) < 5:
        for g in _dedupe_games(balance_games + extra):
            if g not in selected:
                selected.append(g)
            if len(selected) == 5:
                break

    return selected[:5]


def build_rare_candidates(ranked_full: list[int], features_row: pd.Series) -> list[int]:
    """
    Candidatos raros/frios controlados:
    - usar ranks 16..30 do ranking
    - priorizar maior atraso e menor frequência curta
    """
    rare_block = ranked_full[15:30]  # ranks 16..30
    scored = []

    for n in rare_block:
        atraso = float(features_row.get(f"atraso_{n}", 0.0))
        freq10 = float(features_row.get(f"freq_w10_{n}", 0.0))
        freq30 = float(features_row.get(f"freq_w30_{n}", 0.0))
        streak = float(features_row.get(f"streak_{n}", 0.0))

        rare_score = (
            atraso * 1.5
            - freq10 * 20.0
            - freq30 * 10.0
            - streak * 0.5
        )
        scored.append((n, rare_score))

    scored = sorted(scored, key=lambda x: x[1], reverse=True)
    return [n for n, _ in scored]


def build_321_mix_games(ranked_full: list[int], ranked_pool: list[int], features_row: pd.Series) -> list[list[int]]:
    """
    Carteira mista:
    jogo 1 -> baseline balance
    jogo 2 -> 3,2,1 conservador
    jogo 3 -> 3,2,1 padrão
    jogo 4 -> 3,1,2 exploratório
    jogo 5 -> 2,3,1 balanceado aberto
    """
    strong = ranked_full[:6]
    medium = ranked_full[6:15]
    rare = build_rare_candidates(ranked_full, features_row)

    baseline = build_rotating_balance_games(ranked_pool)
    games = []

    if baseline:
        games.append(baseline[0])

    candidates = [
        sorted(strong[:3] + medium[:2] + rare[:1]),          # 3,2,1 conservador
        sorted([strong[0], strong[1], strong[3], medium[2], medium[4], rare[1]]),   # 3,2,1 padrão
        sorted([strong[0], strong[2], strong[4], medium[1], rare[0], rare[2]]),      # 3,1,2
        sorted([strong[1], strong[3], medium[0], medium[3], medium[5], rare[3]]),    # 2,3,1
        sorted([strong[0], strong[5], medium[1], medium[6], medium[7], rare[4]]),    # alternativo
    ]

    for g in candidates:
        if len(set(g)) == 6:
            if not games or _pairwise_redundancy_ok(g, games, max_overlap=4):
                games.append(g)
        if len(games) == 5:
            break

    if len(games) < 5:
        for g in _dedupe_games(candidates + baseline):
            if g not in games:
                games.append(g)
            if len(games) == 5:
                break

    return _dedupe_games(games)[:5]


def build_321_pure_games(ranked_full: list[int], features_row: pd.Series) -> list[list[int]]:
    """
    100% baseado em 3,2,1 e variações próximas.
    """
    strong = ranked_full[:6]
    medium = ranked_full[6:15]
    rare = build_rare_candidates(ranked_full, features_row)

    candidates = [
        sorted(strong[:3] + medium[:2] + rare[:1]),                                    # 3,2,1
        sorted([strong[0], strong[1], strong[4], medium[0], medium[3], rare[1]]),     # 3,2,1
        sorted([strong[1], strong[2], strong[5], medium[1], medium[4], rare[2]]),     # 3,2,1
        sorted([strong[0], strong[2], strong[4], medium[2], rare[0], rare[3]]),       # 3,1,2
        sorted([strong[3], strong[5], medium[0], medium[5], medium[6], rare[4]]),     # 2,3,1
    ]

    selected = []
    for g in candidates:
        if len(set(g)) == 6:
            if not selected or _pairwise_redundancy_ok(g, selected, max_overlap=4):
                selected.append(g)

    if len(selected) < 5:
        for g in _dedupe_games(candidates):
            if g not in selected:
                selected.append(g)
            if len(selected) == 5:
                break

    return selected[:5]


# ======================================================================================
# SAFE SUMMARY
# ======================================================================================

def compute_redundancy_metrics(games: list[list[int]]) -> dict:
    if len(games) <= 1:
        return {
            "avg_pairwise_overlap": 0.0,
            "max_pairwise_overlap": 0.0,
            "redundancy_penalty": 0.0,
        }

    overlaps = []
    for i in range(len(games)):
        for j in range(i + 1, len(games)):
            overlaps.append(_overlap_size(games[i], games[j]))

    avg_overlap = float(np.mean(overlaps)) if overlaps else 0.0
    max_overlap = float(np.max(overlaps)) if overlaps else 0.0
    redundancy_penalty = max(0.0, avg_overlap - 3.0) + max(0.0, max_overlap - 4.0) * 0.5

    return {
        "avg_pairwise_overlap": avg_overlap,
        "max_pairwise_overlap": max_overlap,
        "redundancy_penalty": redundancy_penalty,
    }


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

    unique_quadra_concursos = sum(1 for c in concursos_data if c["best_game"] and c["best_game"]["hits"] == 4)
    unique_quina_concursos = sum(1 for c in concursos_data if c["best_game"] and c["best_game"]["hits"] == 5)
    unique_sena_concursos = sum(1 for c in concursos_data if c["best_game"] and c["best_game"]["hits"] == 6)
    unique_terno_concursos = sum(1 for c in concursos_data if c["best_game"] and c["best_game"]["hits"] == 3)

    unique_prize_score_total = sum(c["best_game"]["prize_score"] for c in concursos_data if c["best_game"] is not None)

    redundancy_rows = [c["redundancy"] for c in concursos_data]
    avg_pairwise_overlap = float(np.mean([r["avg_pairwise_overlap"] for r in redundancy_rows])) if redundancy_rows else 0.0
    max_pairwise_overlap = float(np.max([r["max_pairwise_overlap"] for r in redundancy_rows])) if redundancy_rows else 0.0
    redundancy_penalty = float(np.mean([r["redundancy_penalty"] for r in redundancy_rows])) if redundancy_rows else 0.0

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
        "prize_score_unique_total": unique_prize_score_total,
        "num_ternos_all": sum(1 for g in all_generated_games if g["hits"] == 3),
        "num_quadras_all": sum(1 for g in all_generated_games if g["hits"] == 4),
        "num_quinas_all": sum(1 for g in all_generated_games if g["hits"] == 5),
        "num_senas_all": sum(1 for g in all_generated_games if g["hits"] == 6),
        "num_ternos_best": sum(1 for g in best_games if g["hits"] == 3),
        "num_quadras_best": sum(1 for g in best_games if g["hits"] == 4),
        "num_quinas_best": sum(1 for g in best_games if g["hits"] == 5),
        "num_senas_best": sum(1 for g in best_games if g["hits"] == 6),
        "num_ternos_unique": unique_terno_concursos,
        "num_quadras_unique": unique_quadra_concursos,
        "num_quinas_unique": unique_quina_concursos,
        "num_senas_unique": unique_sena_concursos,
        "max_hits_all": max(hits_all_games) if hits_all_games else 0,
        "max_hits_best": max(best_hits_list) if best_hits_list else 0,
        "min_hits_all": min(hits_all_games) if hits_all_games else 0,
        "min_hits_best": min(best_hits_list) if best_hits_list else 0,
        "avg_hits_all_games": float(np.mean(hits_all_games)) if hits_all_games else 0.0,
        "avg_hits_best_games": float(np.mean(best_hits_list)) if best_hits_list else 0.0,
        "avg_pairwise_overlap": avg_pairwise_overlap,
        "max_pairwise_overlap": max_pairwise_overlap,
        "redundancy_penalty": redundancy_penalty,
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
        redundancy = compute_redundancy_metrics(games)

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
            "redundancy": redundancy,
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
# MODES
# ======================================================================================

def predict_mode(df: pd.DataFrame, features_df: pd.DataFrame, test_df: pd.DataFrame, results: dict, pool_size: int, mode_name: str) -> dict:
    games_by_concurso = {}

    # map concurso -> feature row
    feat_map = features_df.set_index("concurso")

    for i, concurso in enumerate(test_df["concurso"].values):
        ranked_full = _build_ranked_pool_for_concurso(results, i, 30)
        ranked_pool = ranked_full[:pool_size]
        feat_row = feat_map.loc[int(concurso)]

        if mode_name == "safe_balance_5":
            games = build_rotating_balance_games(ranked_pool)
        elif mode_name == "safe_hybrid_5":
            games = build_safe_hybrid_games(ranked_pool)
        elif mode_name == "safe_321_mix_5":
            games = build_321_mix_games(ranked_full, ranked_pool, feat_row)
        elif mode_name == "safe_321_pure_5":
            games = build_321_pure_games(ranked_full, feat_row)
        else:
            raise ValueError(f"Modo inválido: {mode_name}")

        fallback = sorted(ranked_pool[:6])
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
        label=mode_name.upper(),
        pool_size=pool_size,
        extra_info={"mode_name": mode_name},
    )


def predict_random_games(df: pd.DataFrame, test_df: pd.DataFrame, games_per_concurso: int = 5, seed: int = 42) -> dict:
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


# ======================================================================================
# RANDOM SAFE METRICS
# ======================================================================================

def run_random_baseline_trials(
    df: pd.DataFrame,
    test_df: pd.DataFrame,
    games_per_concurso: int = 5,
    random_trials: int = 300,
    seed_base: int = 1000,
) -> list[dict]:
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
    return summaries


def _distribution_counter(values: list[int | float]) -> dict:
    c = Counter(values)
    return dict(sorted(c.items(), key=lambda x: x[0]))


def summarize_random_trials(random_summaries: list[dict]) -> dict:
    keys = [
        "hit_rate_all_games",
        "hit_rate_best_games",
        "avg_hits_all_games",
        "avg_hits_best_games",
        "prize_score_total",
        "best_prize_score_total",
        "prize_score_unique_total",
        "num_quadras_all",
        "num_quadras_unique",
        "num_quinas_all",
        "num_quinas_unique",
        "avg_pairwise_overlap",
        "max_pairwise_overlap",
        "redundancy_penalty",
    ]

    out = {
        "label": f"ALEATÓRIO ({len(random_summaries)} trials)",
        "random_trials": len(random_summaries),
    }

    for key in keys:
        vals = np.array([s[key] for s in random_summaries], dtype=float)
        out[f"{key}_min"] = float(np.min(vals))
        out[f"{key}_max"] = float(np.max(vals))
        out[f"{key}_mean"] = float(np.mean(vals))
        out[f"{key}_median"] = float(np.median(vals))

    out["quadras_unique_distribution"] = _distribution_counter([int(s["num_quadras_unique"]) for s in random_summaries])
    out["quinas_unique_distribution"] = _distribution_counter([int(s["num_quinas_unique"]) for s in random_summaries])
    out["prize_unique_distribution"] = _distribution_counter([int(s["prize_score_unique_total"]) for s in random_summaries])

    return out


def empirical_p_value(random_summaries: list[dict], model_value: float, key: str) -> float:
    vals = np.array([s[key] for s in random_summaries], dtype=float)
    return float((np.sum(vals >= model_value) + 1) / (len(vals) + 1))


def percentile_vs_random(random_summaries: list[dict], model_value: float, key: str) -> float:
    vals = np.array([s[key] for s in random_summaries], dtype=float)
    return float(100.0 * np.mean(vals <= model_value))


# ======================================================================================
# SAFE SCORE
# ======================================================================================

def safe_score(summary: dict) -> float:
    score = 0.0
    score += summary["avg_hits_all_games"] * 30.0
    score += summary["hit_rate_all_games"] * 120.0
    score += summary["avg_hits_best_games"] * 10.0
    score += summary["prize_score_unique_total"] * 1.5
    score += summary["num_ternos_unique"] * 1.0
    score += summary["num_quadras_unique"] * 15.0
    score += summary["num_quinas_unique"] * 60.0
    score -= summary["redundancy_penalty"] * 20.0
    return score


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
    print(f"Ternos totais:                        {summary['num_ternos_all']}")
    print(f"Quadras totais:                       {summary['num_quadras_all']}")
    print(f"Quinas totais:                        {summary['num_quinas_all']}")
    print(f"Prize score total:                    {summary['prize_score_total']}")

    print("\n--- SAFE: prêmio único por concurso ---")
    print(f"Ternos únicos:                        {summary['num_ternos_unique']}")
    print(f"Quadras únicas:                       {summary['num_quadras_unique']}")
    print(f"Quinas únicas:                        {summary['num_quinas_unique']}")
    print(f"Prize score único total:              {summary['prize_score_unique_total']}")

    print("\n--- Melhor jogo de cada concurso ---")
    print(f"Hit rate melhor jogo:                 {summary['hit_rate_best_games']:.4f} ({summary['hit_rate_best_games'] * 100:.2f}%)")
    print(f"Média de acertos do melhor jogo:      {summary['avg_hits_best_games']:.4f}")
    print(f"Maior acerto (melhor jogo):           {summary['max_hits_best']}")
    print(f"Prize score total (melhor jogo):      {summary['best_prize_score_total']}")

    print("\n--- Redundância ---")
    print(f"Overlap médio entre jogos:            {summary['avg_pairwise_overlap']:.4f}")
    print(f"Overlap máximo entre jogos:           {summary['max_pairwise_overlap']:.4f}")
    print(f"Penalidade de redundância:            {summary['redundancy_penalty']:.4f}")
    print(f"SAFE score:                           {safe_score(summary):.4f}")


def print_random_distribution_summary(random_summary: dict) -> None:
    print("\n" + "=" * 100)
    print("BASELINE ALEATÓRIO ROBUSTO (SAFE)")
    print("=" * 100)
    print(f"Trials aleatórios:                    {random_summary['random_trials']}")
    print(f"Quadras únicas min/max/média/med:     {random_summary['num_quadras_unique_min']:.0f} / {random_summary['num_quadras_unique_max']:.0f} / {random_summary['num_quadras_unique_mean']:.4f} / {random_summary['num_quadras_unique_median']:.4f}")
    print(f"Distribuição quadras únicas:          {random_summary['quadras_unique_distribution']}")
    print(f"Quinas únicas min/max/média/med:      {random_summary['num_quinas_unique_min']:.0f} / {random_summary['num_quinas_unique_max']:.0f} / {random_summary['num_quinas_unique_mean']:.4f} / {random_summary['num_quinas_unique_median']:.4f}")
    print(f"Distribuição quinas únicas:           {random_summary['quinas_unique_distribution']}")
    print(f"Prize único min/max/média/med:        {random_summary['prize_score_unique_total_min']:.0f} / {random_summary['prize_score_unique_total_max']:.0f} / {random_summary['prize_score_unique_total_mean']:.4f} / {random_summary['prize_score_unique_total_median']:.4f}")
    print(f"Distribuição prize único:             {random_summary['prize_unique_distribution']}")
    print(f"Redundância média aleatória:          {random_summary['redundancy_penalty_mean']:.4f}")


def print_vs_random_safe(model_summary: dict, random_avg: dict, random_trials: list[dict]) -> None:
    print("\n" + "=" * 100)
    print("COMPARATIVO: MODELO VS ALEATÓRIO (SAFE)")
    print("=" * 100)

    def safe_div(a, b):
        return (a / b) if b not in (0, 0.0) else float("inf") if a > 0 else 0.0

    print(f"Modelo:        {model_summary['label']}")
    print(f"Aleatório avg: {random_avg['label']}")

    print("\n--- Consistência ---")
    print(f"Hit rate modelo:                  {model_summary['hit_rate_all_games']:.4f}")
    print(f"Hit rate aleatório médio:         {random_avg['hit_rate_all_games_mean']:.4f}")
    print(f"Lift hit rate:                    {safe_div(model_summary['hit_rate_all_games'], random_avg['hit_rate_all_games_mean']):.4f}x")
    print(f"Percentil hit rate:               {percentile_vs_random(random_trials, model_summary['hit_rate_all_games'], 'hit_rate_all_games'):.2f}")
    print(f"p-valor hit rate:                 {empirical_p_value(random_trials, model_summary['hit_rate_all_games'], 'hit_rate_all_games'):.4f}")

    print(f"Média acertos modelo:             {model_summary['avg_hits_all_games']:.4f}")
    print(f"Média acertos aleatório:          {random_avg['avg_hits_all_games_mean']:.4f}")
    print(f"Lift média acertos:               {safe_div(model_summary['avg_hits_all_games'], random_avg['avg_hits_all_games_mean']):.4f}x")
    print(f"Percentil média acertos:          {percentile_vs_random(random_trials, model_summary['avg_hits_all_games'], 'avg_hits_all_games'):.2f}")
    print(f"p-valor média acertos:            {empirical_p_value(random_trials, model_summary['avg_hits_all_games'], 'avg_hits_all_games'):.4f}")

    print("\n--- SAFE: prêmio único ---")
    print(f"Prize único modelo:               {model_summary['prize_score_unique_total']:.2f}")
    print(f"Prize único aleatório médio:      {random_avg['prize_score_unique_total_mean']:.4f}")
    print(f"Lift prize único:                 {safe_div(model_summary['prize_score_unique_total'], random_avg['prize_score_unique_total_mean']):.4f}x")
    print(f"Percentil prize único:            {percentile_vs_random(random_trials, model_summary['prize_score_unique_total'], 'prize_score_unique_total'):.2f}")
    print(f"p-valor prize único:              {empirical_p_value(random_trials, model_summary['prize_score_unique_total'], 'prize_score_unique_total'):.4f}")

    print(f"Quadras únicas modelo:            {model_summary['num_quadras_unique']}")
    print(f"Quadras únicas aleatório média:   {random_avg['num_quadras_unique_mean']:.4f}")
    print(f"Percentil quadras únicas:         {percentile_vs_random(random_trials, model_summary['num_quadras_unique'], 'num_quadras_unique'):.2f}")
    print(f"p-valor quadras únicas:           {empirical_p_value(random_trials, model_summary['num_quadras_unique'], 'num_quadras_unique'):.4f}")

    print(f"Quinas únicas modelo:             {model_summary['num_quinas_unique']}")
    print(f"Quinas únicas aleatório média:    {random_avg['num_quinas_unique_mean']:.4f}")
    print(f"Percentil quinas únicas:          {percentile_vs_random(random_trials, model_summary['num_quinas_unique'], 'num_quinas_unique'):.2f}")
    print(f"p-valor quinas únicas:            {empirical_p_value(random_trials, model_summary['num_quinas_unique'], 'num_quinas_unique'):.4f}")

    print("\n--- Redundância ---")
    print(f"Penalidade modelo:                {model_summary['redundancy_penalty']:.4f}")
    print(f"Penalidade aleatório média:       {random_avg['redundancy_penalty_mean']:.4f}")
    print(f"Overlap médio modelo:             {model_summary['avg_pairwise_overlap']:.4f}")
    print(f"Overlap médio aleatório:          {random_avg['avg_pairwise_overlap_mean']:.4f}")


# ======================================================================================
# MAIN
# ======================================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ML Mega Sena v8.5 SAFE MIX - fortes + médias + raras"
    )
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--pool-size", type=int, default=10)
    parser.add_argument("--top-report", type=int, default=20)
    parser.add_argument("--random-trials", type=int, default=300)
    parser.add_argument("--random-seed-base", type=int, default=1000)

    args = parser.parse_args()

    if not DATA_PATH.exists():
        print(f"Erro: Arquivo não encontrado: {DATA_PATH}")
        sys.exit(1)

    df = load_mega_sena(DATA_PATH)

    print("=" * 100)
    print("MEGA SENA ML v8.5 SAFE MIX - 3,2,1")
    print("=" * 100)
    print(f"Concursos carregados: {len(df)}")
    print(f"Período: {df['data'].iloc[0]} até {df['data'].iloc[-1]}")
    print(f"Window:              {args.window}")
    print(f"Pool:                {args.pool_size}")
    print(f"Random trials:       {args.random_trials}")
    print(f"Random seed base:    {args.random_seed_base}")

    features_df = build_frequency_features(df, window=args.window)
    features_df = build_target(df, features_df)
    train_df, test_df = split_by_date(df, features_df)

    print(f"Treino: {len(train_df)} concursos")
    print(f"Teste:  {len(test_df)} concursos")

    results = train_and_evaluate(train_df=train_df, test_df=test_df)

    mode_names = [
        "safe_balance_5",
        "safe_hybrid_5",
        "safe_321_mix_5",
        "safe_321_pure_5",
    ]

    summaries = []
    for mode_name in mode_names:
        summaries.append(
            predict_mode(
                df=df,
                features_df=features_df,
                test_df=test_df,
                results=results,
                pool_size=args.pool_size,
                mode_name=mode_name,
            )
        )

    best_safe = sorted(
        summaries,
        key=lambda s: (
            safe_score(s),
            s["num_quinas_unique"],
            s["num_quadras_unique"],
            s["prize_score_unique_total"],
            s["avg_hits_all_games"],
            -s["redundancy_penalty"],
        ),
        reverse=True
    )[0]

    random_trials = run_random_baseline_trials(
        df=df,
        test_df=test_df,
        games_per_concurso=best_safe["games_per_concurso"],
        random_trials=args.random_trials,
        seed_base=args.random_seed_base,
    )
    random_summary = summarize_random_trials(random_trials)

    print_games_with_matches(best_safe, top_n=args.top_report)

    print("\n" + "=" * 100)
    print("COMPARATIVO INTERNO v8.5")
    print("=" * 100)
    for s in summaries:
        print(
            f"{s['label']:<18} | jogos={s['games_per_concurso']} | "
            f"hit_rate={s['hit_rate_all_games']:.4f} | "
            f"avg={s['avg_hits_all_games']:.4f} | "
            f"quadras_total={s['num_quadras_all']:<2} | "
            f"quadras_unicas={s['num_quadras_unique']:<2} | "
            f"quinas_unicas={s['num_quinas_unique']:<2} | "
            f"prize_total={s['prize_score_total']:<4} | "
            f"prize_unico={s['prize_score_unique_total']:<4} | "
            f"red_pen={s['redundancy_penalty']:.4f} | "
            f"safe_score={safe_score(s):.4f}"
        )

    print(f"\nMelhor cenário SAFE MIX: {best_safe['label']}")

    for s in summaries:
        print_summary_metrics(s)

    print_random_distribution_summary(random_summary)
    print_vs_random_safe(best_safe, random_summary, random_trials)


if __name__ == "__main__":
    main()