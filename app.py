"""
app.py - v8.6 PARALLEL

Objetivo:
- manter a referência saudável da v8.4
- manter o 3,2,1 da v8.5
- adicionar a estrutura do usuário:
    3 números do top 10
    2 números do 11-40
    1 número do 41-60
- comparar tudo em paralelo no mesmo benchmark SAFE
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
DEZENA_COLS = ["dezena_1", "dezena_2", "dezena_3", "dezena_4", "dezena_5", "dezena_6"]

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

        for n in range(1, 61):
            freq_features[f"freq_{n}"] = float(np.sum(recent_numbers == n) / len(recent_draws))

            last_seen = -1
            for j in range(len(recent_draws) - 1, -1, -1):
                if n in recent_draws[j]:
                    last_seen = j
                    break
            delay_features[f"atraso_{n}"] = float((len(recent_draws) - last_seen) if last_seen >= 0 else (window + 1))

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

            multi_window_features[f"trend_10_60_{n}"] = multi_window_features[f"freq_w10_{n}"] - multi_window_features[f"freq_w60_{n}"]
            multi_window_features[f"trend_5_30_{n}"] = multi_window_features[f"freq_w5_{n}"] - multi_window_features[f"freq_w30_{n}"]

        last_draw = sorted(all_draws[i - 1].tolist())
        somas = [int(np.sum(d)) for d in recent_draws]

        row = {
            "concurso": int(df.iloc[i]["concurso"]),
            "idx": i,
            **freq_features,
            **delay_features,
            **streak_features,
            **multi_window_features,
            "soma_ultimo": int(np.sum(last_draw)),
            "pares_ultimo": int(np.sum(np.array(last_draw) % 2 == 0)),
            "impares_ultimo": 6 - int(np.sum(np.array(last_draw) % 2 == 0)),
            "soma_media_janela": float(np.mean(somas)),
            "soma_std_janela": float(np.std(somas)),
            "amplitude": float(last_draw[-1] - last_draw[0]),
            "mediana": float(np.median(last_draw)),
        }
        features_list.append(row)

    return pd.DataFrame(features_list)


def build_target(df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    for n in range(1, 61):
        targets = []
        for _, row in features_df.iterrows():
            idx = int(row["idx"])
            drawn = df.iloc[idx][DEZENA_COLS].astype(int).values
            targets.append(1 if n in drawn else 0)
        features_df[f"target_{n}"] = targets
    return features_df


def split_by_date(df_original: pd.DataFrame, features_df: pd.DataFrame):
    features_df = features_df.merge(
        df_original[["concurso", "data_parsed"]],
        on="concurso",
        how="left",
    )
    train = features_df[features_df["data_parsed"] < CUTOFF_DATE].copy()
    test = features_df[features_df["data_parsed"] >= CUTOFF_DATE].copy()
    return train, test


def get_feature_columns(df: pd.DataFrame):
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


def _safe_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_brier(y_true, y_prob):
    try:
        return float(brier_score_loss(y_true, y_prob))
    except Exception:
        return float("nan")


def train_and_evaluate(train_df: pd.DataFrame, test_df: pd.DataFrame):
    feature_cols = get_feature_columns(train_df)
    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values

    results_per_number = {}
    per_concurso_probas = {}

    for n in range(1, 61):
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
            "model": model,
        }
        per_concurso_probas[n] = proba_col

    return {
        "results_per_number": results_per_number,
        "feature_columns": feature_cols,
        "per_concurso_probas": per_concurso_probas,
    }


# ======================================================================================
# GAME HELPERS
# ======================================================================================

def _build_ranked_full_for_concurso(results: dict, i: int) -> list[int]:
    per_concurso_probas = results["per_concurso_probas"]
    probas = {n: float(per_concurso_probas[n][i]) for n in range(1, 61)}
    ranked_numbers = [n for n, _ in sorted(probas.items(), key=lambda x: x[1], reverse=True)]
    return ranked_numbers


def _overlap_size(g1, g2):
    return len(set(g1) & set(g2))


def _dedupe_games(games):
    out = []
    seen = set()
    for g in games:
        key = tuple(sorted(g))
        if key not in seen:
            seen.add(key)
            out.append(sorted(g))
    return out


def _pairwise_redundancy_ok(candidate, selected, max_overlap=4):
    return all(_overlap_size(candidate, prev) <= max_overlap for prev in selected)


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
        if not selected or _pairwise_redundancy_ok(g, selected, 4):
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
    for a, b in [(tail[0], tail[1]), (tail[0], tail[2]), (tail[1], tail[2])]:
        extra.append(sorted(core + [a, b]))

    selected = []
    for g in balance_games[:3] + extra[:3]:
        if not selected or _pairwise_redundancy_ok(g, selected, 4):
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


def build_rare_candidates_mid(ranked_full: list[int], features_row: pd.Series) -> list[int]:
    rare_block = ranked_full[15:30]
    scored = []
    for n in rare_block:
        atraso = float(features_row.get(f"atraso_{n}", 0.0))
        freq10 = float(features_row.get(f"freq_w10_{n}", 0.0))
        freq30 = float(features_row.get(f"freq_w30_{n}", 0.0))
        streak = float(features_row.get(f"streak_{n}", 0.0))
        rare_score = atraso * 1.5 - freq10 * 20.0 - freq30 * 10.0 - streak * 0.5
        scored.append((n, rare_score))
    scored = sorted(scored, key=lambda x: x[1], reverse=True)
    return [n for n, _ in scored]


def build_321_mix_games(ranked_full: list[int], ranked_pool: list[int], features_row: pd.Series) -> list[list[int]]:
    strong = ranked_full[:6]
    medium = ranked_full[6:15]
    rare = build_rare_candidates_mid(ranked_full, features_row)
    baseline = build_rotating_balance_games(ranked_pool)

    games = []
    if baseline:
        games.append(baseline[0])

    candidates = [
        sorted(strong[:3] + medium[:2] + rare[:1]),
        sorted([strong[0], strong[1], strong[3], medium[2], medium[4], rare[1]]),
        sorted([strong[0], strong[2], strong[4], medium[1], rare[0], rare[2]]),
        sorted([strong[1], strong[3], medium[0], medium[3], medium[5], rare[3]]),
        sorted([strong[0], strong[5], medium[1], medium[6], medium[7], rare[4]]),
    ]

    for g in candidates:
        if len(set(g)) == 6:
            if not games or _pairwise_redundancy_ok(g, games, 4):
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


def build_user_321_games(ranked_full: list[int], features_row: pd.Series) -> list[list[int]]:
    """
    Estrutura do usuário:
    3 do top 10
    2 do 11-40
    1 do 41-60
    """
    top10 = ranked_full[:10]
    mid_11_40 = ranked_full[10:40]
    tail_41_60 = ranked_full[40:60]

    # prioriza atrasadas/baixa freq dentro da cauda 41-60
    tail_scored = []
    for n in tail_41_60:
        atraso = float(features_row.get(f"atraso_{n}", 0.0))
        freq10 = float(features_row.get(f"freq_w10_{n}", 0.0))
        freq30 = float(features_row.get(f"freq_w30_{n}", 0.0))
        score = atraso * 1.5 - freq10 * 20.0 - freq30 * 10.0
        tail_scored.append((n, score))
    tail_scored = [n for n, _ in sorted(tail_scored, key=lambda x: x[1], reverse=True)]

    # meio também com algum critério leve
    mid_scored = []
    for n in mid_11_40:
        atraso = float(features_row.get(f"atraso_{n}", 0.0))
        freq10 = float(features_row.get(f"freq_w10_{n}", 0.0))
        score = atraso * 0.8 - freq10 * 10.0
        mid_scored.append((n, score))
    mid_scored = [n for n, _ in sorted(mid_scored, key=lambda x: x[1], reverse=True)]

    candidates = [
        sorted([top10[0], top10[1], top10[2], mid_scored[0], mid_scored[1], tail_scored[0]]),
        sorted([top10[0], top10[2], top10[4], mid_scored[2], mid_scored[4], tail_scored[1]]),
        sorted([top10[1], top10[3], top10[5], mid_scored[3], mid_scored[6], tail_scored[2]]),
        sorted([top10[0], top10[6], top10[8], mid_scored[5], mid_scored[8], tail_scored[3]]),
        sorted([top10[2], top10[4], top10[7], mid_scored[7], mid_scored[10], tail_scored[4]]),
    ]

    selected = []
    for g in candidates:
        if len(set(g)) == 6:
            if not selected or _pairwise_redundancy_ok(g, selected, 4):
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

def compute_redundancy_metrics(games):
    if len(games) <= 1:
        return {"avg_pairwise_overlap": 0.0, "max_pairwise_overlap": 0.0, "redundancy_penalty": 0.0}

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


def summarize_generated_games(concursos_data, all_generated_games, games_per_concurso, pool_size, label, extra_info=None):
    hits_all_games = [g["hits"] for g in all_generated_games]
    total_hits_all = sum(hits_all_games)
    total_numbers_all = len(all_generated_games) * 6 if all_generated_games else 0
    hit_rate_all_games = (total_hits_all / total_numbers_all) if total_numbers_all > 0 else 0.0
    prize_score_total = sum(g["prize_score"] for g in all_generated_games)

    best_games = [c["best_game"] for c in concursos_data if c["best_game"] is not None]
    best_hits_list = [g["hits"] for g in best_games]
    total_best_hits = sum(best_hits_list)
    total_best_numbers = len(best_games) * 6 if best_games else 0
    hit_rate_best = (total_best_hits / total_best_numbers) if total_best_numbers > 0 else 0.0
    best_prize_score_total = sum(g["prize_score"] for g in best_games)

    unique_quadra_concursos = sum(1 for c in concursos_data if c["best_game"] and c["best_game"]["hits"] == 4)
    unique_quina_concursos = sum(1 for c in concursos_data if c["best_game"] and c["best_game"]["hits"] == 5)
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
        "prize_score_total": prize_score_total,
        "best_prize_score_total": best_prize_score_total,
        "prize_score_unique_total": unique_prize_score_total,
        "num_ternos_all": sum(1 for g in all_generated_games if g["hits"] == 3),
        "num_quadras_all": sum(1 for g in all_generated_games if g["hits"] == 4),
        "num_quinas_all": sum(1 for g in all_generated_games if g["hits"] == 5),
        "num_ternos_unique": unique_terno_concursos,
        "num_quadras_unique": unique_quadra_concursos,
        "num_quinas_unique": unique_quina_concursos,
        "max_hits_all": max(hits_all_games) if hits_all_games else 0,
        "max_hits_best": max(best_hits_list) if best_hits_list else 0,
        "avg_hits_all_games": float(np.mean(hits_all_games)) if hits_all_games else 0.0,
        "avg_hits_best_games": float(np.mean(best_hits_list)) if best_hits_list else 0.0,
        "avg_pairwise_overlap": avg_pairwise_overlap,
        "max_pairwise_overlap": max_pairwise_overlap,
        "redundancy_penalty": redundancy_penalty,
    }
    if extra_info:
        summary.update(extra_info)
    return summary


def _evaluate_games_for_concursos(df, test_df, games_by_concurso, label, pool_size, extra_info=None):
    concursos_data = []
    all_generated_games = []

    for concurso in test_df["concurso"].values:
        concurso = int(concurso)
        original_row = df[df["concurso"] == concurso].iloc[0]
        actual_numbers = sorted(original_row[DEZENA_COLS].astype(int).values.tolist())
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
                "data": original_row["data"],
                "local": get_draw_location(original_row),
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
            "data": original_row["data"],
            "local": get_draw_location(original_row),
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

def predict_mode(df, features_df, test_df, results, pool_size, mode_name):
    games_by_concurso = {}
    feat_map = features_df.set_index("concurso")

    for i, concurso in enumerate(test_df["concurso"].values):
        ranked_full = _build_ranked_full_for_concurso(results, i)
        ranked_pool = ranked_full[:pool_size]
        feat_row = feat_map.loc[int(concurso)]

        if mode_name == "safe_balance_5":
            games = build_rotating_balance_games(ranked_pool)
        elif mode_name == "safe_321_mix_5":
            games = build_321_mix_games(ranked_full, ranked_pool, feat_row)
        elif mode_name == "safe_user_321_5":
            games = build_user_321_games(ranked_full, feat_row)
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


def predict_random_games(df, test_df, games_per_concurso=5, seed=42):
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

def run_random_baseline_trials(df, test_df, games_per_concurso=5, random_trials=300, seed_base=1000):
    summaries = []
    for k in range(random_trials):
        summaries.append(
            predict_random_games(df=df, test_df=test_df, games_per_concurso=games_per_concurso, seed=seed_base + k)
        )
    return summaries


def _distribution_counter(values):
    c = Counter(values)
    return dict(sorted(c.items(), key=lambda x: x[0]))


def summarize_random_trials(random_summaries):
    keys = [
        "hit_rate_all_games",
        "avg_hits_all_games",
        "prize_score_unique_total",
        "num_quadras_unique",
        "num_quinas_unique",
        "avg_pairwise_overlap",
        "redundancy_penalty",
    ]
    out = {
        "label": f"ALEATÓRIO ({len(random_summaries)} trials)",
        "random_trials": len(random_summaries),
    }
    for key in keys:
        vals = np.array([s[key] for s in random_summaries], dtype=float)
        out[f"{key}_mean"] = float(np.mean(vals))
        out[f"{key}_median"] = float(np.median(vals))
        out[f"{key}_min"] = float(np.min(vals))
        out[f"{key}_max"] = float(np.max(vals))

    out["quadras_unique_distribution"] = _distribution_counter([int(s["num_quadras_unique"]) for s in random_summaries])
    out["prize_unique_distribution"] = _distribution_counter([int(s["prize_score_unique_total"]) for s in random_summaries])
    return out


# ======================================================================================
# SCORE / REPORT
# ======================================================================================

def safe_score(summary):
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


def print_summary_metrics(summary):
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


def main():
    parser = argparse.ArgumentParser(description="ML Mega Sena v8.6 PARALLEL")
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--pool-size", type=int, default=10)
    parser.add_argument("--random-trials", type=int, default=300)
    parser.add_argument("--random-seed-base", type=int, default=1000)
    args = parser.parse_args()

    if not DATA_PATH.exists():
        print(f"Erro: Arquivo não encontrado: {DATA_PATH}")
        sys.exit(1)

    df = load_mega_sena(DATA_PATH)
    features_df = build_frequency_features(df, window=args.window)
    features_df = build_target(df, features_df)
    train_df, test_df = split_by_date(df, features_df)
    results = train_and_evaluate(train_df=train_df, test_df=test_df)

    mode_names = [
        "safe_balance_5",   # referência v8.4
        "safe_321_mix_5",   # referência v8.5
        "safe_user_321_5",  # nova estrutura do usuário
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
        games_per_concurso=5,
        random_trials=args.random_trials,
        seed_base=args.random_seed_base,
    )
    random_summary = summarize_random_trials(random_trials)

    print("=" * 100)
    print("MEGA SENA ML v8.6 PARALLEL")
    print("=" * 100)
    print(f"Concursos carregados: {len(df)}")
    print(f"Treino: {len(train_df)} concursos")
    print(f"Teste:  {len(test_df)} concursos")

    print("\n" + "=" * 100)
    print("COMPARATIVO INTERNO v8.6")
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

    print(f"\nMelhor cenário PARALLEL: {best_safe['label']}")

    for s in summaries:
        print_summary_metrics(s)

    print("\n" + "=" * 100)
    print("BASELINE ALEATÓRIO ROBUSTO (SAFE)")
    print("=" * 100)
    print(f"Trials aleatórios:                    {random_summary['random_trials']}")
    print(f"Quadras únicas média:                {random_summary['num_quadras_unique_mean']:.4f}")
    print(f"Prize único médio:                   {random_summary['prize_score_unique_total_mean']:.4f}")
    print(f"Redundância média:                   {random_summary['redundancy_penalty_mean']:.4f}")
    print(f"Distribuição quadras únicas:         {random_summary['quadras_unique_distribution']}")
    print(f"Distribuição prize único:            {random_summary['prize_unique_distribution']}")


if __name__ == "__main__":
    main()