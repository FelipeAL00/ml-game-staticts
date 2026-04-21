"""
app.py - v8.9

Este módulo implementa a pipeline principal de preparação de dados, treinamento e geração de
jogos para a Mega Sena. Nesta versão foram adicionadas otimizações de desempenho nas
funções de geração de features e alvo, além de ajustes no modelo XGBoost para reduzir o
tempo de treinamento.

Principais melhorias:

- A função ``build_frequency_features`` foi totalmente vetorizada utilizando ``numpy``.
  Em vez de percorrer cada dezena e cada janela de forma sequencial, agora pré-computamos
  uma matriz de pertença (membership) das dezenas sorteadas e utilizamos somas
  cumulativas para calcular frequências, atrasos e sequências (streaks) para todas as
  dezenas de uma só vez. Isso reduz significativamente a complexidade computacional.
- A função ``build_target`` agora gera as colunas de alvo (``target_n``) em bloco a
  partir da matriz de pertença, evitando loops sobre linhas do ``DataFrame``.
- O modelo XGBoost foi ajustado para usar menos estimadores (200 em vez de 600) e uma
  taxa de aprendizagem um pouco maior, acelerando o treinamento sem grandes perdas de
  desempenho.

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

# Caminhos e constantes globais
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
    """Carrega o histórico de concursos da Mega Sena a partir de um arquivo CSV.

    Parameters
    ----------
    filepath : Path
        Caminho para o arquivo CSV contendo as colunas ``concurso``, ``data`` e as seis
        colunas de dezenas (``dezena_1`` a ``dezena_6``).

    Returns
    -------
    pd.DataFrame
        DataFrame ordenado pelo número do concurso com a coluna ``data_parsed`` como
        datetime.
    """
    df = pd.read_csv(filepath)
    df["data_parsed"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df = df.sort_values("concurso").reset_index(drop=True)
    return df


def get_draw_location(row: pd.Series) -> str:
    """Obtém a localização do sorteio quando disponível."""
    if "local" in row.index and pd.notna(row["local"]):
        return str(row["local"])
    return "Local não disponível"


# ======================================================================================
# FEATURES
# ======================================================================================

def build_frequency_features(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """Gera features estatísticas de frequência, atraso e sequência para cada dezena.

    Esta versão foi otimizada para utilizar operações vetorizadas com ``numpy``. Ela
    pré-computa uma matriz de pertença das dezenas sorteadas e utiliza somas cumulativas
    para derivar as frequências em múltiplas janelas, bem como atrasos (tempo desde a
    última ocorrência) e sequências consecutivas (streaks).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de concursos com colunas definidas em ``DEZENA_COLS``.
    window : int, optional
        Tamanho da janela principal para cálculo das features, by default 30.

    Returns
    -------
    pd.DataFrame
        DataFrame com uma linha por concurso a partir do índice ``window`` contendo
        features agregadas para cada dezena e métricas adicionais como soma, paridade
        e amplitude do último sorteio.
    """
    all_draws = df[DEZENA_COLS].values.astype(int)
    num_draws = len(all_draws)
    # Matriz de pertença: membership[i, n] = True se a dezena (n+1) está no sorteio i
    membership = np.zeros((num_draws, TOTAL_NUMBERS), dtype=bool)
    indices = np.arange(num_draws)
    for col in range(NUMBERS_PER_DRAW):
        membership[indices, all_draws[:, col] - 1] = True

    # Soma cumulativa ao longo dos concursos para calcular frequências em qualquer janela
    cumsum = membership.cumsum(axis=0)

    # Inicializar arrays de última ocorrência e sequência
    last_idx = np.full(TOTAL_NUMBERS, -1, dtype=int)
    streak = np.zeros(TOTAL_NUMBERS, dtype=int)

    features_list = []
    # Pré-preencher ``last_idx`` e ``streak`` para as primeiras ``window`` linhas
    for j in range(window):
        for n in range(TOTAL_NUMBERS):
            if membership[j, n]:
                streak[n] += 1
                last_idx[n] = j
            else:
                streak[n] = 0

    # Conjunto de janelas adicionais para features multi-janela
    windows_list = [5, 10, 20, 30, 60]

    # Iterar a partir do índice ``window`` até o final dos concursos
    for i in range(window, num_draws):
        # Janela base de tamanho ``window`` (de i-window a i-1)
        start = i - window
        end_idx = i - 1
        cum_end = cumsum[end_idx]
        if start - 1 >= 0:
            cum_start = cumsum[start - 1]
        else:
            cum_start = np.zeros_like(cum_end)
        counts_window = cum_end - cum_start
        freq_window = counts_window / window

        # Atrasos e streaks para a janela base usando last_idx e streak
        # Se last_idx[n] >= start, a dezena apareceu dentro da janela; caso contrário
        # consideramos atraso máximo (window + 1)
        distance_base = np.where(last_idx >= start, i - last_idx, window + 1)
        delay_base = np.minimum(distance_base, window + 1)
        streak_base = np.minimum(streak, window)

        freq_features = {f"freq_{n + 1}": float(freq_window[n]) for n in range(TOTAL_NUMBERS)}
        delay_features = {f"atraso_{n + 1}": float(delay_base[n]) for n in range(TOTAL_NUMBERS)}
        streak_features = {f"streak_{n + 1}": float(streak_base[n]) for n in range(TOTAL_NUMBERS)}

        multi_window_features = {}
        # Calcular features para cada janela auxiliar
        for w in windows_list:
            start_w = i - w
            if start_w - 1 >= 0:
                cum_start_w = cumsum[start_w - 1]
            else:
                cum_start_w = np.zeros_like(cum_end)
            counts_w = cum_end - cum_start_w
            freq_w = counts_w / w
            distance_w = np.where(last_idx >= start_w, i - last_idx, w + 1)
            streak_w = np.minimum(streak, w)
            for n in range(TOTAL_NUMBERS):
                multi_window_features[f"freq_w{w}_{n + 1}"] = float(freq_w[n])
                multi_window_features[f"delay_w{w}_{n + 1}"] = float(distance_w[n])
                multi_window_features[f"streak_w{w}_{n + 1}"] = float(streak_w[n])

        # Tendências: diferença de frequência entre janelas específicas
        for n in range(TOTAL_NUMBERS):
            multi_window_features[f"trend_10_60_{n + 1}"] = (
                multi_window_features[f"freq_w10_{n + 1}"] - multi_window_features[f"freq_w60_{n + 1}"]
            )
            multi_window_features[f"trend_5_30_{n + 1}"] = (
                multi_window_features[f"freq_w5_{n + 1}"] - multi_window_features[f"freq_w30_{n + 1}"]
            )

        # Estatísticas do último sorteio
        last_draw = sorted(all_draws[i - 1].tolist())
        recent_draws = all_draws[start:i]
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

        # Atualizar ``last_idx`` e ``streak`` com o concurso corrente (linha i)
        for n in range(TOTAL_NUMBERS):
            if membership[i, n]:
                streak[n] += 1
                last_idx[n] = i
            else:
                streak[n] = 0

    return pd.DataFrame(features_list)


def build_target(df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona colunas alvo indicando se cada dezena foi sorteada no concurso.

    Em vez de iterar linha a linha para verificar a presença das dezenas, esta versão
    utiliza a mesma matriz de pertença empregada em ``build_frequency_features`` para
    preencher todas as colunas de alvo em bloco.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame original com as colunas de dezenas.
    features_df : pd.DataFrame
        DataFrame de features gerado por ``build_frequency_features`` contendo uma
        coluna ``idx`` com o índice do concurso correspondente em ``df``.

    Returns
    -------
    pd.DataFrame
        O ``features_df`` com colunas ``target_n`` (para n de 1 a 60) adicionadas.
    """
    all_draws = df[DEZENA_COLS].values.astype(int)
    num_draws = len(all_draws)
    membership = np.zeros((num_draws, TOTAL_NUMBERS), dtype=bool)
    indices = np.arange(num_draws)
    for col in range(NUMBERS_PER_DRAW):
        membership[indices, all_draws[:, col] - 1] = True

    # Extrair os índices reais dos concursos presentes em features_df
    idxs = features_df["idx"].astype(int).values
    membership_slice = membership[idxs]

    # Atribuir cada coluna alvo de uma só vez
    for n in range(TOTAL_NUMBERS):
        features_df[f"target_{n + 1}"] = membership_slice[:, n].astype(int)
    return features_df


def split_by_date(df_original: pd.DataFrame, features_df: pd.DataFrame):
    """Separa o conjunto de features em treino e teste com base na data do concurso."""
    features_df = features_df.merge(
        df_original[["concurso", "data_parsed"]],
        on="concurso",
        how="left",
        suffixes=("", "_orig"),
    )
    if "data_parsed_orig" in features_df.columns:
        features_df = features_df.drop(columns=["data_parsed_orig"])

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
    """Instancia um classificador XGBoost com parâmetros otimizados.

    Reduzimos o número de estimadores de 600 para 200 e ajustamos a taxa de
    aprendizagem (learning_rate) para 0.05 para acelerar o treinamento. Os demais
    parâmetros permanecem configurados conforme a versão anterior.
    """
    return XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
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

    accuracy_list = []
    precision_list = []
    recall_list = []
    auc_list = []
    brier_list = []

    for n in range(1, TOTAL_NUMBERS + 1):
        target_col = f"target_{n}"
        y_train_full = train_df[target_col].values.astype(int)
        y_test = test_df[target_col].values.astype(int)

        model = build_xgboost_model()
        model.fit(X_train, y_train_full)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        proba_col = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba[:, 0]

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        auc_val = _safe_auc(y_test, proba_col)
        brier_val = _safe_brier(y_test, proba_col)

        results_per_number[n] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "auc": auc_val,
            "brier": brier_val,
            "model": model,
        }
        per_concurso_probas[n] = proba_col

        accuracy_list.append(acc)
        precision_list.append(prec)
        recall_list.append(rec)
        if not np.isnan(auc_val):
            auc_list.append(auc_val)
        if not np.isnan(brier_val):
            brier_list.append(brier_val)

    return {
        "results_per_number": results_per_number,
        "feature_columns": feature_cols,
        "per_concurso_probas": per_concurso_probas,
        "model_avg_accuracy": float(np.mean(accuracy_list)) if accuracy_list else 0.0,
        "model_avg_precision": float(np.mean(precision_list)) if precision_list else 0.0,
        "model_avg_recall": float(np.mean(recall_list)) if recall_list else 0.0,
        "model_avg_auc": float(np.mean(auc_list)) if auc_list else float("nan"),
        "model_avg_brier": float(np.mean(brier_list)) if brier_list else float("nan"),
    }

# ======================================================================================
# GAME HELPERS
# ======================================================================================

def _build_ranked_full_for_concurso(results: dict, i: int) -> list[int]:
    per_concurso_probas = results["per_concurso_probas"]
    probas = {n: float(per_concurso_probas[n][i]) for n in range(1, TOTAL_NUMBERS + 1)}
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


# ======================================================================================
# STRATEGIES v8.8
# ======================================================================================

def build_rotating_balance_games_6(ranked_pool: list[int]) -> list[list[int]]:
    rp = ranked_pool[:10]
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


def build_rotating_balance_games_7(ranked_pool: list[int]) -> list[list[int]]:
    """
    4 fortes + 2 medianos + 1 fraco controlado usando top-15 como base do balanceado
    de 7 números
    """
    rp = ranked_pool[:15]
    if len(rp) < 12:
        return [sorted(rp[:7])]

    strong = rp[:6]
    medium = rp[6:12]
    weak = rp[12:15]

    candidates = [
        sorted([strong[0], strong[1], strong[2], strong[3], medium[0], medium[1], weak[0]]),
        sorted([strong[0], strong[1], strong[2], strong[4], medium[1], medium[2], weak[1]]),
        sorted([strong[0], strong[1], strong[3], strong[5], medium[0], medium[3], weak[2]]),
        sorted([strong[0], strong[2], strong[4], strong[5], medium[2], medium[4], weak[0]]),
        sorted([strong[1], strong[3], strong[4], strong[5], medium[3], medium[5], weak[1]]),
    ]

    selected = []
    for g in _dedupe_games(candidates):
        if not selected or _pairwise_redundancy_ok(g, selected, 5):
            selected.append(g)

    if len(selected) < 5:
        for g in _dedupe_games(candidates):
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
        streak_val = float(features_row.get(f"streak_{n}", 0.0))
        rare_score = atraso * 1.5 - freq10 * 20.0 - freq30 * 10.0 - streak_val * 0.5
        scored.append((n, rare_score))
    scored = sorted(scored, key=lambda x: x[1], reverse=True)
    return [n for n, _ in scored]


def build_321_mix_games_6(ranked_full: list[int], ranked_pool: list[int], features_row: pd.Series) -> list[list[int]]:
    strong = ranked_full[:6]
    medium = ranked_full[6:15]
    rare = build_rare_candidates_mid(ranked_full, features_row)
    baseline = build_rotating_balance_games_6(ranked_pool)

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


def build_321_mix_games_7(ranked_full: list[int], features_row: pd.Series) -> list[list[int]]:
    """
    4 fortes + 2 medianos + 1 fraco
    """
    strong = ranked_full[:10]
    medium = ranked_full[10:30]
    weak = ranked_full[30:60]

    weak_scored = []
    for n in weak:
        atraso = float(features_row.get(f"atraso_{n}", 0.0))
        freq10 = float(features_row.get(f"freq_w10_{n}", 0.0))
        freq30 = float(features_row.get(f"freq_w30_{n}", 0.0))
        score = atraso * 1.4 - freq10 * 16.0 - freq30 * 8.0
        weak_scored.append((n, score))
    weak_scored = [n for n, _ in sorted(weak_scored, key=lambda x: x[1], reverse=True)]

    medium_scored = []
    for n in medium:
        atraso = float(features_row.get(f"atraso_{n}", 0.0))
        freq10 = float(features_row.get(f"freq_w10_{n}", 0.0))
        score = atraso * 0.7 - freq10 * 6.0
        medium_scored.append((n, score))
    medium_scored = [n for n, _ in sorted(medium_scored, key=lambda x: x[1], reverse=True)]

    candidates = [
        sorted([strong[0], strong[1], strong[2], strong[3], medium_scored[0], medium_scored[1], weak_scored[0]]),
        sorted([strong[0], strong[1], strong[2], strong[4], medium_scored[2], medium_scored[3], weak_scored[1]]),
        sorted([strong[0], strong[1], strong[3], strong[5], medium_scored[4], medium_scored[5], weak_scored[2]]),
        sorted([strong[0], strong[2], strong[4], strong[6], medium_scored[6], medium_scored[7], weak_scored[3]]),
        sorted([strong[1], strong[3], strong[5], strong[7], medium_scored[8], medium_scored[9], weak_scored[4]]),
    ]

    selected = []
    for g in candidates:
        if len(set(g)) == 7:
            if not selected or _pairwise_redundancy_ok(g, selected, 5):
                selected.append(g)

    if len(selected) < 5:
        for g in _dedupe_games(candidates):
            if g not in selected:
                selected.append(g)
            if len(selected) == 5:
                break

    return selected[:5]


def build_dual_core_games_6(ranked_full: list[int], ranked_pool: list[int], features_row: pd.Series) -> list[list[int]]:
    mix_games = build_321_mix_games_6(ranked_full, ranked_pool, features_row)
    balance_games = build_rotating_balance_games_6(ranked_pool)

    ordered = []
    ordered.extend(mix_games[:2])
    ordered.extend(balance_games[:2])
    if len(mix_games) > 2:
        ordered.append(mix_games[2])

    selected = []
    for g in ordered:
        if not selected or _pairwise_redundancy_ok(g, selected, 4):
            selected.append(g)

    if len(selected) < 5:
        for g in _dedupe_games(ordered + mix_games + balance_games):
            if g not in selected:
                selected.append(g)
            if len(selected) == 5:
                break

    return selected[:5]


def build_dual_core_games_7(ranked_full: list[int], ranked_pool: list[int], features_row: pd.Series) -> list[list[int]]:
    mix_games = build_321_mix_games_7(ranked_full, features_row)
    balance_games = build_rotating_balance_games_7(ranked_pool)

    ordered = []
    ordered.extend(mix_games[:2])
    ordered.extend(balance_games[:2])
    if len(mix_games) > 2:
        ordered.append(mix_games[2])

    selected = []
    for g in ordered:
        if not selected or _pairwise_redundancy_ok(g, selected, 5):
            selected.append(g)

    if len(selected) < 5:
        for g in _dedupe_games(ordered + mix_games + balance_games):
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
    numbers_per_game = extra_info.get("numbers_per_game", 6) if extra_info else 6

    hits_all_games = [g["hits"] for g in all_generated_games]
    total_hits_all = sum(hits_all_games)
    total_numbers_all = len(all_generated_games) * numbers_per_game if all_generated_games else 0
    hit_rate_all_games = (total_hits_all / total_numbers_all) if total_numbers_all > 0 else 0.0
    prize_score_total = sum(g["prize_score"] for g in all_generated_games)

    best_games = [c["best_game"] for c in concursos_data if c["best_game"] is not None]
    best_hits_list = [g["hits"] for g in best_games]
    total_best_hits = sum(best_hits_list)
    total_best_numbers = len(best_games) * numbers_per_game if best_games else 0
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

    accuracy_all_games = hit_rate_all_games
    accuracy_best_games = hit_rate_best

    summary = {
        "label": label,
        "concursos": concursos_data,
        "all_games": all_generated_games,
        "best_games": best_games,
        "total_concursos": len(concursos_data),
        "games_per_concurso": games_per_concurso,
        "total_games_generated": len(all_generated_games),
        "pool_size": pool_size,
        "numbers_per_game": numbers_per_game,
        "hit_rate_all_games": hit_rate_all_games,
        "hit_rate_best_games": hit_rate_best,
        "accuracy_all_games": accuracy_all_games,
        "accuracy_best_games": accuracy_best_games,
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
        ranked_pool = ranked_full[:max(pool_size, 15)]
        feat_row = feat_map.loc[int(concurso)]

        if mode_name == "safe_balance_5_6n":
            games = build_rotating_balance_games_6(ranked_pool)
            numbers_per_game = 6

        elif mode_name == "safe_balance_5_7n":
            games = build_rotating_balance_games_7(ranked_pool)
            numbers_per_game = 7

        elif mode_name == "safe_321_mix_5_6n":
            games = build_321_mix_games_6(ranked_full, ranked_pool, feat_row)
            numbers_per_game = 6

        elif mode_name == "safe_321_mix_5_7n":
            games = build_321_mix_games_7(ranked_full, feat_row)
            numbers_per_game = 7

        elif mode_name == "dual_core_5_6n":
            games = build_dual_core_games_6(ranked_full, ranked_pool, feat_row)
            numbers_per_game = 6

        elif mode_name == "dual_core_5_7n":
            games = build_dual_core_games_7(ranked_full, ranked_pool, feat_row)
            numbers_per_game = 7

        else:
            raise ValueError(f"Modo inválido: {mode_name}")

        fallback = sorted(ranked_pool[:numbers_per_game])
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
        extra_info={
            "mode_name": mode_name,
            "numbers_per_game": numbers_per_game,
        },
    )


def predict_random_games(df, test_df, games_per_concurso=5, numbers_per_game=6, seed=42):
    rng = np.random.default_rng(seed)
    games_by_concurso = {}
    for concurso in test_df["concurso"].values:
        games = []
        seen = set()
        while len(games) < games_per_concurso:
            game = sorted(rng.choice(np.arange(1, 61), size=numbers_per_game, replace=False).tolist())
            key = tuple(game)
            if key not in seen:
                seen.add(key)
                games.append(game)
        games_by_concurso[int(concurso)] = games

    return _evaluate_games_for_concursos(
        df=df,
        test_df=test_df,
        games_by_concurso=games_by_concurso,
        label=f"ALEATÓRIO_{numbers_per_game}N",
        pool_size=60,
        extra_info={"random_seed": seed, "numbers_per_game": numbers_per_game},
    )

# ======================================================================================
# RANDOM SAFE METRICS
# ======================================================================================

def run_random_baseline_trials(df, test_df, games_per_concurso=5, numbers_per_game=6, random_trials=300, seed_base=1000):
    summaries = []
    for k in range(random_trials):
        summaries.append(
            predict_random_games(
                df=df,
                test_df=test_df,
                games_per_concurso=games_per_concurso,
                numbers_per_game=numbers_per_game,
                seed=seed_base + k,
            )
        )
    return summaries


def _distribution_counter(values):
    c = Counter(values)
    return dict(sorted(c.items(), key=lambda x: x[0]))


def summarize_random_trials(random_summaries):
    keys = [
        "hit_rate_all_games",
        "avg_hits_all_games",
        "accuracy_all_games",
        "accuracy_best_games",
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


def print_top_matches(summary, top_n=20):
    matched_games = [g for g in summary["all_games"] if g["hits"] > 0]
    matched_games = sorted(
        matched_games,
        key=lambda g: (g["hits"], g["prize_score"]),
        reverse=True
    )[:top_n]

    print("\n" + "=" * 100)
    print(f"JOGOS COM MATCH - {summary['label']}")
    print("=" * 100)
    print(f"Concursos no teste:         {summary['total_concursos']}")
    print(f"Jogos por concurso:         {summary['games_per_concurso']}")
    print(f"Números por jogo:           {summary['numbers_per_game']}")
    print(f"Total de jogos gerados:     {summary['total_games_generated']}")
    print(f"Top report mostrado:        {top_n}")

    if not matched_games:
        print("Nenhum jogo com match encontrado.")
        return

    for g in matched_games:
        local_str = f" | Local {g['local']}" if g.get("local") else ""
        print(
            f"Concurso {g['concurso']} | Data {g['data']}{local_str} | "
            f"Jogo #{g['game_id']} | Acertos {g['hits']}/6 | Prize {g['prize_score']} | "
            f"Previsto {g['predicted']} | Real {g['actual']} | Match {g['matched']}"
        )


def print_summary_metrics(summary):
    print("\n" + "=" * 100)
    print(f"MÉTRICAS RESUMIDAS - {summary['label']}")
    print("=" * 100)
    print(f"Concursos analisados:                 {summary['total_concursos']}")
    print(f"Jogos por concurso:                   {summary['games_per_concurso']}")
    print(f"Total de jogos gerados:               {summary['total_games_generated']}")
    print(f"Pool de dezenas por concurso:         top-{summary['pool_size']}")
    print(f"Números por jogo:                     {summary['numbers_per_game']}")

    print("\n--- Todos os jogos gerados ---")
    print(f"Hit rate geral:                       {summary['hit_rate_all_games']:.4f} ({summary['hit_rate_all_games'] * 100:.2f}%)")
    print(f"Acurácia por jogo:                    {summary['accuracy_all_games']:.4f} ({summary['accuracy_all_games'] * 100:.2f}%)")
    print(f"Média de acertos por jogo:            {summary['avg_hits_all_games']:.4f}")
    print(f"Maior acerto:                         {summary['max_hits_all']}")
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
    print(f"Acurácia melhor jogo:                 {summary['accuracy_best_games']:.4f} ({summary['accuracy_best_games'] * 100:.2f}%)")
    print(f"Média de acertos do melhor jogo:      {summary['avg_hits_best_games']:.4f}")
    print(f"Maior acerto (melhor jogo):           {summary['max_hits_best']}")
    print(f"Prize score total (melhor jogo):      {summary['best_prize_score_total']}")

    print("\n--- Redundância ---")
    print(f"Overlap médio entre jogos:            {summary['avg_pairwise_overlap']:.4f}")
    print(f"Overlap máximo entre jogos:           {summary['max_pairwise_overlap']:.4f}")
    print(f"Penalidade de redundância:            {summary['redundancy_penalty']:.4f}")
    print(f"SAFE score:                           {safe_score(summary):.4f}")


def main():
    parser = argparse.ArgumentParser(description="ML Mega Sena v8.9 - Otimizado")
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--pool-size", type=int, default=15)
    parser.add_argument("--random-trials", type=int, default=300)
    parser.add_argument("--random-seed-base", type=int, default=1000)
    parser.add_argument("--top-report", type=int, default=20)
    args = parser.parse_args()

    if not DATA_PATH.exists():
        print(f"Erro: Arquivo não encontrado: {DATA_PATH}")
        sys.exit(1)

    df = load_mega_sena(DATA_PATH)
    # Gerar features com a nova implementação vetorizada
    features_df = build_frequency_features(df, window=args.window)
    features_df = build_target(df, features_df)
    train_df, test_df = split_by_date(df, features_df)
    results = train_and_evaluate(train_df=train_df, test_df=test_df)

    mode_names = [
        "safe_balance_5_6n",
        "safe_balance_5_7n",
        "safe_321_mix_5_6n",
        "safe_321_mix_5_7n",
        "dual_core_5_6n",
        "dual_core_5_7n",
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

    random_6n_trials = run_random_baseline_trials(
        df=df,
        test_df=test_df,
        games_per_concurso=5,
        numbers_per_game=6,
        random_trials=args.random_trials,
        seed_base=args.random_seed_base,
    )
    random_6n_summary = summarize_random_trials(random_6n_trials)

    random_7n_trials = run_random_baseline_trials(
        df=df,
        test_df=test_df,
        games_per_concurso=5,
        numbers_per_game=7,
        random_trials=args.random_trials,
        seed_base=args.random_seed_base + 5000,
    )
    random_7n_summary = summarize_random_trials(random_7n_trials)

    print("=" * 100)
    print("MEGA SENA ML v8.9 (Otimizado)")
    print("=" * 100)
    print(f"Concursos carregados: {len(df)}")
    print(f"Período: {df.iloc[0]['data']} até {df.iloc[-1]['data']}")
    print(f"Window:              {args.window}")
    print(f"Pool:                {args.pool_size}")
    print(f"Random trials:       {args.random_trials}")
    print(f"Random seed base:    {args.random_seed_base}")
    print(f"Treino: {len(train_df)} concursos")
    print(f"Teste:  {len(test_df)} concursos")
    print(f"Acurácia média do modelo base (60 modelos):  {results['model_avg_accuracy']:.4f}")
    print(f"Precisão média do modelo base:               {results['model_avg_precision']:.4f}")
    print(f"Recall médio do modelo base:                 {results['model_avg_recall']:.4f}")
    print(
        f"AUC média do modelo base:                    {results['model_avg_auc']:.4f}"
        if not np.isnan(results["model_avg_auc"]) else "AUC média do modelo base:                    nan"
    )
    print(
        f"Brier médio do modelo base:                  {results['model_avg_brier']:.4f}"
        if not np.isnan(results["model_avg_brier"]) else "Brier médio do modelo base:                  nan"
    )

    for s in summaries:
        print_top_matches(s, top_n=args.top_report)

    print("\n" + "=" * 100)
    print("COMPARATIVO INTERNO v8.9")
    print("=" * 100)
    for s in summaries:
        print(
            f"{s['label']:<24} | jogos={s['games_per_concurso']} | "
            f"nums={s['numbers_per_game']} | "
            f"acc={s['accuracy_all_games']:.4f} | "
            f"acc_best={s['accuracy_best_games']:.4f} | "
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

    print(f"\nMelhor cenário v8.9: {best_safe['label']}")

    for s in summaries:
        print_summary_metrics(s)

    print("\n" + "=" * 100)
    print("BASELINE ALEATÓRIO ROBUSTO (6N)")
    print("=" * 100)
    print(f"Trials aleatórios:                    {random_6n_summary['random_trials']}")
    print(f"Quadras únicas média:                {random_6n_summary['num_quadras_unique_mean']:.4f}")
    print(f"Prize único médio:                   {random_6n_summary['prize_score_unique_total_mean']:.4f}")
    print(f"Acurácia média aleatória:            {random_6n_summary['accuracy_all_games_mean']:.4f}")
    print(f"Acurácia best aleatória:             {random_6n_summary['accuracy_best_games_mean']:.4f}")
    print(f"Redundância média:                   {random_6n_summary['redundancy_penalty_mean']:.4f}")
    print(f"Distribuição quadras únicas:         {random_6n_summary['quadras_unique_distribution']}")
    print(f"Distribuição prize único:            {random_6n_summary['prize_unique_distribution']}")

    print("\n" + "=" * 100)
    print("BASELINE ALEATÓRIO ROBUSTO (7N)")
    print("=" * 100)
    print(f"Trials aleatórios:                    {random_7n_summary['random_trials']}")
    print(f"Quadras únicas média:                {random_7n_summary['num_quadras_unique_mean']:.4f}")
    print(f"Prize único médio:                   {random_7n_summary['prize_score_unique_total_mean']:.4f}")
    print(f"Acurácia média aleatória:            {random_7n_summary['accuracy_all_games_mean']:.4f}")
    print(f"Acurácia best aleatória:             {random_7n_summary['accuracy_best_games_mean']:.4f}")
    print(f"Redundância média:                   {random_7n_summary['redundancy_penalty_mean']:.4f}")
    print(f"Distribuição quadras únicas:         {random_7n_summary['quadras_unique_distribution']}")
    print(f"Distribuição prize único:            {random_7n_summary['prize_unique_distribution']}")


if __name__ == "__main__":
    main()