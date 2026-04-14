"""
app.py - v4
Pipeline de Machine Learning para análise da Mega Sena mantendo a estrutura
original (1 linha por concurso + 60 targets), mas trocando o treinamento
por dezena para LightGBM.

Melhorias:
1. Mantém a lógica original que vinha entregando baseline melhor
2. Usa LightGBM por dezena
3. Ensemble opcional: LightGBM + LogisticRegression
4. Métricas por print:
   - accuracy / precision / recall / auc / brier por dezena
   - hit_rate top-6
   - avg_hits_per_game
   - % jogos >=1, >=2, >=3 acertos
   - lift vs aleatório
"""

import argparse
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    brier_score_loss,
)
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path(__file__).resolve().parent / "data" / "raw" / "mega_sena_2000_2026.csv"
CUTOFF_DATE = pd.Timestamp("2026-01-01")
TOTAL_NUMBERS = 60
NUMBERS_PER_DRAW = 6
BASELINE_RANDOM_HIT_RATE = NUMBERS_PER_DRAW / TOTAL_NUMBERS  # 0.10

DEZENA_COLS = [
    "dezena_1",
    "dezena_2",
    "dezena_3",
    "dezena_4",
    "dezena_5",
    "dezena_6",
]


def load_mega_sena(filepath: Path) -> pd.DataFrame:
    """Carrega os dados e converte a data."""
    df = pd.read_csv(filepath)
    df["data_parsed"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df = df.sort_values("concurso").reset_index(drop=True)
    return df


def build_frequency_features(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """
    Calcula features por concurso usando apenas histórico até o concurso anterior.
    Mantém o desenho original do pipeline.
    """
    all_draws = df[DEZENA_COLS].values.astype(int)
    features_list = []

    for i in range(window, len(df)):
        recent_draws = all_draws[max(0, i - window):i]
        recent_numbers = recent_draws.flatten()

        # Frequência por dezena
        freq_features = {}
        for n in range(1, TOTAL_NUMBERS + 1):
            freq_features[f"freq_{n}"] = float(np.sum(recent_numbers == n) / len(recent_draws))

        # Atraso por dezena
        delay_features = {}
        for n in range(1, TOTAL_NUMBERS + 1):
            last_seen = -1
            for j in range(len(recent_draws) - 1, -1, -1):
                if n in recent_draws[j]:
                    last_seen = j
                    break
            delay_features[f"atraso_{n}"] = float(
                (len(recent_draws) - last_seen) if last_seen >= 0 else (window + 1)
            )

        # Streak por dezena
        streak_features = {}
        for n in range(1, TOTAL_NUMBERS + 1):
            streak = 0
            for j in range(len(recent_draws) - 1, -1, -1):
                if n in recent_draws[j]:
                    streak += 1
                else:
                    break
            streak_features[f"streak_{n}"] = float(streak)

        # Multi-window leve, ainda no mesmo desenho
        multi_window_features = {}
        for n in range(1, TOTAL_NUMBERS + 1):
            for w in (5, 10, 20, 30, 60):
                subset = all_draws[max(0, i - w):i]
                if len(subset) == 0:
                    freq = 0.0
                    delay = float(w + 1)
                else:
                    flat = subset.flatten()
                    freq = float(np.sum(flat == n) / len(subset))

                    last_seen = -1
                    for j in range(len(subset) - 1, -1, -1):
                        if n in subset[j]:
                            last_seen = j
                            break
                    delay = float((len(subset) - last_seen) if last_seen >= 0 else (w + 1))

                multi_window_features[f"freq_w{w}_{n}"] = freq
                multi_window_features[f"delay_w{w}_{n}"] = delay

            multi_window_features[f"trend_10_60_{n}"] = (
                multi_window_features[f"freq_w10_{n}"] - multi_window_features[f"freq_w60_{n}"]
            )

        # Contexto do último concurso
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
    """Cria target_1 ... target_60."""
    for n in range(1, TOTAL_NUMBERS + 1):
        targets = []
        for _, row in features_df.iterrows():
            idx = int(row["idx"])
            drawn = df.iloc[idx][DEZENA_COLS].astype(int).values
            targets.append(1 if n in drawn else 0)
        features_df[f"target_{n}"] = targets
    return features_df


def split_by_date(df_original: pd.DataFrame, features_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide treino e teste por data."""
    features_df = features_df.merge(
        df_original[["concurso", "data_parsed"]],
        on="concurso",
        how="left",
    )
    train = features_df[features_df["data_parsed"] < CUTOFF_DATE].copy()
    test = features_df[features_df["data_parsed"] >= CUTOFF_DATE].copy()
    return train, test


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Retorna colunas de features."""
    exclude = {"concurso", "idx", "data_parsed"}
    target_cols = {c for c in df.columns if c.startswith("target_")}
    return [c for c in df.columns if c not in exclude and c not in target_cols]


def _build_single_model(model_name: str):
    """Cria modelo."""
    if model_name == "lightgbm":
        return LGBMClassifier(
            objective="binary",
            n_estimators=350,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=6,
            min_child_samples=20,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.2,
            reg_lambda=0.2,
            class_weight={0: 1.0, 1: 9.0},
            random_state=42,
            verbosity=-1,
        )

    if model_name == "logistic_regression":
        return LogisticRegression(
            max_iter=3000,
            C=0.3,
            penalty="l2",
            solver="lbfgs",
            class_weight="balanced",
            random_state=42,
        )

    raise ValueError(f"Modelo inválido: {model_name}")


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Calcula AUC quando possível."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Calcula Brier quando possível."""
    try:
        return float(brier_score_loss(y_true, y_prob))
    except Exception:
        return float("nan")


def train_and_evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str = "lightgbm",
    ensemble: bool = False,
) -> dict:
    """
    Treina 1 modelo por dezena, mantendo a estrutura original.
    """
    feature_cols = get_feature_columns(train_df)
    X_train_raw = train_df[feature_cols].values
    X_test_raw = test_df[feature_cols].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    results_per_number = {}
    all_probabilities = {}
    per_concurso_probas = {}

    if ensemble:
        model_names_used = ["lightgbm", "logistic_regression"]
        weights = np.array([0.85, 0.15], dtype=float)
        print("\nTreinando ensemble: LightGBM(0.85) + LogisticRegression(0.15)")
    else:
        model_names_used = [model_name]
        weights = np.array([1.0], dtype=float)
        print(f"\nTreinando modelo por dezena: {model_name}")

    for n in range(1, TOTAL_NUMBERS + 1):
        target_col = f"target_{n}"
        y_train = train_df[target_col].values
        y_test = test_df[target_col].values

        model_prob_cols = []
        model_pred_cols = []
        models_trained = []

        for m_name in model_names_used:
            model = _build_single_model(m_name)

            if m_name == "logistic_regression":
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_proba = model.predict_proba(X_test_scaled)
            else:
                model.fit(X_train_raw, y_train)
                y_pred = model.predict(X_test_raw)
                y_proba = model.predict_proba(X_test_raw)

            proba_col = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba[:, 0]
            model_prob_cols.append(proba_col)
            model_pred_cols.append(y_pred)
            models_trained.append(model)

        avg_proba_col = np.average(model_prob_cols, axis=0, weights=weights)
        avg_proba = float(np.mean(avg_proba_col))

        # Predição hard apenas para métricas por dezena
        hard_vote_pred = (np.mean(model_pred_cols, axis=0) >= 0.5).astype(int)

        acc = float(accuracy_score(y_test, hard_vote_pred))
        precision = float(precision_score(y_test, hard_vote_pred, zero_division=0))
        recall = float(recall_score(y_test, hard_vote_pred, zero_division=0))
        auc = _safe_auc(y_test, avg_proba_col)
        brier = _safe_brier(y_test, avg_proba_col)

        results_per_number[n] = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "auc": auc,
            "brier": brier,
            "avg_probability": avg_proba,
            "actual_frequency": float(np.mean(y_test)),
            "predicted_frequency": float(np.mean(hard_vote_pred)),
            "model": models_trained[0],
        }
        all_probabilities[n] = avg_proba
        per_concurso_probas[n] = avg_proba_col

    ranking = sorted(all_probabilities.items(), key=lambda x: x[1], reverse=True)

    return {
        "results_per_number": results_per_number,
        "ranking": ranking,
        "feature_columns": feature_cols,
        "per_concurso_probas": per_concurso_probas,
        "ensemble": ensemble,
    }


def analyze_last_year(df: pd.DataFrame, test_df: pd.DataFrame, results: dict, top_n: int = 10) -> dict:
    """Analisa top previstas vs top reais no período de teste."""
    ranking = results["ranking"]
    top_numbers = [n for n, _ in ranking[:top_n]]

    test_concursos = test_df["concurso"].values
    test_original = df[df["concurso"].isin(test_concursos)]
    all_drawn = test_original[DEZENA_COLS].values.astype(int).flatten()
    actual_freq = Counter(all_drawn)

    hits = sum(actual_freq.get(n, 0) for n in top_numbers)
    most_common_actual = actual_freq.most_common(top_n)

    actual_top = {n for n, _ in most_common_actual}
    predicted_top = set(top_numbers)
    overlap = actual_top & predicted_top

    return {
        "top_predicted": top_numbers,
        "top_actual": most_common_actual,
        "overlap": overlap,
        "overlap_count": len(overlap),
        "hits_in_test": hits,
        "total_draws": len(test_concursos) * 6,
        "test_concursos_count": len(test_concursos),
    }


def predict_full_games(df: pd.DataFrame, test_df: pd.DataFrame, results: dict) -> dict:
    """
    Para cada concurso de teste, pega as 6 dezenas com maior probabilidade.
    """
    per_concurso_probas = results["per_concurso_probas"]
    test_concursos = test_df["concurso"].values
    num_test = len(test_concursos)

    games = []
    total_hits = 0
    total_numbers = 0

    for i in range(num_test):
        concurso = int(test_concursos[i])
        original_row = df[df["concurso"] == concurso].iloc[0]
        actual_numbers = sorted(original_row[DEZENA_COLS].astype(int).values.tolist())
        data_jogo = original_row["data"]

        probas = {n: float(per_concurso_probas[n][i]) for n in range(1, TOTAL_NUMBERS + 1)}
        sorted_probas = sorted(probas.items(), key=lambda x: x[1], reverse=True)
        predicted_numbers = sorted([n for n, _ in sorted_probas[:NUMBERS_PER_DRAW]])

        actual_set = set(actual_numbers)
        predicted_set = set(predicted_numbers)
        hits = len(actual_set & predicted_set)

        total_hits += hits
        total_numbers += NUMBERS_PER_DRAW

        games.append({
            "concurso": concurso,
            "data": data_jogo,
            "actual": actual_numbers,
            "predicted": predicted_numbers,
            "hits": hits,
            "top_proba": sorted_probas[0][1],
            "sixth_proba": sorted_probas[5][1],
        })

    hits_list = [g["hits"] for g in games]
    hits_distribution = Counter(hits_list)

    hit_rate = (total_hits / total_numbers) if total_numbers > 0 else 0.0
    avg_hits_per_game = (total_hits / num_test) if num_test > 0 else 0.0
    pct_ge_1 = float(np.mean([h >= 1 for h in hits_list])) if hits_list else 0.0
    pct_ge_2 = float(np.mean([h >= 2 for h in hits_list])) if hits_list else 0.0
    pct_ge_3 = float(np.mean([h >= 3 for h in hits_list])) if hits_list else 0.0
    lift = (hit_rate / BASELINE_RANDOM_HIT_RATE) if BASELINE_RANDOM_HIT_RATE > 0 else 0.0

    return {
        "games": games,
        "total_games": num_test,
        "total_hits": total_hits,
        "total_numbers": total_numbers,
        "avg_hits_per_game": avg_hits_per_game,
        "hit_rate": hit_rate,
        "hits_distribution": dict(sorted(hits_distribution.items())),
        "max_hits": max(hits_list) if hits_list else 0,
        "pct_games_ge_1_hit": pct_ge_1,
        "pct_games_ge_2_hits": pct_ge_2,
        "pct_games_ge_3_hits": pct_ge_3,
        "lift_vs_random": float(lift),
    }


def print_model_metrics(results: dict) -> None:
    """Imprime métricas agregadas dos 60 modelos."""
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


def print_summary_report(results: dict, analysis: dict, prediction: dict, train_size: int, test_size: int) -> None:
    """Imprime relatório principal."""
    ranking = results["ranking"]

    print("\n" + "=" * 90)
    print("RELATÓRIO FINAL - MEGA SENA ML v4 (LIGHTGBM POR DEZENA)")
    print("=" * 90)

    print(f"\nTreino: {train_size} concursos")
    print(f"Teste:  {test_size} concursos")
    print(f"Corte temporal: {CUTOFF_DATE.strftime('%d/%m/%Y')}")

    print("\n" + "-" * 90)
    print("TOP 10 DEZENAS MAIS PROVÁVEIS")
    print("-" * 90)
    for i, (n, prob) in enumerate(ranking[:10], start=1):
        actual_freq = results["results_per_number"][n]["actual_frequency"]
        print(f"{i:2d}. Dezena {n:02d} | Proba média={prob:.4f} | Freq real teste={actual_freq:.4f}")

    print("\n" + "-" * 90)
    print("TOP 10 DEZENAS MAIS SORTEADAS NO TESTE")
    print("-" * 90)
    for i, (n, count) in enumerate(analysis["top_actual"], start=1):
        print(f"{i:2d}. Dezena {n:02d} | Sorteada {count} vezes")

    print("\n" + "-" * 90)
    print("COMPARAÇÃO TOP PREVISTO vs TOP REAL")
    print("-" * 90)
    print(f"Top previsto: {sorted(analysis['top_predicted'])}")
    print(f"Top real:     {sorted([n for n, _ in analysis['top_actual']])}")
    print(f"Overlap:      {sorted(analysis['overlap'])} ({analysis['overlap_count']}/10)")

    print("\n" + "-" * 90)
    print("MÉTRICAS PRINCIPAIS DE JOGO (TOP-6 POR CONCURSO)")
    print("-" * 90)
    print(f"Hit rate top-6:              {prediction['hit_rate']:.4f} ({prediction['hit_rate'] * 100:.2f}%)")
    print(f"Avg hits per game:           {prediction['avg_hits_per_game']:.4f} de 6")
    print(f"% jogos com >=1 acerto:      {prediction['pct_games_ge_1_hit'] * 100:.2f}%")
    print(f"% jogos com >=2 acertos:     {prediction['pct_games_ge_2_hits'] * 100:.2f}%")
    print(f"% jogos com >=3 acertos:     {prediction['pct_games_ge_3_hits'] * 100:.2f}%")
    print(f"Máximo de acertos em jogo:   {prediction['max_hits']}")
    print(f"Lift vs aleatório (10%):     {prediction['lift_vs_random']:.4f}x")

    print("\n" + "-" * 90)
    print("DISTRIBUIÇÃO DE ACERTOS")
    print("-" * 90)
    for hits, count in prediction["hits_distribution"].items():
        pct = (count / prediction["total_games"]) * 100 if prediction["total_games"] > 0 else 0.0
        print(f"{hits} acertos: {count:3d} jogos ({pct:6.2f}%)")

    print("\n" + "=" * 90)


def print_detailed_game_report(prediction: dict) -> None:
    """Imprime detalhe por concurso."""
    print("\n" + "=" * 90)
    print("DETALHE POR CONCURSO")
    print("=" * 90)
    for g in prediction["games"]:
        matched = sorted(set(g["actual"]) & set(g["predicted"]))
        print(
            f"Concurso {g['concurso']} | Data {g['data']} | "
            f"Previsto {g['predicted']} | Real {g['actual']} | "
            f"Acertos {g['hits']}/6 | Match {matched}"
        )


def run_mega_sena_analysis(
    model_name: str = "lightgbm",
    window: int = 30,
    ensemble: bool = False,
) -> dict:
    """Executa pipeline completo."""
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

    print("\n" + "=" * 90)
    print("4. TREINAMENTO")
    print("=" * 90)
    results = train_and_evaluate(
        train_df=train_df,
        test_df=test_df,
        model_name=model_name,
        ensemble=ensemble,
    )

    print("\n" + "=" * 90)
    print("5. ANÁLISE")
    print("=" * 90)
    analysis = analyze_last_year(df, test_df, results, top_n=10)
    prediction = predict_full_games(df, test_df, results)

    print_model_metrics(results)
    print_summary_report(results, analysis, prediction, len(train_df), len(test_df))
    print_detailed_game_report(prediction)

    return {
        "results": results,
        "analysis": analysis,
        "prediction": prediction,
        "train_size": len(train_df),
        "test_size": len(test_df),
    }


def main():
    parser = argparse.ArgumentParser(
        description="ML Mega Sena v4 - LightGBM por dezena com métricas por print"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lightgbm",
        choices=["lightgbm", "logistic_regression"],
        help="Modelo a usar quando não estiver em ensemble",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=30,
        help="Janela base para features (default: 30)",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Usa ensemble LightGBM + LogisticRegression",
    )

    args = parser.parse_args()

    run_mega_sena_analysis(
        model_name=args.model,
        window=args.window,
        ensemble=args.ensemble,
    )


if __name__ == "__main__":
    main()