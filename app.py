"""
app.py - Ponto de entrada principal do projeto ML Game Statistics (Mega Sena).

Pipeline de Machine Learning robusto para análise da Mega Sena:
1. Carregar dados históricos (2000-2026)
2. Engenharia de features avançada (frequência, tendência, co-ocorrência,
   codificação cíclica, regressão à média, gaps, etc.)
3. Divisão temporal: treino (2000-2020), teste (2021-2026)
4. Treinamento com ensemble ponderado por CV (RF + GB + XGBoost + LR)
5. Calibração de probabilidades (isotonic regression)
6. Limiar de decisão otimizado via curva precisão-recall
7. Métricas completas: F1, PR-AUC, ROC-AUC, MCC, Balanced Accuracy
8. Relatório de previsão de jogos completos

Uso:
    python app.py                           # Análise completa
    python app.py --no-plots                # Sem gráficos
    python app.py --model gradient_boosting # Modelo único
    python app.py --ensemble                # Ensemble ponderado
    python app.py --stacking                # Stacking com meta-learner
    python app.py --save                    # Salvar modelo e gráficos
"""

import argparse
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.features import build_advanced_features
from src.imbalance import find_optimal_threshold, apply_threshold, smote_resample, compute_scale_pos_weight
from src.calibration import calibrate_model, reliability_summary
from src.metrics import compute_all_metrics, aggregate_metrics, print_aggregated_metrics
from src.ensemble import build_base_models, cv_weighted_ensemble, weighted_predict_proba, build_stacking_classifier


DATA_PATH = Path(__file__).resolve().parent / "data" / "raw" / "mega_sena_2000_2026.csv"
CUTOFF_DATE = pd.Timestamp("2021-01-01")
TOTAL_NUMBERS = 60
NUMBERS_PER_DRAW = 6

MODELS = {
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "logistic_regression": LogisticRegression,
    "xgboost": XGBClassifier,
}


def load_mega_sena(filepath: Path) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df["data_parsed"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df = df.sort_values("concurso").reset_index(drop=True)
    return df


def build_target(df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    dezena_cols = ["dezena_1", "dezena_2", "dezena_3", "dezena_4", "dezena_5", "dezena_6"]

    idx_array = features_df["idx"].values.astype(int)
    draws_matrix = df[dezena_cols].values.astype(int)

    for n in range(1, TOTAL_NUMBERS + 1):
        targets = np.array([
            1 if n in draws_matrix[idx] else 0
            for idx in idx_array
        ])
        features_df[f"target_{n}"] = targets

    return features_df


def split_by_date(df_original: pd.DataFrame, features_df: pd.DataFrame) -> tuple:
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


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {"concurso", "idx", "data_parsed"}
    target_cols = {c for c in df.columns if c.startswith("target_")}
    return [c for c in df.columns if c not in exclude and c not in target_cols]


def _prepare_arrays(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
    feature_cols = get_feature_columns(train_df)

    X_train_raw = train_df[feature_cols].values.astype(float)
    X_test_raw = test_df[feature_cols].values.astype(float)

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train_raw)
    X_test_imp = imputer.transform(X_test_raw)

    X_train_imp = np.nan_to_num(X_train_imp, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_imp = np.nan_to_num(X_test_imp, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train_imp)
    X_test = scaler.transform(X_test_imp)

    return X_train, X_test, feature_cols, scaler


def _train_single_model(model_name: str, scale_pos_weight: float):
    if model_name == "logistic_regression":
        return LogisticRegression(
            max_iter=2000, C=0.5, class_weight="balanced", random_state=42
        )
    elif model_name == "xgboost":
        return XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=1,
            verbosity=0,
        )
    elif model_name == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=3, random_state=42,
        )
    else:
        return RandomForestClassifier(
            n_estimators=300, max_depth=10, max_features="sqrt",
            class_weight="balanced", min_samples_leaf=3, random_state=42, n_jobs=-1,
        )


def train_and_evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str = "xgboost",
    top_n: int = 10,
    ensemble: bool = False,
    stacking: bool = False,
    cv_splits: int = 3,
    use_smote: bool = True,
    calibrate: bool = True,
) -> dict:
    """Train model(s) for each number and evaluate on the test period.

    Key improvements over v1:
    - RobustScaler instead of StandardScaler (robust to outliers)
    - Median imputation for NaN values
    - SMOTE resampling for class imbalance
    - Optimal threshold from precision-recall curve
    - Probability calibration via isotonic regression
    - Weighted ensemble based on CV ROC-AUC

    Args:
        train_df: Training DataFrame.
        test_df: Test DataFrame.
        model_name: Base model name (used when ensemble=False and stacking=False).
        top_n: Number of top predicted numbers to report.
        ensemble: If True, use CV-weighted ensemble (RF + GB + XGBoost + LR).
        stacking: If True, use stacking with meta-learner.
        cv_splits: Number of TimeSeriesSplit folds for CV weighting.
        use_smote: If True, apply SMOTE resampling before training.
        calibrate: If True, calibrate probabilities on a validation split.

    Returns:
        Dictionary with detailed results.
    """
    X_train, X_test, feature_cols, scaler = _prepare_arrays(train_df, test_df)

    results_per_number = {}
    all_probabilities = {}
    per_concurso_probas = {}
    metrics_list = []

    mode_label = "STACKING" if stacking else ("ENSEMBLE" if ensemble else model_name.upper())
    print(f"\n   Modo: {mode_label} | SMOTE: {use_smote} | Calibracao: {calibrate}")
    print(f"   Treinando modelos para cada dezena (1-{TOTAL_NUMBERS})...")

    val_split = int(len(X_train) * 0.8)

    for n in range(1, TOTAL_NUMBERS + 1):
        target_col = f"target_{n}"
        y_train_full = train_df[target_col].values.astype(int)
        y_test = test_df[target_col].values.astype(int)

        y_train_cv = y_train_full[:val_split]
        y_val = y_train_full[val_split:]
        X_train_cv = X_train[:val_split]
        X_val = X_train[val_split:]

        scale_pos_weight = compute_scale_pos_weight(y_train_cv)

        if use_smote:
            X_train_cv_res, y_train_cv_res = smote_resample(X_train_cv, y_train_cv, sampling_strategy=0.3)
        else:
            X_train_cv_res, y_train_cv_res = X_train_cv, y_train_cv

        if stacking:
            model = build_stacking_classifier(scale_pos_weight=scale_pos_weight)
            model.fit(X_train_cv_res, y_train_cv_res)
            proba_val = model.predict_proba(X_val)[:, 1]
            proba_test = model.predict_proba(X_test)[:, 1]
        elif ensemble:
            base_models = build_base_models(scale_pos_weight=scale_pos_weight)
            ens_result = cv_weighted_ensemble(base_models, X_train_cv_res, y_train_cv_res, n_splits=cv_splits)
            trained_models = ens_result["models"]
            weights = ens_result["weights"]
            proba_val = weighted_predict_proba(trained_models, weights, X_val)
            proba_test = weighted_predict_proba(trained_models, weights, X_test)
            model = trained_models[0]
        else:
            model = _train_single_model(model_name, scale_pos_weight)
            model.fit(X_train_cv_res, y_train_cv_res)
            proba_val = model.predict_proba(X_val)[:, 1]
            proba_test = model.predict_proba(X_test)[:, 1]

        if calibrate and len(np.unique(y_val)) > 1 and len(y_val) >= 10:
            try:
                calibrated = calibrate_model(model, X_val, y_val, method="isotonic")
                proba_test = calibrated.predict_proba(X_test)[:, 1]
                proba_val = calibrated.predict_proba(X_val)[:, 1]
            except Exception:
                pass

        if len(np.unique(y_val)) > 1:
            threshold = find_optimal_threshold(y_val, proba_val, method="f1")
        else:
            threshold = NUMBERS_PER_DRAW / TOTAL_NUMBERS

        threshold = float(np.clip(threshold, 0.05, 0.5))

        y_pred = apply_threshold(proba_test, threshold)
        avg_proba = float(np.mean(proba_test))

        m = compute_all_metrics(y_test, y_pred, proba_test)
        cal = reliability_summary(y_test, proba_test)

        results_per_number[n] = {
            **m,
            "ece": cal["ece"],
            "avg_probability": avg_proba,
            "optimal_threshold": threshold,
            "model": model,
        }
        all_probabilities[n] = avg_proba
        per_concurso_probas[n] = proba_test
        metrics_list.append({**m, "n": n})

    ranking = sorted(all_probabilities.items(), key=lambda x: x[1], reverse=True)
    agg_metrics = aggregate_metrics(metrics_list)

    return {
        "results_per_number": results_per_number,
        "ranking": ranking,
        "feature_columns": feature_cols,
        "per_concurso_probas": per_concurso_probas,
        "aggregated_metrics": agg_metrics,
        "ensemble": ensemble,
        "stacking": stacking,
    }


def analyze_last_year(
    df: pd.DataFrame,
    test_df: pd.DataFrame,
    results: dict,
    top_n: int = 10,
) -> dict:
    dezena_cols = ["dezena_1", "dezena_2", "dezena_3", "dezena_4", "dezena_5", "dezena_6"]
    ranking = results["ranking"]
    top_numbers = [n for n, _ in ranking[:top_n]]

    test_concursos = test_df["concurso"].values
    test_original = df[df["concurso"].isin(test_concursos)]
    all_drawn = test_original[dezena_cols].values.astype(int).flatten()
    actual_freq = Counter(all_drawn)

    hits = sum(actual_freq.get(n, 0) for n in top_numbers)
    most_common_actual = actual_freq.most_common(top_n)
    actual_top = {n for n, _ in most_common_actual}
    predicted_top = set(top_numbers)
    overlap = actual_top & predicted_top
    total_draws = len(test_concursos) * 6
    top_k_recall = hits / total_draws if total_draws > 0 else 0.0

    return {
        "top_k": top_n,
        "top_predicted": top_numbers,
        "top_actual": most_common_actual,
        "overlap": overlap,
        "overlap_count": len(overlap),
        "hits_in_test": hits,
        "total_draws": total_draws,
        "top_k_recall": top_k_recall,
        "test_concursos_count": len(test_concursos),
    }


def predict_full_games(
    df: pd.DataFrame,
    test_df: pd.DataFrame,
    results: dict,
) -> dict:
    dezena_cols = ["dezena_1", "dezena_2", "dezena_3", "dezena_4", "dezena_5", "dezena_6"]
    per_concurso_probas = results["per_concurso_probas"]
    test_concursos = test_df["concurso"].values
    num_test = len(test_concursos)

    games = []
    total_hits = 0
    total_numbers = 0

    for i in range(num_test):
        concurso = int(test_concursos[i])
        original_row = df[df["concurso"] == concurso].iloc[0]
        actual_numbers = sorted(original_row[dezena_cols].astype(int).values.tolist())
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

    return {
        "games": games,
        "total_games": num_test,
        "total_hits": total_hits,
        "total_numbers": total_numbers,
        "avg_hits_per_game": total_hits / num_test if num_test > 0 else 0,
        "hit_rate": total_hits / total_numbers if total_numbers > 0 else 0,
        "hits_distribution": dict(sorted(hits_distribution.items())),
        "max_hits": max(hits_list) if hits_list else 0,
    }


def print_full_games_report(prediction: dict) -> None:
    games = prediction["games"]

    print("\n" + "=" * 80)
    print("   PREVISAO DE JOGOS COMPLETOS - MEGA SENA 2026")
    print("=" * 80)
    print(f"\n   Total de jogos analisados: {prediction['total_games']}")
    print(f"   Media de acertos por jogo: {prediction['avg_hits_per_game']:.2f} de 6")
    print(f"   Taxa de acerto (dezenas): {prediction['hit_rate'] * 100:.1f}%")
    print(f"   Maximo de acertos em um jogo: {prediction['max_hits']}")

    print("\n" + "-" * 80)
    print("   DISTRIBUICAO DE ACERTOS")
    print("-" * 80)
    for hits, count in sorted(prediction["hits_distribution"].items()):
        bar = "#" * (count * 2)
        pct = count / prediction["total_games"] * 100
        print(f"   {hits} acertos: {count:3d} jogos ({pct:5.1f}%) {bar}")

    print("\n" + "-" * 80)
    print("   DETALHAMENTO POR CONCURSO")
    print("-" * 80)
    print(f"   {'Conc.':>6}  {'Data':>12}  {'Previsto':^38}  {'Real':^38}  {'Acertos':>7}")
    print("   " + "-" * 105)

    for g in games:
        actual_set = set(g["actual"])
        predicted_set = set(g["predicted"])
        matched = actual_set & predicted_set

        prev_marked = ", ".join(
            f"*{n:02d}*" if n in matched else f" {n:02d} " for n in g["predicted"]
        )
        real_marked = ", ".join(
            f"*{n:02d}*" if n in matched else f" {n:02d} " for n in g["actual"]
        )

        print(f"   {g['concurso']:>6}  {g['data']:>12}  {prev_marked:<38}  {real_marked:<38}  {g['hits']:>3}/6")

    print("\n" + "-" * 80)
    print("   RESUMO DE ASSERTIVIDADE")
    print("-" * 80)
    print(f"   Assertividade por dezena:  {prediction['hit_rate'] * 100:.1f}%")
    print(f"   Total de dezenas corretas: {prediction['total_hits']} de {prediction['total_numbers']}")
    print(f"   Media de acertos/jogo:     {prediction['avg_hits_per_game']:.2f} de 6")

    best_games = sorted(games, key=lambda g: g["hits"], reverse=True)[:5]
    print("\n   Melhores jogos (mais acertos):")
    for g in best_games:
        matched = set(g["actual"]) & set(g["predicted"])
        matched_str = ", ".join(f"{n:02d}" for n in sorted(matched))
        print(f"     Concurso {g['concurso']} ({g['data']}): {g['hits']}/6 acertos -- dezenas: {matched_str}")

    print("\n" + "=" * 80)


def print_report(results: dict, analysis: dict, train_size: int, test_size: int) -> None:
    ranking = results["ranking"]

    print("\n" + "=" * 60)
    print("   RELATORIO DE ANALISE - MEGA SENA")
    print("=" * 60)

    print(f"\n   Periodo de treino: janeiro 2000 - dezembro 2020 ({train_size} concursos)")
    print(f"   Periodo de teste:  janeiro 2021 - dezembro 2026 ({test_size} concursos)")

    print("\n" + "-" * 60)
    print("   TOP 10 DEZENAS MAIS PROVAVEIS (previsao do modelo)")
    print("-" * 60)
    for i, (n, prob) in enumerate(ranking[:10], 1):
        r = results["results_per_number"][n]
        thr = r.get("optimal_threshold", 0.5)
        print(
            f"   {i:2d}. Dezena {n:02d}  |  Prob: {prob:.4f}  |"
            f"  Limiar: {thr:.3f}  |  F1: {r.get('f1', 0):.4f}  |  PR-AUC: {r.get('pr_auc', 0):.4f}"
        )

    print("\n" + "-" * 60)
    print("   TOP 10 DEZENAS MAIS SORTEADAS NO PERIODO DE TESTE (real)")
    print("-" * 60)
    for i, (n, count) in enumerate(analysis["top_actual"], 1):
        print(f"   {i:2d}. Dezena {n:02d}  |  Sorteada {count} vezes")

    print("\n" + "-" * 60)
    print("   COMPARACAO: PREVISAO vs REALIDADE")
    print("-" * 60)
    print(f"   Dezenas previstas no top {analysis['top_k']}:  {sorted(analysis['top_predicted'])}")
    print(f"   Dezenas reais no top {analysis['top_k']}:      {sorted([n for n, _ in analysis['top_actual']])}")
    print(f"   Acertos (overlap):            {sorted(analysis['overlap'])} ({analysis['overlap_count']}/{analysis['top_k']})")
    print(f"   Top-{analysis['top_k']} recall no teste:    {analysis['top_k_recall']:.4f}")

    print("\n" + "-" * 60)
    print("   DEZENAS MENOS PROVAVEIS (frias)")
    print("-" * 60)
    for i, (n, prob) in enumerate(ranking[-10:], 1):
        r = results["results_per_number"][n]
        print(
            f"   {i:2d}. Dezena {n:02d}  |  Prob: {prob:.4f}  |"
            f"  F1: {r.get('f1', 0):.4f}  |  ECE: {r.get('ece', float('nan')):.4f}"
        )

    print("\n" + "-" * 60)
    print("   METRICAS AGREGADAS (todos os 60 classificadores)")
    print("-" * 60)
    print_aggregated_metrics(results["aggregated_metrics"])

    print("\n" + "=" * 60)


def plot_analysis(results: dict, analysis: dict, save_dir: Path) -> list[str]:
    save_dir.mkdir(parents=True, exist_ok=True)
    saved_plots = []

    ranking = results["ranking"]
    numbers = [n for n, _ in ranking]
    probs = [p for _, p in ranking]

    fig, ax = plt.subplots(figsize=(16, 6))
    colors = ["#2ecc71" if i < 10 else "#e74c3c" if i >= 50 else "#3498db" for i in range(len(numbers))]
    ax.bar(range(len(numbers)), probs, color=colors)
    ax.set_xticks(range(len(numbers)))
    ax.set_xticklabels([f"{n:02d}" for n in numbers], rotation=90, fontsize=8)
    ax.set_xlabel("Dezena")
    ax.set_ylabel("Probabilidade Media")
    ax.set_title("Ranking de Probabilidade por Dezena (Verde=Top10, Vermelho=Bottom10)")
    plt.tight_layout()
    path = save_dir / "ranking_probabilidades.png"
    plt.savefig(path, dpi=150)
    plt.close()
    saved_plots.append(str(path))

    top_predicted = analysis["top_predicted"]
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(top_predicted))
    predicted_probs = [results["results_per_number"][n]["avg_probability"] for n in top_predicted]
    actual_freqs = [results["results_per_number"][n]["actual_prevalence"] for n in top_predicted]
    width = 0.35
    ax.bar([i - width / 2 for i in x], predicted_probs, width, label="Probabilidade Prevista", color="#3498db")
    ax.bar([i + width / 2 for i in x], actual_freqs, width, label="Frequencia Real", color="#e74c3c")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{n:02d}" for n in top_predicted])
    ax.set_xlabel("Dezena")
    ax.set_ylabel("Proporcao")
    ax.set_title("Top 10 Dezenas Previstas: Probabilidade vs Frequencia Real")
    ax.legend()
    plt.tight_layout()
    path = save_dir / "previsto_vs_real.png"
    plt.savefig(path, dpi=150)
    plt.close()
    saved_plots.append(str(path))

    actual_freq_all = {n: results["results_per_number"][n]["actual_prevalence"] for n in range(1, TOTAL_NUMBERS + 1)}
    grid = np.zeros((6, 10))
    labels_grid = np.zeros((6, 10), dtype=int)
    for n in range(1, TOTAL_NUMBERS + 1):
        row = (n - 1) // 10
        col = (n - 1) % 10
        grid[row][col] = actual_freq_all[n]
        labels_grid[row][col] = n

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(grid, annot=labels_grid, fmt="d", cmap="YlOrRd", ax=ax,
                cbar_kws={"label": "Frequencia no periodo de teste"}, linewidths=1)
    ax.set_title("Frequencia Real de Cada Dezena no Periodo de Teste")
    ax.set_xlabel("Coluna")
    ax.set_ylabel("Linha")
    plt.tight_layout()
    path = save_dir / "heatmap_frequencia.png"
    plt.savefig(path, dpi=150)
    plt.close()
    saved_plots.append(str(path))

    f1_scores = [results["results_per_number"][n]["f1"] for n in range(1, TOTAL_NUMBERS + 1)]
    pr_aucs = [results["results_per_number"][n].get("pr_auc", float("nan")) for n in range(1, TOTAL_NUMBERS + 1)]
    pr_aucs_clean = [v for v in pr_aucs if not np.isnan(v)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(f1_scores, bins=20, color="#3498db", edgecolor="white")
    axes[0].axvline(np.mean(f1_scores), color="#e74c3c", linestyle="--",
                    label=f"Media: {np.mean(f1_scores):.4f}")
    axes[0].set_xlabel("F1 Score")
    axes[0].set_ylabel("Quantidade de Modelos")
    axes[0].set_title("Distribuicao de F1 Score dos 60 Modelos")
    axes[0].legend()

    if pr_aucs_clean:
        axes[1].hist(pr_aucs_clean, bins=20, color="#2ecc71", edgecolor="white")
        axes[1].axvline(np.mean(pr_aucs_clean), color="#e74c3c", linestyle="--",
                        label=f"Media: {np.mean(pr_aucs_clean):.4f}")
        axes[1].set_xlabel("PR-AUC")
        axes[1].set_ylabel("Quantidade de Modelos")
        axes[1].set_title("Distribuicao de PR-AUC dos 60 Modelos")
        axes[1].legend()

    plt.tight_layout()
    path = save_dir / "distribuicao_metricas.png"
    plt.savefig(path, dpi=150)
    plt.close()
    saved_plots.append(str(path))

    thresholds = [results["results_per_number"][n]["optimal_threshold"] for n in range(1, TOTAL_NUMBERS + 1)]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(1, TOTAL_NUMBERS + 1), thresholds, color="#3498db", alpha=0.7)
    ax.axhline(NUMBERS_PER_DRAW / TOTAL_NUMBERS, color="#e74c3c", linestyle="--",
               label=f"Limiar teorico ({NUMBERS_PER_DRAW / TOTAL_NUMBERS:.2f})")
    ax.set_xlabel("Dezena")
    ax.set_ylabel("Limiar Otimo")
    ax.set_title("Limiar de Decisao Otimizado por Dezena")
    ax.legend()
    plt.tight_layout()
    path = save_dir / "limiares_otimos.png"
    plt.savefig(path, dpi=150)
    plt.close()
    saved_plots.append(str(path))

    return saved_plots


def run_mega_sena_analysis(
    model_name: str = "xgboost",
    show_plots: bool = True,
    save: bool = False,
    window: int = 30,
    ensemble: bool = False,
    stacking: bool = False,
    cv_splits: int = 3,
    use_smote: bool = True,
    calibrate: bool = True,
) -> dict:
    print("=" * 60)
    print("1. CARREGANDO DADOS DA MEGA SENA")
    print("=" * 60)
    if not DATA_PATH.exists():
        print(f"   Erro: Arquivo nao encontrado: {DATA_PATH}")
        sys.exit(1)

    df = load_mega_sena(DATA_PATH)
    print(f"   {len(df)} concursos carregados")
    print(f"   Periodo: {df['data'].iloc[0]} a {df['data'].iloc[-1]}")

    print("\n" + "=" * 60)
    print("2. ENGENHARIA DE FEATURES AVANCADA")
    print("=" * 60)
    print(f"   Janela primaria: {window} concursos")
    print("   Incluindo: codificacao ciclica, co-ocorrencia, tendencias,")
    print("   gaps sequenciais, regressao a media, features multi-janela")
    features_df = build_advanced_features(df, window=window, cooccurrence_window=60)
    print(f"   Features calculadas: {len(get_feature_columns(features_df))} colunas")
    print(f"   Concursos com features: {len(features_df)}")

    print("\n   Construindo variaveis alvo...")
    features_df = build_target(df, features_df)
    print(f"   Targets criados para dezenas 1-{TOTAL_NUMBERS}")

    print("\n" + "=" * 60)
    print("3. DIVISAO TEMPORAL DOS DADOS")
    print("=" * 60)
    train_df, test_df = split_by_date(df, features_df)
    print(f"   Treino (2000-2020):  {len(train_df)} concursos")
    print(f"   Teste  (2021-2026):  {len(test_df)} concursos")
    print(f"   Corte temporal: {CUTOFF_DATE.strftime('%d/%m/%Y')}")

    print("\n" + "=" * 60)
    print("4. TREINAMENTO E AVALIACAO ROBUSTA")
    print("=" * 60)
    print("   Melhorias aplicadas:")
    print("   - RobustScaler (resistente a outliers)")
    print("   - Imputacao por mediana (valores ausentes)")
    if use_smote:
        print("   - SMOTE (balanceamento de classes 90/10)")
    print("   - Limiar de decisao otimizado (curva PR)")
    if calibrate:
        print("   - Calibracao isotonica de probabilidades")
    if stacking:
        print("   - Stacking com meta-learner (RF + GB + XGB + LR)")
    elif ensemble:
        print("   - Ensemble ponderado por CV ROC-AUC (RF + GB + XGB + LR)")
    else:
        print(f"   - Modelo: {model_name}")
    print(f"   - Metricas: F1, PR-AUC, ROC-AUC, MCC, Balanced Acc, ECE")

    results = train_and_evaluate(
        train_df,
        test_df,
        model_name=model_name,
        ensemble=ensemble,
        stacking=stacking,
        cv_splits=cv_splits,
        use_smote=use_smote,
        calibrate=calibrate,
    )

    print("\n" + "=" * 60)
    print("5. ANALISE POR DEZENA")
    print("=" * 60)
    analysis = analyze_last_year(df, test_df, results)

    print("\n" + "=" * 60)
    print("6. PREVISAO DE JOGOS COMPLETOS")
    print("=" * 60)
    prediction = predict_full_games(df, test_df, results)

    print_report(results, analysis, len(train_df), len(test_df))
    print_full_games_report(prediction)

    if show_plots or save:
        print("\n" + "=" * 60)
        print("7. GERANDO GRAFICOS")
        print("=" * 60)
        plots_dir = Path(__file__).resolve().parent / "data" / "processed"
        saved_plots = plot_analysis(results, analysis, plots_dir)
        for p in saved_plots:
            print(f"   Salvo: {p}")

    return {
        "results": results,
        "analysis": analysis,
        "prediction": prediction,
        "train_size": len(train_df),
        "test_size": len(test_df),
    }


def main():
    parser = argparse.ArgumentParser(
        description="ML Mega Sena - Analise preditiva robusta"
    )
    parser.add_argument("--model", type=str, default="xgboost", choices=list(MODELS.keys()))
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--ensemble", action="store_true", help="Ensemble ponderado por CV")
    parser.add_argument("--stacking", action="store_true", help="Stacking com meta-learner")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--cv-splits", type=int, default=3)
    parser.add_argument("--no-smote", action="store_true", help="Desativar SMOTE")
    parser.add_argument("--no-calibration", action="store_true", help="Desativar calibracao")

    args = parser.parse_args()

    run_mega_sena_analysis(
        model_name=args.model,
        show_plots=not args.no_plots,
        save=args.save,
        window=args.window,
        ensemble=args.ensemble,
        stacking=args.stacking,
        cv_splits=args.cv_splits,
        use_smote=not args.no_smote,
        calibrate=not args.no_calibration,
    )


if __name__ == "__main__":
    main()
