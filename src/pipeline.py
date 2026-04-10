import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from os import PathLike
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.features import build_enhanced_features
from src.imbalance import (
    balance_with_smote,
    compute_scale_pos_weight,
    optimize_threshold_precision_recall,
)
from src.metrics import compute_classification_metrics
from src.calibration import calibrate_model, expected_calibration_error

TOTAL_NUMBERS = 60
NUMBERS_PER_DRAW = 6


def load_mega_sena(filepath: str | PathLike) -> pd.DataFrame:
    """Carrega os dados da Mega Sena e converte a coluna de data."""
    df = pd.read_csv(filepath)
    df["data_parsed"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df = df.sort_values("concurso").reset_index(drop=True)
    return df


def generate_features(df: pd.DataFrame, window: int = 30, windows: tuple[int, ...] = (10, 30, 60)) -> pd.DataFrame:
    """Gera o conjunto de features a partir dos dados brutos."""
    return build_enhanced_features(df, window=window, windows=windows)


def build_target(df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    """Cria variável alvo para cada dezena (1-60)."""
    dezena_cols = [f"dezena_{i}" for i in range(1, 7)]

    for n in range(1, TOTAL_NUMBERS + 1):
        targets = []
        for _, row in features_df.iterrows():
            idx = int(row["idx"])
            drawn = df.iloc[idx][dezena_cols].astype(int).values
            targets.append(1 if n in drawn else 0)
        features_df[f"target_{n}"] = targets

    return features_df


def split_by_date(df_original: pd.DataFrame, features_df: pd.DataFrame, cutoff_date: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide o conjunto por data usando corte temporal."""
    features_df = features_df.merge(
        df_original[["concurso", "data_parsed"]],
        on="concurso",
        how="left",
    )
    train = features_df[features_df["data_parsed"] < cutoff_date].copy()
    test = features_df[features_df["data_parsed"] >= cutoff_date].copy()
    return train, test


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Retorna somente as colunas de entrada do modelo."""
    exclude = {"concurso", "idx", "data_parsed"}
    target_cols = {c for c in df.columns if c.startswith("target_")}
    return [c for c in df.columns if c not in exclude and c not in target_cols]


def _build_single_model(model_name: str, tuned: bool = False, scale_pos_weight: float | None = None):
    """Cria uma instância de modelo com hiperparâmetros padrão ou otimizados."""
    if tuned:
        if model_name == "random_forest":
            return RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=3,
                max_features="sqrt",
                class_weight="balanced",
                random_state=42,
            )
        elif model_name == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_split=5,
                min_samples_leaf=3,
                random_state=42,
            )
        elif model_name == "xgboost":
            params = {
                "n_estimators": 200,
                "max_depth": 5,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "use_label_encoder": False,
                "eval_metric": "logloss",
                "random_state": 42,
                "n_jobs": 1,
            }
            if scale_pos_weight is not None:
                params["scale_pos_weight"] = scale_pos_weight
            return XGBClassifier(**params)
        else:
            return LogisticRegression(
                max_iter=2000,
                C=0.5,
                penalty="l2",
                solver="lbfgs",
                class_weight="balanced",
                random_state=42,
            )
    else:
        if model_name == "logistic_regression":
            return LogisticRegression(max_iter=1000, random_state=42)
        elif model_name == "xgboost":
            params = {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "use_label_encoder": False,
                "eval_metric": "logloss",
                "random_state": 42,
                "n_jobs": 1,
            }
            if scale_pos_weight is not None:
                params["scale_pos_weight"] = scale_pos_weight
            return XGBClassifier(**params)
        else:
            return {
                "random_forest": RandomForestClassifier,
                "gradient_boosting": GradientBoostingClassifier,
                "logistic_regression": LogisticRegression,
            }[model_name](n_estimators=100, random_state=42)


def _compute_time_series_cv(
    X: np.ndarray,
    y: np.ndarray,
    model_names: list[str],
    cv_splits: int = 5,
    ensemble: bool = False,
) -> dict:
    """Avalia o treino usando TimeSeriesSplit."""
    if len(y) < 2:
        return {"cv_accuracy": np.nan, "cv_balanced_accuracy": np.nan, "cv_roc_auc": np.nan}

    n_splits = min(cv_splits, max(1, len(y) - 1))
    if n_splits < 2:
        return {"cv_accuracy": np.nan, "cv_balanced_accuracy": np.nan, "cv_roc_auc": np.nan}

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_accuracies = []
    fold_aucs = []
    fold_balances = []

    for train_idx, val_idx in tscv.split(X):
        X_train_cv, X_val_cv = X[train_idx], X[val_idx]
        y_train_cv, y_val_cv = y[train_idx], y[val_idx]

        all_proba_cols = []
        all_preds = []

        for m_name in model_names:
            model = _build_single_model(m_name, tuned=ensemble)
            model.fit(X_train_cv, y_train_cv)

            y_pred_cv = model.predict(X_val_cv)
            y_proba_cv = model.predict_proba(X_val_cv)
            proba_col_cv = y_proba_cv[:, 1] if y_proba_cv.shape[1] > 1 else y_proba_cv[:, 0]

            all_proba_cols.append(proba_col_cv)
            all_preds.append(y_pred_cv)

        avg_proba_cv = np.mean(all_proba_cols, axis=0)
        maj_pred_cv = (np.mean(all_preds, axis=0) >= 0.5).astype(int)
        fold_accuracies.append(accuracy_score(y_val_cv, maj_pred_cv))

        try:
            fold_aucs.append(roc_auc_score(y_val_cv, avg_proba_cv))
        except ValueError:
            pass

        try:
            fold_balances.append(balanced_accuracy_score(y_val_cv, maj_pred_cv))
        except ValueError:
            pass

    return {
        "cv_accuracy": float(np.mean(fold_accuracies)) if fold_accuracies else np.nan,
        "cv_balanced_accuracy": float(np.nanmean(fold_balances)) if fold_balances else np.nan,
        "cv_roc_auc": float(np.nanmean(fold_aucs)) if fold_aucs else np.nan,
    }


def train_and_evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str = "random_forest",
    top_n: int = 10,
    ensemble: bool = False,
    cv_splits: int = 5,
) -> dict:
    """Treina e avalia os modelos para cada dezena."""
    feature_cols = get_feature_columns(train_df)
    X_train_raw = train_df[feature_cols].values
    X_test_raw = test_df[feature_cols].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    results_per_number = {}
    all_probabilities = {}
    per_concurso_probas = {}

    if ensemble:
        model_names_used = ["random_forest", "gradient_boosting", "logistic_regression", "xgboost"]
    else:
        model_names_used = [model_name]

    for n in range(1, TOTAL_NUMBERS + 1):
        target_col = f"target_{n}"
        y_train = train_df[target_col].values
        y_test = test_df[target_col].values

        X_train_balanced, y_train_balanced = balance_with_smote(X_train, y_train)
        scale_weight = compute_scale_pos_weight(y_train)

        all_proba_cols = []
        all_proba_cols_train = []
        all_preds = []
        models_trained = []

        for m_name in model_names_used:
            model = _build_single_model(m_name, tuned=ensemble, scale_pos_weight=scale_weight)
            model.fit(X_train_balanced, y_train_balanced)

            if m_name == "xgboost":
                model = calibrate_model(model, X_train_balanced, y_train_balanced, method="isotonic", cv=3)

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            proba_col = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba[:, 0]
            all_proba_cols.append(proba_col)

            y_proba_train = model.predict_proba(X_train_balanced)
            train_proba_col = y_proba_train[:, 1] if y_proba_train.shape[1] > 1 else y_proba_train[:, 0]
            all_proba_cols_train.append(train_proba_col)

            all_preds.append(y_pred)
            models_trained.append(model)

        avg_proba_col = np.mean(all_proba_cols, axis=0)
        avg_proba_train = np.mean(all_proba_cols_train, axis=0)
        decision_threshold = optimize_threshold_precision_recall(y_train_balanced, avg_proba_train)
        optimized_pred = (avg_proba_col >= decision_threshold).astype(int)

        cv_results = _compute_time_series_cv(
            X_train,
            y_train,
            model_names_used,
            cv_splits=cv_splits,
            ensemble=ensemble,
        )

        classification_metrics = compute_classification_metrics(y_test, optimized_pred, avg_proba_col)
        ece = expected_calibration_error(y_test, avg_proba_col)

        results_per_number[n] = {
            "accuracy": accuracy_score(y_test, optimized_pred),
            "balanced_accuracy": classification_metrics["balanced_accuracy"],
            "precision": classification_metrics["precision"],
            "recall": classification_metrics["recall"],
            "f1": classification_metrics["f1"],
            "pr_auc": classification_metrics["pr_auc"],
            "roc_auc": classification_metrics["roc_auc"],
            "mcc": classification_metrics["mcc"],
            "cv_accuracy": cv_results["cv_accuracy"],
            "cv_balanced_accuracy": cv_results["cv_balanced_accuracy"],
            "cv_roc_auc": cv_results["cv_roc_auc"],
            "avg_probability": float(np.mean(avg_proba_col)),
            "expected_calibration_error": ece,
            "decision_threshold": decision_threshold,
            "actual_frequency": float(np.mean(y_test)),
            "predicted_frequency": float(np.mean(optimized_pred)),
            "model": models_trained[0],
        }
        all_probabilities[n] = float(np.mean(avg_proba_col))
        per_concurso_probas[n] = avg_proba_col

    ranking = sorted(all_probabilities.items(), key=lambda x: x[1], reverse=True)

    return {
        "results_per_number": results_per_number,
        "ranking": ranking,
        "feature_columns": feature_cols,
        "per_concurso_probas": per_concurso_probas,
        "ensemble": ensemble,
        "cv_splits": cv_splits,
    }


def analyze_last_year(
    df: pd.DataFrame,
    test_df: pd.DataFrame,
    results: dict,
    top_n: int = 10,
) -> dict:
    """Analisa o desempenho da previsão no último ano."""
    dezena_cols = [f"dezena_{i}" for i in range(1, 7)]
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
    total_draws = len(test_concursos) * NUMBERS_PER_DRAW
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
    """Prevê jogos completos para cada concurso de teste."""
    dezena_cols = [f"dezena_{i}" for i in range(1, 7)]
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
    """Imprime relatório de previsão de jogos completos."""
    games = prediction["games"]

    print("\n" + "=" * 80)
    print("   PREVISÃO DE JOGOS COMPLETOS - MEGA SENA 2026")
    print("=" * 80)
    print(f"\n   Total de jogos analisados: {prediction['total_games']}")
    print(f"   Média de acertos por jogo: {prediction['avg_hits_per_game']:.2f} de 6")
    print(f"   Taxa de acerto (dezenas): {prediction['hit_rate'] * 100:.1f}%")
    print(f"   Máximo de acertos em um jogo: {prediction['max_hits']}")

    print("\n" + "-" * 80)
    print("   DISTRIBUIÇÃO DE ACERTOS")
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
        prev_str = ", ".join(f"{n:02d}" for n in g["predicted"])
        real_str = ", ".join(f"{n:02d}" for n in g["actual"])

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
    print(f"   Média de acertos/jogo:     {prediction['avg_hits_per_game']:.2f} de 6")

    best_games = sorted(games, key=lambda g: g["hits"], reverse=True)[:5]
    print("\n   Melhores jogos (mais acertos):")
    for g in best_games:
        matched = set(g["actual"]) & set(g["predicted"])
        matched_str = ", ".join(f"{n:02d}" for n in sorted(matched))
        print(f"     Concurso {g['concurso']} ({g['data']}): {g['hits']}/6 acertos — dezenas: {matched_str}")

    print("\n" + "=" * 80)


def print_report(results: dict, analysis: dict, train_size: int, test_size: int) -> None:
    """Imprime relatório completo da análise."""
    ranking = results["ranking"]

    print("\n" + "=" * 60)
    print("   RELATÓRIO DE ANÁLISE - MEGA SENA")
    print("=" * 60)

    print(f"\n   Período de treino: janeiro 2000 - dezembro 2020 ({train_size} concursos)")
    print(f"   Período de teste:  janeiro 2021 - dezembro 2026 ({test_size} concursos)")

    print("\n" + "-" * 60)
    print("   TOP 10 DEZENAS MAIS PROVÁVEIS (previsão do modelo)")
    print("-" * 60)
    for i, (n, prob) in enumerate(ranking[:10], 1):
        actual_freq = results["results_per_number"][n]["actual_frequency"]
        print(f"   {i:2d}. Dezena {n:02d}  |  Probabilidade: {prob:.4f}  |  Freq. real no teste: {actual_freq:.4f}")

    print("\n" + "-" * 60)
    print("   TOP 10 DEZENAS MAIS SORTEADAS NO ÚLTIMO ANO (real)")
    print("-" * 60)
    for i, (n, count) in enumerate(analysis["top_actual"], 1):
        print(f"   {i:2d}. Dezena {n:02d}  |  Sorteada {count} vezes")

    print("\n" + "-" * 60)
    print("   COMPARAÇÃO: PREVISÃO vs REALIDADE")
    print("-" * 60)
    print(f"   Dezenas previstas no top {analysis['top_k']}:  {sorted(analysis['top_predicted'])}")
    print(f"   Dezenas reais no top {analysis['top_k']}:      {sorted([n for n, _ in analysis['top_actual']])}")
    print(f"   Acertos (overlap):            {sorted(analysis['overlap'])} ({analysis['overlap_count']}/{analysis['top_k']})")
    print(f"   Top-{analysis['top_k']} recall no teste:    {analysis['top_k_recall']:.4f}")

    print("\n" + "-" * 60)
    print("   DEZENAS MENOS PROVÁVEIS (frias)")
    print("-" * 60)
    for i, (n, prob) in enumerate(ranking[-10:], 1):
        actual_freq = results["results_per_number"][n]["actual_frequency"]
        print(f"   {i:2d}. Dezena {n:02d}  |  Probabilidade: {prob:.4f}  |  Freq. real no teste: {actual_freq:.4f}")

    accuracies = [r["accuracy"] for r in results["results_per_number"].values()]
    cv_accuracies = [r["cv_accuracy"] for r in results["results_per_number"].values() if not np.isnan(r["cv_accuracy"])]
    cv_balanced_accuracies = [r["cv_balanced_accuracy"] for r in results["results_per_number"].values() if not np.isnan(r["cv_balanced_accuracy"])]
    cv_aucs = [r["cv_roc_auc"] for r in results["results_per_number"].values() if not np.isnan(r["cv_roc_auc"])]
    balanced_accuracies = [r["balanced_accuracy"] for r in results["results_per_number"].values() if not np.isnan(r["balanced_accuracy"])]
    pr_aucs = [r["pr_auc"] for r in results["results_per_number"].values() if not np.isnan(r["pr_auc"])]
    mccs = [r["mcc"] for r in results["results_per_number"].values() if not np.isnan(r["mcc"])]
    eces = [r["expected_calibration_error"] for r in results["results_per_number"].values() if not np.isnan(r["expected_calibration_error"])]

    print("\n" + "-" * 60)
    print("   MÉTRICAS GERAIS")
    print("-" * 60)
    print(f"   Acurácia média dos modelos:            {np.mean(accuracies):.4f}")
    print(f"   Acurácia mínima:                       {np.min(accuracies):.4f}")
    print(f"   Acurácia máxima:                       {np.max(accuracies):.4f}")
    print(f"   Balanced Accuracy média dos modelos:   {np.mean(balanced_accuracies):.4f}")
    print(f"   F1 Score médio dos modelos:            {np.mean([r['f1'] for r in results['results_per_number'].values() if not np.isnan(r['f1'])]):.4f}")
    print(f"   PR-AUC médio dos modelos:              {np.mean(pr_aucs):.4f}")
    print(f"   MCC médio dos modelos:                 {np.mean(mccs):.4f}")
    if eces:
        print(f"   ECE médio dos modelos:                 {np.mean(eces):.4f}")
    if cv_accuracies:
        print(f"   Acurácia média CV (TimeSeriesSplit):    {np.mean(cv_accuracies):.4f}")
    if cv_balanced_accuracies:
        print(f"   Balanced Accuracy média CV:             {np.mean(cv_balanced_accuracies):.4f}")
    if cv_aucs:
        print(f"   ROC AUC média CV:                      {np.mean(cv_aucs):.4f}")
    print(f"   Total de concursos analisados no teste: {analysis['test_concursos_count']}")

    print("\n" + "=" * 60)


def plot_analysis(results: dict, analysis: dict, save_dir: Path) -> list[str]:
    """Gera gráficos da análise e salva em arquivos."""
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
    ax.set_ylabel("Probabilidade Média")
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
    actual_freqs = [results["results_per_number"][n]["actual_frequency"] for n in top_predicted]

    width = 0.35
    ax.bar([i - width / 2 for i in x], predicted_probs, width, label="Probabilidade Prevista", color="#3498db")
    ax.bar([i + width / 2 for i in x], actual_freqs, width, label="Frequência Real", color="#e74c3c")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{n:02d}" for n in top_predicted])
    ax.set_xlabel("Dezena")
    ax.set_ylabel("Proporção")
    ax.set_title("Top 10 Dezenas Previstas: Probabilidade vs Frequência Real")
    ax.legend()
    plt.tight_layout()
    path = save_dir / "previsto_vs_real.png"
    plt.savefig(path, dpi=150)
    plt.close()
    saved_plots.append(str(path))

    actual_freq_all = {n: results["results_per_number"][n]["actual_frequency"] for n in range(1, TOTAL_NUMBERS + 1)}
    grid = np.zeros((6, 10))
    labels_grid = np.zeros((6, 10), dtype=int)
    for n in range(1, TOTAL_NUMBERS + 1):
        row = (n - 1) // 10
        col = (n - 1) % 10
        grid[row][col] = actual_freq_all[n]
        labels_grid[row][col] = n

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        grid,
        annot=labels_grid,
        fmt="d",
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": "Frequência no último ano"},
        linewidths=1,
    )
    ax.set_title("Frequência Real de Cada Dezena no Último Ano (Teste)")
    ax.set_xlabel("Coluna")
    ax.set_ylabel("Linha")
    plt.tight_layout()
    path = save_dir / "heatmap_frequencia.png"
    plt.savefig(path, dpi=150)
    plt.close()
    saved_plots.append(str(path))

    accuracies = [results["results_per_number"][n]["accuracy"] for n in range(1, TOTAL_NUMBERS + 1)]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(accuracies, bins=20, color="#3498db", edgecolor="white")
    ax.axvline(np.mean(accuracies), color="#e74c3c", linestyle="--", label=f"Média: {np.mean(accuracies):.4f}")
    ax.set_xlabel("Acurácia")
    ax.set_ylabel("Quantidade de Modelos")
    ax.set_title("Distribuição de Acurácia dos 60 Modelos (um por dezena)")
    ax.legend()
    plt.tight_layout()
    path = save_dir / "distribuicao_acuracia.png"
    plt.savefig(path, dpi=150)
    plt.close()
    saved_plots.append(str(path))

    return saved_plots
