"""
app.py - Ponto de entrada principal do projeto ML Game Statistics (Mega Sena).

Pipeline de Machine Learning para análise da Mega Sena:
1. Carregar dados históricos (2020-2026)
2. Engenharia de features (frequência, atraso, paridade, soma, etc.)
3. Treinar modelo nos primeiros 5 anos (2020 - março 2025)
4. Analisar e prever o último ano (abril 2025 - abril 2026)
5. Gerar relatório com estatísticas e análises

Uso:
    python app.py                           # Executa a análise completa
    python app.py --no-plots                # Sem gráficos
    python app.py --model gradient_boosting # Escolher modelo
    python app.py --save                    # Salvar modelo treinado
"""

import argparse
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


DATA_PATH = Path(__file__).resolve().parent / "data" / "raw" / "mega_sena_2020_2026.csv"
CUTOFF_DATE = pd.Timestamp("2025-04-01")
TOTAL_NUMBERS = 60  # Mega Sena: números de 1 a 60

MODELS = {
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "logistic_regression": LogisticRegression,
}


def load_mega_sena(filepath: Path) -> pd.DataFrame:
    """Carrega os dados da Mega Sena e converte a coluna de data."""
    df = pd.read_csv(filepath)
    df["data_parsed"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df = df.sort_values("concurso").reset_index(drop=True)
    return df


def build_frequency_features(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """Calcula features de frequência para cada dezena (1-60) em janelas deslizantes.

    Para cada concurso, calcula:
    - Frequência de cada dezena nos últimos N concursos
    - Atraso (quantos concursos desde a última aparição)
    - Paridade e soma do último concurso
    - Soma média e desvio padrão da janela

    Args:
        df: DataFrame com os dados da Mega Sena.
        window: Tamanho da janela deslizante.

    Returns:
        DataFrame com features calculadas por concurso.
    """
    dezena_cols = ["dezena_1", "dezena_2", "dezena_3", "dezena_4", "dezena_5", "dezena_6"]
    all_draws = df[dezena_cols].values.astype(int)

    features_list = []

    for i in range(window, len(df)):
        recent_draws = all_draws[max(0, i - window):i]
        recent_numbers = recent_draws.flatten()

        freq = Counter(recent_numbers)
        freq_features = {f"freq_{n}": freq.get(n, 0) / len(recent_draws) for n in range(1, TOTAL_NUMBERS + 1)}

        delay_features = {}
        for n in range(1, TOTAL_NUMBERS + 1):
            last_seen = -1
            for j in range(len(recent_draws) - 1, -1, -1):
                if n in recent_draws[j]:
                    last_seen = j
                    break
            delay_features[f"atraso_{n}"] = (len(recent_draws) - last_seen) if last_seen >= 0 else window + 1

        last_draw = all_draws[i - 1]
        soma_ultimo = int(np.sum(last_draw))
        pares_ultimo = int(np.sum(last_draw % 2 == 0))
        impares_ultimo = 6 - pares_ultimo

        somas = [int(np.sum(d)) for d in recent_draws]
        soma_media = np.mean(somas)
        soma_std = np.std(somas)

        row = {
            "concurso": df.iloc[i]["concurso"],
            "idx": i,
            **freq_features,
            **delay_features,
            "soma_ultimo": soma_ultimo,
            "pares_ultimo": pares_ultimo,
            "impares_ultimo": impares_ultimo,
            "soma_media_janela": soma_media,
            "soma_std_janela": soma_std,
        }
        features_list.append(row)

    return pd.DataFrame(features_list)


def build_target(df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    """Cria a variável alvo: para cada dezena (1-60), se ela foi sorteada no concurso.

    Args:
        df: DataFrame original da Mega Sena.
        features_df: DataFrame com as features calculadas.

    Returns:
        DataFrame com uma coluna target para cada dezena (target_1 ... target_60).
    """
    dezena_cols = ["dezena_1", "dezena_2", "dezena_3", "dezena_4", "dezena_5", "dezena_6"]

    for n in range(1, TOTAL_NUMBERS + 1):
        targets = []
        for _, row in features_df.iterrows():
            idx = int(row["idx"])
            drawn = df.iloc[idx][dezena_cols].astype(int).values
            targets.append(1 if n in drawn else 0)
        features_df[f"target_{n}"] = targets

    return features_df


def split_by_date(df_original: pd.DataFrame, features_df: pd.DataFrame) -> tuple:
    """Divide os dados por data: primeiros 5 anos para treino, último ano para teste.

    Args:
        df_original: DataFrame original da Mega Sena.
        features_df: DataFrame com features e targets.

    Returns:
        Tuple (train_df, test_df).
    """
    features_df = features_df.merge(
        df_original[["concurso", "data_parsed"]],
        on="concurso",
        how="left",
    )
    train = features_df[features_df["data_parsed"] < CUTOFF_DATE].copy()
    test = features_df[features_df["data_parsed"] >= CUTOFF_DATE].copy()
    return train, test


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Retorna as colunas de features (exclui concurso, idx, targets, data)."""
    exclude = {"concurso", "idx", "data_parsed"}
    target_cols = {c for c in df.columns if c.startswith("target_")}
    return [c for c in df.columns if c not in exclude and c not in target_cols]


def train_and_evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str = "random_forest",
    top_n: int = 10,
) -> dict:
    """Treina um modelo para cada dezena e avalia no último ano.

    Para cada número (1-60), treina um classificador binário que prevê
    se aquele número será sorteado no próximo concurso.

    Args:
        train_df: DataFrame de treino (5 primeiros anos).
        test_df: DataFrame de teste (último ano).
        model_name: Nome do modelo.
        top_n: Quantidade de dezenas mais prováveis para recomendar.

    Returns:
        Dicionário com resultados detalhados.
    """
    feature_cols = get_feature_columns(train_df)
    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values

    results_per_number = {}
    all_probabilities = {}

    print(f"\n   Treinando modelos para cada dezena (1-{TOTAL_NUMBERS})...")

    for n in range(1, TOTAL_NUMBERS + 1):
        target_col = f"target_{n}"
        y_train = train_df[target_col].values
        y_test = test_df[target_col].values

        model_cls = MODELS[model_name]
        if model_name == "logistic_regression":
            model = model_cls(max_iter=1000, random_state=42)
        else:
            model = model_cls(n_estimators=100, random_state=42)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        proba_col = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba[:, 0]
        avg_proba = float(np.mean(proba_col))

        acc = accuracy_score(y_test, y_pred)

        results_per_number[n] = {
            "accuracy": acc,
            "avg_probability": avg_proba,
            "actual_frequency": float(np.mean(y_test)),
            "predicted_frequency": float(np.mean(y_pred)),
            "model": model,
        }
        all_probabilities[n] = avg_proba

    ranking = sorted(all_probabilities.items(), key=lambda x: x[1], reverse=True)

    return {
        "results_per_number": results_per_number,
        "ranking": ranking,
        "feature_columns": feature_cols,
    }


def analyze_last_year(
    df: pd.DataFrame,
    test_df: pd.DataFrame,
    results: dict,
    top_n: int = 10,
) -> dict:
    """Analisa o desempenho das previsões no último ano.

    Args:
        df: DataFrame original.
        test_df: DataFrame de teste.
        results: Resultados do treinamento.
        top_n: Número de dezenas top para analisar.

    Returns:
        Dicionário com análise detalhada.
    """
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

    return {
        "top_predicted": top_numbers,
        "top_actual": most_common_actual,
        "overlap": overlap,
        "overlap_count": len(overlap),
        "hits_in_test": hits,
        "total_draws": len(test_concursos) * 6,
        "test_concursos_count": len(test_concursos),
    }


def print_report(results: dict, analysis: dict, train_size: int, test_size: int) -> None:
    """Imprime o relatório completo da análise."""
    ranking = results["ranking"]

    print("\n" + "=" * 60)
    print("   RELATÓRIO DE ANÁLISE - MEGA SENA")
    print("=" * 60)

    print(f"\n   Período de treino: março 2020 - março 2025 ({train_size} concursos)")
    print(f"   Período de teste:  abril 2025 - abril 2026 ({test_size} concursos)")

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
    print(f"   Dezenas previstas no top 10:  {sorted(analysis['top_predicted'])}")
    print(f"   Dezenas reais no top 10:      {sorted([n for n, _ in analysis['top_actual']])}")
    print(f"   Acertos (overlap):            {sorted(analysis['overlap'])} ({analysis['overlap_count']}/10)")

    print("\n" + "-" * 60)
    print("   DEZENAS MENOS PROVÁVEIS (frias)")
    print("-" * 60)
    for i, (n, prob) in enumerate(ranking[-10:], 1):
        actual_freq = results["results_per_number"][n]["actual_frequency"]
        print(f"   {i:2d}. Dezena {n:02d}  |  Probabilidade: {prob:.4f}  |  Freq. real no teste: {actual_freq:.4f}")

    accuracies = [r["accuracy"] for r in results["results_per_number"].values()]
    print("\n" + "-" * 60)
    print("   MÉTRICAS GERAIS")
    print("-" * 60)
    print(f"   Acurácia média dos modelos:  {np.mean(accuracies):.4f}")
    print(f"   Acurácia mínima:             {np.min(accuracies):.4f}")
    print(f"   Acurácia máxima:             {np.max(accuracies):.4f}")
    print(f"   Total de concursos analisados no teste: {analysis['test_concursos_count']}")

    print("\n" + "=" * 60)


def plot_analysis(results: dict, analysis: dict, save_dir: Path) -> list[str]:
    """Gera gráficos da análise e salva em arquivos.

    Args:
        results: Resultados do treinamento.
        analysis: Análise do último ano.
        save_dir: Diretório para salvar os gráficos.

    Returns:
        Lista de caminhos dos gráficos salvos.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    saved_plots = []

    # 1. Ranking de probabilidades
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

    # 2. Comparação: previsto vs real
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

    # 3. Heatmap de frequência por dezena no último ano
    actual_freq_all = {}
    for n in range(1, TOTAL_NUMBERS + 1):
        actual_freq_all[n] = results["results_per_number"][n]["actual_frequency"]

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

    # 4. Distribuição de acurácia dos modelos
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


def run_mega_sena_analysis(
    model_name: str = "random_forest",
    show_plots: bool = True,
    save: bool = False,
    window: int = 30,
) -> dict:
    """Executa a análise completa da Mega Sena.

    Args:
        model_name: Modelo de ML a usar.
        show_plots: Se True, gera gráficos.
        save: Se True, salva gráficos e modelo.
        window: Tamanho da janela para features de frequência.

    Returns:
        Dicionário com todos os resultados.
    """
    # 1. Carregar dados
    print("=" * 60)
    print("1. CARREGANDO DADOS DA MEGA SENA")
    print("=" * 60)
    if not DATA_PATH.exists():
        print(f"   Erro: Arquivo não encontrado: {DATA_PATH}")
        print("   Execute primeiro a coleta de dados.")
        sys.exit(1)

    df = load_mega_sena(DATA_PATH)
    print(f"   {len(df)} concursos carregados")
    print(f"   Período: {df['data'].iloc[0]} a {df['data'].iloc[-1]}")
    print(f"   Distribuição por ano:")
    for year, count in df["data_parsed"].dt.year.value_counts().sort_index().items():
        print(f"     {year}: {count} concursos")

    # 2. Engenharia de features
    print("\n" + "=" * 60)
    print("2. ENGENHARIA DE FEATURES")
    print("=" * 60)
    print(f"   Janela deslizante: {window} concursos")
    features_df = build_frequency_features(df, window=window)
    print(f"   Features calculadas: {len(get_feature_columns(features_df))} colunas")
    print(f"   Concursos com features: {len(features_df)}")

    # 3. Construir targets
    print("\n   Construindo variáveis alvo...")
    features_df = build_target(df, features_df)
    print(f"   Targets criados para dezenas 1-{TOTAL_NUMBERS}")

    # 4. Dividir por tempo
    print("\n" + "=" * 60)
    print("3. DIVISÃO TEMPORAL DOS DADOS")
    print("=" * 60)
    train_df, test_df = split_by_date(df, features_df)
    print(f"   Treino (5 primeiros anos): {len(train_df)} concursos")
    print(f"   Teste (último ano):        {len(test_df)} concursos")
    print(f"   Corte temporal: {CUTOFF_DATE.strftime('%d/%m/%Y')}")

    # 5. Treinar e avaliar
    print("\n" + "=" * 60)
    print("4. TREINAMENTO E AVALIAÇÃO")
    print("=" * 60)
    print(f"   Modelo: {model_name}")
    results = train_and_evaluate(train_df, test_df, model_name=model_name)

    # 6. Análise do último ano
    print("\n" + "=" * 60)
    print("5. ANÁLISE DO ÚLTIMO ANO")
    print("=" * 60)
    analysis = analyze_last_year(df, test_df, results)

    # 7. Relatório
    print_report(results, analysis, len(train_df), len(test_df))

    # 8. Gráficos
    if show_plots or save:
        print("\n" + "=" * 60)
        print("6. GERANDO GRÁFICOS")
        print("=" * 60)
        plots_dir = Path(__file__).resolve().parent / "data" / "processed"
        saved_plots = plot_analysis(results, analysis, plots_dir)
        for p in saved_plots:
            print(f"   Salvo: {p}")

    return {
        "results": results,
        "analysis": analysis,
        "train_size": len(train_df),
        "test_size": len(test_df),
    }


def main():
    """Função principal — parse de argumentos e execução da análise."""
    parser = argparse.ArgumentParser(
        description="ML Mega Sena - Análise preditiva dos resultados da Mega Sena"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="random_forest",
        choices=list(MODELS.keys()),
        help="Modelo de ML a ser usado (default: random_forest)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=30,
        help="Tamanho da janela deslizante para features de frequência (default: 30)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Desativar geração de gráficos",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Salvar gráficos e modelo treinado",
    )

    args = parser.parse_args()

    run_mega_sena_analysis(
        model_name=args.model,
        show_plots=not args.no_plots,
        save=args.save,
        window=args.window,
    )


if __name__ == "__main__":
    main()
