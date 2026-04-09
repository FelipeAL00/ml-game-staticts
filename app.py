"""
app.py - Ponto de entrada principal do projeto ML Game Statistics.

Este arquivo orquestra todo o pipeline de Machine Learning:
1. Carregar dados
2. Preprocessar (tratar valores faltantes, escalar features, codificar labels)
3. Treinar modelo
4. Avaliar modelo
5. Visualizar resultados

Uso:
    python app.py
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.data_loader import load_csv, load_json, get_data_path
from src.preprocessing import handle_missing_values, scale_features, encode_labels
from src.model import split_data, train_model, evaluate_model, save_model, MODELS
from src.visualization import (
    plot_feature_importance,
    plot_confusion_matrix,
    plot_correlation_matrix,
    plot_distribution,
)


def load_data(filepath: str) -> pd.DataFrame:
    """Carrega os dados a partir de um arquivo CSV ou JSON.

    Args:
        filepath: Caminho para o arquivo de dados.

    Returns:
        DataFrame com os dados carregados.
    """
    path = Path(filepath)
    if path.suffix == ".csv":
        return load_csv(filepath)
    elif path.suffix == ".json":
        return load_json(filepath)
    else:
        raise ValueError(f"Formato não suportado: {path.suffix}. Use .csv ou .json")


def preprocess(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Executa o pipeline de preprocessamento dos dados.

    Args:
        df: DataFrame com os dados brutos.
        target_column: Nome da coluna alvo (target).

    Returns:
        DataFrame preprocessado.
    """
    print(f"[Preprocessing] Shape original: {df.shape}")
    print(f"[Preprocessing] Valores faltantes:\n{df.isnull().sum()}\n")

    # Tratar valores faltantes
    df = handle_missing_values(df, strategy="mean")

    # Codificar colunas categóricas (exceto a target se for categórica)
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_cols:
        df, _ = encode_labels(df, col)

    # Escalar features numéricas (exceto a target)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col != target_column]
    if feature_cols:
        df, _ = scale_features(df, feature_cols)

    print(f"[Preprocessing] Shape final: {df.shape}")
    return df


def run_pipeline(
    filepath: str,
    target_column: str,
    model_name: str = "random_forest",
    test_size: float = 0.2,
    show_plots: bool = True,
    save: bool = False,
) -> dict:
    """Executa o pipeline completo de ML.

    Args:
        filepath: Caminho para o arquivo de dados.
        target_column: Nome da coluna alvo.
        model_name: Nome do modelo a ser treinado.
        test_size: Proporção dos dados para teste.
        show_plots: Se True, exibe visualizações.
        save: Se True, salva o modelo treinado.

    Returns:
        Dicionário com o modelo treinado e métricas de avaliação.
    """
    # 1. Carregar dados
    print("=" * 50)
    print("1. CARREGANDO DADOS")
    print("=" * 50)
    df = load_data(filepath)
    print(f"   Dados carregados: {df.shape[0]} linhas, {df.shape[1]} colunas")
    print(f"   Colunas: {list(df.columns)}\n")

    # 2. Visualizar correlações (antes do preprocessamento)
    if show_plots:
        print("=" * 50)
        print("2. VISUALIZAÇÕES EXPLORATÓRIAS")
        print("=" * 50)
        plot_correlation_matrix(df)
        for col in df.select_dtypes(include=["number"]).columns[:3]:
            plot_distribution(df, col)

    # 3. Preprocessar
    print("=" * 50)
    print("3. PREPROCESSAMENTO")
    print("=" * 50)
    df = preprocess(df, target_column)

    # 4. Treinar modelo
    print("\n" + "=" * 50)
    print("4. TREINAMENTO DO MODELO")
    print("=" * 50)
    X_train, X_test, y_train, y_test = split_data(df, target_column, test_size=test_size)
    print(f"   Modelo: {model_name}")
    print(f"   Train: {X_train.shape[0]} amostras | Test: {X_test.shape[0]} amostras")

    model = train_model(X_train, y_train, model_name=model_name)
    print("   Modelo treinado com sucesso!\n")

    # 5. Avaliar modelo
    print("=" * 50)
    print("5. AVALIAÇÃO DO MODELO")
    print("=" * 50)
    results = evaluate_model(model, X_test, y_test)
    print(f"   Acurácia: {results['accuracy']:.4f}")
    print(f"\n   Classification Report:\n{results['classification_report']}")

    # 6. Visualizar resultados
    if show_plots:
        print("=" * 50)
        print("6. VISUALIZAÇÕES DOS RESULTADOS")
        print("=" * 50)
        plot_confusion_matrix(results["confusion_matrix"])
        if hasattr(model, "feature_importances_"):
            plot_feature_importance(model, list(X_train.columns))

    # 7. Salvar modelo (opcional)
    if save:
        model_path = save_model(model, f"{model_name}_game_stats.joblib")
        print(f"\n   Modelo salvo em: {model_path}")

    return {"model": model, "results": results}


def main():
    """Função principal — parse de argumentos e execução do pipeline."""
    parser = argparse.ArgumentParser(
        description="ML Game Statistics - Pipeline de Machine Learning para estatísticas de jogos"
    )
    parser.add_argument(
        "data_file",
        type=str,
        help="Caminho para o arquivo de dados (CSV ou JSON)",
    )
    parser.add_argument(
        "target",
        type=str,
        help="Nome da coluna alvo (target) para previsão",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="random_forest",
        choices=list(MODELS.keys()),
        help="Modelo de ML a ser usado (default: random_forest)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proporção dos dados para teste (default: 0.2)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Desativar visualizações",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Salvar o modelo treinado",
    )

    args = parser.parse_args()

    # Verificar se o arquivo existe
    if not Path(args.data_file).exists():
        print(f"Erro: Arquivo não encontrado: {args.data_file}")
        sys.exit(1)

    run_pipeline(
        filepath=args.data_file,
        target_column=args.target,
        model_name=args.model,
        test_size=args.test_size,
        show_plots=not args.no_plots,
        save=args.save,
    )


if __name__ == "__main__":
    main()
