# ML Game Statistics

Projeto de Machine Learning para análise e previsão de estatísticas de jogos usando Python.

## Tecnologias

- Python 3.10+
- Pandas, NumPy, Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook

## Como usar

### Instalação

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Executar o pipeline

```bash
# Uso básico
python app.py data/raw/seu_arquivo.csv coluna_alvo

# Escolher modelo
python app.py data/raw/dados.csv resultado --model gradient_boosting

# Sem gráficos + salvar modelo
python app.py data/raw/dados.csv resultado --no-plots --save

# Modelos disponíveis: random_forest, gradient_boosting, logistic_regression
```

### Estrutura do projeto

```
ml-game-staticts/
├── app.py                  # Ponto de entrada principal (pipeline completo)
├── requirements.txt        # Dependências do projeto
├── src/
│   ├── data_loader.py      # Carregamento de dados (CSV/JSON)
│   ├── preprocessing.py    # Preprocessamento e feature engineering
│   ├── model.py            # Treinamento e avaliação de modelos
│   └── visualization.py    # Visualizações e gráficos
├── data/
│   ├── raw/                # Dados brutos
│   └── processed/          # Dados processados
├── models/                 # Modelos treinados salvos
└── notebooks/              # Jupyter Notebooks para exploração
```

## Em breve

- Coleta e análise de dados de jogos
- Modelos de Machine Learning para previsões
- Visualizações e dashboards
