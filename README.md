# 📈 Business Analysis com Regressão Linear Múltipla

Este repositório contém um estudo de caso usando **regressão linear múltipla** aplicado à análise de negócios. O objetivo é prever uma variável de interesse (ex: faturamento, lucro, desempenho) com base em múltiplas variáveis explicativas.

---

## 🧠 Objetivo do Projeto

O projeto utiliza **Python e bibliotecas de ciência de dados** para:

- Realizar análise exploratória de dados
- Construir e avaliar um modelo de regressão linear múltipla
- Interpretar os coeficientes do modelo
- Identificar insights estratégicos com base nos dados

---

## 📁 Estrutura dos Arquivos

| Arquivo/Notebook           | Descrição                                                        |
|----------------------------|------------------------------------------------------------------|
| `regressao_multipla.ipynb` | Notebook principal com todo o fluxo de análise e modelagem       |
| `/data/` (sugerido)         | Dados utilizados na análise (CSV, XLSX, etc.)                   |
| `/img/` (sugerido)          | Gráficos e visualizações exportadas                            |

---

## 🔧 Bibliotecas Utilizadas

- `pandas` para manipulação de dados  
- `numpy` para cálculos numéricos  
- `matplotlib` e `seaborn` para visualização  
- `sklearn` para regressão e métricas de avaliação  

Instale os pacotes necessários com:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn

🔍 Etapas da Análise
1. Carregamento dos Dados
python
Copiar
Editar
import pandas as pd
df = pd.read_csv('dados.csv')
2. Análise Exploratória
Histograma de variáveis

Matriz de correlação

Boxplots para outliers

3. Regressão Linear Múltipla
python
Copiar
Editar
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X = df[['variavel1', 'variavel2', 'variavel3']]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
modelo = LinearRegression().fit(X_train, y_train)
4. Avaliação do Modelo
python
Copiar
Editar
from sklearn.metrics import mean_squared_error, r2_score

y_pred = modelo.predict(X_test)
print("R²:", r2_score(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
5. Interpretação dos Coeficientes
python
Copiar
Editar
coeficientes = pd.DataFrame({'Variável': X.columns, 'Coeficiente': modelo.coef_})
print(coeficientes)
📊 Exemplos de Gráficos
python
Copiar
Editar
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlação entre variáveis")
plt.show()
📌 Conclusão
A regressão linear múltipla é uma ferramenta poderosa para prever métricas de negócio e entender o impacto de múltiplas variáveis independentes sobre um resultado. Este projeto mostra uma aplicação prática, desde a limpeza de dados até a interpretação estatística.

📚 Referências
Hands-On Machine Learning - Aurélien Géron

Scikit-learn documentation

Wes McKinney – Python for Data Analysis

