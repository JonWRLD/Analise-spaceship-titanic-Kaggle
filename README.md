# 🚀 Análise de Dados: Spaceship Titanic

Análise completa de dados feita do zero com Python: descrição das variáveis, diagnóstico e limpeza, análise exploratória com testes estatísticos e análise preditiva para classificar quais passageiros foram **transportados para outra dimensão**.

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1yD_CcDQxvKG69RqJA4vQiFJotMuKOrP_)
![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-modelagem-F7931E?logo=scikitlearn&logoColor=white)

---

## 📌 Sobre o projeto

Trabalho desenvolvido para a disciplina de **Análise de Dados**. A base descreve passageiros da nave *Spaceship Titanic*, que colidiu com uma anomalia do espaço-tempo; parte deles foi transportada para outra dimensão.

**Pergunta central:** quais características dos passageiros estão associadas a terem sido transportados, e com que precisão é possível prever isso?

- **Base de dados:** [Spaceship Titanic (Kaggle)](https://www.kaggle.com/competitions/spaceship-titanic/data), arquivo `train.csv` (8.693 passageiros e 14 colunas)
- **Variável-alvo:** `Transported` (classificação binária)
- **Ambiente:** Google Colab

## 🔎 Etapas da análise

1. **Descrição das variáveis:** tipo e significado de cada uma das 14 colunas originais
2. **Diagnóstico inicial:** dimensões, estatísticas descritivas, valores ausentes e duplicados
3. **Limpeza e tratamento**
   - Extração de variáveis de `PassengerId` (grupo e tamanho do grupo) e `Cabin` (deck, número e lado)
   - Imputação com **regras de negócio** antes de recorrer a estatística (quem está em criossono não gasta; `HomePlanet` pela moda do grupo de viagem)
   - Imputação por mediana (numéricas) e moda (categóricas) para o restante
   - Variáveis derivadas: `TotalSpend` e `NoSpend`
   - Análise de outliers pelo critério do IQR (valores mantidos, pois refletem a assimetria natural dos gastos)
4. **Análise exploratória (EDA)**
   - Distribuição do alvo, variáveis numéricas e categóricas vs. alvo
   - Testes estatísticos: **qui-quadrado + V de Cramér** (categóricas) e **Mann-Whitney + correlação bisserial por postos** (numéricas)
   - Correlação de Spearman
5. **Preparação para modelagem:** `Pipeline` com `ColumnTransformer` (padronização, `log(1 + x)` nos gastos e *one-hot encoding*), divisão treino/teste 80/20 estratificada
6. **Análise preditiva:** baseline, regressão logística e árvore de decisão, com validação cruzada estratificada (5 folds)
7. **Interpretação e conclusões:** *odds ratios*, importância das variáveis e leitura da árvore

## 📊 Principais resultados

| Modelo | Acurácia (teste) | AUC-ROC |
|---|---|---|
| Baseline (classe majoritária) | 0,504 | n/d |
| Regressão Logística | 0,777 | 0,860 |
| Árvore de Decisão (profundidade 8) | 0,779 | 0,873 |

**Destaques:**

- Os dois modelos superam o baseline com folga, e a diferença de acurácia entre eles é desprezível na prática.
- `CryoSleep` e `NoSpend` são as variáveis mais associadas ao alvo (V de Cramér de cerca de 0,46 e 0,48). Passageiros em criossono têm taxa de transporte de cerca de 0,82, contra cerca de 0,33 para os demais.
- Gastos altos estão associados a **menor** chance de transporte, e `Deck` e `HomePlanet` também carregam sinal (decks B e C e origem em Europa têm taxas acima da média).
- `CryoSleep`, `NoSpend` e `TotalSpend` são muito redundantes entre si, o que exige cautela ao interpretar os coeficientes da regressão isoladamente.

## ▶️ Como executar

O arquivo `analise_spaceship_titanic.py` é a exportação de um notebook do Colab e contém comandos de shell (`!pip`, `!kaggle`, `!unzip`). Por isso, ele deve ser executado em um **notebook (Google Colab ou Jupyter)**, e não diretamente com `python`.

### Opção 1: Google Colab (recomendada)

1. Clique no botão **Abrir no Colab** no topo deste README, ou envie o código para um notebook novo.
2. Siga os pré-requisitos do Kaggle abaixo.
3. Execute as células em ordem (*Runtime → Run all*).

### Opção 2: Jupyter local

```bash
git clone https://github.com/<seu-usuario>/<seu-repositorio>.git
cd <seu-repositorio>
pip install numpy pandas matplotlib seaborn scipy scikit-learn kaggle jupyter
jupyter notebook
```

Depois, cole o conteúdo de `analise_spaceship_titanic.py` em um notebook e execute as células.

### Pré-requisitos do Kaggle

A base é baixada automaticamente pela API do Kaggle. Antes de rodar:

1. **Gere um token da API:** Kaggle → *Settings* → *API* → *Create New Token*. O token aparece como um texto para copiar (algo como `KGAT_...`); não é preciso baixar arquivo.
2. **Aceite as regras da competição** na [página do Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) (*Join Competition* / *I Understand and Accept*). Sem isso, o download retorna erro 403.

Ao executar a célula de configuração, o token é solicitado por `getpass` (fica oculto) e usado apenas na sessão, **sem ser salvo no código**.

## 🧰 Tecnologias

- **Python**
- **pandas** e **NumPy** para manipulação de dados
- **Matplotlib** e **seaborn** para visualização
- **SciPy** para testes estatísticos
- **scikit-learn** para pré-processamento, modelagem e métricas

## 📁 Estrutura

```
.
├── analise_spaceship_titanic.py   # Código completo da análise (exportado do Colab)
└── README.md
```

> A pasta `dados/` com o `train.csv` é criada automaticamente durante a execução.

## ⚠️ Limitações

- A imputação por mediana/moda foi calculada com a base inteira antes da divisão treino/teste. Como há poucos ausentes (no máximo 2,5% por coluna), o efeito esperado é pequeno, mas o ideal seria imputar dentro do `Pipeline`.
- Foi usado apenas o `train.csv`; o desempenho foi medido em um teste interno de 20%, não no teste oficial da competição.
- A multicolinearidade entre `CryoSleep`, `NoSpend` e `TotalSpend` não foi tratada.
- Os modelos são deliberadamente simples, alinhados ao nível da disciplina.

## 🔭 Próximos passos

- Remover variáveis redundantes e comparar com o modelo atual
- Testar Random Forest e Gradient Boosting
- Ajustar hiperparâmetros com `GridSearchCV`
- Criar variáveis de família a partir do sobrenome

## 👤 Autor

**João Ricardo Gomes Ferreira**: estudante de Engenharia da Computação na Universidade Presbiteriana Mackenzie.

## 📄 Licença

Projeto acadêmico. A base de dados pertence ao Kaggle e está sujeita às regras da competição.
