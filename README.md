# Machine learning models for predicting work-related sickness absence due to mental disorders using national surveillance data in Brazil

This repository contains the data, scripts, and reproducible analytical workflow for the article:

**Machine learning models for predicting work-related sickness absence due to mental disorders using national surveillance data in Brazil.**

## Repository structure

```
├── dados_Beatriz_TCC_2025.xlsx         # Raw dataset used in the analyses
├── Script_Leitura_Pre_Processamento.R  # Data import, cleaning, and preprocessing
├── Script_Arvore_Decisao.R             # Decision Tree model
├── Script_Random_Forest.R              # Random Forest model
├── Script_XGBoost.R                    # XGBoost model
└── README.md
```

## Study overview

This study evaluates the predictive performance of three supervised machine learning algorithms—**Decision Tree**, **Random Forest**, and **XGBoost**—for identifying work-related sickness absence due to mental disorders using data from Brazil’s national surveillance system (SINAN, 2006–2024).

Key contributions include:

* Demonstration of feasibility of ML techniques in large national administrative surveillance systems.
* Evidence that structural and service-related factors are the strongest predictors of sickness absence.
* Strengthening of occupational health surveillance in low- and middle-income country contexts.

## Reproducibility

### 1. Requirements

Recommended R packages:

```
tidyverse
caret
rpart
randomForest
xgboost
janitor
readxl
Metrics
```

Install with:

```r
install.packages(c("tidyverse", "caret", "rpart", "randomForest",
                   "xgboost", "janitor", "readxl", "Metrics"))
```

### 2. Workflow

1. **Script_Leitura_Pre_Processamento.R** – imports, cleans, preprocesses, splits data.
2. **Script_Arvore_Decisao.R** – trains and evaluates Decision Tree model.
3. **Script_Random_Forest.R** – trains and evaluates Random Forest model.
4. **Script_XGBoost.R** – trains and evaluates XGBoost with tuning.

## Data

`dados_Beatriz_TCC_2025.xlsx` contains the anonymized raw dataset for all analyses.

## Machine Learning models

Each script includes:

* Data loading
* Preprocessing
* Model training
* Hyperparameter tuning
* Performance metrics
* Variable importance analysis

## Contact

**Letícia Martins Raposo, D.Sc.**
Federal University of the State of Rio de Janeiro
[leticia.raposo@uniriotec.br](mailto:leticia.raposo@uniriotec.br)
