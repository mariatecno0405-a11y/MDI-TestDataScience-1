# TEST 1 - DATA SCIENCE - CLASSIFICATION
**Author:** María Donoso  
**Dataset:** Becker, B. & Kohavi, R. (1996). *Adult* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20

## Objective

Build a fully reproducible pipeline to:
1. Realise an Exploratory Data Analysis (EDA) on UCI Adult Income dataset.
2. Train, backtest and compare different classification models.
3. Choose the best model along metrics.
4. Show results and conclusions.

## Reproducibility
- **Python**: 3.11+  
- Key packages: `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `xgboost`, `mlflow`
### Option A — venv + pip
```bash
python -m venv .venv
# Linux/Mac
source .venv/bin/activate
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

pip install -U pip
pip install -r requirements.txt
```

### Option B — conda (optional)
```bash
conda env create -f test1-ds.yml
conda activate test1-ds
```

## Repository Layout
```rb
MDI-TestDataScience-1/           
├── README.md
├── requirements.txt 
├── test1-ds.yml        # (optional - conda)
├── notebooks/
│   ├── 01_EDA.ipynb               # Data loading + cleaning + EDA
│   ├── 02_modeling.ipynb          # Data modeling + metrics + conclusions
├── data/
│   ├── raw/                       # raw CSV placed here by user if needed
│   ├── processed/                
├── models/
│   └── trained_model.joblib
├── src/                            # example structure (not valid to reproduce this repository)
│   ├── __init__.py
│   ├── data/
│   │   ├── load_data.py             
│   ├── features/
│   │   ├── preprocess.py
│   ├── models/
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── utils.py
│   └── evaluation/
│       └── evaluate.py
├── notebooks_html/                 # notebooks on HTML
│   └── 01_EDA.html  
├── tests/
│   └── test_data.py
├── mlflow/ or runs/
├── deployment/
│   ├── app.py            # FastAPI serving endpoints
│   ├── Dockerfile
│   └── requirements-deploy.txt   
└── reports/
    ├── report.tex     
```
## How To Run

1) Execute `01_EDA.ipynb` end-to-end → generates `data/processed/adult.parquet`.  
2) Execute `02_modeling.ipynb` end-to-end → trains models, selects the best one, calculates metrics, and plots evaluation.  

To export the HTML notebooks:
```bash
jupyter nbconvert --to html 01_EDA.ipynb 02_modeling.ipynb
```

## Data Ingestion

The selected dataset is open source and free, and comes from the University of California Irvine machine learning repository (Becker, B. & Kohavi, R. (1996). Adult [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20). It contains census-like information about different adults, such as their age, workclass, occupation or capital-gain and loss. Our main goal is to predict whether annual income of an individual exceeds $50K/yr. The number of features is 14, being one of them our target value ("income").

The EDA notebook loads the dataset from `data/raw/adult.csv` or from UCI (note that some environments may block outbound connections). **If offline**, download the CSV from the UCI repository.



**Attribute Documentation** 
- `age`: integer.
- `workclass`: categorical (Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked).
- `fnlwgt`: integer.
- `education`: categorical (Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool).
- `education-num`: integer.
- `marital-status`: categorical ( Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse).
- `occupation`: categorical (Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces).
- `relationship`: categorical (Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried).
- `race`: categorical (White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black).
- `sex`: binary (Female, Male).
- `capital-gain`: integer.
- `capital-loss`: integer.
- `hours-per-week`: integer.
- `native-country`: categorical (United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands)
- `income`: binary (50K, <=50K)

## Exploratory Data Analysis (notebook `01_EDA.ipynb`)

- Loads Adult from `data/raw/adult.csv` or from UCI URL.  
- Cleans: strip whitespace, replace `'?'` with `NaN`, cast numeric/categorical types, unify target to `{0,1}`.  
- Splits into **train/valid/test** (60/20/20) with **stratification**.  
- EDA visuals: class balance, numeric histograms, categorical cardinality.  
- Saves splits to `data/processed/{train,valid,test}.parquet`.

## Modeling (notebook `02_modeling.ipynb`)

- Preprocessing with **ColumnTransformer**:
  - Numeric: `SimpleImputer(median)` + `StandardScaler`  
  - Categorical: `SimpleImputer(most_frequent)` + `OneHotEncoder(handle_unknown='ignore')`
- Models:
  - `LogisticRegression` (with/without `class_weight='balanced'`)
  - `RandomForestClassifier`
  - `HistGradientBoostingClassifier`
  - *(optional)* XGBoost/LightGBM if available
- CV: `StratifiedKFold(n_splits=5)`; **scores:** Accuracy, Precision, Recall, F1, ROC-AUC, **PR-AUC**.  
- **Model selection:** best average **PR-AUC**.  
- **Test evaluation:** confusion matrix + ROC + PR curves.  
- **Persist:** `models/trained_model.joblib` + `models/model_card.json`.  
- **Inference example** with a single row.

## Metrics and Conclusions

- **Imbalanced** dataset → **PR-AUC** is the primary selection metric.  
- Report full table of CV metrics and final **test** metrics.  
- Provide model interpretation if needed (e.g., permutation importances on the pipeline).
Accuracy is one metric for evaluating classification models. Informally, accuracy is the fraction of predictions our model got right. Formally, accuracy has the following definition:![Accuracy](https://github.com/Ansu-John/Classification-Models/blob/main/resources/Accuracy1.png)

For binary classification, accuracy can also be calculated in terms of positives and negatives as follows:![Accuracy](https://github.com/Ansu-John/Classification-Models/blob/main/resources/Accuracy2.png)

Where TP = True Positives, TN = True Negatives, FP = False Positives, and FN = False Negatives.

Accuracy alone doesn't tell the full story when you're working with a **class-imbalanced data set**, where there is a significant disparity between the number of positive and negative labels. Metrics for evaluating class-imbalanced problems are precision and recall.

### Confusion matrix

A confusion matrix is a tool for summarizing the performance of a classification algorithm. A confusion matrix will give us a clear picture of classification model performance and the types of errors produced by the model. It gives us a summary of correct and incorrect predictions broken down by each category. The summary is represented in a tabular form.

Four types of outcomes are possible while evaluating a classification model performance. These four outcomes are described below:-

+ **True Positives (TP)** – True Positives occur when we predict an observation belongs to a certain class and the observation actually belongs to that class.
+ **True Negatives (TN)** – True Negatives occur when we predict an observation does not belong to a certain class and the observation actually does not belong to that class.
+ **False Positives (FP)** – False Positives occur when we predict an observation belongs to a certain class but the observation actually does not belong to that class. This type of error is called Type I error.
+ **False Negatives (FN)** – False Negatives occur when we predict an observation does not belong to a certain class but the observation actually belongs to that class. This is a very serious error and it is called Type II error.

![Error Types](https://github.com/Ansu-John/Classification-Models/blob/main/resources/errorTypes.png)

These four outcomes are summarized in a confusion matrix given below.

![ConfusionMatrix](https://github.com/Ansu-John/Classification-Models/blob/main/resources/confusionMatrix.png)

### Classification Report
Classification report is another way to evaluate the classification model performance. It displays the precision, recall, f1 and support scores for the model. 

![ClassificationReport](https://github.com/Ansu-John/Classification-Models/blob/main/resources/ClassificationReport.png)

#### Precision
Precision can be defined as the percentage of correctly predicted positive outcomes out of all the predicted positive outcomes. It can be given as the ratio of true positives (TP) to the sum of true and false positives (TP + FP).

So, Precision identifies the proportion of correctly predicted positive outcome. It is more concerned with the positive class than the negative class.

Mathematically, precision can be defined as the ratio of TP to (TP + FP).

#### Recall
Recall can be defined as the percentage of correctly predicted positive outcomes out of all the actual positive outcomes. It can be given as the ratio of true positives (TP) to the sum of true positives and false negatives (TP + FN). Recall is also called Sensitivity.

Recall identifies the proportion of correctly predicted actual positives.

Mathematically, recall can be given as the ratio of TP to (TP + FN). True Positive Rate is synonymous with Recall and can be given as the ratio of TP to (TP + FN).

#### f1-score
f1-score is the weighted harmonic mean of precision and recall. The best possible f1-score would be 1.0 and the worst would be 0.0. f1-score is the harmonic mean of precision and recall. So, f1-score is always lower than accuracy measures as they embed precision and recall into their computation. The weighted average of f1-score should be used to compare classifier models, not global accuracy.

### Receiver Operating Characteristics (ROC) Curve
Another tool to measure the classification model performance visually is ROC Curve. ROC Curve stands for Receiver Operating Characteristic Curve. An ROC Curve is a plot which shows the performance of a classification model at various classification threshold levels.

The ROC Curve plots the True Positive Rate (TPR) against the False Positive Rate (FPR) at various threshold levels. True Positive Rate (TPR) is also called Recall. It is defined as the ratio of TP to (TP + FN). False Positive Rate (FPR) is defined as the ratio of FP to (FP + TN).

**ROC AUC** stands for Receiver Operating Characteristic - Area Under Curve. It is a technique to compare classifier performance. In this technique, we measure the area under the curve (AUC). A perfect classifier will have a ROC AUC equal to 1, whereas a purely random classifier will have a ROC AUC equal to 0.5.

So, ROC AUC is the percentage of the ROC plot that is underneath the curve.

![ROC](https://github.com/Ansu-John/Classification-Models/blob/main/resources/ROC.png)

