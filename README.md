# TEST 1 - DATA SCIENCE - CLASSIFICATION
**Author:** María Donoso  
**Dataset:** Becker, B. & Kohavi, R. (1996). *Adult* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20

## Objective
Build and evaluate several **classification** models (with clean EDA, proper preprocessing, and fair baselines) and justify the chosen model using appropriate **metrics** and **plots**.

**Target:** predict whether annual income exceeds **$50K/yr** from census features.
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
This layout shows what a production repository could look like (modules under src/, CI, Docker, etc.). For this TEST 1, you only need to run the notebooks under notebooks/.
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
├── src/                            # src code directory (example only; not required to run for this test)
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
├── docs/                 # notebooks on HTML
│   ├── 01_EDA.html             
│   ├── 02_modeling.html 
├── tests/
│   └── test_data.py
├── mlflow/ or runs/      # run `mlflow ui` command-line to visualize tracking server with all experiments
├── deployment/           # FastAPI serving endpoints (example only; not required to run for this test)
│   ├── app.py            
│   ├── Dockerfile
│   └── requirements-deploy.txt     
```
## How To Run

1) Execute `01_EDA.ipynb` end-to-end → generates `data/processed/adult.parquet`.  
2) Execute `02_modeling.ipynb` end-to-end → trains models, selects the best one, calculates metrics, and plots evaluation.  

To export the HTML notebooks:
```bash
jupyter nbconvert --output-dir "../docs/" --to html 01_EDA.ipynb 02_modeling.ipynb
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

- Loads data from `data/raw/adult.csv` or from UCI URL.  
- Cleans: strip whitespace, replace `'?'` with `NaN`, cast numeric/categorical types, unify target to `{0,1}`.    
- EDA visuals: class balance, numeric histograms, categorical cardinality.  
- Saves splits to `data/processed/adult.parquet`.

## Modeling (notebook `02_modeling.ipynb`)
- Splits data into **train/test** (80/20) with **stratification**.
- Preprocessing with **ColumnTransformer**:
  - Numeric: `SimpleImputer(median)` + `StandardScaler`  
  - Categorical: `SimpleImputer(most_frequent)` + `OneHotEncoder(handle_unknown='ignore')`
- Models:
  - `LogisticRegression`
  - `RandomForestClassifier`
  - `XGBoost`
- CV: `StratifiedKFold(n_splits=5)`; **scores:** Accuracy, Precision, Recall, F1, ROC-AUC, **PR-AUC**.  
- **Model selection:** best average **PR-AUC**.  
- **Test evaluation:** confusion matrix + ROC + PR curves.  
- **Output:** `models/trained_model.joblib`.
- **Inference example** with a single adult data.

## Performance Metrics
Once our supervised learning model have been trained, it is very important to analyze **performance metrics**, as they are essential to evaluate the accuracy and efficiency of our model. First of all, we need to understand the following important concepts: 
- **True Positives (TP):** correctly predicted positive cases. 
- **False Positives (FP):** incorrectly predicted positive cases. 
- **False Negatives (FN):** incorrectly predicted negative cases. 
- **True Negatives (TN):** correctly predicted negative cases.

Classification problems aim to predict discrete categories. To evaluate the performance of classification models, we use the following metrics:
- **Accuracy:** it indicates the proportion of correct predictions made by the model out of all predictions.

**Accuracy** alone doesn't tell the full story when you're working with a **class-imbalanced data set**, where there is a significant disparity between the number of positive and negative labels. It gives a False Positive sense of achieving high accuracy. Metrics for evaluating class-imbalanced problems are **precision** and **recall**:

- **Precision:** it calculates the proportion of detected positives that are actually correct, measuring the model’s ability to avoid false positives. It is defined by: $$Precision = \frac{TP}{TP + FP}$$
- **Recall (or Sensitivity):** it calculates the proportion of true positives among all actual positives. It is defined by: $$Precision = \frac{TP}{TP + FN}$$ 

A **confusion matrix** is a tool for summarizing the performance of a classification algorithm. It will give us a clear picture of classification model performance and the types of errors produced by the model. It gives us a summary of correct and incorrect predictions broken down by each category. **Classification report** is another way to evaluate the classification model performance. It displays the precision, recall, f1 and support scores for the model. 
- **F1 score:** it is the harmonic mean of **precision** and **recall**. It is useful when we need a balance between precision and recall as it combines both into a single number. A high F1 score means the model performs well on both metrics. Its range is [0,1] and it is defined by: $$ F1= 2\cdot \frac{Precision \cdot Recall}{Precision + Recall}$$ 

The last, but not least, metric we are going to consider is **Area Under Curve (AUC) and ROC Curve.** The AUC value represents the probability that the model will rank a randomly chosen positive example higher than a randomly chosen negative example. AUC ranges from 0 to 1 with higher values showing better model performance. 
- **True Positive Rate (TPR):** it is equal to **recall**.
- **True Negative Rate (TNR):** also called specificity, it measures how many actual negative instances were correctly identified by the model. It is defined by: $$TNR = \frac{TN}{TN + FP}$$
- **False Positive Rate (FPR):** it measures how many actual negative instances were incorrectly classified as positive. It’s a key metric when the cost of false positives is high such as in fraud detection. It is defined by: $$FPR = \frac{FP}{TN + FP}$$
- **False Negative Rate (FNR):** it measures how many actual positive instances were incorrectly classified as negative. It is defined by: $$FNR = \frac{FN}{TN + FP}$$

The **ROC Curve** is a graphical representation of the **True Positive Rate (TPR)** vs the **False Positive Rate (FPR)** at different classification thresholds. The curve helps us visualize the trade-offs between sensitivity (TPR) and specificity (1 - FPR) across various thresholds. **Area Under Curve (AUC)** quantifies the overall ability of the model to distinguish between positive and negative classes.

- **AUC = 1:** Perfect model (always correctly classifies positives and negatives).
- **AUC = 0.5:** Model performs no better than random guessing.
- **AUC < 0.5:** Model performs worse than random guessing (showing that the model is inverted).


