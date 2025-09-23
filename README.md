# TEST 1 - DATA SCIENCE - CLASSIFICATION
Build and evaluate various machine learning classification models using Python.

## GOAL
The main goal of this test is to choose an open source dataset to realise an exploratory analysis (EDA), train and test some of the classification models that best fit.
El objetivo de esta prueba es escoger un dataset abierto para realizar un análisis exploratiorio (EDA), así como entrenar y testear varios de los modelos de clasificación que mejor se ajustan al problema.

## DATASET
In our case, the selected dataset is open source and free, and comes from the University of California Irvine machine learning repository (Becker, B. & Kohavi, R. (1996). Adult [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20). It contains information about different adults, such as their age, workclass, occupation or capital-gain and loss. Our main goal is to predict whether annual income of an individual exceeds $50K/yr based on census data. The number of features is 14, being one of them our target value.


**Variable Information** 
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

#### Dataset import
```rb
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
adult = fetch_ucirepo(id=2) 
  
# data (as pandas dataframes) 
X = adult.data.features 
y = adult.data.targets 
  
# metadata 
print(adult.metadata) 
  
# variable information 
print(adult.variables) 
```


## Instrucciones para reproducir

### Crear entorno
```rb
python -m venv .venv
# activar:
# Linux / Mac
source .venv/bin/activate
# Windows (Powershell)
.\.venv\Scripts\Activate.ps1
# actualizar pip
pip install -U pip
pip install -r requirements.txt
```

```rb 
conda env create -f test1-ds.yml
conda activate test1-ds
```





## Estructura del repo
```rb
MDI-TestDataScience-1/           
├── README.md
├── LICENSE
├── requirements.txt 
├── environment.yml (opcional - conda)
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_results_and_conclusions.ipynb
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── make_dataset.py             # descarga / limpieza
│   ├── features/
│   │   ├── build_features.py
│   ├── models/
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── model_utils.py
│   └── viz/
│       └── plots.py
├── tests/
│   └── test_data.py
├── models/
│   └── trained_model.joblib
├── mlflow/ or runs/
├── deployment/
│   ├── app.py            # FastAPI serving endpoints
│   ├── Dockerfile
│   └── requirements-deploy.txt
├── .github/
│   └── workflows/
│       └── ci.yml
├── notebooks_html/
│   └── 01_EDA.html     
└── report/
    ├── report.tex       # si haces LaTeX/beamer
    └── report.pdf
```






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

# REFERENCE

https://www.geeksforgeeks.org/understanding-logistic-regression/

https://medium.com/swlh/decision-tree-classification-de64fc4d5aac

https://builtin.com/data-science/random-forest-algorithm

https://medium.com/swlh/random-forest-classification-and-its-implementation-d5d840dbead0

https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761

https://www.datacamp.com/community/tutorials/k-nearest-neighbor-classification-scikit-learn

https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python

https://www.datacamp.com/community/tutorials/naive-bayes-scikit-learn

https://developers.google.com/machine-learning/crash-course/classification/accuracy

https://towardsdatascience.com/model-evaluation-techniques-for-classification-models-eac30092c38b
