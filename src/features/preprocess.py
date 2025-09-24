import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def build_preprocessor(categorical: list, numerical: list):
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])
    preprocessor = ColumnTransformer([
        ("categorical", cat_transformer, categorical),
        ("numerical", num_transformer, numerical)
    ])
    return preprocessor
