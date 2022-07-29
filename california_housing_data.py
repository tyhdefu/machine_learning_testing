import pandas as pd


def get_training_data() -> pd.DataFrame:
    return pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")


def get_test_data() -> pd.DataFrame:
    return pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv")