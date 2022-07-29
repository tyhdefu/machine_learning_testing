import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

import model_helper
import california_housing_data


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 0)

    train_df = california_housing_data.get_training_data()
    test_df = california_housing_data.get_test_data()

    #print(train_df.head(5))

    train_df["median_house_value"] /= 1000
    test_df["median_house_value"] /= 1000

    print("Describe:")
    print(train_df.describe())

    print("Correlation matrix:")
    print(train_df.corr())

    feature_columns = []

    income = tf.feature_column.numeric_column("median_income")

    feature_columns.append(income)
    fp_feature_layer = layers.DenseFeatures(feature_columns)

    learning_rate = 0.01
    model = model_helper.build_model(learning_rate, fp_feature_layer)

    label_name = "median_house_value"
    batch_size = 100
    epochs = 30

    epochs, rmse = model_helper.train_model(model=model, dataset=train_df, epochs=epochs, batch_size=batch_size, label_name=label_name)

    test_features = {name: np.array(value) for name, value in test_df.items()}
    test_label = np.array(test_features.pop(label_name))
    print(model.evaluate(x=test_features, y=test_label, batch_size=batch_size))
    model_helper.predict(model, label_name, feature="median_income", dataset=test_df)

