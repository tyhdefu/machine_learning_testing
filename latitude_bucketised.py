import california_housing_data
import model_helper

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

if __name__ == '__main__':
    train_df = california_housing_data.get_training_data()
    test_df = california_housing_data.get_test_data()

    train_df["median_house_value"] /= 1000
    test_df["median_house_value"] /= 1000

    feature_name="latitude"
    latitude_numeric_column = tf.feature_column.numeric_column(feature_name)

    latitude_boundaries = list(np.arange(min(train_df[feature_name]), max(train_df[feature_name]), 1.0))
    print(latitude_boundaries)

    latitude_bucketised = tf.feature_column.bucketized_column(latitude_numeric_column, latitude_boundaries)

    latitude_bucketised_feature = tf.feature_column.indicator_column(latitude_bucketised)
    feature_columns = [latitude_bucketised_feature]

    feature_layer = layers.DenseFeatures(feature_columns)

    learning_rate = 0.01
    model = model_helper.build_model(learning_rate=learning_rate, feature_layer=feature_layer)

    target_label = "median_house_value"
    epochs = 30
    batch_size = 10
    epochs, rmse = model_helper.train_model(model=model, dataset=train_df, label_name=target_label,
                                            epochs=epochs, batch_size=batch_size)

    print("Evaluating model")
    test_features = {name: np.array(value) for name, value in test_df.items()}
    test_label = np.array(test_features.pop(target_label))
    print(model.evaluate(x=test_features, y=test_label, batch_size=batch_size))

    model_helper.predict(model, feature=None, label_name=target_label, dataset=test_df, n=20)