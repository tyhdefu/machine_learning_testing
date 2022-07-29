import california_housing_data
import model_helper

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


def bucketise(feature_name: str):
    numeric_column = tf.feature_column.numeric_column(feature_name)

    resolution_in_degrees = 0.4
    boundaries = list(np.arange(min(train_df[feature_name]), max(train_df[feature_name]), resolution_in_degrees))
    return tf.feature_column.bucketized_column(numeric_column, boundaries)


if __name__ == '__main__':
    train_df = california_housing_data.get_training_data()
    test_df = california_housing_data.get_test_data()

    train_df["median_house_value"] /= 1000
    test_df["median_house_value"] /= 1000

    latitude_bucketised = bucketise("latitude")

    longitude_bucketised = bucketise("longitude")

    latitude_x_longitude = tf.feature_column.crossed_column([longitude_bucketised, latitude_bucketised], hash_bucket_size=100)
    crossed_feature = tf.feature_column.indicator_column(latitude_x_longitude)

    # Adding median income improves the model significantly.
    feature_columns = [crossed_feature, tf.feature_column.numeric_column("median_income")]

    feature_layer = layers.DenseFeatures(feature_columns)

    learning_rate = 0.1
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