import pandas as pd
import tensorflow as tf
import numpy as np


def build_model(learning_rate, feature_layer) -> tf.keras.models.Sequential:
    model = tf.keras.models.Sequential()

    model.add(feature_layer)

    model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))

    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
                  loss="mean_squared_error",
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model


def train_model(model, dataset, epochs: int, batch_size: int, label_name: str):
    features = {name: np.array(value) for name, value in dataset.items()}

    label = np.array(features.pop(label_name))
    history = model.fit(x=features, y=label, batch_size=batch_size,
                        epochs=epochs, shuffle=True)

    epochs = history.epoch

    hist = pd.DataFrame(history.history)
    rmse = hist["root_mean_squared_error"]

    return epochs, rmse


def predict(model, label_name: str, feature: str, dataset, n: int = 10):
    # Since our feature layer includes feature names, it is important we pass those in during prediction.
    print(dataset[:n])
    if feature is not None:
        print(dataset[:n][feature])
    x = dict(dataset[:n])
    # print("Dictionary: " + str(x))
    predicted = model.predict_on_batch(x)

    predicted_simple = []
    for pred in predicted:
        predicted_simple.append(pred[0])

    print(predicted_simple)

    output_dict = {}
    if feature is not None:
        output_dict[feature + " (feature)"] = list(dataset[feature][:n].values)

    output_dict[label_name + " (label)"] = list(dataset[label_name][:n].values)
    output_dict["predicted"] = predicted_simple

    predict_df = pd.DataFrame.from_dict(output_dict)
    print(predict_df)
