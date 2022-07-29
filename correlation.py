import pandas as pd

import california_housing_data

if __name__ == '__main__':
    pd.options.display.max_columns = None
    pd.options.display.width = 0
    train_df = california_housing_data.get_training_data()

    train_df["rooms_per_person"] = train_df["total_rooms"] / train_df["population"]
    train_df["bedrooms_per_room"] = train_df["total_bedrooms"] / train_df["total_rooms"]

    print(train_df.corr())