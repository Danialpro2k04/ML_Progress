import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def preprocess_data(path, target_column):
    df = pd.read_csv(path)

    df = df.drop_duplicates()

    df.dropna(inplace=True)

    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    X_train = train_df.drop(target_column, axis=1)
    y_train = train_df[target_column]

    X_test = test_df.drop(target_column, axis=1)
    y_test = test_df[target_column]

    return X_train, y_train, X_test, y_test
