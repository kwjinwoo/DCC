import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


def basic_preprocess(df_, num_columns=None, label_columns=None, oh_columns=None, train=True,
                     scaler=None):
    df = df_.copy()

    if train:
        scaler = StandardScaler()
        num_df = df[num_columns]
        num_sc = scaler.fit_transform(num_df)
        num_sc = pd.DataFrame(num_sc, columns=num_columns)

        

