import pandas as pd
from sklearn.preprocessing import LabelEncoder


def main(df_):
    df = df_.copy()

    # fill na
    df = df.fillna('unknown')

    # drop FLAG_MOBIL
    df = df.drop(['FLAG_MOBIL'], axis=1)

    # one-hot
    one_hot_list = ['income_type', 'edu_type', 'family_type', 'occyp_type', 'house_type']
    for col in one_hot_list:
        dummy = pd.get_dummies(df[col], prefix=col)
        df = df.drop([col], axis=1)
        df = pd.concat([df, dummy], axis=1)

    # label
    df['gender'] = LabelEncoder().fit_transform(df['gender'])
    df['car'] = LabelEncoder().fit_transform(df['car'])
    df['reality'] = LabelEncoder().fit_transform(df['reality'])

    return df
