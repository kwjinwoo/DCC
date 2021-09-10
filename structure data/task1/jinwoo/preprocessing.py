import pandas as pd
from sklearn.preprocessing import LabelEncoder


def make_train_set(df_):
    df = df_.copy()

    x_train = df.drop(['voted'], axis=1)
    y_train = df['voted']

    x_train['age_group'] = LabelEncoder().fit_transform(x_train['age_group'])
    x_train['gender'] = LabelEncoder().fit_transform(x_train['gender'])

    race_dummies = pd.get_dummies(x_train['race'], prefix='race_')
    x_train = x_train.drop(['race'], axis=1)
    x_train = pd.concat([x_train, race_dummies], axis=1)

    religion_dummies = pd.get_dummies(x_train['religion'], prefix='religion_')
    x_train = x_train.drop(['religion'], axis=1)
    x_train = pd.concat([x_train, religion_dummies], axis=1)

    return x_train, y_train, x_train.columns



