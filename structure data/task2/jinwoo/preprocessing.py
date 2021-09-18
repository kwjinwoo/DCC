import pandas as pd


def main(df_):
    df = df_.copy()

    # fill na
    df = df.fillna('unknown')

    # drop FLAG_MOBIL
    df = df.drop(['FLAG_MOBIL'], axis=1)

    # one-hot
    df =
    return df
