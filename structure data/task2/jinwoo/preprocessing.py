import pandas as pd


def main(df_):
    df = df_.copy()

    # drop FLAG_MOBIL
    df = df.drop(['FLAG_MOBIL'], axis=1)

    # family_size 를 이산 변수로 변환
    mask = df['family_size'] >= 5
    df.loc[mask, 'family_size'] = 5
    df['family_size'] = df['family_size'].astype('int')

    return df
