import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder


# 기본적인 전처리
# numeric 변수에는 standard scaling 적용, label 변수에는 label encoding, category 변수에는 one-hot encoding
# train 모드시, scaler와 encoder를 함께 반환
# test 모드시, output df만 반환. scaler와 encoder가 반드시 함께 전달되어야 함
# df_ : 변환할 df
# num_columns : numeric 변수의 이름을 담은 리스트
# label_columns : label encoding을 할 변수의 이름을 담은 리스트
# oh_columns : one-hot encoding을 할 변수의 이름을 담은 리스트
# train : 훈련 모드 설정. True면 훈련모드, False면 test 모드
# scaler : test 모드 시 필요한 scaler
# oh_encoder : test 모드 시 필요한 one-hot scaler
# 필요 모듈
# import pandas as pd
# from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
# 주의사항 !!!!!
# 변수로 전달된 column들을 가지고만 전처리 진행 --> 전달된 column들로만 구성된 df를 반환
def basic_preprocess(df_, num_columns=None, label_columns=None, oh_columns=None, train=True,
                     scaler=None, oh_encoder=None):
    df = df_.copy()
    final_columns = []  # 최종 df column 이름 리스트
    
    # train mode
    if train:
        # scaling
        scaler = StandardScaler()   # scaler 선언
        num_df = df[num_columns]
        num_sc = scaler.fit_transform(num_df)   # scaling
        num_sc = pd.DataFrame(num_sc, columns=num_columns)  # df로 변환
        final_columns.extend(num_columns)
        
        # label encoding
        enc = LabelEncoder()    # label encoder 선언
        label_df = df[label_columns]
        # 각 변수에 대해 label encoding
        for col in label_columns:
            label_df[col] = enc.fit_transform(label_df[col])
        final_columns.extend(label_columns)
        
        # one hot encoding
        oh_encoder = OneHotEncoder(sparse=False)    # one-hot encoder 선언. sparse matrix 생성 x
        oh_df = df[oh_columns]
        oh_df = oh_encoder.fit_transform(oh_df)
        
        # one-hot 변환 후 이름 생성
        names = oh_encoder.categories_  # category들
        columns = []
        for pre, name in zip(oh_columns, names):
            for i in range(len(name)):
                temp = pre + '_' + name[i]  # 각 category에 원래 column 이름을 prefix로 붙여줌
                columns.append(temp)
        oh_df = pd.DataFrame(oh_df, columns=columns)    # df로 만들기
        final_columns.extend(columns)
        
        # 최종 df
        final_df = pd.concat([num_sc, label_df, oh_df], axis=1, ignore_index=True)
        final_df.columns = final_columns
        return final_df, scaler, oh_encoder
    # test 모드
    else:
        num_df = df[num_columns]
        num_sc = scaler.transform(num_df)   # 전달 받은 scaler로 변환
        num_sc = pd.DataFrame(num_sc, columns=num_columns)
        final_columns.extend(num_columns)

        enc = LabelEncoder()
        label_df = df[label_columns]
        for col in label_columns:
            label_df[col] = enc.fit_transform(label_df[col])
        final_columns.extend(label_columns)

        oh_df = df[oh_columns]
        oh_df = oh_encoder.transform(oh_df) # 전달 받은 encoder로 변환

        names = oh_encoder.categories_
        columns = []
        for pre, name in zip(oh_columns, names):
            for i in range(len(name)):
                temp = pre + '_' + name[i]
                columns.append(temp)
        oh_df = pd.DataFrame(oh_df, columns=columns)
        final_columns.extend(columns)

        final_df = pd.concat([num_sc, label_df, oh_df], axis=1, ignore_index=True)
        final_df.columns = final_columns
        return final_df
    