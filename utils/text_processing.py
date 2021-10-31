"""
자연어 처리시 사용할 수 있는 함수
made by jinwoo
"""
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import re


# 문장 전처리 함수
# 정규표현식으로 전처리 함
# default로 숫자, 한글, 영어 제외 모두 제거
# pandas의 apply 함수와 함께 사용해도 좋을 듯
# x : 전처리할 문장
# reg : 정규표현식
def reg_preprocessing(x, reg=r'[^\d가-힣a-zA-Z ]'):
    x = re.sub(reg, '', x)

    return x


# 문장을 토큰화 한 후 index로 변경
# 문장을 sequence로 변경
# text : 토큰화 할 text(train set)
# max_len : 훈련 시킬 문장의 최대 길이
# oov : 처음 보는 단어에 대한 token. fasttext는 필요 x
# num_words : 사용할 단어의 최소 빈도 수. model fit 할 때의 수랑 맞춰야 함
# pad_option : padding 및 trunc 때 앞에서 부터 할지 뒤에서 부터 할지
# 필요 module
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
def make_token_and_tokenizer(text, max_len=100, oov=None, num_words=None, pad_option='post'):
    tokenizer = Tokenizer(num_words=num_words, oov_token=oov)
    tokenizer.fit_on_texts(text)
    token_index = tokenizer.texts_to_sequences(text)
    vocab_size = len(tokenizer.word_index) + 1
    token_index = pad_sequences(token_index, maxlen=max_len, padding=pad_option, truncating=pad_option)

    return token_index, vocab_size, tokenizer


# tf Embedding layer에 weight로 사용할 matrix를 만드는 함수
# word_dict : 단어와 인덱스 번호. tokenizer.word_index. 0은 padding
# model : embedding 모델(fasttext를 생각하고 만듦)
# vocab_size : make_token_and_tokenizer에서 얻어지는 vocab_size.
# embedding_size : 임베딩 차원. fasttext는 100이 default
# 필요 module
# 없음. 모델만 잘 넣어주면 됨
def make_embedding_matrix(word_dict, model, vocab_size, embedding_size=100):
    embedding_matrix = np.zeros((vocab_size, embedding_size))

    for word, index in word_dict.items():
        embedding_matrix[index] = model.wv[word]

    return embedding_matrix
