from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# 문장을 토큰화 한 후 index로 변경
# 문장을 sequence로 변경
# text : 토큰화 할 text(train set)
# max_len : 훈련 시킬 문장의 최대 길이
# oov : 처음 보는 단어에 대한 token
# num_words : 사용할 단어의 최소 빈도 수
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


