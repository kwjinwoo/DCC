"""
이미지 처리 문제를 해결할 때 사용 할 수 있는 함수들
이미지 처리는 tensorflow를 통해 처리한다고 가정하고 만듦
made by jinwoo
"""
import tensorflow as tf
from tensorflow import keras


# augemtation layer를 만들어 주는 함수
# rotation_factor : rotation 비율
# contrast_factor : contrast 비율 ( 색 대비 )
# seed : 랜덤 시드
# 훈련시에만 활성화 되고 모델 평가에는 활성화 되지 않음
# 모델 summary를 하려면 input을 지정해 주어야 함
# 필요 module
# import tensorflow as tf
# from tensorflow import keras
def make_aug_layer(rotation_factor=0.2, contrast_factor=0.4, seed=42):
    augmentation_layer = keras.models.Sequential([
        keras.layers.RandomFlip('horizontal_and_vertical', seed=seed),
        keras.layers.RandomRotation(factor=rotation_factor, seed=seed),
        keras.layers.RandomContrast(factor=contrast_factor, seed=seed)
    ])
    return augmentation_layer


# 이미지 데이터를 resize 하고 scaling 하는 함수
# dataset을 만들때 map에 사용(batch 하기 전에 사용)
# 이미지가 w * h * channel인 형태 기대
# image_size : 변경할 이미지 크기. 튜플로 받음(map 할때 전달 못하니까 사용 할때 마다 default를 바꿔줘야 할듯)
# scaling_factor : scaling 시 나누어 줄 값. 이미지의 max 값을 넣음
# 필요 module
# import tensorflow as tf
# from tensorflow import keras
def resize_and_scaling(x, y, image_size=(32, 32), scaling_factor=255.0):
    x = tf.image.resize(x, image_size)
    x = x / scaling_factor

    return x, y
