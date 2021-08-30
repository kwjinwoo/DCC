# CIFAR-10 이미지 분류

## Case1(Default)

* Model : CNN / ResNet34 / EfficientNet-b4
* Hyperparameter : Batch_size = 32 / Epochs = 5
* Optimizer : Adam
* Loss Function : CrossEntropy

### Accuracy(Best)

* CNN : 60.79%
* ResNet34 : 69.73%
* EfficientNet-b4 : 84.54%


### Training Time

* CNN : 1m 15s
* ResNet34 : 4m 29s
* EfficientNet-b4 : 18m 14s


#### Result

* FashionMNIST 데이터의 경우 세 모델간 Accuracy차이가 크지 않았는데 CIFAR-10의 경우 EfficientNet-b4의 성능이 확실히 뛰어남을 확인할 수 있었습니다.
* 비록 시간이 4~18배 정도 차이가 나긴하지만 데이터가 크지 않은 경우 EfficientNet을 사용하는 게 좋을 것 같습니다.

## Case2(Data Augmentation(좌우 반전))

* Model : CNN / ResNet34 / EfficientNet-b4
* Hyperparameter : Batch_size = 32 / Epochs = 5
* Optimizer : Adam
* Loss Function : CrossEntropy
* 데이터를 받아올 때 RandomHorizontalFlip()옵션을 추가해 50% 확률로 좌우 반전 시켰습니다.
* Augmentation데이터에 대해 Nomalize과정을 진행했습니다. 

### Accuracy(Best)

* CNN : 60.79%(Default) -> 65.79%
* ResNet34 : 69.73%(Default) -> 74.60%
* EfficientNet-b4 : 84.54%(Default) -> 84.27%


### Training Time

* CNN : 1m 56s
* ResNet34 : 5m 14s
* EfficientNet-b4 : 19m 6s


#### Result

* 가장 간단한 좌우 반전 Augmentation을 추가해 학습을 진행했는데 모델에 대한 Accuracy가 Default대비 약 5%갸량 올라갔습니다. 
* 다만 EfficientNet의 경우 Accuracy가 다른 모델들에 비해 많이 큰 것은 맞지만 Default대비 유의미한 성능향상으로 이어지지는 못했습니다.

