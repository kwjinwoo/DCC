# CIFAR-10 이미지 분류

## Case1

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

