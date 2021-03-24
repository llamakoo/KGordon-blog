---
layout: post
title: EfficientNet
date: 2019-11-23
comments: true
categories: [paper-review]
tags: [deeplearning]
---

> In this paper, we systematically study model scaling and identify that carefully **balancing network** depth, width, and resolution can lead to better performance. Based on this observation, we propose **a new scaling method that uniformly scales all dimensions of depth/width/resolution** using a simple yet highly effective **compound coefficient**. 

![Model Scaling](https://drive.google.com/uc?id=1tgBVYooCbdxLcHi6eByFyXQfEGMVPKY7){: width="100%" height="100%"}

- Table of Contents
{:toc .large-only}

## Paper Info
---
> Title : EfficientNet, Rethinking Model Scaling for Convolutional Neural Network

> Paper link : [https://arxiv.org/pdf/1905.11946.pdf](https://arxiv.org/pdf/1905.11946.pdf)

> Author : Mingxing Tan, Quoc V. Le

> Publised year : 23 Nov 2019

> keywords : Model Scaling, Classification

---

<br>
## Introduction
---

> In previous work, it is common to scale only one of the three dimensions – depth, width, and image size. Though it is possible to scale two or three dimensions arbitrarily, arbitrary scaling requires tedious manual tuning and still often yields sub-optimal accuracy and efficiency. ...  In particular, we investigate the central question: is there a principled method to scale up ConvNets that can achieve better accuracy and efficiency?

- 기존 연구에서는 성능을 올리기 위해 depth, width, image size 중 하나만 조절하여 수동으로 튜닝을 하였다. 하지만 이러한 방법으로는 최적값을 찾기 어렵다. 저자들은 <span style="background-color:#BFFF00">"3가지 요소를 적절하게 조절하여 성능을 최적화하는 방법이 없을까?"</span> 라는 질문에서 부터 시작하여 논지를 전개한다. 

> Unlike conventional practice that arbitrary scales these factors, our method uniformly scales network width, depth, and resolution with a set of fixed scaling coefficients. 

- 본 논문에서는 그런 방법의 일환으로 임의의 값이 아닌, 고정된 scaling coefficients로 동일하게 적용할 수 있는 **"Compound scaling method"**를 소개한다.
- ImageNet competition에서 [GPipe(Huang et al.,2018)](https://arxiv.org/abs/1811.06965)는 84.3%의 가장 높은 accuracy를 보였지만, 파라미터가 556M으로 많은 메모리를 차지하게 된다는 단점이 존재한다.
- ImageNet으로 학습된 Classification model은 Object detection과 같은 분야에서 backbone network로 많이 사용된다. Model accuracy 뿐만이 아닌, memory를 차지하는 비중과 inference latency도 중요한 요소이기 때문에 네트워크를 효율적으로 만들 필요성이 있다.

> In this paper, we aim to study model efficiency for super large ConvNets that surpass state-of-the-art accuracy. To achieve this goal, we resort to model scaling.

- Model의 효율성을 높이려면, Model compression을 하거나 [SqueezeNets(Gholami et al.,2018)](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w33/Gholami_SqueezeNext_Hardware-Aware_Neural_CVPR_2018_paper.pdf), [MobileNets(Howard et al.,2017)](https://arxiv.org/abs/1704.04861), [ShuffleNets(Zhang et al.,2018)](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/0642.pdf)과 같은 handcraft model을 사용하곤 하였다. [MnasNet(Tan et al.,2019)](https://arxiv.org/pdf/1807.11626)은 ConvNet의 width, depth, kernel type/size를 조절하여 handcraft model 보다 더 좋은 효율을 보이는 mobile-size 모델이다.
- 하지만 MNasNet의 기법은 더 큰 모델(design space가 넓어 tuning이 어려운 모델)에 적용하기 어렵다는 단점이 있다. 따라서 저자들은 더 큰 모델에도 적용할 수 있는 기법에 대해 연구하였다.

<br>
## Compound Model Scaling
### Problem Formulation
- ConvNet Layer $$i$$의 함수는 $$\mathrm{Y}_i = \mathcal{F}_i(\mathrm{X}_i)$$로 정의된다. ($$\mathrm{Y}_i$$ : output tensor, $$\mathcal{F}_i$$ : operator, $$\mathrm{X}_i$$ : input tensor)
- ConvNet $$\mathcal{N}$$은 layer들 간의 결합으로 표현할 수 있다. ($$\mathcal{N}=\mathcal{F_k}\odot\ldots\odot\mathcal{F_2}\odot\mathcal{F_1}(\mathrm{X_1})=\bigodot_{j=1 \ldots k}\mathcal{F_j}(\mathrm{X_1})$$)
- ConvNet은 여러개의 stage로 나뉘고, 각 stage의 layer들은 일반적으로 동일한 구조를 가진다. 
- 기존의 방법들은 최적 레이어 구조 $$\mathcal{F_i}$$를 찾으려 하였다. 하지만 $$\mathcal{F_i}$$를 고정하면 design space가 줄어들기 때문에 model scaling이 쉬워진다.
- 그럼에도 불구하고 $$\mathrm{L_i}, \mathrm{C_i}, \mathrm{H_i}, \mathrm{W_i}$$를 각각 layer마다 조절하여 최적의 값을 찾는 것은 어려운 문제이다. 따라서 모든 layer를 같은 비율로 줄이는 것을 제약조건으로 정한다.

<p align="center"><img src="https://drive.google.com/uc?id=1qfZoJYrAqOol_1bjyton796H7ORx2T4X" width="75%" height="100%"></p>
<br>
- Model scaling을 위의 식과 같은 <span style="background-color:#BFFF00"> 메모리 공간에 대한 제약조건이 주어질 때, model accuracy를 최대화 하는 최적화 문제</span>로 바꾸어 풀고자 한다.
- $$w,d,r$$은 각각 네트워크의 width, depth, resolution에 곱해지는 coefficients이며, $$\hat{\mathcal{F_i}},\hat{\mathrm{L_i}},\hat{\mathrm{H_i}},\hat{\mathrm{W_i}},\hat{\mathrm{C_i}}$$는 각각 baseline 네트워크의 predefined parameter이다.

### Scaling Dimensions
> The main difficulty of the problem is that the optimal d, w, r depend on each other and the values change under different resource constrain.

- 위 최적화 문제에서 가장 큰 어려움은 각각의 $$w,d,r$$이 의존적인 값이며 매번 다른 메모리 조건에 따라 값들이 변한다는 것이다. 이런 이유로 인해 기존의 방법들은 하나의 값만 조절하는 기법을 사용했다. 3가지의 파라미터를 조정하면 다음과 같은 영향을 미친다.

    1. - **Depth** : 네트워크의 depth가 커질 수록, 더 복잡한 feature를 capture 할 수 있다. 하지만 vanishing graident 문제가 발생한다.
    2. - **Width** : 네트워크의 width가 넓어질 수록, 더 미세한 feature를 capture 할 수 있다. 하지만 depth가 충분히 깊지 않다면, 추상정보(high-level features)를 획득하기 어렵다.
    3. - **Resolution** : 입력 영상의 해상도가 클 수록, 더 정밀한 패턴을 capture 할 수 있다. 

<p align="center"><img src="https://drive.google.com/uc?id=1FnnOo3yfR0FnXrDOA2w3QZxRTjd3BWd9" width="100%" height="100%"></p>

- Figure3에서 처럼 각각 파라미터의 값을 올릴 수록 성능이 올라간다. 하지만 어느정도 올라가면 성능이 수렴하게 된다(=accuracy gain이 적어진다). 

### Compound Scaling

<p align="center"><img src="https://drive.google.com/uc?id=1Q86GB1Q99Y-oNMjkz2g6bWTtI9qkTC-I" width="70%" height="100%"></p>

> In order to pursue better accuracy and efficiency, it is critical to balance all dimensions of network width, depth, and resolution during ConvNet scaling.

- 직관적으로, 입력 이미지의 해상도가 커지면 더 많은 픽셀정보를 담기 위해서 모델의 depth, width도 커져야 한다. 이에 대한 실험 결과는 Figure4와 같다. depth, resolution을 각각 조절하는 것 보다, 두가지 다 조절하는 것이 성능이 더 좋다는 것을 알 수 있다.
- <span style="background-color:#BFFF00">따라서 실험을 통해 알 수 있는 사실은 depth, width, resolution을 적절하게 조합하여 scaling하면 효율적이며 높은 성능을 얻을 수 있다는 것이다.</span>

<p align="center"><img src="https://drive.google.com/uc?id=1bASUuxuC7pzMLZbFRLTYjZeLuhRn1dGn" width="70%" height="100%"></p>

- 저자들은 3가지 파라미터를 원칙에 입각하여 변경할 수 있는, **Compound scaling method**를 제안한다.
- $$\phi$$는 width, depth, resolution에 uniformly하게 곱해지는 계수이며, $$\alpha, \beta, \gamma$$는 grid search를 통해 얻은 상수들이다.
- 직관적으로, $$\phi$$는 모델 확장에 사용할 수 있는 리소스를 제어하는 변수이며, 가용메모리의 용량이 클 수록 값을 올릴 수 있다. $$\alpha, \beta, \gamma$$는 네트워크의 width, depth, resolution의 할당 비중을 나타내는 값이다.
- Convolution 연산이 대부분이기 때문에, FLOPS 또한 convolution에 비례한다. 그리고 convolution 연산은 $$d, w^2, r^2$$에 비례한다. 따라서 FLOPS는 $$(\alpha\cdot\beta^2\cdot\gamma^2)^\phi$$에 비례하게 된다. 저자들은 ($$\alpha\cdot\beta^2\cdot\gamma^2$$)를 2에 근접한 값이 나오도록 설정하여, 총 FLOPS가 $$2^\phi$$가 되도록 하였다.

<br>
## EfficientNet Architecture
---

<p align="center"><img src="https://drive.google.com/uc?id=1JW0aOK1Y1T7J-Oe863kda2k9eoPJF-Fr" width="70%" height="100%"></p>

> Since model scaling does not change layer operators $$\hat{\mathcal{F_i}}$$ in baseline network, having a good baseline network is also critical. We will evaluate our scaling method using existing ConvNets, but in order to better demonstrate the effectiveness of our scaling method, we have also developed a new mobile-size baseline, called EfficientNet.

- 아무리 model scaling을 효과적으로 하더라도, baseline network가 좋지 않으면 성능 향상에 한계가 있다. 따라서 저자들은 EfficientNet-B0라는 새로운 mobile-size network를 설계하였다(Table1).

- EfficientNet-B0는 latency보다 FLOPS를 목표로 최적화하였는데, 이는 특정 하드웨어를 목표로 하는 것이 아니기 때문이다.
- [NAS(Neural architecture search)](https://arxiv.org/abs/1611.01578)를 활용하여 구현한 EfficientNet-B0를 baseline으로 compound scaling을 다음과 같이 수행한다.

    1. - **STEP1** : $$\phi=1$$로 고정후, 메모리 자원이 2배라고 가정할 때 최적의 $$\alpha, \beta, \gamma$$ 값을 수식 2, 3을 기반으로 grid search로 찾는다.
    2. - **STEP2** : $$\alpha, \beta, \gamma$$을 고정하고, $$\phi=1$$을 변화시켜 scale up한다. 

### MBConv
<p align="center">
    <img src="https://drive.google.com/uc?id=1Z6GL7hoBsWzT8EiAAK2LZA-rPWGr4lcu" width="100%" height="100%">
    <em><a href="https://towardsdatascience.com/mobilenetv2-inverted-residuals-and-linear-bottlenecks-8a4362f4ffd5">MBconv block (figure from [MobileNetV2: Inverted Residuals and Linear Bottlenecks]</a></em>
</p>
- MBConv는 MobileNet V2에서 제안된 block이며, efficientNet은 MBConv를 사용한 MNasNet과 비슷한 구조의 모델을 baseline(EfficientNet B0)으로 구축하였다.
- MBConv의 특징은 크게 3가지로 구분할 수 있다. 자세한 설명은 링크 참조[(MobileNetV2: Inverted Residuals and Linear Bottlenecks)](https://towardsdatascience.com/mobilenetv2-inverted-residuals-and-linear-bottlenecks-8a4362f4ffd5) 

    1. - **Depthwise convolution + Pointwise convolution** : 기본 convolution 연산을 2단계로 분리하여 연산량을 줄인다.
    2. - **Inverse residuals** : Residual 블록은 channel을 압축하는 레이어와 확장하는 레이어로 구성된다. 기존 residual 방식에서는 채널이 확장된 레이어 끼리 연결이 되는 반면, MBconv는 적은 채널끼리 skip connection을 형성하게 되어 연산량이 감소된다.
    3. - **Linear bottleneck** : Relu 활성화 함수로 인한 정보 손실을 방지하기 위해, 각 블록의 마지막 계층에서 선형 활성화 함수를 사용한다.

<br>
## Experiments
---
<p align="center"><img src="https://drive.google.com/uc?id=1Own0qoGYxXg5hc8oQpFGhLpZhiSTNa1o" width="70%" height="100%"></p>
- EfficientNet이 다른 네트워크에 비해 훨씬 좋은 성능을 보여주는 것을 알 수 있다.
- 특히 연산량을 줄였음에도 불구하고, accuracy가 더 올라간 것은 compound scaling이 매우 효과적이다라는 것을 보여준다.
- 메모리를 줄이는 것을 타겟으로 하였지만, 모델이 작아지면서 연산량이 줄어 결과적으로 inference latency도 작아지는 결과를 보여주었다(논문 Table 4 참고). 

<p align="center"><img src="https://drive.google.com/uc?id=1vYlpLb9-x1jlx7ryp8IEvzY-zKuOE9Fm" width="100%" height="100%"></p>
- CAM(Class activation map)을 보아도 compound scaling을 하였을 때 타겟 객체에 초점이 더 맞는 것을 보여준다. 

<br>
## Conclusion
---
- Model을 scaling 할 때 width, depth, resolution을 **최적의 비율**로 변화시키면, accuracy와 efficiency 측면에서 좋은 성능의 모델을 생성할 수 있다.