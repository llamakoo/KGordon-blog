---
layout: post
title: EfficientDet
date: 2020-04-03
comments: true
categories: [paper-review]
tags: [detector]
---

> In this paper, we systematically study neural network architecture design choices for object detection and propose several key optimizations to improve efficiency. First, we propose **a weighted bi-directional feature pyramid network (BiFPN)**, which allows easy and fast multi-scale feature fusion; Second, we propose **a compound scaling method** that uniformly scales the resolution, depth, and width for all backbone, feature network, and box/class prediction networks at the same time.

<p align="center"><img src="https://drive.google.com/uc?id=1DFwNw-nHL0A0Wzw7gCl_OT45Bi25yVFR" width="100%" height="100%"></p>

- Table of Contents
{:toc .large-only}

## Paper Info
---
> Title : EfficientNet, Scalable and Efficient Object Detection

> Paper link : [https://arxiv.org/abs/1911.09070](https://arxiv.org/abs/1911.09070)

> Author : Mingxing Tan, Ruoming Pang, Quoc V. Le

> Publised year : 3 Apr 2020

> keywords : Object Detection

---

<br>
## Introduction
---
- EfficientDet의 저자는 EfficientNet의 저자(Mingxing Tan)와 동일하다.
- EfficientNet은 Image classification에서 뛰어난 성능향상을 보여주었으며, 본 논문에서는 EfficientNet 네트워크 및 기법을 object detection에 적용하여 마찬가지로 탁월한 결과를 보여준다.
- 기존의 Detector들은 accuracy가 높으면, 연산량(FLOPS)이 많았다. 반대로 연산량이 적어 efficiency가 높으면, accuracy가 낮은 경향이 있었다.
- 또한 기존 연구들은 특정 또는 적은 resource에만 초점을 맞추었다. 하지만 실제로 작은 mobile device에서 datacenter까지 다른 특성의 application을 고려하는 경우가 많기 때문에 resource 특성을 고려한 detector를 설계하고자 하였다.

> A natural question is: Is it possible to build a scalable detection architecture with both higher accuracy and better efficiency across a wide spectrum of resource constraints (e.g., from 3B to 300B FLOPs)? 

- 따라서 저자들의 목표는 <span style="background-color:#BFFF00"> "다양한 스펙트럼의 환경(주로 메모리 용량)에서 사용자의 취지에 맞게 조절이 가능하며, 높은 성능과 효율을 지닌 detection 구조 설계"</span> 이다.

### Challenge
- feature fusion에서 많이 사용되는 FPN은 다양한 객체 크기를 잡는데 효과적이다. FPN에서는 feature fusion을 모두 동일한 비중으로 가져가는데, 실제로는 feature들이 다른 해상도를 가지기 때문에 출력에 동등하게 기여하지 않는다. <span style="color:#DF8B00">$$\rightarrow$$ BiFPN</span>
- Backbone 네트워크와 box/class prediction 네트워크를 적절하게 scaling해야 효율적이면서 높은 성능을 이끌어 낼 수 있다.  <span style="color:#DF8B00">$$\rightarrow$$ compound scaling method</span>

<br>
## BiFPN
---
<p align="center"><img src="https://drive.google.com/uc?id=1w5ksW1Pi1PPrVv-UH0oj7RvSk_P2coSM" width="100%" height="100%"></p>

### Problem Formulation
- Object detection은 Classification과 다르게 객체의 크기가 다양하며, 이를 모두 탐지해야 한다는 문제가 있다. 따라서 다양한 해상도의 feature의 정보를 사용해야하며 이들을 효과적으로 fusion하는 것(multi-scale feature fusion)이 중요하다.
- Multi-scale features를 $$\vec{P^{in}}=(\vec{P^{in}_{l_1}},\vec{P^{in}_{l_2}},\dots)$$로 정의할 때, $$\vec{P^{in}_{l_i}}$$는 level $$l_i$$에서의 feature이다.
- <span style="background-color:#BFFF00">목표는 입력 feature를 효과적으로 융합하는 $$f$$를 찾는 것이다.</span> $$\vec{P^{out}}=f(\vec{P^{in}})$$
- $$\vec{P^{in}_{l_i}}$$의 해상도는 입력 영상 해상도의 $$1/2^i$$의 배이다. 예를들면, 입력 영상 해상도가 $$640\times640$$인 경우, level 3의 feature는 $$80\times80 (=640/2^3)$$의 해상도를 갖는다.

### Cross-Scale Connections
<p align="center"><img src="https://drive.google.com/uc?id=1tAv7xlPJoa5QT_EO9c1YetsntexNLGHd" width="70%" height="100%"></p>
- Figure 2에는 기존의 detector들이 feature fusion을 어떻게 행했는지에 대해 나와있다.
- **FPN** : top-down 방식으로 상위 레벨의 feature가 하위 레벨의 feature에 융합되는 형태이며, 단방향이다. 단방향이기 때문에 하위 레밸의 정보가 상위 레벨의 feature로 전달되지 않는 단점이 있다.
- **PANet** : FPN의 단점을 개선하기 위해서 top-down뿐만이 아닌, bottom-up으로도 연결한다.
- **NAS-FPN** : 학습을 통해 최적의 아키텍쳐를 구현한다. 하지만 수천개의 GPU가 요구되며, 불규칙한 구조로 수정하기가 어렵다.
- PANet이 FPN, NAS-FPN에 비하여 accuracy가 높지만 파라미터 수와 연산량이 많다.

- 저자들은 모델의 성능을 향상시키기 위해 다음과 같은 기법을 사용하였다.

1. feature fusion이 일어나지 않는 노드를 제거 <span style="color:#DF6C00">$$\rightarrow$$ 영향이 적은 노드를 제거하여 연산량 개선</span>
2. 동일 레벨에서 입력 노드와 출력 노드를 연결하는 extra edge를 추가 <span style="color:#DF6C00">$$\rightarrow$$ 추가비용을 치르지 않고 더 많은 feature를 fusion하는 효과</span>
> Third, unlike PANet that only has one top-down and one bottom-up path, we treat each bidirectional (top-down & bottom-up) path as one feature network layer, and repeat the same layer multiple times to enable more high-level feature fusion.
3. <span style="background-color:#BFFF00">더 많은 상위 레벨의 feature fusion을 위해, 레이어를 반복하여 추가</span> <span style="color:#DF8B00">$$\rightarrow$$ 한번의 top-down, bottom-up 연결만 구현된 PANet과 다르게  bidirectional path(top-down & bottom-up)를 하나의 feature network layer로 보아, 여러번 반복하여 추가함으로서 더 많은 feature fusion이 가능하게 한다.</span>

### Weighted Feature Fusion
- Feature를 fusing하는 가장 단순한 방법은, 각기 다른 해상도의 feature를 bicubic과 같은 interpolation으로 resize 한 후, 더하는 것이다.
- 기존의 feature fusing 방법들은 입력 feature 들을 동등한 비중으로 fusion 하였다. 하지만 입력 feature 들의 해상도가 다르기 때문에, 일반적으로 출력 feature에 다른 비중으로 기여하게 된다.
- 저자들은 이점에 착안하여 <span style="background-color:#BFFF00">입력 feature 들의 비중을 나타내는 값인 weight를 정의하고, 네트워크가 입력 feature들의 중요도를 학습하게 설계하였다.</span>

1. Unbounded Fusion
: $$O=\Sigma_iw_i\cdot I_i$$ 
: $$\rightarrow$$ $$w_i$$ : learnable weight, $$I_i$$ : feature map
: $$\rightarrow$$ weight가 unbounded 되어, 학습시 불안정하기 때문에 정규화가 필요하다.

2. Softmax-based fusion
: $$O=\Sigma_i \frac{e^{w_i}}{\Sigma_je^{w_j}} \cdot I_i$$
: $$\rightarrow$$ 각 weight에 softmax를 취하여 정규화한다. 
: $$\rightarrow$$ 왜 softmax 인가? softmax는 $$j$$개의 값이 존재하면, 각 값의 편차를 확대시켜 (큰 값은 더 크게, 작은 값은 더 작게) [0,1] 범위의 값들로 정규화한다.
: $$\rightarrow$$ 하지만 단순히 Softmax를 하게 되면, GPU에서 연산 속도가 느려지게 된다. 따라서 연산이 빠른 방법을 고안하게 되었다.

3. **Fast normalized fusion**
: $$O=\Sigma_i \frac{e^{w_i}}{\epsilon + \Sigma_je^{w_j}} \cdot I_i$$
: $$\rightarrow$$ $$w_i$$에 Relu를 취해 양수가 되도록 한다. $$\epsilon$$은 0.0001과 같은 작은 수치로 정한다.
: $$\rightarrow$$ 정규화된 weight는 0에서 1사이 값을 가지지만, softmax 연산을 안하므로 더 효율적이다.
: <p align="center"><img src="https://drive.google.com/uc?id=1PWABfKbTtz-Le2FBiJ-yjlQwKVKqXKB4" width="100%" height="100%"></p>
: $$\rightarrow$$ 실험 결과에서도 Fast normalized fusion의 학습 방식이, Softmax를 적용할 때와 비슷한 학습 양상을 보인다. 또한 연산 속도도 1.3배 정도 빨라지게된다.
: <p align="center"><img src="https://drive.google.com/uc?id=1TS2XsSWBRJpMqOD0gTiDYjvjl15Z3-rK" width="100%" height="100%"></p>
: $$\rightarrow$$ BiFPN을 도식화 하면 위 그림과 같다. node는 operation으로 edge는 feature map으로 볼 수 있다. 실제로 저자들은 더 높은 효율적인 구조를 위해 depthwise separable convolution을 사용하였으며, conv 연산마다 batch normalization을 적용하였다.

<br>
## EfficientDet
---

> Aiming at optimizing both accuracy and efficiency, we would like to develop a family of models that can meet a wide spectrum of resource constraints. A key challenge here is how to scale up a baseline EfficientDet model.

- EffiicentDet은 마찬가지로 EfficientNet의 **Compound Scaling** 기법([EfficientNet 논문 리뷰 참고](https://llamakoo.github.io/blog/2020/04/17/EfficientNet/))을 사용하여 model scaling을 통해 다양한 크기의 모델을 구현하였다.
- Network의 width, depth, input resolution을 scaling하는 것은 EfficientNet과 동일하지만, scaling 해야 할 네트워크(backbone network, BiFPN network, class/box network)가 많다.
- 따라서 네트워크들 전부에 대해서 grid search로 하나씩 찾는 것은 매우 어려운 작업이기 때문에 heuristic한 방법을 사용하였다.
- EfficientNet의 모델(B0 ~ B6)과 동일하게 구성하며,  BiFPN 네트워크의 width, depth는 각각 $$W_{bifpn}=64\cdot(1.35^\phi)$$, $$D_{bifpn}=3+\phi$$ 식에 맞게 설정한다. Box/class prediction 네트워크는 $$D_{box}=D_{class}=3+\lfloor\phi+3\rfloor$$ 식에 맞게 설정한다. 마지막으로 input resolution은 $$R_{input}=512+\phi\cdot128$$에 맞춘다. 
<p align="center"><img src="https://drive.google.com/uc?id=1TjXFYhYXStd_gUzp_JNSwb-OwjQz9NQS" width="80%" height="100%"></p>
- Depth, width, resolution을 각각 scaling하는 것 보다, 전부 scaling하는 것이 가장 좋은 성능을 보여준다.

<br>
## Experiment
---
<p align="center"><img src="https://drive.google.com/uc?id=1Bi5dyls3y0SAzRFwijUc_ToRmSrWuM8v" width="100%" height="100%"></p>
- 실험 결과는 매우 뛰어난 성능을 보여준다. 가장 작은 모델인 EfficientDet-D0와 기존의 대표적인 detector인 YOLOv3을 비교할 때, AP는 비슷한 수치를 보이지만 FLOPs 차이가 28배 나는 것을 보아 D0가 훨씬 효율적인 모델이라 할 수 있다.
- RetinaNet, NAS-FPN등의 다른 detector와 비교하여도 성능은 비슷하거나 우위에 있으면서도 FLOPs는 더 적다는 장점을 보인다.

<br>
## Conclusion
---

- 한 줄 요약 : BiFPN을 통한 효과적인 feature fusion과, compound scaling을 통한 고성능, 고효율의 detector.


<br>