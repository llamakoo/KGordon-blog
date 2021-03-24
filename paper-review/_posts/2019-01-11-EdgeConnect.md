---
layout: post
title: EdgeConnect
date: 2019-01-11
comments: true
categories: [paper-review]
tags: [neural_rendering, inpainting]
---

> Over the last few years, deep learning techniques have yielded significant improvements in **image inpainting**. However, many of these techniques fail to reconstruct reasonable structures as they are commonly over-smoothed and/or blurry. This paper develops a new approach for image inpainting that does a better job of **reproducing filled regions exhibiting fine details**. We propose **a two-stage** adversarial model EdgeConnect that comprises of an edge generator followed by an image completion network. The edge generator **hallucinates edges of the missing region** (both regular and irregular) of the image, and the image completion network **fills in the missing regions using hallucinated edges as a priori**. - Abstract of the paper

- Table of Contents
{:toc .large-only}

## Summary
---
- 본 알고리즘은 두개의 generator를 거쳐 inpainting을 수행한다.
- 첫번째 단계에서 사라진 영역의 edge를 추정하고 전체 이미지의 edge map을 생성하며, 두번째 단계에서는 생성한 edge map과 비어있는 영역이 존재하는 hold 이미지를 입력받아
주변 context와 어울리는 hole을 채운 이미지를 생성한다.
- 네트워크는 end-to-end로 학습되어지며, 추정한 edge map 덕분에 더 디테일한 이미지를 생성할 수 있다.

## Image Inpainting
---
- Image inpainting은 이미지의 사라진 영역을 채우는 것이 목적이다. 
  사람은 visual inconsitencies에 민감하기 때문에 인지적으로 그럴듯하게 hole을 채워야 한다.
- 고전적인 inpainting 기법은 diffusion-based와 patch-based로 분류할 수 있다.
  Diffusion-based 방식은 diffusive 과정을 통하여 배경의 데이터를 propagate한다.
  반면, Patch-based 방식은 소스가 되는 이미지의 모음에서 가장 유사한 패치를 통해 hole을 채운다.
  하지만 두가지 방식 모두 복잡한 디테일을 생성하기에는 많은 한계가 존재하였다.
- 최근 딥러닝이 대두되면서 image inpainting 기법에도 많은 발전이 이루어졌으며, 학습된 데이터 분포를 통해 missing pixel을 채우는 방식이다.
  고전적인 방법에서 거의 불가능했던, 일관성 있는 context를 생성할 수 있다.
- 하지만 지금까지 연구되어진 많은 inpainting 기법들은 대부분 과하게 blurry한 이미지를 생성하는 경향이 있다.

<figure>
  <img src="https://drive.google.com/uc?id=1uH8c-nk0bLqC8F3HwNtxgn0YyzbYXooW" style="width:100%">
  <figcaption>(b)는 고전적인 방법 중 하나의 결과영상으로, 복잡한 디테일을 생성하기 어려운 경향이 있다.<br>
  Figure from "Image inpainting for irregular holes using partial convolutions.", Liu, Guilin, et al.,ECCV,2018.</figcaption>
</figure>

### Challenge
> Then, how does one force an image inpainting network to generate fine details?

- 미세한 디테일까지 생성하는 inpainting 네트워크는 어떻게 설계할 것 인가?

## EdgeConnect
---

<figure>
  <img src="https://drive.google.com/uc?id=1Tc042FFo1dW8I-KgPnqsRW6PX6Qr7kZ-" style="width:100%">
  <figcaption>EdgeConnect 네트워크는 two-stage로 inpainting을 수행한다. </figcaption>
</figure>

- 2개의 generator로 구성된다. (학습시 각각의 discriminator가 붙는다)
- 첫번째, edge generator는 빈 영역의 edge를 추정하여 완성된 edge map을 생성한다. 두번째, image completion network는
  생성된 edge map과 hole이 존재하는 rgb 이미지를 입력받아 최종 inpainting된 영상을 생성한다.
- 각 generator는 동일한 구조를 가지며, discriminator로는 70x70 PatchGAN을 사용한다. 

### Edge Generator

$$
\begin{aligned} %!!15
\min _{G_{1}} \max _{D_{1}} \mathcal{L}_{G_{1}}=\min _{G_{1}}\left(\lambda_{a d v, 1} \max _{D_{1}}\left(\mathcal{L}_{a d v, 1}\right)+\lambda_{F M} \mathcal{L}_{F M}\right)
\end{aligned}
$$

An objective function of edge generator
{:.figcaption}

- edge generator에는 3종류의 데이터가 입력된다. $$\mathbf{C}_{\text {pred }}=G_{1}\left(\tilde{\mathbf{I}}_{\text {gray }}, \tilde{\mathbf{C}}_{g t}, \mathbf{M}\right)$$
- edge generator를 학습할 때, objective function은 adversarial loss와 feature-matching loss간의 합으로 구성된다.
- feature-matching loss은 생성된 이미지가 실제 이미지와 유사하도록 하는 학습과정을 더 안정적이게 해준다.
- 최근 대다수의 이미지 생성 논문들은 feature-matching loss와 perceptual loss를 결합하여 사용하는데,
  학습된 VGG 네트워크를 사용하는 perceptual loss는 edge 정보를 생성하는데 적합하지 않으므로 본 edge generator를 학습할때는 적합하지 않아서 배제하였다고 한다.
- generator와 discriminator에 SN(Spectral normalization)을 적용하였다. 이는 네트워크의 gradient값과 파라미터가
  갑자기 변동하는 것을 경감해주는데 이점이있다.

### Image Completion Network

$$
\begin{aligned} %!!15
\mathcal{L}_{G_{2}}=\lambda_{\ell_{1}} \mathcal{L}_{\ell_{1}}+\lambda_{a d v, 2} \mathcal{L}_{a d v, 2}+\lambda_{p} \mathcal{L}_{\text {perc }}+\lambda_{s} \mathcal{L}_{\text {style }}
\end{aligned}
$$

An objective function of image completion network
{:.figcaption}
- image completion network에는 두가지 데이터가 입력된다. edge generator가 추정한 edge map과 hole이 있는 rgb 영상이다.
- objective function은 L1 loss, Adversarial loss, perceptual loss, style loss가 사용된다.
  이는 최근 이미지 생성 논문들에서 많이 쓰이는 조합이다.
- edge generator와는 다르게 SN을 사용하지 않았는데, 저자는 SN을 사용할 경우 학습시간이 방대하게 증가하기 때문이다고 한다.

## Experiments
---
- train label로 사용할 edge map은 Canny edge detector를 사용하여 만들었다.
- train label로 사용하는 mask는 2가지가 있는데, 사각형 모양의 mask와 irregular한 모양의 mask를 사용하였다.
- 학습 방법으로는 2가지 generator를 각각 따로 학습하며, 학습이 완료되고 다시 2개의 generator를 jointly하게 학습한다.
  jointly하게 학습할 때는 image completion network의 discriminator만 사용한다.

<figure>
  <img src="https://drive.google.com/uc?id=1fBEgpx-_AykJ8KB0l2Hda6gWxk5FwacE" style="width:100%">
  <figcaption>다른 inpainting 기법들과의 비교 결과. 기타 방법들 대비 좋은 성능을 보인다.</figcaption>
</figure>

- 저자들은 edge가 주어지면, image completion network가 이미지의 구조정보는 고려하지 않고 칼라 분포만 학습하면 되기 때문에 효과적인 결과를 보인다고 추측한다.

<figure>
  <img src="https://drive.google.com/uc?id=1N2QfDNkND2gqgrbnJKmMtzTpjlcA99fv" style="width:100%">
</figure>

- Figure4를 보면, 왼쪽부터 원본 이미지, 입력 이미지, 생성된 edge map, inpainted 이미지의 결과를 보여준다.
- 또한 정량적 평가로 SSIM, PSNR, FID 수치를 비교하였으며 기타 알고리즘 대비 높은 수치를 보여준다.
- 저자들은 또한 여러가지 ablation 평가를 진행하였는데 간단히 종합해보면
    - edge 정보가 주어질 때 inpainting도 더욱 효과적인 결과를 보여준다.
    - edge map에 edge 성분이 너무 과하여도 너무 적어도 좋지 않으며, 적당한 edge 성분일 때 좋은 결과를 보여준다.
    - Canny edge detector가 아닌 기타 edge detector를 사용하였지만 결과에서 큰 차이를 보이지 않았다.
    - edge map을 사용자가 조절하여 변형된 구조의 이미지를 생성할 수도 있다. 또한 본 알고리즘으로 이미지의 원하지 않는 부분을 지울수 있다.


## Paper Info
---
> Title : EdgeConnect, Generative Image Inpainting with Adversarial Edge Learning

> Paper link : [https://arxiv.org/abs/1901.00212](https://arxiv.org/abs/1901.00212)

> Publised year : 1 Jan 2019

> Authors : Kamyar Nazeri, Eric Ng, Tony Joseph, Faisal Z. Qureshi, Mehran Ebrahimi

---


<br>