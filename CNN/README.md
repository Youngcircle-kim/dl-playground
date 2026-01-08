# Introduction

CNN은 Convolutional Neural Network의 약자로서, 영상 및 이미지 처리를 위해 만들어진 DNN(Deep Neural Network)의 한 종류이다. 컴퓨터에게 영상과 이미지는 많은 픽셀들로, 영상의 경우 여러 프레임까지 포함된 다차원 숫자 데이터로 표현된다. 이런 데이터를 기존의 MLP(Multi Layer Perceptron)에 그대로 입력하면, 모든 픽셀이 모든 뉴런과 완전 연결되는 구조 때문에 파라미터 수가 폭발하고, 이미지를 1차원 벡터로 평탄화하는 과정에서 인접한 픽셀 사이의 공간 구조 정보를 제대로 활용하지 못하는 문제가 발생한다.

CNN은 이미지나 영상과 같은 다차원 데이터를 그대로 입력으로 사용하면서, 합성곱 연산을 통해 국소 영역만 부분적으로 연결하고 가중치를 공유하는 구조를 사용한다. 이를 통해 파라미터 수를 크게 줄이는 동시에, 2D/3D 공간 구조를 유지한 채 계층적으로 특징을 추출할 수 있다. 또한 풀링(pooling) 등의 연산으로 공간 해상도와 차원을 점진적으로 줄여 연산량을 감소시키면서도, 중요한 시각적 특징은 보존하여 위와 같은 문제를 효과적으로 완화한다.

# Basic Concept

CNN의 가장 기본적인 형태는 다음과 같다.

1. Convolution
2. ReLU
3. Max pooling

![CNN Architecture](./assets/CNN%20Architecture.png)
Figure 1. CNN Basic Architecture

### Convolution

수학적인 Convolution Operation의 정의는 다음과 같다. 두 함수를 결합해 한 함수의 효과를 다른 함수에 적용하는 연산이다.
$f(t)$와 $g(t)$라는 Convolution 하는 식은 다음과 같이 나타낼 수 있다.

$$
(f * g)(t) = \int\_{-\infty}^{\infty} f(\tau) g(t - \tau) \, d\tau
$$

이 식은 $g$를 뒤집고(Flipping)($\tau$로 이동(Shifting)) $f$와 곱한(Multiplication) 후 적분(Summation)한다.

하지만 CNN에서는 엄격한 수학적 Convolution이 아닌 Cross-Correlation을 사용한다. 즉, 필터를 뒤집는 Flipping 단계를 하지 않고 입력과 직접 곱하며 슬라이딩한다. 수식은 아래와 같다.

$$
S(i,j) = (I * K)(i,j) = \sum_m \sum_n I(i+m, j+n) \cdot K(m,n)
$$

![Convolution Process](assets/Convolution%20process.png)
Figure 2. Convolution 과정

Figure 2는 Convolution 과정을 나타낸다. 노란색 박스를 시작으로 파란색 => 회색 => 진홍색으로 진행된다. 각각의 계산은 이해를 위해 노란색을 대표 예시로 아래와 같이 진행된다.

$$
1*0 + 2*8 + 4*3 + 5*0 = 30
$$

위와 같이 각 요소들끼리 곱을 한 다음 덧셈으로 계산이 된다.
진행 방식은 행 우선 순회(Row-major order, Z자 방향, 왼쪽 위에서 오른쪽 아래)로 진행된다.

이러한 과정을 통해 CNN은 MLP에 대비되어 입력 데이터의 공간적 정보를 반영하고, 학습 시키는 파라미터 수를 눈에 띄게 줄였다.

### ReLU

Activation Function의 일종으로 수학적 정의는 다음과 같다.

$$
f(x)=max(0,x)
$$

정의역이 음수일 경우 상수함수이고, 양수일 경우 $y=x$인 함수다.

이 단순한 비선형성 때문에 ReLU는 신경망의 학습 속도를 개선하고, 이전에 많이 사용되던 `sigmoid function`이나 `tanh fuction`보다 효율적으로 깊은 네트워크를 학습시킬 수 있다.

이는 `Foward`시 음수 값을 0으로 바꿔 계산 효율성을 올려준다. 또한 `Back propagation`시 음수 영역은 0, 양수 영역은 1이 되어 효율성을 올린다. 물론 0으로 바뀌는 문제 때문에 `죽은 뉴런 문제`라는 단점을 가지고 있다.

### Max Pooling

Max Pooling은 Convolutional layer 다음에 주로 오는 다운샘플링 기법이다.
이는 지정된 영역 (예: 2x2) 내에서 가장 큰 값만 선택해 출력한다.

Convolutional layer의 커널과 비슷하게 지정된 영역을 설정하고 stride 또한 설정해 **겹치지 않게** 슬라이딩하여 최대 값만 뽑는 방식이다. 이를 통해 채널 수는 그대로, 높이/너비만 영역과 원본데이터에 맞춰 줄어든다.

예시는 아래와 같다.
![Max Pooling](assets/Max%20pooling.png)
Figure 3. Example of max pooling

Figure 3는 지정된 영역(2x2)에서 가장 큰 요소인 8만 가져와서 다운 샘플링한 예시이다.

이렇게 다운 샘플링을 하여 feature map 크기를 줄여 계산량을 감소시킨다. 또한 특징의 정확한 위치보다는 존재 여부에 집중할 수 있게 한다. 이는 객체가 이미지내에서 이동하거나 회전해도 동일하게 인식하는데 도움을 준다. 마지막으로 불필요한 세부 정보를 제거하여 모델이 더 일반화된 특징에 집중하도록 돕습니다.

# Conclusion

이러한 구조적 특징 때문에 CNN은 다음과 같은 장점을 가진다.

- **국소 연결(Local Connectivity)** 과 **가중치 공유(Weight Sharing)** 를 통해 파라미터 수를 크게 줄여, 고차원 이미지 데이터에서도 효율적인 학습이 가능하다.
- 입력 데이터의 **공간적 구조를 유지**하면서 특징을 추출하므로, 이미지의 위치 관계와 패턴을 효과적으로 학습할 수 있다.
- 계층적 구조를 통해 저수준 특징(엣지, 코너)부터 고수준 특징(객체, 형태)까지 **점진적인 특징 표현 학습**이 가능하다.
- 동일한 필터를 전체 이미지에 적용함으로써 **이동 불변성(translation invariance)** 을 어느 정도 확보할 수 있다.

CNN은 다음과 같은 한계를 가진다.

- **공간적 국소성(Locality)에 강하게 의존**하므로, 이미지 전체의 전역적 관계(Global Context)를 학습하는 데 한계가 있다.
- **입력 크기와 구조에 민감**하여, 해상도나 형태가 크게 달라질 경우 성능이 저하될 수 있다.
- 깊은 네트워크에서는 **gradient 소실·폭주 문제**가 발생할 수 있으며, 이를 완화하기 위한 추가적인 기법(BatchNorm, Skip Connection 등)이 필요하다.
- **Pooling 연산**으로 인해 세부적인 위치 정보가 손실될 수 있어, 정밀한 위치 인식이 중요한 작업에는 불리할 수 있다.
- 대규모 데이터와 연산 자원이 필요하며, **학습 비용이 높다**.
- 회전, 스케일 변화 등 **복잡한 기하학적 변형에 대해 완전한 불변성을 제공하지는 못한다**.

# References

1. https://blog.naver.com/simula/223924642257
2. https://brunch.co.kr/@donghoon0310/44
3. https://wikidocs.net/120168
