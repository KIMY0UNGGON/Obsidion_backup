MLE를 일반화한 방법. ERM은 모델의 학습 목표를 정의하는 프레임워크, activation fuction과는 다름

경험적 위험(RISK) : 모델의 예측 오차가 발생할 기댓값. 모델이 새로운 데이터를 얼마나 잘못 예측할지에 대한 가능성. 즉, Loss의 기댓값

모델의 파라미터를 최적화하여 학습 데이터에 대한 LOSS를 최소화하는 방법.

정의는 다음과 같음.

$L(\theta) = \frac{1}{N} \sum_{n=1}^N \ell(y_n, \theta; x_n)$ ⇒ 평균 log LOSS

ERM은 모델이 학습을 얼마나 잘했는지 측정하는 LOSS함수의 평균을 최소화하는 과정.

MLE는 NLL을 사용하는 특수한 ERM의 사례로 볼 수 있음. ERM은 이것을 일반화 하여 다양한 LOSS함수를 사용함.

오분류율 최소화:

오분류율이란 현재 예측이 맞는지 안맞는지.
분류 문제에서 예측 Label과 정답 label이 같은지 여부를 기준으로 LOSS를 정의 가능. ⇒ 0-1LOSS

예측이 맞으면 LOSS = 0 , 예측이 틀리면 LOSS = 1로 정의

정의는 다음과 같음.

${\ell_{01}(y_n, \theta; x_n) = \begin{cases} 0 & \text{if } y_n = f(x_n; \theta) \\ 1 & \text{if } y_n \neq f(x_n; \theta) \end{cases}}$

$f(x_n; \theta)$는 예측 label, $y_n$은 정답 label.

다음과 같이 다시 재정의 가능.

$\tilde{y} = f(x_n; \theta)$ ⇒ y 틸라이드는 예측 label이라 정의.

$\ell_{01}(\tilde{y}, y_n) = Ⅱ(\tilde{y}y_n < 0)$
예측값과, 정답이 다르면 1, 같으면 0. => 예측성공은 0, 실패는 1

0-1손실은 미분이 불가능 ⇒ gradient descent가 적용이 어려움. ⇒ 최소화가 힘들다는 단점이 존재.

Surrogate Loss(대리 손실 함수):

0-1 손실의 최적화가 안되는 문제를 해결하기 위해 사용.

대리 Loss함수는 모두 0-1손실의 상한. 이를 최소화하면 오분류율도 간접적으로 줄어듬.

convex한 형태여서 최적화(gradient descent)가 쉬움. ⇒ global minimum도 찾을 수 있음

1. Log Loss ⇒ 이진 분류에선 시그모이드로 Label의 확률을 예측.
    
    $\sigma(\tilde{y} \eta) = \frac{1}{1 + e^{-\tilde{y} \eta}}$로 라벨의 확률을 예측.
    
    $\eta = f(x;\theta)$로 로그 Odds.
    
    $\ell_{ll}(\tilde{y}, \eta) = \log(1 + e^{-\tilde{y} \eta})$
    
    $\tilde{y} \eta$가 클수록 LOSS가 작아짐.
    
    로지스틱 회귀에서 주로 사용됨.
    
2. Hinge Loss(힌지 손실)
    
    0-1Loss의 또다른 Convex 상한.
    
    다음과 같이 정의됨.
    
    $\ell_{hinge}(\tilde{y}, \eta) = \max(0, 1 - \tilde{y} \eta)$
    
    $\tilde{y} \eta$가 1 이상이면 Loss가 0이되고, 1미만이면 선형적으로 증가.
    
    SVM에서 주로 사용되고 부분적으로 미분가능, 최적화도 가능함.
    

결과 Label의 분포를 보고서 손실함수를 결정할 수 있음. 웬만하면 대부분 가이드라인이 정해져있음.

<figure>
<img src="손실함수그래프.png" >
<figcaption>손실함수 그래프</figcaption>
</figure>

