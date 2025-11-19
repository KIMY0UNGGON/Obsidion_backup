
<hr>

## 파라미터간의 공분산 정보가, FIM이다

데이터가 Parameter에 대해 얼마나 많은 정보를 제공하는지 측정하는 행렬.
MLE의 불확실성을 계산하는 도구로서 활용이 됨.


MLE로 값을 추정할때 데이터의 수가 굉장히 많아지면 MLE로 추정한 값은 가우시안 분포를 따르게 된다.

즉 불확실성 :  $\hat\theta \approx N(\theta^*, Var), Var=\frac{1}{N \times 정보}$

 FIM은 데이터가 Parameter를 얼마나 잘 결정하는지를 알아냄.
   만약 정보가 많으면 FIM이 크고, Parameter를 잘 결정한 것. -> 즉, Var(분산)이 작고 신뢰도가 높음
   만약 정보가 적으면 FIM은 작고 Parameter가 애매함 -> 분산은 커지고, 신뢰도가 낮아짐. 불확실성이 높아짐.


식 정의 : $F(\theta) = \mathbb{E_{x \sim p(x|\theta)}}[\nabla l(\theta)[\nabla l(\theta)]^T]$ 
    $l(\theta) = log p(x|\theta) (score function \nabla l)$

F는 Likelihood의 정보양을 측정. 파라미터 변화시에 Likelihood가 얼마나 민감한지를 측정한다.

$\nabla l(\theta)$는 score function으로 $\frac{∂ℓ}{∂θ}$ 즉, log likelihood를 $\theta$로 편미분한값.

