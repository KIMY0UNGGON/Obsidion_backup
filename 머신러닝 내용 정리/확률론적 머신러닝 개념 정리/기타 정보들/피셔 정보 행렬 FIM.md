

데이터가 Parameter에 대해 얼마나 많은 정보를 제공하는지 측정하는 행렬.
MLE의 불확실성을 계산하는 도구로서 활용이 됨.

식 정의 : $F(\theta) = \mathbb{E}[\nabla l(\theta)[\nabla l(\theta)]^T]$ 
    $l(\theta) = log p(x|\theta) (score function \nabla l)$

F는 Likelihood의 정보양을 측정. 파라미터 변화시에 Likelihood가 얼마나 민감한지를 측정한다.

