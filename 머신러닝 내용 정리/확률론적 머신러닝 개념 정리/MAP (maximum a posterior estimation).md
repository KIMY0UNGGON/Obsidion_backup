 사전 분포를 도입해, 데이터 만으로 추정하는 MLE가 아니라 '사전 지식 + 데이터' 모두 반영하는 MAP로 추정.
 MLE는 prior(사전분포)를 무시한 특수한 케이스라 볼 수 있음
  [[정규화(Regularization)]] 에서 사용하는 사전 분포의 정규화 parameter를 1로 한 뒤에 $\theta$값을 최대화 한 것.
  정규화 parameter는 정규화의 강도를 나타냄
  Log posterior(로그 사후 확률)을 최대화 하는 것이 MAP추정

정의 :  $\hat\theta = argmax_\theta log\ p(\theta|D) = argmax_\theta[log\ p(D|\theta)+log\ p(\theta) - const]$

### 예시 1 : 베르누이 분포에서의 MAP 추정
  
  베르누이에서 MLE의 경우 zero-count problem이 생길 수 있음

  zero-count problem : 관측 되지 않은 이벤트의 확률이 0으로 할당되는 문제. 즉, 관측되지 않은 이벤트의 일어날 가능성을 0으로 가정하므로 베르누이에선 문제가 생길 수 있음. 그래서 MAP로 패널티항 추가

  베타분포를 사전분포로 사용. Beta($\theta$|a,b) a,b > 1 일때 $\theta가\ \frac{a}{(a+b)}$근처의 값들에 가까워짐
  정의 : $l(\theta) = log\ p(D|\theta) + log\ p(\theta) = [N_1\ log\theta + N_0\ log(1-\theta)] + [(a-1)log(\theta)+(b-1)log(1-\theta)]$ 
   
   $N_1 + N_0 = N$  N은 시도 횟수 $N_1$ 은 성공 횟수, $N_0$는 실패 횟수

   log 사전 = log($\frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)}x^{a-1}(1-x)^{b-1}$) = log$\frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)}$ + (a-1)log$\ x$ + (b-1)log $(1-x)$ 
     (const는 정규화에서 무시. 상수값인 감마분포는 무시됨)
   
   $\theta_{MAP} = \frac{N_1+a-1}{N_1+N_0+a+b-2}$

   a = b = 2면 $\theta$ = 0.5에 가까움. $\theta_{MAP} = \frac{N_1+1}{N_1+N_0+2}$ 


 add-one smoothing : zero-count problem을 해결하기 위한 간단한 정규화 기법
    카테고리 분포, 베르누이 분포 같은 이산확률 모델에서의 zero-count 문제를 피하게해줌.
    과적합 방지와, 새로운 데이터의 확률을 0으로 만들지 않도록 함
    분모에 카테고리 수 만큼 더해줌. 그후 분자에 1을 더해줌
    베르누이는 카테고리가 2이므로 위의 a= b = 2와 마찬가지로 N+2를 분모로, $N_1$+1을 분자로 사용하여 MAP를 추정 


### 예시2 : 다변량 가우시안에 대한 MAP 추정

 고차원에서의 $\Sigma$(공분산행렬)의 추정치가 특이(singular)해질 수 있음. **행렬이 특이하면, 미분이 어렵고**, **역행렬 계산 불가능**함.

 즉, 데이터가 적을 수록 고차원으로 모델을 학습시키는데 차원이 더욱 높아질 수록 공분산 행렬이 특이해질 수 있다.  데이터의 노이즈에 대해서도 맞추려고 학습하기 때문.
그래서 MAP추정을 사용해 해결함. (데이터를 많이 수집해서 학습시키거나, data agumented로 데이터 돌려막기를 해서 차원을 줄이는 해결법도 존재함)
 MVN은 다변량 데이터(이미지, 유전자 표현 등의 고차원 벡터)에 자주 사용됨.

 특이행렬 : det(행렬) = 0이거나 역행렬이 존재하지 않는 행렬


#### 1.Shrinkage 추정

 공분산 행렬 $\Sigma$에 사용하기 편리한 사전 분포는 [[inverse Wishart]] 분포를 사용.
 즉 MVN의 공분산 행렬의 MAP추정.
 
 inverse Wishart 사전분포로 수축 추정을 함.

정의 : $\hat\Sigma_{map} = \frac{\tilde S+S_{\bar y}}{\tilde N+N} = \lambda\Sigma_0+(1-\lambda)\hat\Sigma_{mle}$

 $\lambda$ = $\frac{\tilde N}{\tilde N + N}$ 으로 $\lambda$는 사전분포의 비중. 정규화의 강도를 제어하는 parameter로 사용됨
 $\lambda$ = 0이면 prior을 무시하고 MLE만 사용
 $\lambda$ = 1이면 MLE를 무시하고 prior만 따름.
 $\lambda$ = 0.5~0.9  : $\lambda$비율 만큼 MLE를 prior쪽으로 shrink(끌어당기는) 정도. 데이터 샘플 수 가 적으면 람다의 값을 키워 prior를 더 믿음

 사전 분포의 데이터 샘플 수는 하이퍼 파라미터로 사용함

 shrinkage 추정에선 MLE를 대각행렬쪽으로 축소하는 효과를 줌. 그렇기에 inverse wishart를 대각행렬로 사용
 

 위의 정의를 다시 사용하면 $\hat\Sigma_{map} = \lambda\ diag(\hat\Sigma_{mle})+ (1-\lambda)\hat\Sigma_{mle}$ 


 대각 행렬쪽으로 축소하는 이유 :
 
  MLE 공분산행렬의 경우, 고차원에서 역행렬이 계산이 불가능하는 경우가 생김. 그리고, off-diagonal 요소 과대/과소 추정으로 훈련데이터에 과적합할 수 있음
   그래서 고차원에서 MLE는 모델 성능이 줄어듦으로 MAP를 사용

**off-diagonal** : 주 대각선이 아닌 모든요소

   1. 안정성확보 : 고차원에서 $\Sigma$의 고유값이 극단적임. 그래서 대각 shrink로 균형맞춤
   2. 과적합 방지: off-diagonal을 0 shrink하여 불필요한 과적합을 줄임
   3. 계산 효율 : 대각행렬만 계산해서 계산 효율성이 높음
   4. 해석이 용이함.



### 2. weight decay(가중치 소멸)

 과적합 해결을 위해선 여러 해결법이 존재.
 가장 쉬운건 다항식의 차수(차원)를 줄이는것. 대신 이 방법은 데이터가 충분해야 가능함
 
 이것보다 더 일반적인 방법은 weight의 크기에 페널티를 주는것이다.
 이떄 사전분포는 $\mu$ = 0인 정규분포 p(w)를 사용한다

식 정의 : $\hat w_{MAP} = argmin_{w}\ NLL(w) + \lambda\frac{{\left\lVert w\right\rVert}^{2}_{2}}{2}$  NLL은 음수이기에 페널티항을 더해줌

${\left\lVert w\right\rVert}^{2}_{2}$ = $\sum_{d=1}^{D}w^2_d$  

$\theta$가 아닌 w로 표시한 이유는 weight 벡터의 크기에 페널티를 주는것이기 때문 다른 parameter에는 적용안함.
즉, weight의 절대값이 큰것들을 축소시키기 위해 L2 정규화를 사용함

이 식을 L2 정규화라고 부르거나 weight decay라 부름
$\lambda$가 클수록 parameter가 큰 값에 대해 더 많은 페널티를 받아서 모델이 덜 유연, 덜 과적함 해짐
 선형회귀에선 ridge regression이라 부름 ^46cd43