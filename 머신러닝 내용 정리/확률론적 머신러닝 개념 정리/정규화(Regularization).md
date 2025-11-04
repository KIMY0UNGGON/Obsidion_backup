Overfitting(과적합) :
	MLE와 ERM에서 LOSS를 최소화 하는 parameter를 선택했지만 새로운 입력에 대해 Loss가 작아지지 않았을 때 과적합이라 함.
	즉, 학습데이터에 너무 fit하게 학습되었다는 뜻.
	모델이 과적합되면 일반화하지 못할 수 있는 문제가 있음.
	 이를 해결하기 위해 정규화를 사용

정규화:
	**과적합 문제를 해결**하는 주요방법.
	모델이 훈련데이터에 지나치게 맞춰지지 않도록 복잡도를 제약하거나 Parameter에 사전 지식을 반영하여 일반화된 모델을 만듦
	NLL이나 경험적 위험(empirical risk)에 패널티(penalty term)을 추가하는 방식
	정의 : $L(\theta; \lambda) = \left[ \frac{1}{N} \sum_{n=1}^N \ell(y_n, \theta; x_n) \right] + \lambda C(\theta)$
	 $\lambda>=0$ 으로 람다는 정규화 hyper parameter. 패널티의 강도를 조절함.
	 $C(\theta)$ : 파라미터 $\theta$의 복잡도를 측정하는 패널티 함수.
	  일반적으로는 $-\log p(\theta)$를 사용. $p(\theta)$는 $\theta$에 대한 사전분포. $\theta$가 사전분포를 벗어나면 패널티를 부여. $p(\theta)$가 작아질수록 $-\log p(\theta)$가 커짐.
	패널티는 Loss값이 0에 가까워지지 않기 하기 위해 부여함. 제대로 학습시에는 패널티는 부여안함. 
	너무 학습 데이터에 fit하지 않게 하기 위함.
람다를 1로 설정하고 $p(\theta)$를 적절히 스케일링하면 다음을 최소화 하는 것과 Loss가 동일.
$L(\theta; \lambda) = -\left[ \sum_{n=1}^N \log p(y_n|x_n, \theta) + \log p(\theta) \right] = -[\log p(D|\theta) + \log p(\theta)]$
 위의 식을 모드로 변경하면 다음과 같음 -이므로 argmin이 아닌 argmax를 사용
$\hat{\theta} = \arg\max_\theta \log p(\theta|D) = \arg\max_\theta [\log p(D|\theta) + \log p(\theta) - \text{const}]$ 로 정의가 가능.
즉 $\log p(\theta|D) = \log p(D|\theta) + \log p(\theta) - \text{const}$
이를 MAP(최대 사후 확률 추정)이라함.  MAP(Maximum a posterior estimation)
	이때의 const는 $logp(D)$로 posterior 정규화 상수. 즉 D, 데이터 셋에 대한 로그사전확률
	 $p(\theta|D) = \frac{p(D|\theta)p(\theta)}{p(D)}$이므로 이를 로그를 씨웠으니 const가 logp(D)이다.


$\log p(\theta|D) = \log p(D|\theta) + \log p(\theta) - p(D)$

 여기서 $logp(D|\theta)+logp(\theta)$는 MLE가 되고, const-> p(D)는 우리가 알고 있는 데이터셋의 확률이 된다. 즉, 우리가 알고 있는 데이터 셋의 확률이 패널티가 된다.

 ex)
	동전을 세번 던졌을 때 3번 모두 앞면이 나왔다 가정하면,
	MLE는 3번 던져서 모두 앞면이 나왔으니 다음 것도 앞면이겠네 라고 생각하고 있는거고, 패널티는 "동전 던지기의 확률은 0.5 잖아" 그러니 뒷면도 나올 수 있어 하고 과적합을 막아주는것



예시 1: 베르누이에 대한 MAP 추정

 동전 던지기에서 데이터 샘플이 1개에 그 한개의 데이터가 앞면이면 MLE는 1이됨.
  이러면 앞으로의 모든 동전 던지기가 모두 앞면이 나온다고 예측하는 과적합이 일어남.
  이 과적합을 해결하기 위해 패널티를 추가. 이때 베타 분포의 사전확률을 사용함

MAP 추정은 다음 파일 참조[[MAP (maximum a posterior estimation)]]




### 정규화 강도 선택: validation set

정규화 강도 $\lambda$가 작으면 ERM에 초점을 맞춰 과적합이 발생할 수 있고, 또 너무 $\lambda$가 크면 Prior분포에 가까워 과소적합이 발생할 수 있음

그래서 학습데이터 셋의 모든 데이터를 학습에 사용하지 않고, 훈련데이터와 검증 데이터로 나눠서 사용
대부분 훈련데이터 : 80%, 검증데이터 : 20%로 나눔
훈련데이터로 학습한 모델을 검증데이터로 검증하여 가장 낮은 loss의 모델을 선택
 그 후 다시 full data로 학습하여 모델을 생성함
validation set으로 하이퍼 파라미터를 정한다고 보면됨
 grid search를 사용. 

 이때 모델의 성능을 측정하는데 수학적 용어로 Empirical Risk(위험)을 사용

식 정의들 : 

경험적 위험:
$R_\lambda(\theta,\ D) = \frac{1}{|D|} \sum_{(x,y) \in D} \ell(y,f(x;\theta))+\lambda C(\theta)$ 

 첫번째 식은 empirical risk이고 ,두번째 $\lambda C(\theta)$가 정규화 페널티
 $\ell$은 손실함수. MVN에선 MSE

 train 추정
 $\hat\theta_\lambda(D_{train}) = argmin_\theta R_\lambda(\theta, D_{train})$ 


 risk 검증. 현재 모델의 성능을 validation set으로 확인.

$R_{val}^{\lambda} = R_0(\hat\theta_{\lambda}(D_{train}),D_{valid}) = \frac{1}{|D_{valid}|}\sum_{(x,y) \in D_{valid}}\ell(y,f(x;\hat\theta_\lambda)$ 

$R_0$는 $\lambda$값이 0인 검증값.
이 검증값들을 통해 가장 좋은 $\lambda$값을 찾는게 목표.

최적 $\lambda$선택

$\lambda^* = argmin_{\lambda \in S} R^\lambda_{val}$ 

그후 선택한 $\lambda$값을 가지고 validation set과 train set을 합쳐 모델을 재학습

최종 모델
$\hat\theta^* = argmin_\theta R_{\lambda^{*}}(\theta,D)$ 


#### Cross-Validation(교차 검증)
 훈련 데이터가 작으면 검증세트를 따로 빼기 힘듦. 따로 빼서 검증하더라도 학습이 제대로 되었는지 신뢰가 불가능함
 데이터가 너무 적어서 제대로 된 모델이 안나오기 때문

 그래서 Cross-Validation을 사용
 훈련 데이터를 K개의 fold로 나눠서 학습을 시키는 것이 cross-validation

 랜덤하게 섞은 데이터 셋을 5개로 나눠서 각 나눠진 fold들 중 하나씩 검증데이터로 사용하고 나머지 4개는 학습함.
 아래와 같이 모델 5개를 만들어 각각 validation set을 다르게 줘서 확인.
 아래와 같은 방식을 round-robin 방식이라 함. 아래는 K=5인 경우

 <figure>
 <img src="img/k-fold.png">
 </figure>
 

식 정의:

$R_{cv}^{\lambda} = \frac{1}{K}\sum_{k=1}^K R_0(\hat\theta_\lambda(D^{-k}), D_k)$ 

 $D^{-k}$는 validation set이 아닌 학습데이터 fold4개, $D_k$는 validation set
 위의 식처럼 튜닝하려는 하이퍼파라미터로 만든 모델들의 Loss값의 평균을 내서 하이퍼파라미터를 지정

$\lambda^* = argmin_\lambda R_{cv}^\lambda$ 로 $\lambda$를 선택해서 full data로 재학습.

$\lambda$ = {0.1, 0.01, 0.45} 면 5개의 모델을 3번 하이퍼파라미터를 다르게 학습시켜서 Loss 평균이 가장 좋은 람다값을 선택.

K = N인 경우 N-1개의 fold로 학습하고, 나머지 하나로 테스트하는 leave-one-out cross-validation(LOOCV)
위의 방식들로 딥러닝과 회귀에서 하이퍼파라미터를 선택함. 100개 미만의 데이터 셋에서는 특히 유용함



표준 오차 규칙(The One Standard Error Rule)


  Cross-Validation은 $\lambda$가 가장 작은 추정치를 줌.
 즉, Loss가 가장 작은 $\lambda$를 선택하게 하는데 , Cross-validation은 이 추정치가 얼마나 정확한지를 알려주지 않음
  즉 validation set에 과적합 되었을 수도 있음.
그래서 현재 이 Cross-validation의 결과가 믿을만한지 실제로 쓸만한지는 안알려줌. 
즉, 불확실성은 모름

그래서 추정치의 불확실성에 대한 빈도주의 측도를 사용
  => 추정치의 샘플링 분포(sampling distribution)의 평균인 표준 오차

 표준오차 구하는 법 정의

$L_{n}^{\lambda} = \ell(y_n, f(x_n; \hat\theta_\lambda(D^{-n})$  를 n번째 fold가 validation set일때의 Loss으로 지정.

CV의 경험적 평균 $\hat\mu = \frac{1}{N}\sum_{n=1}^{N}L_n^\lambda$
CV의 경험적 분포 $\hat\sigma^2 = \frac{1}{N}\sum_{n=1}^{N}(L^{\lambda}_n-\hat\mu)^2$

 이 경험적 평균과 분포를 기준으로 표준오차를 정의

 표준오차 : $se(\hat\mu) = \sqrt\frac{\hat\sigma^2}{N}$  표준오차는 평균$\hat\mu$에 대한 불확실성을 측정함
 se는 안정적인 모델과 하이퍼파라미터를 선택할때 사용됨


 사용 방법 : Loss 평균이 가장 작은 모델의 Loss 평균과 se를 더해서 평가. min Loss + se(min Loss)보다 큰 Loss 평균 모델은 안정적이지 않다고 판단되어 선택하지 않음




| λ (복잡도)   | $\hat{R}_λ$ (평균 loss) | SE  | $\hat{R}$ ≤ 상한? |
| --------- | --------------------- | --- | --------------- |
| 0.01 (복잡) | 1.8                   | 0.4 | Yes (1.8 ≤ 2.2) |
| 0.1 (중간)  | 2.1                   | 0.2 | Yes (2.1 ≤ 2.2) |
| 1.0 (단순)  | 2.5                   | 0.1 | No (2.5 > 2.2)  |



#### 예시 : [[회귀#^cbafe1| Ridge regression]]

  Ridge Regression에 CV 추정사용




### 조기종료(early stopping)


 실무 및 복잡한 모델에서 효과적인 매우 간단한 형태의 정규화
 최적화 알고리즘이 반복적이기 때문에 훈련세트에 대해 너무 과적합(많이 암기)하는 것 같으면 조기에 최적화 과정을 종료하는 방식.



### 더 많은 데이터 사용

 더 많은 데이터를 사용하면 같은 모델의 적은 데이터를 사용한 학습모델과 비교해 과적합할 가능성이 적음.

