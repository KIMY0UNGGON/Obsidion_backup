Maximum Likelihood estimation

 훈련데이터에 가장 높은 확률을 부여하는 Parameter를 선택하는 것.
 즉 모델을 훈련시킬때 나온 결과가 Label(정답)에 가까운 Parameter를 찾는 것.

Likelihood:
	 $p(D|\theta)$  => Parameter가 $\theta$일때 데이터가 나올 확률. 

MLE:

 $p(D|\theta)$ 일때 이 함수, 확률을 1에 가깝게 하는 $\theta$를 찾는 것.
정의: $\hat\theta_{mle} = argmax_\theta\,p(D|\theta)$

즉 Parameter를 계속 바꾸면서 정답을 찾을 확률이 더 높은 Parameter를 찾는 과정.

모델을 만들때 우리는 데이터들이 i.i.d를 따른다고 가정함.
 i.i.d => Independent and Identically Distrubuted
  독립적이고 **동일한** 분포를 따른다는 뜻.

 즉 학습을 할 **데이터셋의 데이터들이 모두 독립적이고 동일한 분포**를 따른다고 가정을 함.
 D = 데이터셋

**데이터들은 독립**이므로 데이터셋의 데이터들이 서로 동시에 일어날 확률은 데이터셋의 확률들을 서로 곱해서 구할 수 있다. 

그래서 다음과 같은 정의가 나온다.

$p(D|\theta) = p(Data1|\theta) \times p(Data2|\theta) \dots \times p(Data_n|\theta) = \prod_{i=1}^{n} p(Data_i|\theta)$ => 정답일 확률

 데이터 셋은 동일한 분포를 가진다고 가정하므로, 동일한 형태의 확률함수를 사용한다.
 예를 들어 주사위 던지기의 경우 같은 데이터 셋에선 동일한 주사위를 사용한다.
  어떤 데이터는 6면 주사위, 다른 데이터는 11면 주사위를 던지지 않는다.
  어떤 데이터가 6면 주사위를 던지면 다른 데이터들도 모두 6면 주사위를 던지는 것이다.


보통은 위의 Likelihood에서 **Log를 붙여서 사용**한다.
  Log를 사용하는 이유는 확률들의 값이 0~1 사이의 값들이라 계속 곱하면 Likelihood가 0에 가까워지기 때문에 이를 방지하기 위함이다.

Log Likelihood는 다음과 같이 정의된다.

 $log\ p(D|\theta) = \sum_{n=1}^{N}log\,p(Data_n,\theta)$ 
  $p(Data_n,\ \theta) = p(y_n|x_n,\ \theta)$로도 표현 가능하다. $y_n|x_n$은 출력값|입력값으로, 입력데이터가 $x_n$일때 출력데이터가 $y_n$일 확률이다.

위의 **Log Likelihood를 최대화** 하는것이 MLE.
 식은 다음과 같다.

$\hat\theta = argmax_\theta\,log\,p(D|\theta) = argmax_\theta\,\sum_{n=1}^{N}log\,p(Data_n,\theta)$

대부분의 최적화 알고리즘에서는 cost function을 최소화하도록 설계함.

cost function(비용함수): **모델이 얼마나 잘못 예측했는지**를 측정하는 기준이 되는 함수.

그러므로 Log Likelihood에 negative를 씌워서 argmin을 찾도록 변경함
Log Likelihood에 negative를 씌운것을 Negative Log Likelihood(NLL)이라 함.
식은 다음과 같음.

$NLL(\theta) = -log\,p(D|\theta) = -\sum_{n=1}^{N}log\,p(Data_n|\theta)$

이 NLL에 argmin을 취하면 MLE를 구할 수 있음
식은 다음과 같음.

$\hat\theta = argmin_\theta-\sum_{n=1}^{N}log\,p(Data_n|\theta)$


  