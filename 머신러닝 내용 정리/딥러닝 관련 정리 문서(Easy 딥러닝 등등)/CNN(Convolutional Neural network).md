


탄생 배경: Overfitting과 관련되어 있음.
 DNN은 깊으면 깊을 수록 여러함수로 나타낼 수 있어 자유도가 너무 높아짐. 그래서 학습데이터의 에러를 0으로 만들 수 있음 -> 이게 Overfit

 여기서 문제가 새로운 데이터에 대해서 너무 멍청한, 무지한 상태가 됨.
 ex) 강아지 사진을 집어넣었을 때 학습한 강아지 사진이 아니면, 새로운 강아지 사진이면 강아지가 아니라고 인식을 해버림

 fully-connected layer는 너무 비선형으로( 너무 복잡하고, 자세하게 ) 생각.

 효율적이고 Overfit하지 않는 머신을 찾기 위해 탄생함. -> Convolutional로 구조를 바꾸자.
 조금은 융통성있는 뉴럴 네트워크를 만들자.

Covolutuion(합성곱) : 뒤집고 밀면서 곱하고 더했다(?) 아직 이해 못함
 데이터 X가 있을때 어떤 행렬 h가 있다고 가정.

<figure>
<img src = "img/CNN_img.png">
</figure>

이 행렬 h를 x와 곱하고 더하고를 계속 반복함.


<figure>
<img src="img/cnn_anime.gif">
</figure>
결과는 3x3행렬이 나오게 됨


<figure>
<img src="img/cnn_filter.png">
</figure>

x filter 같은 경우는 좌우의 차이가 크면 클수록 결과값이 커짐.
만약 결과 값이 똑같으면 곱하고 더했을 때 값이 0이 되어  버림.

y filter는 위 아래의 값의 차이가 크면 결과값이 커짐.

x,y필터를 통과시키면 윤곽을 알수가 있음.

인간은 사물이나 생물을 윤곽을 먼저 보고 생각하고 인지를 함.
Convolution이라는 건 이미지의 특징을 뽑는 다는것. 그 특징들을 뽑아서 학습을 시키면 그냥 이미지를 통채로 집어넣은 것보다 효과적일 것이라고 기대해서 사용.

image의 feature들을 뽑아서 입력값으로 넣어주면 overfit 문제를 해결할 수 있지 않을까 라고 나온게 CNN


Stride : 지금은 convolution을 할때 h를 한칸씩 이동하고 있으 