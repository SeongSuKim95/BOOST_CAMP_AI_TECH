# Week 5 : CV basic 

# Contents 

1. Studies 

- Object Detection

    1. Focal loss는 object detection에만 사용될 수 있을까요?
    2. CornerNet / CenterNet은 어떤 형식으로 네트워크가 구성되어 있을까요?

- CNN Visualization
    
    1. 왜 filter visualization에서 주로 첫번째 convolutional layer를 목표로할까요?
    2. Occlusion map에서 heatmap이 의미하는 바가 무엇인가요?
    3. Grad-CAM에서 linear combination의 결과를 ReLU layer를 거치는 이유가 무엇인가요?

- Instance Panoptic Segmenetation

    1. Mask R-CNN과 Faster R-CNN은 어떤 차이점이 있을까요? (ex. 풀고자 하는 task, 네트워크 구성 등)
    2. Panoptic segmentation과 instance segmentation은 어떤 차이점이 있을까요?
    3. Landmark localization은 human pose estimation 이외의 어떤 도메인에 적용될 수 있을까요?

- Multi-modal

    1. Multi-modal learning에서 feature 사이의 semantic을 유지하기 위해 어떤 학습 방법을 사용했나요?
    2. Captioning task를 풀 때, attention이 어떻게 사용될 수 있었나요?
    3. Sound source localization task를 풀 때, audio 정보는 어떻게 활용되었나요?

- Recent Trends on Vision Transformer
    1. Transformer 기본 구조
    2. ViT applications
    3. Questions
        - ZeroCap에서 이미지에 적합한 caption을 생성하기위해, CLIP 정보를 어떻게 활용했나요?
        - Transformer의 어떤 특징이 Unified Model 구성을 용이하게했나요?

2. 과제 
    - 과제 4 : Conditional Generative Adverserial Network
    - 과제 5 : CLIP
---

# Studies
## Object Detection

>1. Focal loss는 object detection에만 사용될 수 있을까요?

- 강의에서도 설명되었듯이, Focal loss는 YOLO,SSD와 같은 one-stage detector의 성능 개선을 위해 고안되었다. (One stage detector란, localization과 classification을 동시에 처리하는 detector를 의미)
- Localization을 먼저 수행한 후, classification을 진행하는 two-stage detector와 달리(ex: RCNN) one-stage detector는 배경에 대한 negative bbox의 수가 많아서, 이미지 속 객체에 대한 positive bbox의 수가 이보다 현저히 적음을 고려할 떄 학습 중 class imbalance 문제를 야기한다. (Dense sampling of anchor box에 의한 현상)
- 또 한가지 중요한점은, 배경에 대한 bbox들은 모델 입장에서 **easy negative** 라는 점이다. 따라서, 이들은 모델이 높은 확률로 객체가 아님을 잘 구분할 수 있기 때문에 각각이 loss에 끼치는 영향력은 작지만, 그 수가 많기 때문에 모델 입장에선 불필요한 학습이 증가하여 학습의 효율을 저하한다.
- 질문으로 돌아가서, 결국 Focal loss는 class imbalance와 easy negative의 영향력을 줄이기 위해 도입된 것이다. 이것이 object detection이 아닌 다른 상황에 대해 어떻게 활용될 수 있는지를 이해하기 위해선, cross-entropy의 단점이 무엇이었는지 부터 생각해보아야 한다.
    
    ## Cross Entropy Loss의 문제점
    - Cross Entropy loss(CE)는 모델이 옳게 분류한 경우 보다 잘못 예측한 경우에 대해서 패널티를 부여하는 것에 더 초점이 맞추어진 loss이다.
    $$ CE = -Y_{gt}log(Y_{pred}) - (1 - Y_{gt})log(1-Y_{pred})$$
    - $Y_{gt}$가 1일 경우
        - $Y_{pred} = 1$ 이면 CE = 0 이 된다. 즉, 모델이 제대로 예측한 경우에 대해선 아무 보상도 패널티도 없다.
        - $Y_{pred} = 0$ 이면 CE = inf 가 된다. 즉, 패널티가 매우 커지게 된다.
    - 또한 Easy negative, 즉 gt가 0인 배경에 대해 매우 낮은 확률의 값으로 예측하는 경우를 살펴 보자.
    - 예를 들어, Foreground의 객체($Y_{gt}$ = 1)인 경우에 모델이 0.95의 확률로 예측하고, Background의 오탐($Y_{gt}$= 0)에 대해 0.05의 확률로 예측했다면 두 경우의 loss 값은 -log(0.95)로 동일하다. 즉, **정답을 근사하게 맞추는 행위의 학습에 끼치는 영향력과 매우 낮은 확률의 오탐의 끼치는 영향력이 같기 때문에**, 오탐이 많을수록 객체에 대한 학습에 집중하기 어려워 지는 것이다. 

    ## Balanced Cross Entropy loss
    - Negative의 수가 많은 것이 문제라면, 이를 해결하기 위해 loss에 가중치를 부여하면 되지 않을까? Balanced CE는 loss 자체에 각 class에 대한 비율을 조절하는 weight w를 곱해주어, 우리가 집중해야하는 class가 loss에 끼치는 영향력이 더 크도록 유도한다.
    $$ Balanced CE = -w_{i}log(p_{i}) $$ 
    - 이는 class sample의 수에 따른 imbalance 문제는 해결할 수 있으나 각 class의 성질, 즉 model 입장에서 예측하기 쉬운 class인지, 어려운 class인지를  고려하지는 못한다.  **각 class의 sample의 수에 대한 비율이, 각 class를 예측하는 확률적 난이도에 비례하지는 않는다는 것이 포인트이다.** 예를 들어, 강아지, 고양이,사자,사과를 classification한다고 가정할 때, 다른 class는 100개씩 있는데 사과 sample은 한개만 있는 상황이라고 해도 모델은 사과를 높은 확률로 예측해낼 수 있을 것이다.

    ## Focal Loss
    - 위 모든 논점들을 고려하여 Focal Loss는 다음과 같이 구성된다.
    $$FocalLoss = -{{\alpha}_{i}}(1-p_{i})^{\gamma}log(p_{i})$$
    
    - <p align="center"><img src=https://user-images.githubusercontent.com/62092317/196978796-b03706f7-9071-4ebe-bc9e-d7196f2ebc63.png width = "500"></p>
        
        1. 위 그래프는 $\gamma$ 가 0 ~ 5로 변화할 때 loss의 변화를 나타내며, 0일때 CE와 같고 값이 커질수록 easy example에 대한 loss가 줄어든다.
        2. 오분류된 확률 $p_{i}$가 작아지게 되면 $(1-p_{i})^{\gamma}$도 1에 가까워지고, $log(p_{i})$도 커져서 loss에 반영된다.
        3. $p_{i}$가 1에 가까워지면 $(1-p_{i})^{\gamma}$은 0에 가까워지고 CE와 동일하게 $log(p_{i})$도 줄어든다.
        4. ${\gamma}$를 focusing parameter라고 하며, easy example에 대한 Loss의 비중을 낮추는 역할을 한다.
        5. Easy example에 대해 가중치를 더 많이 약화시킴으로써 이들이 학습에 끼치는 영향력을 낮출 수 있다.
- 결론 : **Focal loss의 이런 성질을 고려할 떄, detection task 뿐만 아니라 data imbalance가 심한 dataset에 대한 classification, 또는 data sample의 variation이 커서 hard, easy sample의 분포가 고르지 않을 경우 사용하면 효율적인 학습을 할 수 있을 것이라고 생각한다.**

>2. CornerNet / CenterNet은 어떤 형식으로 네트워크가 구성되어 있을까요?[[참고 LINK]](https://deep-learning-study.tistory.com/622)
- CornerNet과 CenterNet은 bounding box를 정의하는 방식에서 차이가 있는데, CornerNet의 경우 (Top-left,Bottom-right) 두 corner로 bounding box를 정의하지만 CenterNet의 경우 (Top-left,Botton-right,center point) 또는 (Width, Height, Center)로 정의된다.

- <p align="center"><img src=https://user-images.githubusercontent.com/62092317/197157914-293e2ea1-d4ec-4db0-933f-3ddeeb10f3c7.png width = "500"></p>

- CorneNet은 속도가 빠르지만, 두 쌍의 특징점으로만 bounding box를 예측하기 때문에 단점이 존재한다.
    1. 객체의 전체 정보를 활용하는데에는 상대적으로 제한되고,
    2. Bounding box의 경계를 예측하는 데에만 집중이 되어있고, 어느 특징점이 연결되어야 하는지에 대한 정보가 부족하여 위 그림 처럼 많은 수의 incorrect한 box를 생성한다.

- <p align="center"><img src=https://user-images.githubusercontent.com/62092317/197158908-a275c67a-9d4a-4c70-b69f-cab41c0f150d.png width = "700"></p>

- 이러한 단점을 해결하기 위해 CenterNet에서는 두 쌍의 특징점이 생성한 영역에서 중심점 정보를 추가로 활용한다. **두 쌍의 특징점으로 예측한 bounding box의 중심 구역에 존재하는 중심점이 실제로 ground truth와 동일한 class를 지니는지 확인하여 최종 결정을 내리는 것이다.** 위 그림에서 기존 cornernet이 생성한 3개의 bounding box중 center point가 실제로 말인 box는 하나이므로, 이를 통해 incorrect한 box들을 제외할 수 있다.

## CNN Visualization

<p align="center"><img src=https://user-images.githubusercontent.com/62092317/196985434-acb6846b-ca89-4b6e-86f8-028486f07017.png
></p>

>1. 왜 filter visualization에서 주로 첫번째 convolutional layer를 목표로할까요?
- 3 channel의 input image를 받기 떄문에, filter의 채널도 3개여서 이후 layer들의 high dimension filter들에 비해 visualize가 용이하다. 해석 가능한 정보를 얻을 수 있는 것이 중요하기 때문에!!
- 주의점 : **Filter visualization과 Activation visualization을 구분하자**

>2. Occlusion map에서 heatmap이 의미하는 바가 무엇인가요?

<p align="center"><img src="https://user-images.githubusercontent.com/62092317/197089400-ccda02aa-3b18-4f0d-84be-73c6d271863d.png" width = "600"></p>

- Saliency test는 모델이 이미지의 어느 부분에서 정보를 많이 얻는지를 판단하기 위해 이미지의 특정 영역을 위 그림과 같이 가린 상태에서 class를 예측하게 한다.
- 만약 가려진 부분이 image의 class를 판별하는데 중요한 근거를 담고 있는 영역이라면 모델의 prediction score가 급격하게 감소하게 된다. 예를 들어, 위 그림의 경우 코끼리 사진의 배경 부분을 가릴때 보다 코키리의 얼굴 부분을 가릴때 모델의 prediction score가 현저히 낮아지는 것을 알 수 있다. 
- Occluded 된 각 영역에 대해, prediction score의 변화를 측정하고 이를 하나의 heatmap 형식으로 나타낸다면 이미지의 중요한 정보가 어디에 분포하고 있는지 알 수 있다. 위 그림의 heatmap에서 빨간색으로 표시된 부분은 score가 급격히 변하는 구간은 이미지상에서 코끼리가 있는 영역과 일치하는 것을 알 수 있다.

>3. Grad-CAM에서 linear combination의 결과를 ReLU layer를 거치는 이유가 무엇인가요? 
- <p align="center"><img src="https://user-images.githubusercontent.com/62092317/197148233-05d53a65-65fe-4bc5-bf07-9d3d587efe77.png" width = "500"></p>
- <p align="center"><img src="https://user-images.githubusercontent.com/62092317/197150018-e510058d-418e-43f5-bdc2-a1dc602232ac.png" width = "500"></p>
- Grad-CAM은 기존의 CAM(Class activation mapping)이 모델의 classification단을 GAP(global average pooling)로 바꿔야 사용할 수 있었던 것과 달리, original network를 변형하지 않으면서 마지막 CNN layer의 출력 feature의 각 channel에 대해서 위 그림과 같은 연산을 수행한다.
- **각 k-th 채널 맵에 대해서, (i,j)번째 feature의 class c 에 대한 gradient를 구한 후 Global average pooling한 값을 importance weight로 사용**하여 해당 채널의 feature들이 class c와 얼마나 연관되어 있는지를 판단한다.
- 이렇게 구한 각 채널에 대한 weight값을 feature map 각각에 대해 곱하여 k개의 map을 weighted summation을 한 후, ReLU를 통과 시켜 Grad-CAM map을 visualize하는 것이다.
- ReLU를 통과할 경우, 각 이미지의 (i,j)위치에 대해 이전 단계에서 weighted summation한 값이 양수인 경우에만 출력이 되므로, 가장 유의미한 값만을 visualize할 수 있다. 따라서, ReLU를 사용하는 이유는 CAM의 기본 목적에 부합하여 activation map을 효율적으로 visualize하기 위함이라고 생각된다. ReLU를 사용하지 않을 경우, 목표하는 객체 이외의 activation 결과들을 visualize상에서 제외시킬 수 없기 떄문이다.
- [[참고 LINK]](https://medium.com/@ninads79shukla/gradcam-73a752d368be)
    > ReLU is the preferred choice in this case as it highlights features having positive influence on the class of interest. **Regions of interest implicitly refer to those pixels whose intensity varies directly with the gradient yc. Without the use of ReLU, it is observed that localisation maps sometimes might include more information than the desired class like negative pixels that probably belong to other categories in the image hence affecting localization performance.**

## Instance Panoptic Segmenetation

>1. Mask R-CNN과 Faster R-CNN은 어떤 차이점이 있을까요? (ex. 풀고자 하는 task, 네트워크 구성 등)
- Mask R-CNN은 기존 object detection task에서 사용되던 Faster R-CNN에 Mask branch를 추가한 구조로 **classification, bounding box regression, predictiing object mask를 동시에 처리하는 모델**
- Mask branch는 작은 크기의 FCN으로 픽셀 wise로 K개의 class 각각에 대해 물체가 있는지 없는지를 판단하는 segmentation mask를 예측
- 추가로, Faster RCNN의 ROI pooling 을 **ROI align layer**로 대체해 물체의 spatial location 즉, 위치정보를 보존
- **Mask Branch의 역할과 ROI Align에 대하여** [[참고 LINK]](https://blahblahlab.tistory.com/139#:~:text=MASK%20RCNN%EC%9D%80%20%EA%B8%B0%EC%A1%B4%20object,%EB%8F%99%EC%8B%9C%EC%97%90%20%EC%B2%98%EB%A6%AC%ED%95%98%EB%8A%94%20%EB%AA%A8%EB%8D%B8%EC%9E%85%EB%8B%88%EB%8B%A4.)
    - Mask Branch
        <p align="center"><img src="https://user-images.githubusercontent.com/62092317/197340309-671892e9-1c26-4800-8d6a-fe4d18c2cb20.PNG" width = 500></p>

        - Mask branch에서는 각각의 ROI에 대해 각각의 클래스 k에 대한 binary mask를 생성 (Binary mask : 클래스 k에 대한 instance가 있다고 생각되면 1, 없으면 0으로 표현)
        - 기존 모델들의 경우 mask와 class prediction을 동시에 수행, 즉 pixel 각각에 대해 모든 class에 대한 확률을 얻은 반면, mask branch에서는 mask와 class prediction이 분리된 셈이다. 이를 논문에서는 "decouple"이라고 표현했다.
        - 하나의 픽셀에 대해 k(class 수)개의 output을 출력하므로, feature map M x M 에 대해 각각의 ROI는 $kM^2$개의 output을 출력하게 된다. 이렇게 생성된 mask는 object의 spatial한 정보를 담고 있다.
    
    - ROI Align
        <p align="center"><img src="https://user-images.githubusercontent.com/62092317/197341360-365c4a76-f5cf-4a43-a6b4-849a9bc854be.PNG" width = 300></p>
        
        <p align="center"><img src="https://user-images.githubusercontent.com/62092317/197341362-2eaf03c4-0666-478a-b773-28bf07c406c2.PNG" width = 300>
        
        - 위 그림과 같이 ROI가 소수점 단위로 pixel의 좌표를 지정할 때, quantization(정수화,반올림)을 하면 정보의 손실과 동시에 불필요한 정보를 포함하게 된다.
        - 이런 현상은 pixel 단위로 segmentation하여 mask를 예측하는 과정에서 안좋은 영향을 끼치게 된다.
        <p align="center"><img src="https://user-images.githubusercontent.com/62092317/197341896-b1fcc568-7227-4dc1-a46a-223a2f27ccfc.PNG" width = 500>

        - 이러한 단점을 해결하기 위해 bilinear interpolation을 활용하여, ROI에 걸친 pixel의 수 만큼의 좌표, prediction 값을 딱 맞게 생성한다.[[참고 LINK]](https://firiuza.medium.com/roi-pooling-vs-roi-align-65293ab741db)
        - **기존 방식이 image의 고정된 pixel 좌표에 맞추어 ROI를 제단한 것과 반대로, ROI에 맞추어 필요한 pixel(소수점)의 좌표를 격자 형식으로 할당한 후 보간법을 사용하여 해당 위치의 prediction 값을 구했다는 점에서 신선하다고 느꼈다.**

>2. Panoptic segmentation과 instance segmentation은 어떤 차이점이 있을까요?
- Semantic vs Instance vs panoptic segmentation [[참고 LINK]](https://pyimagesearch.com/2022/06/29/semantic-vs-instance-vs-panoptic-segmentation/)

- 이 세 방식의 차이는 **things(셀 수 있는 class)** 와 **stuff(셀 수 없는 class)**를 어떻게 다루느냐에서 기인한다.
    - Semantic segmentation은 객체의 class만을 구분할 뿐, 객체 각각을 구분하지는 않는다. 하늘이나 길과 같은 물체로 셀 수 없는 class도 분할하여 인식한다.
    - Instance segmentation은 객체의 class뿐만 아니라, 객체 각각의 경계를 구분하여 물체가 겹친 경우에도 이를 구분하여 셀 수 있도록 인식한다. 그러나, semantic segmentation과 달리 하늘,도로 등 정해진 형태가 없는 물체의 경우엔 라벨을 부여하지 않는다. 
    - **Panoptic segmentation은 두 방식을 결합한 방식으로 셀 수 있는 class(ex:차,사람)에 대해선 instance segmentation을 , 셀 수 없는 class(ex:하늘,도로)에 대해서는 semantic segmentation을 수행한다.**

>3. Landmark localization은 human pose estimation 이외의 어떤 도메인에 적용될 수 있을까요?

- Landmark localization이란 특정 물체에 대해 중요시되는 특징점(keypoint)들을 추정하는 task이다.
    - Coordinate regression : 좌표를 추정, Inaccurate and biased
    - Heatmap classification : 각 채널을 하나의 keypoint에 대한 확률맵으로 표현하여 추정, 성능은 개선되지만 계산량이 크다.
- Pose estimation 뿐만 아니라, facial landmark detection와 같이 사람 얼굴 이미지로부터 중요한 특징점(ex: 눈,코,입)등의 위치를 추정하는 분야에도 사용될 수 있다.

## Multi-modal

>1. Multi-modal learning에서 feature 사이의 semantic을 유지하기 위해 어떤 학습 방법을 사용했나요?

- Joint embedding이란?
    - Matching을 하기 위한 공통된 embedding vector를 학습하는 것 
    - 각각의 unimodal model에서 나온 feature vecture를 같은 embedding space로 투영시켜, modal간 관계성을 학습하도록 유도
    - ex: Image tagging
        <p align="center"><img src="https://user-images.githubusercontent.com/62092317/201482054-4232d52c-3e9a-4df6-bd76-da662cfd184d.png" width = 300 height=250></p>
        
        - visual feature와 text feature의 semantic 정보가 유지된채로 같은 embedding space에서 서로의 유사성에 따라 다르게 투영된다.
        
>2. Captioning task를 풀 때, attention이 어떻게 사용될 수 있었나요?

- Image captioning as image-to-sentence (Show and tell)
    - Input의 image를 표현할 때는 CNN, sentence를 출력할 때 RNN을 사용
    - Soft attention을 활용하는 기법
        1. CNN으로 부터 나온 feature map(channel map)으로 부터 attention이 필요한 부분(heat map 기반 spatial attention)을 판단하여 RNN에 전달
        2. RNN은 전달 받은 attention 정보를 condition으로 사용하여 word feature와의 연관성을 파악하여 단어를 순서대로 출력 한다.

>3. Sound source localization task를 풀 때, audio 정보는 어떻게 활용되었나요?

<p align="center"><img src="https://user-images.githubusercontent.com/62092317/201482769-523b85a0-501b-4622-80c1-2853c8bb1cef.png"width = 400></p>

- Video frame과 audio 정보 각각이 visual net과 Audio net에 통과 된 후, attention network에 전달 된다.
- Spatial 정보를 담은 visual feature map 각 위치의 video feature vector와 audio feature간에 내적을 통해 attention을 구한다.(이미지 각 위치와 소리 정보와의 연관성)
- 이를 통해 구해진 sound localization score와 visual spatial feature vector를 element wise로 곱하여 attended visual feature를 구하고, metric learning을 이용하여 전체 network를 학습 한다.

## Recent Trends on Vision Transformer

1. Transformer 기본 구조
- Self-attention(Q,K,V)
    
    <p align="center"><img src="https://user-images.githubusercontent.com/62092317/201505394-f672500c-921e-49b2-939f-7aa5d5c48916.png" width=400></p>

    - 같은 Input vector에 대해 Query,Key,Value로 transformation 이후, Query와 Key간의 내적을 통해 관계성을 파악하고 attention score를 얻음
    - Value와 attention score간 matrix multiplication을 통해 weighted summation으로 aggregation
    - Multi-head attention: 여러 개의 attention head를 통해 input token사이의 다각도의 관계성을 고려
- Positional encoding
    - Self-attention이 order-invariant하기 때문에 token의 순서에 대한 정보를 embedding하여 각 token에 더해줌
- Encoder & Decoder
    - Encoder : MHA, FFN
    - Decoder : Masked multi-head attention

2. ViT application - DETR : End-to-End Object Detection with Transformers [[참고자료 Medium]](https://medium.com/visionwizard/detr-b677c7016a47)

<p align="center"><img src="https://user-images.githubusercontent.com/62092317/201513073-d8793de2-d62d-4de7-abe4-1032f9ac496b.png" width=500></p>

- Object detection 문제를 set prediction 문제로 치환
- 기존 detection model들의 hand-designed components(ex:non-max suppression) 없이 detection 수행
- CNN을 사용하여 feature를 추출
- Encoder, Decoder 구조를 함께 사용
    - Encoder
        - CNN에서 나온 feature를 1x1 convolution으로 채널수를 조정한 후 transformer encoder에 넣는다.
        - CNN feature를 바로 사용하지 않는 이유는, receptive field를 고려할 때 global한 특성을 담지 못하기 때문
    - Decoder
        - Object queries를 특정 위치에 대한 object 존재여부를 학습
- Prediction heads
    - N bounding boxes(N >> #actual objects in an image)를 예측
    - No object(background class 개념과 유사)를 표현하기 위해 additional special class label을 사용
- Prediction heads에서 나온 bounding box들 각각에 대해 ground truth bbox에 대한 bipartite matching을 하여 loss를 계산

3. Questions
    > ZeroCap에서 이미지에 적합한 caption을 생성하기위해, CLIP 정보를 어떻게 활용했나요?

    <p align="center"><img src="https://user-images.githubusercontent.com/62092317/201513643-476280a9-fdfc-462a-b6ec-d924ca8b1b6f.png" width=500></p>

    - ZeroCap : zero-shot image-to-text generation for visual-semantic arithmetic
        - Visual semantic model인 CLIP과 large language model인 GPT-2를 결합 하여 image captioning을 수행
        - Encoder를 통과한 image feature와 단어의 context feature를 동시에 받아 두 정보가 얼마나 align 되는지를 판단
            - $L_{clip}$ : Stimulates the model to generate sentences that describe the given image
         
    > Transformer의 어떤 특징이 Unified Model 구성을 용이하게했나요?

    - Unified transformers이란?
        - Single purpose를 가지고 특정 modality에 대해 하나의 task를 수행하는 것이 아니라, transformer의 flexibility를 이용하여 input,output representation을 handle함으로써 하나의 통합된 model로 다양한 task를 처리
    - **Transformer는 modality에 상관 없이 Input을 tokenize하기만 한다면 그들간의 관계성을 고려하도록 네트워크를 학습하면 되므로, cross-modal간의 generalization이 가능한 unified model로 사용될 수 있다.**
    - Tokenize를 통해 모델 구조의 변경 없이 다양한 input, output representation을
    handle할 수 있다.
# 과제 

- **과제 4 : Conditional Generative Adverserial Network**
    - Random noise vector와 condition label의 embedding dimension을 임의로 150,50으로 설정하고 (concat시 200) generator, discriminator 둘 다 하나의 linear layer만을 통과시켜 실험해보았다.
    ``` python
    ## 처음 설계한 모델
    class Generator(nn.Module):
        # initializers
        def __init__(self):
            super(Generator, self).__init__()
            self.fc1_1 = nn.Linear(100,150) # random noise vector를 위한 fc 
            self.fc1_2 = nn.Linear(10,50) # condition label을 위한 fc
            self.relu_z = nn.ReLU()
            self.relu_y = nn.ReLU()
            ## fill ##

            self.fc_out = nn.Linear(200, 784)
            self.Tanh = nn.Tanh()
        # weight_init
        def weight_init(self, mean, std):
            for m in self._modules:
                normal_init(self._modules[m], mean, std)

        # forward method
        def forward(self, input, label):
            
            noise = self.relu_z(self.fc1_1(input))
            condition_label = self.relu_y(self.fc1_2(label))
            
            x = self.fc_out(torch.cat([noise,condition_label],dim=1))
            x = self.Tanh(x)
            ## fill ##
            return x

    class Discriminator(nn.Module):
        # initializers
        def __init__(self):
            super(Discriminator, self).__init__()
            self.fc1_1 = nn.Linear(784, 150)
            self.fc1_2 = nn.Linear(10, 50)
            self.relu_x = nn.ReLU()
            self.relu_y = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
            ## fill ##
            self.fc_out = nn.Linear(200, 1)
        # weight_init
        def weight_init(self, mean, std):
            for m in self._modules:
                normal_init(self._modules[m], mean, std)

        # forward method
        def forward(self, input, label):

            real = self.relu_x(self.fc1_1(input))
            condition_label = self.relu_y(self.fc1_2(label))
            ## fill ##
            
            x = self.fc_out(torch.cat([real,condition_label],dim=1))
            x = self.sigmoid(x)
            return x
    ```
    - 한개의 linear layer로 학습시 50 Epoch 학습 결과
    - 
    <p align="center"><img src="https://user-images.githubusercontent.com/62092317/196613023-15f29fdc-91a4-48ed-9098-bfa6e58e4810.png" width = "300"></p>

    - Linear layer를 하나만 사용할 경우, model의 representation power가 제한되다보니 생성되는 이미지가 정확하지 않아서, 원 논문의 구현을 참고하여 다시 실험해보았다.[[참고 LINK]](https://deep-learning-study.tistory.com/m/640)

        ``` python
        # 논문을 참고하여 구현한 코드
        class Generator(nn.Module):
            # initializers
            def __init__(self):
                super(Generator, self).__init__()
                self.fc1_1 = nn.Linear(100,100) # random noise vector를 위한 fc 
                self.fc1_2 = nn.Linear(10,10) # condition label을 위한 fc
                ## fill ##
                
                self.gen = nn.Sequential(
                    nn.Linear(110, 128),
                    nn.LeakyReLU(0.2),
                    nn.Linear(128,256),
                    nn.BatchNorm1d(256),
                    nn.LeakyReLU(0.2),
                    nn.Linear(256,512),
                    nn.BatchNorm1d(512),
                    nn.LeakyReLU(0.2),
                    nn.Linear(512,1024),
                    nn.BatchNorm1d(1024),
                    nn.LeakyReLU(0.2),
                    nn.Linear(1024,784),
                    nn.Tanh()
                )
            # weight_init
            def weight_init(self, mean, std):
                for m in self._modules:
                    normal_init(self._modules[m], mean, std)

            # forward method
            def forward(self, input, label):
                
                gen_input = torch.cat((self.fc1_1(input),self.fc1_2(label)),dim=1)
                x = self.gen(gen_input)
                return x

        class Discriminator(nn.Module):
            # initializers
            def __init__(self):
                super(Discriminator, self).__init__()
                self.fc1_1 = nn.Linear(784, 784)
                self.fc1_2 = nn.Linear(10, 10)

                ## fill ##
                self.dis = nn.Sequential(
                    nn.Linear(794,512),
                    nn.LeakyReLU(0.2),
                    nn.Linear(512,512),
                    nn.Dropout(0.4),
                    nn.LeakyReLU(0.2),
                    nn.Linear(512,512),
                    nn.Dropout(0.4),
                    nn.LeakyReLU(0.2),
                    nn.Linear(512,1),
                    nn.Sigmoid()
                )
            # weight_init
            def weight_init(self, mean, std):
                for m in self._modules:
                    normal_init(self._modules[m], mean, std)

            # forward method
            def forward(self, input, label):

                dis_input = torch.cat((self.fc1_1(input),self.fc1_2(label)),dim = 1)
                x = self.dis(dis_input)
                return x
        ```
    - 내가 처음 구현했던 방식과 논문의 구현방식 간엔 4가지의 차이점이 있다.
        - Embedding layer의 입출력 dimension이 동일하다는 것 (noise vector 100 to 100, label vector 10 to 10) 
        - Generator와 Discriminator는 각각 5개의 linear layer를 사용하여 representation power를 높였다.
        - Generator는 layer를 거치면서 dimension의 크기를 2배씩 키워나가는 반면, discriminator는 512로 feature size를 유지한다.  
        - Generator, Discriminator 둘 다 ReLU가 아닌 LeakyReLU를 사용하였다.
    - 논문의 구현 방식으로 50 epoch 학습한 결과, 더 정확한 이미지가 생성되었다.

    - <p align="center"><img src = "https://user-images.githubusercontent.com/62092317/197113025-233c7e6d-bb0e-4416-9553-314debc7f219.png" width = 300></p>

    - 고찰 사항
        1. Generator와 Discriminator의 loss를 관찰한 결과, Generator의 loss가 Discriminator의 loss보다 크기가 크고, 수렴성이 낮다.
            ``` python
            Train Epoch: [0/50]  Step: [3700/7500]G loss: 0.83967  D loss: 0.50499 
            Train Epoch: [10/50] Step: [3700/7500]G loss: 3.63719  D loss: 0.28119
            Train Epoch: [20/50] Step: [3700/7500]G loss: 2.94529  D loss: 0.24260 
            Train Epoch: [30/50] Step: [3700/7500]G loss: 2.63239  D loss: 0.29057 
            Train Epoch: [40/50] Step: [3700/7500]G loss: 1.21396  D loss: 0.58974
            Train Epoch: [49/50] Step: [3700/7500]G loss: 1.05710  D loss: 0.55929
            ```
            - Training loop에서 각 loss를 구해지는 부분을 살펴보면, 다음과 같이 이루어져 있다.
            ``` python
                # Adversarial ground truths
                valid = torch.ones(parser.batch_size, 1).cuda() # real image 라는 것을 표현하기 위한 label
                fake = torch.zeros(parser.batch_size, 1).cuda() # fake image 라는 것을 표현하기 위한 label

                # G Loss - Generator가 노이즈로 부터 생성한 이미지를 discriminator에 통과시켜 real image라고 판단하도록 유도하는 CE
                
                # Generate a batch of images
                gen_imgs = generator(z, gen_labels)
                
                # Loss measures generator's ability to fool the discriminator
                val_output = discriminator(gen_imgs, gen_labels)
                g_loss = cross_entropy(val_output, valid)

                # D Loss 
                # 1.Real loss : 실제 이미지를 실제라고 판단하도록 유도
            
                validity_real = discriminator(real_imgs, labels)
                d_real_loss = cross_entropy(validity_real, valid)
            
                # 2.Fake loss : 생성된 이미지를 가짜라고 판단하도록 유도
                validity_fake = discriminator(gen_imgs.detach(), gen_labels)
                d_fake_loss = cross_entropy(validity_fake, fake)

                d_loss = (d_real_loss + d_fake_loss)/2
            ``` 
            - 즉, G loss는 생성된 이미지를 진짜라고 유도하기 때문에 모델을 속이는 loss인 반면 D loss는 실제를 실제로, 가짜는 가짜로 판단하도록 유도하는 loss이다.
            - 두 loss는 서로 반대되는 방향으로 모델을 supervise 하지만, d loss중 d_real_loss를 최적화하는 것은 비교적 쉬운 일이기 때문에 수렴성이 더 나은것으로 판단된다.

        2. Generator 모델의 Final activation function은 tanh인 반면, Discriminator 모델은 Sigmoid이다.
            - Generator: Input image를 생성하는 generator의 출력을 -1~1로 생성해야 실제 이미지의 feature 분포와 일치시킬 수 있기 때문에, tanh를 쓴다.
            - Discriminator : 입력 image가 real인지 fake인지만을 1,0으로 판단하면 되기 때문에 sigmoid 함수를 사용한다.
        
- **과제 5 : Pre-trained multi modal model applications with CLIP**

    - TO DO Codes-3 
    ``` python
    # CIFAR100 dataset을 다운로드 받습니다.
    cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

    # CIFAR 100 모든 classes의 어구를 만들고 토큰화합니다.
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)

    # 임의의 input 이미지를 선정.
    image_index =  4   #  image_index 값은 0~9999 까지 입력이 가능합니다. 자유롭게 선택해보세요.
    image, _ = cifar100[image_index]
    plt.imshow(image)
    plt.show()

    # 이미지 전처리 및 feature 추출
    image = preprocess(image).unsqueeze(0).to(device) # CLIP 모델의 전처리 모듈 사용 (위 코드 참조) 

    with torch.no_grad():
        image_features = model.encode_image(image)         # 이미지 feature 추출
        text_features = model.encode_text(text)         # 텍스트 feature 추출

    # Cosine Silmilarity 계산
    similarity = cos_sim(image_features, text_features)*100       # similarity의 최대값을 100점처럼 표현하기 위해 편의상 CLIP에서는 100을 곱합니다. (위 코드 참조)
    probs = similarity.softmax(dim=-1)

    K =  3   

    values, indices =  torch.topk(probs,3)
            # softmax 함수로 가장 높은 K개의 확률값 구하기 
    print("\nTop predictions:\n") 
    for value, index in zip(values, indices):
        print(f"{cifar100.classes[index]:>16s}: {100*value.item():.2f}%") 
    ```
    - 출력 결과
    - <p align="center"><img src="https://user-images.githubusercontent.com/62092317/197120864-4a29b5e8-e858-402d-9a97-68a8dc1310c7.png" width = 350></p>
    - <p align="center"><img src="https://user-images.githubusercontent.com/62092317/197121447-bc8a17b3-dec9-454f-aea7-4f4e65c6d006.png" width = 600></p>
    - CIFAR 100의 class는 보통 동식물 또는 사물로 구성되어 있지만, 일부 class는 자연 풍경을 나타내기도 한다. 자연 풍경(large natrueal outdoor scenes) class의 image에 대해선 CLIP이 예측에 실패하는 모습을 볼 수 있었는데, 위 그림에선 **노을지는 바다**를 **사과**라고 예측하였다. 
    - 이런 결과가 나온 이유는, CLIP이 image feature 와 text feature의 contextual similarity가 **해당 사물의 대표적인 속성**에 의존도가 크기 때문으로 예상된다. 위 이미지의 class는 sea인데 sea의 경우 노을지는 바다,푸른 빛의 바다와 같이 feature의 variation이 커서 대표적인 속성이 파악하기 어려운 반면, 사과는 대부분의 sample 이미지가 붉은 색상의 feature를 포함할 것이다. 따라서, 모델 입장에선 붉은 계열의 노을 이미지에 대해, sea를 포함한 다른 class보다 apple과 similarity가 높다고 판단하게 된것으로 예상된다.
    - **즉, CLIP은 객체의 모습/색깔의 variation이 커서 이것을 대표할만한 feature의 특성이 명확하지 않은 class의 경우 어떤 text feature와도 similarity가 높게 나오지 못하여 예측이 어려울 것으로 예상된다.**
    