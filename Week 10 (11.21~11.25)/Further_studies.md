# Week 10 : Object Detection

## Contents 

- Competition

    1. EDA
        - Number of Annotations Per Image
        - Unique Class Per image
        - Annotations Per Class
        - Bbox Area as % of source image
        - Aspect Ratios for Bounding boxes of class
    2. Yolov7 [[Github]](https://github.com/WongKinYiu/yolov7) [[Paper]](https://arxiv.org/abs/2207.02696)
        - 오피스아워
        - Tuning

- Further questions : 다음 의문점들을 풀고 정리한다.

    1. Bbox loss : IoU, GIoU, DIoU, CIoU의 차이점은?

    2. Confidence threshold/ IoU threshold and mAP

    3. Yolov7 
    - Augmentation
        - Mosaic & Mixup [[Medium : YOLOX Explanation]](https://medium.com/mlearning-ai/yolox-explanation-mosaic-and-mixup-for-data-augmentation-3839465a3adf)
    - HyperParams

    4. K-fold ansemble을 하는것이 전체 dataset을 학습하는 것보다 성능이 높을 수 있나? [[About Bagging]](https://sungkee-book.tistory.com/9)

- Paper review
    - MLP mixer [[Github]](https://github.com/lucidrains/mlp-mixer-pytorch) [[Paper]](https://arxiv.org/abs/2105.01601) [[발표자료]](https://docs.google.com/presentation/d/1OrjqBGVU8ATVnXGUexgACAZma4M-Y7CL/edit?usp=share_link&ouid=112634105046419935376&rtpof=true&sd=true)

- Mentoring
    - About Dropout

- 정규식
    - 특정 두 문자 사이의 문자열 추출
---

## Competition

### 1. EDA : trash dataset
- Number of Annotations Per Image

    <p align="center"><img src="https://user-images.githubusercontent.com/62092317/203898410-1d540091-6ed6-43d2-b8ef-e428478d7e4c.png"></p>

    - 하나의 이미지 내의 평균 객체 수가 많다. 객체의 개수가 10개 미만인 case와 10~50개인 case가 거의 비슷한 비율로 데이터셋에 존재한다.
    - COCO dataset과 비교했을 때에도 한 이미지당 객체 수가 월등히 많은 것을 알 수 있다.  

    <p align="center"><img src="https://user-images.githubusercontent.com/62092317/203913667-58ff6c0a-9726-4d8a-9cd8-f572189db958.png" width= 400></p>

    - COCO의 경우 대부분의 이미지가 10개 미만의 객체를 포함하고 있고, 30개 이상의 객체가 있는 sample은 거의 존재하지 않는다.
    
    <p align="center"><img src="https://user-images.githubusercontent.com/62092317/203914297-934d0cbc-92e0-4bdb-8256-9fea18402fae.png" width = 500></p>

    <p align="center"><img src="https://user-images.githubusercontent.com/62092317/203915656-2ee99d66-e197-499e-b37e-8106efc8de82.jpg" width = 200></p>

    <p align="center"><img src="https://user-images.githubusercontent.com/62092317/203916062-d4bd7fde-dc4e-466d-b71e-09ce18163fab.jpg" width = 200></p>
    
    - COCO image dataset과 비교했을 때 trash dataset은 다수의 객체가 존재하는 sample이 많다. 그러나 객체가 1개만 존재할 경우, 대부분의 객체가 이미지의 중앙과 가깝게 위치한다.

- Unique Class Per image
    
    <p align="center"><img src="https://user-images.githubusercontent.com/62092317/203898522-d5f6a7a5-48cb-4cb1-bd1a-0a396dd71277.png"></p>

    - Trash dataset은 한 이미지당 객체의 수가 많음과 동시에, 한 이미지당 객체의 종류도 많다. 또, 쓰레기의 특성상 쌓여있거나 널부러져 있는 등 객체의 위치가 정렬되어 있지 않다 보니 bounding box의 독립성도 보장되지 않는 경향성이 있다. 

- Annotations Per Class

    <p align="center"><img src="https://user-images.githubusercontent.com/62092317/203898640-45281447-bf5f-46db-bcf0-16eadded1004.png" width = 300></p>

    - Data Imbalance가 존재하며, battery의 개수가 유독 적다.
    - Battery의 경우, 물체의 크기는 작지만 실제 이미지를 살펴보면 확대되어 찍혀있는 경우가 많다.
    - Class 중 **General trash** 는 **일반쓰레기**이기 때문에 다양한 종류의 쓰레기를 하나의 class로 묶은 것 같은데, feature의 다양성을 model이 학습할 수 있나?
    - Label smoothing, focal loss를 classifcation loss로 사용해봐야 할 듯

- Bbox Area as % of source image

    <p align="center"><img src="https://user-images.githubusercontent.com/62092317/203911089-8823ac2b-42a5-4f98-9f81-161edf6b76cd.png"></p>

    - Clothing, battery를 제외하고는 전체적으로 비슷한 bbox area를 가진다.
    - 모든 class에서 객체 크기의 편차가 큰 이유
        - 객체가 1개만 있는 경우 : 물체가 중앙에 오며 확대되어 찍힌다.
        - 객체가 여러개 있는 경우 : 전체 쓰레기가 찍히면서 객체 하나 하나는 작게 찍힌다.
    - 따라서 같은 class의 객체라고 비슷한 크기를 가진다는 가정을 할 수 없다.

- Aspect Ratios for Bounding boxes of class

    <p align="center"><img src="https://user-images.githubusercontent.com/62092317/203911224-b66e16b9-a811-4385-892f-76af35ca253d.png"></p>

    - Aspect ratio가 1에서 크게 벗어나지 않고, 모든 class가 비슷한 aspect ratio를 가진다.
    - 쓰레기들의 형태상의 다양성은 크지 않은 편이다.

---
## Further Questions: 다음 의문점들을 풀고 정리한다.

### 1. Bbox loss : IoU, GIoU, DIoU, CIoU의 차이점은?
- IoU
    - Intersection over Union으로 OD 분야에서 prediction bbox와 ground truth가 일치하는 정도를 0과 1사이의 값으로 나타낸 것
    - **IoU가 왜 필요할까?**
        
        <p align="center"><img src="https://user-images.githubusercontent.com/62092317/204812955-00dda64a-5e61-4f35-bb34-05c18b79068d.png"></p>
        <p align="centeR"><img src="https://user-images.githubusercontent.com/62092317/204816626-43caec74-f49b-4aa6-bfb4-8e79ac0edfc4.png"></p>
        
        - Bounding box 의 좌하단, 우상단 지점을 기준으로 두 bbox를 비교할 때 좌표간 L2,L1 norm이 같더라도 box가 겹치는 영역의 크기는 균일하지 않다.(두 그림은 각각 L2,L1 norm distance가 같아도 bbox간 위치가 상이할 수 있음을 보여준다.)
        - 이 때문에 Object Detection에서 단순히 box의 좌표 차이를 통해 loss를 구하는 것보다 IoU를 loss에 활용하는 것이 regression loss에 더 적합하다.
        - 그러나, **IoU는 수식 상 두 bbox간 겹치는 영역이 0인 경우들에 대해 어느 정도의 오차를 가지고 bbox가 떨어져 있는지를 구분할 수 없기 때문에 gradient vanishing 문제가 발생**한다.
- GIoU(Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression[[LINK]](https://arxiv.org/abs/1902.09630))
    
    <p align="center"><img src="https://user-images.githubusercontent.com/62092317/204826809-d9183adf-11c6-4b4b-bd48-26b2e5799a6e.png"></p>

    - 두 bbox를 비교할 때 둘을 포함하는 가장 큰 bbox를 기준으로 측정
    - C box는 A,B를 포함하는 가장 작은 이며, C \ (AUB)는 C 영역에서 A와B의 합집합 영역을 뺀 것을 의미한다.
    - GIoU는 IoU와 다르게 -1~1의 값을 가지며, 두 bbox가 겹치는 영역 없이 딱 붙어 있을 때 0이다.
    - **이를 통해 gt와 겹치지 않는 bbox에 대한 gradient vanishing 문제는 개선 했지만 수렴 속도가 느리고, 겹치는 영역이 없는 경우 bbox가 gt로 수렴하기 위한 방향성(horizontal,vertical)을 가진 정보를 포함하지 않는다는 문제점이 존재**
    
- DIoU(Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression)[[LINK]](https://arxiv.org/pdf/1911.08287.pdf)

    <p align="center"><img src="https://user-images.githubusercontent.com/62092317/204833078-23ba9523-364f-4b38-a7da-64447d6e5446.png"></p>

    - Bbox의 수렴성을 개선하기 위해 IoU와 함께 중심 좌표를 regression한다. 
    - IoU, GIoU가 두 박스의 교집합을 넓히는 것에만 집중하고 bbox가 수렴해야 하는 방향(x,y)은 고려하지 않는 반면 DIoU는 중심 좌표를 regression하기 때문에 gt의 중심점을 향하도록 bbox가 수렴해 나간다.
    - DIoU에선 두 bbox의 중심좌표간 Euclidean distance를 최소화 하는 방향으로 학습이 이루어진다.

- CIoU(Complete-IoU)
    
    <p align="center"><img src="https://user-images.githubusercontent.com/62092317/204853397-10e9614f-efe0-40c4-b262-e5171fde1930.png"></p>

    - Bbox에 대한 좋은 regression loss는 **overlap area, central point distance, aspect ratio**를 고려하는 것임을 고려하여 loss를 구성한다.
    - v는 두 bbox간 aspect ratio의 유사성을 측정하는 항으로, 마지막 항을 통해 prediction이 ground truth의 aspect ratio에 수렴하도록 학습

### 2. Confidence threshold/ IoU threshold and mAP

- IoU threshold, mAP의 오류

- 실험결과와 함께 첨부하기
    - Confidence threshold, IoU threshold에 따른 LB score 변화 
        |conf|IoU|LB score|
        |------|---|---|
        |0.001|0.4|60.93|
        |0.001|0.5|61.48|
        |0.001|0.6|61.41|
        |0.05|0.5|60.03|

### 3. YOLOv7 

#### 오피스 아워

#### Tuning
- Augmentation
    - Mosaic & Mixup [[Medium : YOLOX Explanation]](https://medium.com/mlearning-ai/yolox-explanation-mosaic-and-mixup-for-data-augmentation-3839465a3adf)
        - Mosaic
            - YOLOv4에서 처음 도입 되었으며, [Cutmix](https://arxiv.org/abs/1905.04899)를 더 발전시킨 방법이다.
            - 4개의 image로 부터 랜덤한 영역을 뽑아 하나로 합치는데, 객체의 일부가 잘릴 경우 bbox 또한 잘린 부분만 똑같이 적용되어 합쳐 진다.
            - Steps
                
                <p align="center"><img src="https://user-images.githubusercontent.com/62092317/203998844-83b3ae98-f49b-485e-b3fa-d9efba6d7c7d.png"></p>

                1. 이미지들을 전부 같은 사이즈로 resize
                2. 4개의 이미지를 2x2 격자형태의 큰 하나의 이미지로 합친다.
                3. 각 이미지에 존재하는 bounding box를 새로운 이미지에 맞추어 적용한다.
                4. 새로운 이미지에서 random한 영역을 추출하고, 이 영역에 포함되지 않는 bbox는 제거한다.
                4. 잘라낸 영역에 걸쳐진 bbox들은 좌표를 이미지 boundary에 맞게 재조정한다.

            - Trash Dataset Mosaic

                <p align="center"><img src="https://user-images.githubusercontent.com/62092317/203999947-b7aca8a7-f16e-4c43-b3c7-8cd42afbb5db.jpg"></p>

                - 하나의 이미지로 더 많은 객체를 학습
                - 객체의 형태가 잘린 경우 일부분만으로 class를 예측해야하므로, 강한 augmentation 효과를 줄 수 있음

        - Mixup
            - Mixup은 원래 classification task에서 처음 제안되었으나, object detection task에서도 잘 동작하는 augmentation 기벅이다.
            - 2개의 image를 weigthing parameter $\gamma$에 기반하여 합치는 방법으로, 다음과 같은 수식으로 이루어진다.

            <p align="center"><img src="https://miro.medium.com/max/1122/1*HAt2pgpcqK3J_iqO4RrY8Q.png" width=400></p>

            - 이미지를 합칠 때는, label도 함께 합쳐야 한다. Bounding box annotation의 경우, 픽셀값처럼 weight summation하는 것이 아니라 하나의 이미지에 추가 bbox를 할당하는 방식으로 이루어진다.(아래 이미지 참조) 

            <p align="center"><img src="https://miro.medium.com/max/1400/1*uYy0ru1y3H6ky3X7U3BTqA.png" width= 300></p>

- HyperParams
    - YOLOv7의 augmentation관련 hyperparameter 구성
        ``` python
        hsv_h: 0.015  # image HSV-Hue augmentation (fraction)
        hsv_s: 0.7  # image HSV-Saturation augmentation (fraction)
        hsv_v: 0.4  # image HSV-Value augmentation (fraction)
        degrees: 0.0  # image rotation (+/- deg)
        translate: 0.2  # image translation (+/- fraction)
        scale: 0.2  # image scale (+/- gain)
        shear: 0.0  # image shear (+/- deg)
        perspective: 0.0  # image perspective (+/- fraction), range 0-0.001
        flipud: 0.0  # image flip up-down (probability)
        fliplr: 0.5  # image flip left-right (probability)
        mosaic: 1.0  # image mosaic (probability)
        mixup: 0.15 # image mixup (probability)
        copy_paste: 0.0  # image copy paste (probability)
        paste_in: 0.15  # image copy paste (probability), use 0 for faster training
        loss_ota: 1 # use ComputeLossOTA, use 0 for faster training
        ```
        - hsv_h, hsv_s, hsv_v : Color augmentation 
        - degrees, scale, translate, shear : 이미지를 회전, 확대/축소, 평행 이동, shear 등과 같은 affine transform의 정도
        - mosaic, mixup : mosaic와 mixup을 적용할 확률
        - flipud, fliplr : 상하,좌우 반전을 적용할 확률
        - copy_paste, paste_in은 detection task가 아닌 segmentation task에서 사용되는 copy & paste augmentation의 params
    - Hyperparameter Tuning은 train.py에서 parsing하는 argument 중 evolve를 활용하면 된다.
- 참고자료
    - [[YOLO Auto-anchoring]](https://towardsdatascience.com/training-yolo-select-anchor-boxes-like-this-3226cb8d7f0b)
    - [[YOLOv1v6비교]](https://leedakyeong.tistory.com/entry/Object-Detection-YOLO-v1v6-%EB%B9%84%EA%B5%902)
    - [[YOLO evolve]](https://github.com/ultralytics/yolov5/issues/607)
    - [[YOLO auto-anchoring2]](https://leedakyeong.tistory.com/entry/Object-Detection-YOLO-Optimal-Anchor-Box-YOLO-v5-YOLO-v6-autoanchor)
    - [[YOLO-loss]](https://leedakyeong.tistory.com/entry/Object-Detection-YOLO-v5-v6-Loss)
    - ```python
        class ComputeLossAuxOTA:
        # Compute losses
        def __init__(self, model, autobalance=False):
            super(ComputeLossAuxOTA, self).__init__()
            device = next(model.parameters()).device  # get model device
            h = model.hyp  # hyperparameters

            # Define criteria
            BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
            BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

            # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
            self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

            # Focal loss
            g = h['fl_gamma']  # focal loss gamma
            if g > 0:
                BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

            det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
            self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
            self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
            self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
            for k in 'na', 'nc', 'nl', 'anchors', 'stride':
                setattr(self, k, getattr(det, k))
      ```
    - [[About Loss gain]](https://github.com/ultralytics/yolov5/issues/5371)
    - [[Object Positive weight]](https://github.com/ultralytics/yolov3/issues/375)
### 4. K-fold ansemble을 하는것이 전체 dataset을 학습하는 것보다 성능이 높을 수 있나? [[About Bagging]](https://sungkee-book.tistory.com/9)

- Bagging이란?
    - 기존 학습 데이터(Original Data)로부터 랜덤하게 '복원추출'하여 동일한 사이즈의 데이터셋을 여러개 만들어 앙상블을 구성하는 여러 모델을 학습시키는 방법
    - 이렇게 생성된 새로운 데이터셋을 Bootstrap이라고 한다.
- Bagging의 의미

    <p align="center"><img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FdYlyoL%2Fbtq7U1diMzt%2FiKzomNxsUrDBhvTe021650%2Fimg.png" width=400></p>

    - 각 Boootstrap은 복원 추출ㄹ로 인해 기존 데이터와는 다른, 즉(긍정적으로) 왜곡된 데이터 분포를 갖게 된다. 기존 데이터셋이 갖고 있던 노이즈와는 조금씩 다른 노이즈 분포를 갖도록 변형을 주는 것이다.
    - **기존 데이터셋만 사용하여 학습했을 때는 하나의 특정한 노이즈에만 종속적인 모델이 만들어질 수 있는 위험이 있다.**
    - **따라서, Bagging은 조금씩 분포가 다른 데이터셋을 기반으로 반복적인 학습 및 모델 결합을 통해 이러한 위험을 방지하는 효과를 가진다 다양한 노이즈 분포를 가진 Bootstrap들을 기반으로 개별 모델을 학습하고 또 결합함으로써 노이즈의 변동으로 인한 영향력을 줄일 수 있는 것이다.**
- 즉, **같은 모델에 대해 k-fold ensemble을 하는 것은 학습 데이터에 존재할 수 있는 특정 노이즈에 종속적인 모델이 되지 않도록 하기 위함**이고 **서로 다른 모델에 대해 ensemble을 하는 것은 모델의 다양한 특성을 반영하여 test data를 특정 모델의 편향 없이 판단하기 위함**이다.
---
## Paper review

### MLP mixer [[Github]](https://github.com/lucidrains/mlp-mixer-pytorch) [[Paper]](https://arxiv.org/abs/2105.01601) [[발표자료]](https://docs.google.com/presentation/d/1OrjqBGVU8ATVnXGUexgACAZma4M-Y7CL/edit?usp=share_link&ouid=112634105046419935376&rtpof=true&sd=true) [[Team Notion]](https://www.notion.so/929db26565754c7785805cb0f3a3ea03?v=9d218ff86dc145f184f2e49b1a1d34c1&p=94661ed5e1a5477a9fcadc6391630582&pm=s)

### 논문 리뷰

- Background
    - Inductive Bias

        - 모델이 학습과정에서 본 적이 없는 분포의 데이터를 입력 받았을 때, 해당 데이터에 대한 판단을 내리기 위해 가지고 있는 설계 구조상의 편향

        - CNN의 경우 특정 pixel과 가장 관계가 깊은 pixel은 근처의 영역에 위치할 것이라는 가정
        
        - Invariance VS Equivalance
        
            - Invariance : 변하지 않는다는 의미의 불변성, ex) Pooling operation
            - Equivalance : 변할 수 있지만 같다는 의미의 등가성, ex) Convolutional layer
            
                <p align="center"><img src="https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F2544fb2a-920a-41bf-94e8-fef573936787%2FUntitled.png?table=block&id=b3c90ce6-e9ad-4684-99ee-677b419c0927&spaceId=0252647d-6def-4884-b578-6ff83938f101&width=1920&userId=64c1d066-66f8-412c-abd5-ef5b901b6403&cache=v2" width = 400></p>

            - 위 그림 에서 invariance하다는 표현은 convolution layer는 Equivalance하지만 pooling 과정에서 Invariance 하다는 의미로 사용한 것
        
        <p align="center"><img src="https://hallowed-eris-113.notion.site/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2Fbf9e4d53-a47a-4396-8781-0255eb41e1a2%2FUntitled.png?table=block&id=70027629-2813-4ecb-99a5-d77b746570e5&spaceId=0252647d-6def-4884-b578-6ff83938f101&width=1770&userId=&cache=v2" width= 400></p>

    - ViT는 convolution layer가 존재하지 않아 locality에 관한 Inductive biase가 없음

        <p align="center"><img src="https://hallowed-eris-113.notion.site/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F0f59f7f0-8d8c-4319-b169-6d48e41bcd3f%2FUntitled.png?table=block&id=d3c60f47-414a-420b-9485-9b95a8053f81&spaceId=0252647d-6def-4884-b578-6ff83938f101&width=670&userId=&cache=v2" width= 400></p> 

        - 따라서 Train data가 적으면 strong Inductive bias를 가진 model(BiT)이 우수
        - 하지만 Train data가 많으면 Weak Inductive bias를 가진 model(ViT)이 우수
        
    - CNN은 Texture에 집중, ViT는 Shape에 집중

        <p align="center"><img src="https://hallowed-eris-113.notion.site/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2Fb5f4f221-c508-49e4-b039-efa4c3992723%2FUntitled.png?table=block&id=93fec70a-fb62-417c-a8e8-0501ca65ef8d&spaceId=0252647d-6def-4884-b578-6ff83938f101&width=580&userId=&cache=v2" width= 300></p> 

        - 인간은 Shape에 집중하므로 Inductive bias가 적은 ViT가 인간의 판단에 더 가깝다.

- MLP - Mixer

    - CNN과 ViT가 학습과정 내에서 feature를 mixing하는 방식은 크게 두 가지 이다.
        - (1) at a given spatial location (e.g., 특정 pixel에 대한 연산)
        - (2) between different spatial locations (e.g., 주변 pixel 간 연산)
        - 예를들어 CNN의 경우, 1x1 Conv 동작은 (1), NxN Conv(N>1)과 pooling 동작은 (2)에 해당되며 ViT를 비롯한 self-attention기반 알고리즘은 MLP-blocks 동작은 (1), self-attention 레이어는 (1)와 (2)를 동시에 수행한다.
    - MLP-Mixer 논문의 특징은 두가지 features, 즉 per-location (channel-mixing) 동작과 cross-location (token-mixing) 동작을 분리하고 이 두 동작을 MLPs로 구현하고 있다는 점이다.

    - Mixer는 두가지 형태의 MLP 레이어(Channel-Mixing MLPs & Token-Mixing MLPs)를 포함한다. (아래 그림은 Mixer 레이어의 구조 및 연산식을 설명한다.)
        
        <p align="center"><img src="https://hallowed-eris-113.notion.site/image/https%3A%2F%2Fmiro.medium.com%2Fmax%2F1050%2F1*4mRJPu8Gp9SKew9NBiPdVg.png?table=block&id=f3d11cf8-a68b-4ed7-a0e9-4ffefe9b2749&spaceId=0252647d-6def-4884-b578-6ff83938f101&width=2000&userId=&cache=v2" width= 400></p>

        - Channel-Mixing MLPs: 각 패치에 대해 모든 feature간 mixing을 수행
        - Token-Mixing MLPs: 모든 패치의 동일한 채널에 존재하는 feature간 mixing을 수행

    - Mixer는 “An all-MLP architecture for vision”이라는 부제를 붙였지만, Channel-mixng, Token-mixing의 MLP의 연산 과정을 살펴보면 기존에 사용되던 convolution 연산의 특수한 케이스로 해석될 수 있다.
        - Per-patch fully-connected layer: 입력 이미지로부터 patch를 만들 때 P x P conv (with Stride P, No padding)을 수행하는 것과 동일함
        - Channel-Mixing MLPs: 1x1 conv
        - Token-Mixing MLPs: weight-shared depth-wise convolution

    <p align="center"><img src="https://hallowed-eris-113.notion.site/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2Fddeed7d8-5b82-4243-98c5-56397d24e251%2FUntitled.png?table=block&id=10b41b2f-33ef-45d9-92a8-4d7bfaa81cfa&spaceId=0252647d-6def-4884-b578-6ff83938f101&width=1920&userId=&cache=v2" width=500></p>

    - Mixer는 Image를 patch단위로 해석하는 것은 ViT와 동일하지만 positional embedding을 하지 않음

        - 왜 positional embedding이 필요하지 않은가??
            
            - Token mixing을 할 때 이미지 패치별 특성 위치는 그대로 layer 계층에 입력되어 위치 정보가 반영된다.
            
            - ViT는 복잡도가 quadratic 하지만 MLP-Mixer는 input patch 수에 따라 계산 복잡도가 결정되어 linear하다. 따라서 한번의 행렬 곱에서 각 행이 유지되기 때문이다.

    - 실험 결과

        <p align="center"><img src="https://hallowed-eris-113.notion.site/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F76db60f9-74ed-4740-b34d-8825402f7c2a%2FUntitled.png?table=block&id=d4806202-2701-40a7-ab49-8aff2812a15a&spaceId=0252647d-6def-4884-b578-6ff83938f101&width=1730&userId=&cache=v2" width = 400></p>

        - Data Set의 Size가 작을수록 mixer model의 overfitting이 심함
        - Data Set의 Size가 클수록 performance가 상승하는 폭이 다른 모델보다 큼
---
## Mentoring

### About Dropout

- Dropout은 신경망의 각 layer에서 노드를 일정 확률로 비활성화 시키는건데, 매 layer를 거치면서 이 작업이 이루어진다면 backpropagate 되는 gradient flow가 너무 작아지지 않을까?
- Dropout기법은 이를 방지하기 위해, 살아남은 node 들의 출력 값에 1/(1-p)만큼을 곱하여 출력을 scaling해준다.(p=drop out probability)
    ``` python
    class MyDropout(nn.Module):
    def __init__(self, p=0.5):
        super(MyDropout, self).__init__()
        self.p = p
        # multiplier is 1/(1-p)
        # Set multiplier to 0 when p = 1 to avoid error
        if self.p < 1:
        self.multiplier_ = 1.0 / (1.0 - p)
        else:
        self.multiplier_ = 0.0
        
    def forward(self, input):
        # if model.eval(), don't apply dropout
        if not self.training:
        return input
        
        # So that we have `input.shape` numbers of Bernoulli(1-p) samples
        # --> input의 데이터 사이즈를 고려하여 베르누이(1-p)의 샘플을 만들기 위한 과정
        # --> 0~1사이의 난수를 발생시켜 지정 확률 p보다 큰지 작은지를 boolean으로 설정 
        selected_ = torch.Tensor(input.shape).uniform_(0, 1) > self.p
        
        # To suppert both CPU and GPU
        # --> boolean 값을 0, 1로 바꾸는 작업
        if input.is_cuda:
        selected_ = Variable(selected_.type(torch.cuda.FloatTensor), requires_grad = False)
        else:
        selected_ = Variable(selected_.type(torch.FloatTensor), requires_grad = False)

        # Multiply output by multiplier as described in the paper [1]
        # --> 0,1의 값들을 곱하여 선택된 input을 골라내는 과정
        res = torch.mul(selected_, input)*self.multiplier_
        return res
    ```
- Drop out Code
    - multiplier_
        - 1/(1-p)는 p가 1보다 작은 값이므로, 결과적으로 1보다 큰 값을 곱해주는 효과를 보여준다.
        - 테스트시에는 학습때보다 예상 출력이 더 커질것이다. 왜냐하면 학습할 때에는 0으로 셋팅된 노드가 있기 때문에 weight들이 Fully connected 될 때보다 더 적은 weight들로 propagation을 해야하기 때문에 더 큰 수로 수렴하기 때문이다.
        - 하지만, dropout을 거치고 나면 각 노드에서 drop된 노드를 제외하면 input과 output이 일치하는 identity function이어야한다.
        - 그러므로 dropout output에 1/(1-p)를 곱함으로써 scaling해준다.
    - self.training
        - 실제로 학습할 때, model.train( ) 이라는 statement를 사용하는데, 이것은 모델이 학습중임을 알리는 것으로 model 안에 있는 model.training이라는 멤버변수가 True가 됨을 의미한다.
        - 반대로 테스트를 진행할 때는, model.eval( )이라는 statement를 사용하는데, 이것은 모델이 테스트 중임을 알리는 것으로 model.training이라는 멤버변수가 False가 됨을 의미한다.
    - 베르누이 분포
        - 실제 베르누이 분포를 따르는 변수를 만들기보다, 0~1 사이인 uniform 분포를 활용하여 나온 값들이 p보다 큰지 작은지로 베르누이 분포 변수를 만들어낸다.

- Dropout 과 DropConnect의 차이 [[LINK]](https://stats.stackexchange.com/questions/201569/what-is-the-difference-between-dropout-and-drop-connect)

- torch.nn의 layer를 생성할 떄 Dropout을 적용할 줄만 알았지, 구체적인 구현에 대한 원리와 이유를 모르고 있었다. Deep learning 공부에선 detail이 중요함을 다시금 깨달았다.. 

---

## 정규식

- 특정 두 문자 사이의 문자열 추출하기
    ``` python
    import re

    string = 'dsafasdf"helloworld"dsfafads'
    regex = re.compile('{}(.*){}'.format(re.escape('"'), re.escape('"')))
    text = regex.findall(string)
    print(text[0])

    ```