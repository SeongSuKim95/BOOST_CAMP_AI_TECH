# Week 9 : Object Detection

## Contents 

- Course
    - (1강)Object Detection Overview

- Further questions 
    - 초기 YOLO는 왜 2개의 boundingbox를 예측했을까?
    - EfficientNet의 objective 수식과 parameter들의 타당성에 대하여
    - AuGNet에 대하여
## Course

### (1강) Object Detection Overview
- History
    <p align="center"><img src="https://user-images.githubusercontent.com/62092317/201580445-5ff5b971-b5ff-4d65-afe2-9809cef888cd.png" width = 500></p> 
- Evaluation metric
    - mAP(mean Average Precision): 각 클래스당 AP의 평균
        - Confusion matrix
            - <p align="center"><img src="https://user-images.githubusercontent.com/62092317/201581352-fbc2dc15-81cc-4866-bfb1-81ab08df883a.png" width = 400></p>
        - Precision & Recall
            - Precision : 모든 검출 결과 중에 올바른 검출 결과의 비율
            - Recall : 모든 GT 대비 올바른 검출 결과의 비율
        - PR curve
            - <p align="center"><img src="https://user-images.githubusercontent.com/62092317/201581177-cebb73cd-4217-4c2d-9a85-2d8dfd8f4344.png" width= 400></p>
            - Detection 결과를 confidence 기준 내림차순으로 정렬 후, 누적 TP/FP를 계산하여 Precision 과 Recall을 측정
            - Precision(y축)과 Recall(x축)에 대해 Curve를 Plot
            - Average Precision : PR Curve 아래의 면적
            - mAP: 모든 class에 대한 AP의 평균 값
        - IoU(Intersection Over Union)
            - 예측 bounding box의 정확도를 판단하기 위한 지표
            - Detection box와 Ground Truth box에 대해, Overlapping region / Combined region 

    - FPS(Frame Per Second), Flops
        - FPS : 초당 처리 가능한 frame 수
        - Flops(Floating Point Operations) : Model이 얼마나 빠르게 동작하는지를 연산량 횟수를 통해 측정
            - ex) Convolutional filter
                <p align="center"><img src="https://user-images.githubusercontent.com/62092317/201582226-2e8fd8b2-7a4c-482b-9648-41fa68560b60.png" width=400></p>

### (2강) 2 Stage Detectors

- 2 stage detector : Localization 과 classfication 두 단계를 통해 물체를 검출

- RCNN
    <p align="center"><img src="https://user-images.githubusercontent.com/62092317/201804763-a4ca0213-b305-4bd6-84c9-c72b2e3c5fab.png" width=400></p>

    - Region Proposal을 통해 물체가 있을 만한 후보 영역을 고정된 size로 resize(warping)
        - Warping을 하는 이유는?
            - CNN의 마지막 FC layer의 입력 사이즈가 고정이므로 이미지 사이즈를 이에 맞춰주어야 하므로!
    - 이후 ConvNet(AlexNet)에 넣어 resize된 영역으로 부터 feature를 추출
    - Selective Search : 후보 영역(Region Proposal)을 뽑는 방식

        <p align="center"><img src="https://user-images.githubusercontent.com/62092317/201805304-926b0e55-85e4-4220-850d-c571c961c7d1.png" width=400></p>

        - 이미지를 무수히 많은 작은 영역으로 나눈 후(segmentation)[[LINK]](https://blog.naver.com/laonple/220925179894), 이들을 통합해 나가며 후보 영역을 골라 낸다.
    
    - CNN을 통해 나온 feature를 SVM에 넣어 classification 진행
        - Input : 2000 * 4096 features
        - Output : Class(C+background) + Confidence score
    - CNN을 통해 나온 feature를 regression을 통해 bounding box 예측
        - 예측한 bbox와 GT bbox간 regression 수행
    - Shortcomings
        1. 2000개의 Region을 각각 CNN에 통과
        2. Warping에 의한 RoI의 aspect ratio 변화에 따른 성능 하락 가능성
        3. CNN, SVM classifier, bbox regressor 따로 학습
        4. End-to-End X
- SPP(Spatial Pyramid Pooling)
        <p align="center"><img src="https://user-images.githubusercontent.com/62092317/201807055-06760ca2-74af-4b3f-a9c0-ceccfe9ba693.png" width = 500></p>

    - RCNN과 달리 **Input image를 한번에 CNN에 통과시켜 ROI를 추출하고, Warping X**
    - Spatial Pyramid Pooling

        <p align="center"><img src="https://user-images.githubusercontent.com/62092317/201807225-a9a2ac7a-0b7a-4f84-996a-8171b77021ec.png" width = 300></p>

        - Fixed length representation을 통해 ROI의 size에 상관 없이 고정된 size의 feature vector를 추출
    - RCNN의 shortcoming 1,2번을 해결

- Fast RCNN
    
    - Spatial Pyramid Pooling 대신 RoI Pooling을 수행
        > Selective search는 image에 대해 직접 적용되었으나, CNN을 통과시킨 feature map에선 RoI를 어떻게  뽑아낼 수 있을까?
    - RoI Projection
        <p align="center"><img src="https://user-images.githubusercontent.com/62092317/201808774-c5291ae1-6519-4dd0-ba81-bbe2a6f34e5f.png" width = 400></p>
        - Selective search를 통해 RoI를 뽑고, image를 CNN에 통과시킨 feture map이 있을때 **feature map상에서 ROI와 대응되는 부분을 추출**
        - 이후 maxpooling 등을 통해 추출한 output feature map을 고정된 사이즈로 변환
    - Training
        - multi task loss 사용 : classification loss(CE) + bbox regression(smooth L1 loss)
        - Hierarchical sampling
            - R-CNN의 경우 이미지에 존재하는 RoI를 전부 저장해서 사용
            - Fast R-CNN의 경우 한 배치에 한 이미지의 RoI만을 포함
        - RCNN의 3번 shortcoming까지를 보완
      
- Faster R-CNN
    - Fast R-CNN vs Faster R-CNN
        <p align="center"><img src="https://user-images.githubusercontent.com/62092317/201810930-5ed79546-0924-47da-8f73-719e1493f332.png" width = 400></p>

        - 차이점 : Selective search를 제거하고 RPN(Region Proposal Network)를 도입
    
    - Pipeline
        1. 이미지를 CNN에 넣어 feature maps추출
        2. RPN을 통해 RoI 계산

            <p align="center"><img src="https://user-images.githubusercontent.com/62092317/201811532-50ffe392-b025-472a-82a7-bf38b9dbc981.png" width = 200></p>

            - Anchor Box 개념 사용
                - 이미지를 격자 형태로 나눈 후, 각 셀 마다 셀을 중심으로 다양한 aspect ratio와 size의 anchor box를 생성
                - Cell의 크기와 상관 없이, 객체의 다양한 size에 대응되는 bbox를 생성
                - ex)64개의 cell에 대해 9개의 anchor box를 사용하면, 64*9개의 box 사용

            - RPN: 생성된 anchor box들이 객체를 포함하고 있는지를 판단하고, 객체가 포함되어 있다면 bbox의 위치를 미세 조정(Classification head(객체 분류)
            ,Coordinate head(bbox 미세 조정))

                <p align="center"><img src="https://user-images.githubusercontent.com/62092317/201813771-4a8a338a-b01a-44a9-8305-12a3126ac129.png" width = 400></p>

                - Featuremap shape : 64 * 64
                - Cls prediction : 2 * 9 (객체 유무 True/False * anchor box 9개)
                - Box prediction : 4 * 9 (좌표 x,y,w,h 4개 * anchor box 9개)
                - Top N개의 select된 anchor box를 가지고 coordinate를 재조정

            - NMS(Non max suppression)
                - 유사한 RPN Proposal을 제거하기 위해 사용
                - Class score를 기준으로 proposal을 정렬 후, bbox간 IoU가 0.7이상인 중복된 영역으로 판단 후 제거
- Summary
    <p align="center"><img src="https://user-images.githubusercontent.com/62092317/201814678-e453b726-46cc-4893-af12-f01948cab45f.png" width = 500></p>

### (3강) Object Detection Library

- MMDetection
    - Pytorch 기반의 Object Detection 오픈소스 라이브러리
    - Two stage 모델은 크게 Backbone,NeckDenseHead,RoIHaed 모듈로 나눌 수 있음
        1. Backbone : 입력 이미지를 특징 맵으로 변현
        2. Neck : backbone과 head를 연결, Feature map을 재구성
        3. DenseHead : 특징 맵의 dense location을 수행
        4. RoIHaed : RoI 특징을 입력으로 받아 box 분류, 좌표 회귀등을 예측
    - 각각의 모듈 단위로 config 파일을 이용해 커스터마이징
        - configs를 통해 데이터셋으로부터 모델, scheduler,optimizer 정의 가능
        - 특히, configs에는 다양한 object detection 모델들의 config 파일들이 정의되어 있음
        - 그 중, configs/base/폴더에 가장 기본이 되는 config 파일이 존재
        - 각각의 base/폴더에는 여러 버전의 conifg들이 담겨 있음
- Detectron2
    - Facebook AI Research의 Pytorch 기반 라이브러리
    - MMDetection과 유사하게 config파일을 수정, 이를 바탕으로 파이프라인을 build하고 학습
    - 틀이 갖춰진 기본 config를 상속 받고, 필요한 부분만 수정해 사용

### (4강) Neck

- Neck은 무엇인가?
    - **Fundamental question:  RoI pooling 전에 Input이 CNN을 통과하여 얻은 마지막 layer의 출력 feature map이 아닌, 중간 단계의 feature map들을 사용할 수는 없을까?**
    - 중간 단계에서 추출되는 feature map을 사용하자!
    - Neck의 필요성
        <p align="center"><img src="https://user-images.githubusercontent.com/62092317/201905430-7566edb2-7c10-49b0-8e64-5e663f8fafdc.png" width = 400></p>
        - Receptive field 관점에서 (출력 기준) 작은 feature map일수록 이미지의 더 넓은 영역을 보며, 큰 feature map일수록 더 local한 영역을 봄.
        - 이미지에 포함된 물체의 size는 다양하지만 high level의 feature는 작은 객체를 포착하기 힘듦.
        - **따라서 다양한 크기의 객체를 더 잘 탐지하기 위해** Neck을 통해 중간 단계의 feature map을 더 효율적으로 활용
        - 하위 level의 feature는 객체의 semantic 정보가 약하므로 상대적으로 semantic이 강한 상위 feature와의 교환이 필요
- FPN(Feature Pyramid Network)
    - 각 level의 feature map에서 top-down path way를 추가 하여, pyramid 구조를 통해서 high level의 정보를 low level에 순차적으로 전달
    
        <p align="center"><img src="https://user-images.githubusercontent.com/62092317/201906273-51c9f6c4-a5da-402f-831d-f8f242913b09.png" width = 400></p>

    - Bottom-up, Top-down way의 lateral connection을 사용 (1x1 convolution for C & upsampling in WH)
    
    - Lateral connection을 통해 얻은 모든 feature map에 대해 RoI pooling 후 NMS 수행, Top N개의 bbox를 선택
    
    - ResNet backbone의 4개의 stage에서 추출된 feature map에 lateral connection을 적용하여 4개의 feature map을 얻음. 이후, 각 feature map에서 RoI pooling을 하여 얻은 모든 RoI 중 NMS를 통해 Top N개의 bbox를 선택

    - FPN의 문제점 : ResNet의 stage간 path가 길기 때문에 bottom up path way에서 high level feature의 정보 전달이 어려움

        - PANet(Path Aggregation Network) : Bottom-up path 추가하고 adaptive feature pooling을 수행
- DetectoRS : Recursive한 RoI pooling, feature extraction이 main idea

    <p align="center"><img src="https://user-images.githubusercontent.com/62092317/201912447-c6f98d00-b813-4c9c-9e67-c6a1ff1476b7.png"></p>

    - Recursive Feature Pyramid(RFP): Feature pyramid에서 나온 정보를 backbone으로 recursive하게 전달
    - ASPP(Atrous Convolution)을 통해 convolution 연산의 receptive field를 증가

- BiFPN(Bi-directional Feature Pyramid) 
    - Weighted Feature Fusion
        - FPN과 같이 단순 summation을 하는 것이 아니라 각 feature 별로 가중치를 부여한 뒤 summation
        - Feature별 가중치를 통해 feature를 강조하여 성능 상승

- AugFPN : Improving Multi-scale featrue learning for object Detection
    - FPN의 문제점 Highest feature map에서 발생하는 정보 손실(higher feature map으로부터의 down path가 없기 때문)
    - Residual feature augmentation을 통해 마지막 feature map에도 정보를 보강
    - Soft RoI selection
        - FPN과 같이 하나의 feature map에서 RoI를 계산하는 경우  sub-optimal
        - PANet은 모든 feature map으로 부터 max pooling하였지만, pooling에 의한 정보 손실 발생
        - Soft RoI selection은 channel-wise 가중치를 계산 후 가중치를 사용

### (5강) 1 stage Detectors

- 2 Stage Detector는 Localization과 Classification을 나누어 수행하기 때문에 속도가 느리다!
- 2 Stage VS 1 Stage Detector
    <p align="center"><img src= "https://user-images.githubusercontent.com/62092317/202064275-418add2c-a1eb-485a-a741-918dfd5047f0.png"></p>
- YOLO (You Look Only Once)
    - Pipeline
        1. 입력 이미지를 SxS 그리드 영역으로 나누기
        2. 각 그리드 영역마다 B개의 Bounding box와 Confidence Score계산
        3. 각 그리드 영역마다 C개의 class에 대한 해당 클래스일 확률 계산
    - 단점
        - 7x7 그리드 영역으로 나눠 Bounding box prediction 진행하므로 Grid보다 작은 크기의 물체는 검출 불가능
        - Feature extractor의 마지막 feature만을 사용하므로 정확도 하락
- SSD
    - Extra Convolution layers에서 나온 feature map들 모두 detection 수행
        - **6개의 서로 다른 scale의 feature map 사용**
        - 큰 feature map (early stage feature map)에서는 작은 물체 탐지
        - 작은 feature map(last stage feature map)에서는 큰 물체 탐지
    - Fully connected layer 대신 convolution layer 사용하여 속도 향상
    - **서로 다른 scale과 비율을 가진 미리 게산된 default anchor box 사용**

- YOLO v2
    - 정확도 향상(Better)
        - Batch normalization 사용
        - High resolution classifier
        - Convolution with anchor boxes
            - YOLO v1 grid cell의 첫 10개의 feature(2개의 bbox 정보)를 랜덤으로 초기화 후 학습 했던 것에서 SSD처럼 anchor box를 도입하는 것으로 변경
            - 좌표 값을 그대로 예측하는 것 대신 anchor box로 부터의 offset을 예측하는 문제가 단순하고 학습하기 쉽다
        - Fine-grained features
            - Early feature map을 late feature map에 합쳐주는 pass through layer 도입
            - Multi-scale training(다양한 사이즈의 input image 사용)
    - 속도 향상(Faster)
        - Backbone model을 Google net에서 Darknet-19로 변경
        - 마지막 fc layer를 3x3 convolution layer로 대체 후 1x1 convolution layer 추가
    - 더 많은 class를 예측(Stronger)
        - Classification 데이터셋, detection 데이터셋을 함께 사용
        - WordTree 구성(계층적인 트리)
- YOLO v3
    - Darknet-53 사용
    - Multi-scale Feature maps
        - 서로 다른 3개의 scale 사용
        - Feature pyramid network 사용

- RetinaNet
    - 1 Stage detector Problems
        - Region proposal 이 없는 1 stage detector는 이미지를 grid로 나누고 각 grid마다 bounding box를 무조건 예측
        - 따라서, positive sample(객체 영역) < negative sample(배경 영역)에 의한 class imbalance가 심함
        - Anchor Box 대부분이 Negative sample(background)
    - Solution
        - Focal loss 활용하여 one-stage detector의 단점을 해결

### (6강) EfficientDet

- Model scaling 등장 배경
    - Layer를 너무 깊게 쌓으면 성능의 gain은 줄어들고, 속도와 연산량만 늘어난다. 어떻게 하면 모델의 layer **잘** 쌓을 수 있을까?
        > 즉, 더 높은 정확도와 효율성을 가지면서 ConvNet의 크기를 키우는 방법(scale-up)은 없을까?
    - Width Scaling
        - **더 wide한 네트워크는 미세한 특징**을 잘 잡아내는 경향이 있고, 학습이 쉬움
        - But, 극단적으로 넓지만 얕은 모델은 high-level 특징들을 잘 잡지 못함
    - Depth Scaling
        - 깊은 ConvNet은 더 풍부하고 복잡한 특징들을 잡아낼 수 있고, 새로운 task에 잘 일반화됨
        - But, Gradient vanishing 문제가 존재
- EfficientNet
    - Objective
        <p align="center"><img src="https://user-images.githubusercontent.com/62092317/202098342-9a253371-bba4-44d0-9b54-70af4ca8295b.png" width = 300></p>

        - **제한된 모델의 메모리와 FLOPs 속에서 모델의 성능을 최대화하는 depth,width,resolution의 조합 찾기!**

    - Observation
        1. 네트워크의 폭,깊이,혹은 해상도를 키우면 정확도가 향상된다. 그러나 더 큰 모델에 대해서는 정확도 향상 정도가 감소한다.
        2. 더 나은 정확도와 효율성을 위해서는 ConvNet 스케일링 과정에서 네트워크의 폭, 깊이, 해상도의 균형을 잘 맞춰주는 것이 중요하다.
    - Compound Scaling Method
        <p align="center"><img src="https://user-images.githubusercontent.com/62092317/202101456-6e34f738-9300-4829-89a6-489ef4eeef00.png" width = 200></p>

        - depth, width, resolution에 대해 최적의 쎄타 값을 heuristic하게 search
- EfficientDet
    - **Motivation : Object Detection은 특히나 속도가 중요하다!!**

    - How?
        - Efficient multi-scale feature fusion
            - 서로 다른 정보를 갖고 있는 feature map을 단순합 하는게 맞을까?
            - cross-scale connections: 여러 resolution의 feature map을 가중 합
        - Model scaling
            - EfficientNet과 같은 compound scaling 방식을 제안

### (7강) Advanced Object Detection 1

- Cascade RCNN
    - Motivation : Iou threshold를 어떻게 결정 해야하는가?

        <p align="center"><img src="https://user-images.githubusercontent.com/62092317/202150434-7f8cf8e9-b0d7-4e0f-abf9-c293cef73b81.png" width = 400></p>

        - Input IoU : Region Proposal Network 를 통해 나온 bbox와 gt간 IoU
        - Output IoU : Prediction bbox와 gt간 IoU
        - IoU threshold가 높을 수록, **높은 Input IoU에 대해서는 높은 localization 결과를 보여주고 낮은 Input IoU의 bbox에 대해서는 낮은 localization 결과를 보여준다.**
        - 학습되는 IoU에 따라 대응 가능한 IoU 박스가 다름
        - High quality detection을 수행하기 위해선 IoU threshold를  높여 학습할 필요가 있으나, 성능이 하락하는 문제가 존재
        - ** 서로 다른 IoU threshold에 대해 순차적으로 모델을 학습 하자!**
    - Faster RCNN과 다르게 IoU threshold가 다른 Classifier C1,C2,C3를 학습
    - 여러 개의 RoI head 학습하여 head 별로 IoU threshold를 다르게 설정
    - AP90 (high quality detection)에 대해 큰 성능 향상

- Deformable Convolutional Networks(DCN)
    - 일정한 패턴을 지닌 CNN은 **geometric transformations에 한계를 지님** 
    - Deformable convolution 이란?

        <p align="center"><img src="https://user-images.githubusercontent.com/62092317/202156710-41005fae-566a-4229-baed-9f35385e342c.png" width=300></p>

        - **Grid 영역 내부에 대한 convolution 연산이 아닌, 각 영역에 대한 offset을 학습시켜 연산 위치를 유동적으로 변화**
        - Object detection과 segmentation task에서 성능 향상

- DETR(End to End Object Detection with Transformer)
    - Transformer를 처음으로 object detection에 적용
    - 기존의 object Detection의 hand-crafted post process 단계(non max supression)를 transformer를 이용해 없앰
    - Architecture

        <p align="center"><img src="https://user-images.githubusercontent.com/62092317/202166586-5968c6a4-830f-4f6c-8318-f8a6ee39e5f0.png" width=450></p>
        
        - CNN으로 추출한 image feature를 encoder-decoder 구조에 통과시켜 bbox를 predict
        - N개의 output bbox(한 이미지에 존재하는 object 개수 보다 높게 설정)
            - Groundtruth에서 부족한 object 개수 만큼 no object로 padding처리
            - Groundtruth와 prediction이 일대일 매핑
            - Unique한 N값을 얻으므로 post-processing 과정이 필요 없음

### (8강) Advanced Object Detection 2

- YOLOv4

    - Contribution
        - BOF(Bag of Freebies) : Inference 비용을 늘리지 않고 정확도 향상시키는 방법
        - BOS(Bag of Specials) : Inference 비용을 조금 높이지만 정확도가 크게 향상하는 방법
        - GPS 학습에 더 효율적이고 적합하도록 방법들을 변형
    - Bag of Freebies
        
        <p align="center"><img src="https://user-images.githubusercontent.com/62092317/202174541-06390499-92c7-4af7-ac8c-c4b09045574d.png" width=450></p>

    - Bag of Specials

        <p align="center"><img src="https://user-images.githubusercontent.com/62092317/202174847-b6cd1950-01fd-4c89-8bf8-6bb95af33202.png" width=450></p>

        <p align="center"><img src="https://user-images.githubusercontent.com/62092317/202175323-046e5e6f-77e6-4ba1-9684-0842bc4a91b1.png" width=450></p >

        - Enhance receptive field
            - SPP(Spatial Pyramid Pooling)
        - Attention Module
            - Squeeze and Excitation block 
            - Convolutional Block Attention Module(CBAM)
        - Feature Integration
            - FPN, ASFF
        - Activation function
        - Post-processing method
            - Soft NMS, DIoU NMS
    - Architecture
        - 기존에 사용되던 backbone인 DarkNet을 Cross Stage Partial Network(CSPNet)로 사용
    - Additional Improvement
        - Mosaic
        - Self-Adversarial Training : 4장의 이미지를 하나로 합침
        - Cross mini-batch normalization
    - **이러한 방법들은 모든 dataset에 대해 일반화된 방법론들이 아니므로, 제시된 여러 방법들을 본인의 직관을 가지고 판단하여 적용하는 것이 중요**

- CornerNet
    - Anchorbox 의 단점
        - Anchorbox의 수가 많다보니, positive sample이 적고 대부분이 negative sample(배경)이라 class imbalance가 심함
        - 개수, 사이즈, 비율과 같은 hyper-parameter를 고려해야함
    - CornerNet은 중심점 기반의 4개 점을 예측하는 것이 아닌, corner 두 개(top-left corners, bottom-right corners)만을 예측
    - Hourglass : Global,local 정보를 모두 추출
        - Encoder : convolution layer + maxpooling layer
        - Decoder : Encdoer과정에서 스케일별로 추출한 feature를 조합
    - Detecting corner
        - 2개의 heatmap을 통해서 예측, 각 채널은 class에 해당하는 corner의 위치를 나타내는 binary mask
    - Grouping corner
        - Top-left코너와 bottom-right코너의 짝을 맞춰주는 과정
        - Top-left코너와 bottom-right코너의 임베딩 벡터간 거리가 작으면 같은 물체의 bbox위에 있다고 판단
    - CornerPooling
        - 코너에는 특징적인 부분이 없으므로, 코너에 객체에 대한 정보를 집약시키기 위함

### 멘토링

- 멘토님의 질문 : Pretrained weight를 쓰는 것이 항상 좋을까요?
    - ImageNet 도 자체적인 bias를 갖고 있다.
    - 우리가 봤을때 시각적인 점,선,면에 대한 개념이 유지되는 데이터셋이라면 ImageNet pretrained를 쓸 수 있다. 
