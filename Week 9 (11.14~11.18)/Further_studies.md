# Week 9 : Object Detection

## Contents 

- Course
    - (1강) Object Detection Overview
    - (2강) 2 Stage Detectors
    - (3강) Object Detection Library
    - (4강) Neck
    - (5강) 1 Stage Detectors
    - (6강) EfficientDet
    - (7강) Advanced Object Detection 1
    - (8강) Advanced Object Detection 2
    - (9강) Ready for Competition

- Further questions : 다음 의문점들을 풀고 정리한다.

    1. **초기 YOLO는 왜 2개의 boundingbox를 예측했을까?**
    2. **EfficientNet의 objective 수식과 parameter들의 타당성에 대하여**
    3. **AuGNet에 대하여**
        - Adaptive Spatial Fusion
        - Soft RoI selection
    4. **Convolution 구현 살펴보기**
        - Atrous Convolution
        - Deformable Convolution

- About mission
    - FasterRCNN : RoI pooling을 통해 얻은 roi결과들과 gt bbox간 IoU를 구할때 , 왜 gt bbox를 positive sample로 간주할까?

- Mentoring
    - Pretrained weight를 사용하는 것이 항상 올바른 선택일까요?

---
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

- (9강) Ready for Competition!
    - mAP에 대한 오해 

        <p align="center"><img src="https://user-images.githubusercontent.com/62092317/202412203-e83eaa1d-3a09-44af-b382-cdbf4d034db9.png" width=300></p>

        - 왼쪽은 bbox threshold를 높게 준 경우, 오른쪽은 bbox threshold를 낮게 준 경우
        - 오른쪽의 경우 bbox가 무분별하게 많으나, mAP가 왼쪽보다 높다. 과연 오른쪽이 더 좋은 detection 결과라고 볼 수 있을까?
            - AP는 bbox의 수에 대한 패널티를 부여하지 않는다. 
            - Bounding box수를 늘려 recall을 1에 가깝게 만들 수록 AP는 기존 보다 높게 측정된다.
            - 즉, 낮은 bbox threshold를 낮출수록 AP 측정 관점에서 유리하다.
        - **추후 모델을 앙상블 하는 경우에 과도한 bounding box에 의한 문제가 발생할 수 있다.**
        - 실제 연구에서는 대부분 0.05를 threshold로 mAP 평가

    - Validation set 찾기
        - Validation set의 스코어가 올랐을 때 Public과 Private score가 모두 상승하는지를 확인
        - Stratified K-fold Validation
            
            <p align="center"><img src="https://user-images.githubusercontent.com/62092317/202416797-a65a6af3-63eb-4573-99f5-6b4627f1d07d.png" width=400></p>

            - 데이터의 분포가 imbalance한 상황에서 fold마다 유사한 데이터 분포를 갖도록 하는 방법
---

## Further Questions

### 1. 초기 YOLO는 왜 2개의 boundingbox를 예측했을까?
- YOLOv1은 하나의 셀에 대해 1x30 의 vector를 predict한다. 30은 2개의 bounding box에 대한 정보(x,y,w,h,c) + 20개의 class에 대한 score이다. (7x7 grid에 대해 최종 예측 7x7x30)
- **그렇다면 왜 1개도 3개도 아닌 2개의 bbox를 예측했어야 했는지가 궁금해졌다. 어차피 여러개도 아닌 2개를 예측할 것이었다면, 나라면 1개를 예측하는 것부터 시작했을 것이다. 저자는 왜 이런 선택을 했을까?**
    - 이에 대한 이유 또는 ablation study가 논문[[YOLOv1]](https://arxiv.org/abs/1506.02640)에 나와 있을줄 알았는데, 다음과 같은 문장말고는 근거를 찾을 수 없었다.

        > For evaluating YOLO on PASCAL VOC, we use S = 7,
        B = 2. PASCAL VOC has 20 labelled classes so C = 20.
        Our final prediction is a 7 × 7 × 30 tensor.
    - 구글링을 하다가 이와 관련된 3개의 글을 찾았다.
        1.  [[YOLO정리글]](https://curt-park.github.io/2017-03-26/yolo/)
            - 개념 정리도 정리지만, 댓글의 질문들을 보며 YOLO에 대한 이해를 높일 수 있다.
            
                <p align="center"><img src="https://user-images.githubusercontent.com/62092317/202445416-e831ea65-9a23-47a7-91e2-d7cc05863c75.png" width=600></p>
        2. [[Inflearn 질문: YOLO는 왜 2개의 bbox를 예측하나요]](https://www.inflearn.com/questions/551824)
        
        3. [[Towards data science: YOLO]](https://towardsdatascience.com/yolo-you-only-look-once-real-time-object-detection-explained-492dc9230006)
            - 이 글에서 어느정도는 궁금증을 해소 할 수 있었다.

                > YOLO predicts **multiple bounding boxes per grid cell.** At training time we only want one bounding box predictor to be responsible for each object. We assign one predictor to be “responsible” for predicting an object based on which prediction has the highest current IOU with the ground truth. **This leads to specialization between the bounding box predictors.** Each predictor gets better at predicting certain sizes, aspect ratios, or classes of object, improving overall recall.
                
                > **YOLO imposes strong spatial constraints on bounding box predictions since each grid cell only predicts two boxes and can only have one class. This spatial constraint limits the number of nearby objects that our model can predict.** Our model struggles with small objects that appear in groups, such as flocks of birds.

            - 정리하자면, 기본적으로 YOLO는 한 cell내에 존재하는 다중 객체에 대한 detection을 수행할 수 있게 하기 위해 cell마다 1개의 bbox가 아닌 복수 개(2개)의 bbox를 예측한다. 학습 되는 과정 속에서 두개의 bbox중 하나만을 원하기 때문에 gt와 IoU가 가장 높은 bbox를 선택하게 되고 이것이 prediction 방식을 더 **구체화(specialize)** 시킨다는 것이다. **만약 bbox를 1개만 예측했다면, 단순히 비교군이 없기 때문에 IoU가 적당히 높은 수준에서 학습이 멈추고 size, aspect ratio에 대한 세밀한 detection을 하지 못하는 것이 아닐까?** 1 stage detector이기 때문에 이런 설정이 필요한 것으로 보인다.
            - 그러나, 2개만을 예측하는 것(더 많이 예측하지 않는 것)을 YOLO의 limitation으로 보기도 한다. 그 이유는, cell 주변에 2개 이상의 작은 object가 있을 때 대응하지 못하기 때문이다.

### 2. EfficientNet의 objective 수식의 타당성에 대하여

- EfficientNet의 objective는 너무 당연해 보인다. network의 성능을 maximize하는 depth, width, resolution을 구하는 것.. 너무 당연해 보이지 않는가? **그러나, 그걸 찾아나가는 방식을 당연시해서는 안된다고 생각한다.**

- 저자는 왜 **exponential equation**을 토대로 $\alpha, \beta,\gamma$ 을 찾으려고 했을까?
    - 맨 처음 들었던 의문점이다. $d=\phi\alpha, w= \phi\beta, r = \phi\gamma$ 와 이 linear equation으로 search해도 되지 않았을까?

    <p align="center"><img src="https://user-images.githubusercontent.com/62092317/202458891-ccc6c587-2698-4f00-905d-84e4302dbd1e.png" width = 600></p>


    - 오른쪽은 논문에서 제시한 compound scaling method의 수식이고, 왼쪽은 $\phi$ 값에 따른 최적의 $\alpha, \beta, \gamma$ 값을 찾을 때 linear한 식과 exponential한 식의 차이를 나타낸 것이다.
    - 지수함수의 경우 같은 $\phi$에 대해, 변수의 값이 linear하게 증가할때 수식의 값은 빠르게 증폭된다.
    - 이러한 경향성이 **Layer가 쌓임에 따라 Deep Neural Network의 representation power가 증가하는 경향성**과 비슷하다는 가정을 한것이 아닐까? 또한, non-linearity가 보장되는 nerual network의 capacity가 linear하게 증가할 것이라는 가정은 non-sense처럼 보인다. 따라서 neural network의 표현력이 증가하는 경향성을 가장 간단하게 잘 표현할 수 있는 수단이 exponential이었을 것이라고 추측해본다.

- Constraint $\alpha \times \beta^2 \times \gamma^2 = 2$ 는 어떻게 정했을까?
    - 위 그림에서 알 수 있듯이 저자는 small grid search를 통해 $\alpha,\beta,\gamma$ 값을 정했다고 한다.
    - $\alpha \times \beta^2 \times \gamma=k$ 라고 가정해보자.
    - 각 parameter $\alpha,\beta,\gamma$ 를 동일한 초기값에서 시작하여 searching한다고 할 때 그 값은 $\sqrt[3]{k}$ 일 것이다.
    - k=2 일 경우 $\sqrt[3]{2}$의 값은 1.25992, 논문에서 사용한 $\alpha=1.2, \beta=1.1,\gamma=1.15$ 와 비교해볼때 small grid search로 찾기에 적합한 초기값임을 알 수 있다.
    - 논문에서 $\phi$는 모델의 scaling된 정도를 model 이름의 B뒤에 붙여 naming하기 위해 사용된다. 예를 들어, EfficientNet-B0에서 $\phi=0$이다.
    - B0~7까지 $\phi$의 값은 0,0.5,1,2,3.5,5,6,7 이다.

        <p align="center"><img src="https://user-images.githubusercontent.com/62092317/202469894-ee9e2cfc-52d5-4670-9731-972d0ef70ba9.png" width=600></p>

    - Efficient-Net B7($\phi=7$)의 경우 k=2 일때 $\alpha=1.2^7=3.58, \beta=1.1^7=1.95,\gamma=1.15^7=2.83$ 이고, 이 값들은 7제곱 값임에도 $\phi$를 0부터 7까지 키워나가며 단계적인 성능 향상을 보여주기에 알맞은 scaling value라고 생각하지 않았을까?
    - 만약 k=3 이었다면?
        - $\sqrt[3]{3}=1.44, 1.44^7=12.83$ 이 된다. **모델의 깊이를 10배 이상 키우며 성능 향상을 도모하는 것은 효율적인 모델 크기를 찾고자하는 논문의 의도**에 맞지 않는다.
        
    - 결국 k=1보다 커야하는 상황에서 논문의 objective와 알맞는 유일한 값은 **2** 뿐이었던 것!!

    
### 3. [AugFPN](https://arxiv.org/abs/1912.05384)에 대하여

- Adaptive Spatial Fusion

    <p align="center"><img src="https://user-images.githubusercontent.com/62092317/202600558-da613400-355a-46bb-b949-64399d18f36f.png" width = 400></p>

    <p align="center"><img src="https://user-images.githubusercontent.com/62092317/202602115-a007a1c6-d776-4449-ae48-cc82491fbbda.png" width = 400></p>

    - AugFPN은 P5가 다른 level의 feature map과 달리 상위 feature map으로 부터의 정보 전달을 받지 못하여 발생하는 정보 손실을 지적하여, M6 feature map을 만드는 **Residual Feature Augmentation(이하 RFA)** 을 제안
    - 논문에 제시된 RFA의 구조를 살펴보면 다음과 같은 순서로 C5로 부터 M6를 생성한다.
        1. Ratio-invariant adaptive pooling(RAP)를 통해 $h \times w$ size를 가진 서로 C5를 다른 3개의 작은 feature map으로 down scaling(alpha = 0.1,0.2,0.3)
        2. 이후 scaling 된 feature map들을 1x1 convolution에 통과시켜 각각 채널 크기를 256으로 조정(figure엔 표기 생략)
        3. 3개의 feature map을 bilinear interpolation을 통해 원래의 크기인 $ h \times w$ 로 다시 upscaling하고, **Adaptive Spatial Fusion(이하 ASF)** 을 수행
        4. ASF는 feature map을 weighted aggregation하여 최종 feature map을 생성하기 위한 방법으로, ASF 그림의 왼쪽 route를 통해 각 feature map을 사용하여 가중치를 생성하고 이를 원래의 feature map에 channel-wise multiplication 후 summation
        5. 가중치를 생성하는 과정
            - 4개의 feature map을 물리적으로 concat하여 하나의 feature vector로 만듦
            - 이후, 이를 1x1 convolution에 통과시켜 feature level간 fusion을 학습
            - Fusion된 feature map을 3x3 convolution에 통과시켜 spatial한 정보를 추출
            - 추출된 값이 weight로 사용될 수 있도록 sigmoid에 통과시켜 0~1의 값을 가지도록 조정
    - 위 과정에서 알 수 있듯이, C5를 ratio-invariant하게 줄였다가 다시 원래의 크기로 upsampling한 후 합치는 것을 알 수 있다. 이때 ASF가 꼭 필요할까? 3개의 feature map을 단순히 합치지 않은 이유가 무엇일까?
        - 논문에서 다음과 같은 근거를 찾을 수 있었다.
            >  Considering the **aliasing effect caused by interpolation**, we design a module named Adaptive Spatial Fusion (ASF) to adaptively combine these context features instead of simple summation. 

            - 즉, Upsampling에서의 interpolation을 통해 발생할 수 있는 aliasing effect를 피하기 위해 ASF를 도입한 것
            - 그렇다면 **aliasing effect**란 무엇인가?

                <p align="center"><img src="https://user-images.githubusercontent.com/62092317/202606165-0eb6971e-12f2-4582-a512-05ed08328b31.png"></p>

                - Image processing에서 upsampling시 interpolation에 의해 왼쪽 그림과 같은 **moire pattern**이 발생하게 된다.
                - Moire pattern에 대한 설명[[Stack Exchange: What is moire?]](https://photo.stackexchange.com/questions/11909/what-is-moir%C3%A9-how-can-we-avoid-it)

                    <p align="center"><img src="https://user-images.githubusercontent.com/62092317/202606945-df8c42c8-2615-44e1-a803-ee04f5e15b32.png"></p>

        - RFA의 목적을 다시 생각해보면, feature map C5을 다양한 scale의 관점에서 해석한 M6를 만들기 위해 pooling & upsamping을 했어야했고 이에 수반되는 문제점(aliasing effect)을 해결하기 위해 ASF를 도입한 것으로 해석된다.
    - 위의 분석을 토대로 생각해볼때, Adaptive spatial fusion은 **network가 feature map들의 중요도를 adaptive하게 결정하여 spatial(3x3 conv) filter를 통해 각 feature map을 fusion한다**는 의미를 가진다.
    - Ablation studies for RFA

        <p align="center"><img src="https://user-images.githubusercontent.com/62092317/202609580-e295b523-7333-417d-b921-074dc4ba1841.png" width = 600></p>
        
        - 논문에선 ASF와 RA-AP 사용 여부와 이때 사용하는 feature map의 개수, alpha값 변화에 따른 성능 변화를 연구하였다.
        - Baseline과 비교했을때, 예상과 다르게 feature map을 단순 summation을 하여도 GMP을 제외한 pooling방법에서 AP가 상승된다. **이는, summation에 의해 밝생하는 aliasing effect를 고려하더라도 upsampling이전에 채널수를 맞춰주는 1x1 conv에서 parameter들이 관여하는 영향력이 꽤 크기 때문이 아닐까?**
        - RA-AP는 단순히 pooling 방식임에도 불구하고, 동일한 조건하에 다른 pooling 방식보다 꽤 큰 성능 향상을 가져온다. Network가 multi-scale feature를 잘 추출할 수 있게 하는 일종의 augmentation처럼 동작하는 듯 하다.
        - 4,5번째 행에서 확인할 수 있듯이, ASF 사용시 한개의 feature map만으로 3개의 feature map을 단순 summation 한 것과 비슷한 성능을 얻을 수 있다. 
        - ASF에서 사용하는 feature map의 수가 증가할수록 성능이 향상되는 것으로, feature map간 weighted summation을 고려하여 설계된 ASF의 타당성이 입증된다.
        - **특히, feature map 4개(0.1,0.2,0.3,0.4)를 사용할 경우 3개(0.1,0.2,0.3)을 사용할 때보다 $AP_{l}$이 0.5% 상승하는 것을 알 수 있다. 이는 large object에 대해 다양한 scale의 feature map으로 부터 spatial fusion을 하는 것이 중요하다는 사실을 입증한다.**
- Soft RoI Selection
    - Soft RoI selection(SRS)는 기존 모델들의 문제점을 해결하기 위해 등장했다.
        - FPN과 같이 하나의 feature map에서 RoI를 계산하는 경우 sub-optimal
        - PANet은 모든 level의 feature map를 사용하나, max 값을 가지는 level의 feature만을 사용하기 때문에 정보 손실이 발생
    - **그렇다면 SRS는 어떻게 feature map간 fusion을 진행하는가?**

        <p align="center"><img src="https://user-images.githubusercontent.com/62092317/202616342-6c31a8f3-1766-49bb-8a3b-9f9b8ef5c2dd.png"></p>

        - Steps             
            - 먼저 4개 level feature map $ C \times H \times W$ 을 GMP 한 후 concat 하여 $4C \times 1 \times 1$ 벡터를 생성
            - $4C \times C/4 \times 1 \times 1$(#param=C^2)에 통과시켜  $C/4 \times 1 \times 1$ 벡터로 만들어 각 level의 channel 정보를 fusion
            - 다시 $C/4 \times 4C \times 1 \times 1$(#param=C^2)에 통과시켜 원래 channel수로 expand 하고, 4개로 나누어 각각을 feature map의 weight로 사용
            - feature map과 weight vector를 Channel-wise multiplication한 후 weighted summation 수행
        - Weight를 만들어내는 과정은 ASF모델 내부와 매우 유사한데, **한가지 다른점은 3x3 conv filter 대신 1x1 conv filter만을 사용**했다는 것이다. 논문에선 이 둘을 구분하여 ASF를 spatial fusion으로, SRS를 adaptive channel fusion으로 명명하고 있다.
    - Squeeze and Excitation block과의 유사성[[Squeeze-and-Excitation Networks]](https://arxiv.org/abs/1709.01507)
        - SRS를 보자마자 SE-Net이 떠올랐다.[[참고자료]](https://jayhey.github.io/deep%20learning/2018/07/18/SENet/)
        
        <p align="center"><img src="https://user-images.githubusercontent.com/62092317/202618341-095bb7b1-fc25-4e0b-b3f6-98b3be1e69bb.png"></p>

        - Squeeze : Global information Embedding
            - 각 채널들의 중요한 정보만을 추출
            - GAP를 사용하여 global spatial information을 한개의 channel을 대표하는 descripter로 압축(SRS에서는 GMP 사용)
        - Excitation : Adaptive Recalibration
            - 압축한 중요한 정보들을 재조정(Recalibrate)
            - 채널 간 의존성(channel-wise dependencies)를 계산
        - 채널간 의존성을 계산한다는 측면에서 유사한 구조를 갖는 SRS은 **모델이 각 level feature map들을 살펴보며 중요도를 판단** 할 수 있게 하는 역할을 할 것이다.
    
    - Ablation studies for SRS
        
        <p align="center"><img src="https://user-images.githubusercontent.com/62092317/202619880-d7696b87-4edf-4e9c-8405-84cbfd57fe92.png"></p>

        - SRS사용할 경우 fusion type에 상관 없이 항상 성능이 향상 된다. (fusion type의 sum,max는 PANet에서 extra fc layer를 사용하지 않고 RoI feature를 사용한 것이라고 한다.)
        - **모든 정보를 그대로 사용하는 summation보다 ACF,ASF의 성능이 좋은 것을 볼 때 모델이 feature map간 가중치를 스스로 결정하는 것이 학습에 도움이 되는 것을 알 수 있다.**
        - 추가로, ASF(3x3 conv)를 사용한 경우에 ACF보다 AP가 높은 것을 알 수 있다. **특히, $AP_{l}$에 대해 0.7%의 꽤 큰 성능차이가 나는 것을 확인할 수 있는데, 이는 large object들에 대해선 spatial feature를 추출하는 것이 중요하기 때문으로 보인다. ASF가 의도대로 동작하고 있음을 보여주는 증거가 아닐까?**


### 4. Convolution 구현 살펴보기
- Atrous Convolution

    ``` python
    import torch.nn as nn

    class DilConv(nn.Module):
        def __init__(self, C_in, C_out, kernel_size, stride, padding,dilation, affine=True):
            self.op = nn.Sequential(
                nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding,dilation=dilation, groups=C_in, bias=False),
                nn.Conv2d(C_in, C_out , kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(C_out, affine=affine),
                nn.ReLU(inplace=False)   
            )

        def forward(self, x):
            return self.op(x)
    ```
        
    - Atrous Convolution을 어떻게 구현하는 건지 궁금해서 code를 찾아봤는데, 알고보니 nn.Conv2d의 parameter에 이를 구현할 수 있게 해놓았더라..
    - Parameter중 dilation 이란 argument(default=1)을 조절하면 atrous convolution(=dilated convolution)을 사용할 수 있다.
        
- Deformable Convolution
    - 예상했듯이 offset을 학습시킨다는 개념을 코드로 구현하기는 쉽지 않아보인다.[[LINK]](https://github.com/oeway/pytorch-deform-conv/blob/d61d3aa4da20880c524193a50f6e9b44b921a938/torch_deform_conv/layers.py#L10)
    - 찾아보니 Pytorch 공식 문서에 deform_conv2d로 API화 되어있다.[[LINK]](https://pytorch.org/vision/main/generated/torchvision.ops.deform_conv2d.html)

---
## About Mission

- Mission2 : FasterRCNN
    > RoI pooling을 통해 얻은 roi결과들과 gt bbox간 IoU를 구할때 , 왜 gt bbox를 positive sample로 간주할까?
    - FasterRCNN의 code 중 강의에서 배운 내용과 다르게 구현되어 있는 부분이 있다.
        ``` python
        def __call__(self, roi, bbox, label,
                    loc_normalize_mean=(0., 0., 0., 0.),
                    loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
            n_bbox, _ = bbox.shape

            roi = np.concatenate((roi, bbox), axis=0) ## 의문점이 드는 부분

            pos_roi_per_image = np.round(self.n_sample * self.pos_ratio) # positive image 갯수 = 32
            iou = bbox_iou(roi, bbox) # RoI와 bounding box IoU
            gt_assignment = iou.argmax(axis=1)
            max_iou = iou.max(axis=1)
        ```
        - 위 코드는 ProposalTargetCreator 의 call 함수로, **roi pooling(2000개) 후 각각의 roi에 대해 image의 ground truth bounding box와 ioU를 계산** 한다. (ex : roi 2000개, gt_bbox 10개의 경우 20000번의 ioU 연산을 하는 것)
        - 이후 각 roi 에 대해 가장 ioU가 높은 gt_bbox가 무엇인지를 판단한다.(가장 연관성이 높은 gt)
        - 해당 ioU 값이 pos_threshold보다 크면, 이 roi를 positive sample로 여기며 top N개의 positive sample을 선별 한다.
    - 이 과정에서, roi 와 gt_bbox(code에선 bbox로 표기)를 concat 후 gt_bbox와의 ioU를 계산하면 concat 한 gt_bbox에 대해선 당연히 ioU 값 1을 얻는다.(자기 자신과의 ioU이므로)
    - ioU는 1을 초과할 수 없으므로 이 bbox들은 대해선 pos_threshold 기준을 넘는 것은 물론이고, Top N개의 postive sample에 **무조건** 포함 되어 이후 학습(loss)에 관여하게 된다.
    - **즉, ground truth를 마치 roi 처럼 여기고 이후 ground truth와 비교하는 학습에 포함시키는 행위이다. 왜 이렇게 학습하는 것일까?**
        - 내가 미처 확인하지 못한 사안이 있을지도 몰라서 official code[[LINK]](https://github.com/chenyuntc/simple-faster-rcnn-pytorch/issues/143)를 확인 해보았다.
            - Issue([[About creator_tool.py#143]](https://github.com/chenyuntc/simple-faster-rcnn-pytorch/issues/143),[[About concatenate roi and bbox#56]](https://github.com/chenyuntc/simple-faster-rcnn-pytorch/issues/56))에 이에 대한 질문들이 올라와 있었는데 명쾌한 정답을 찾을 수 없었다.
        - 또 다른 official code[[LINK]](https://github.com/endernewton/tf-faster-rcnn/blob/master/lib/layer_utils/proposal_target_layer.py)엔 해당 부분이 다음과 같이 구현되어 있다.
            ```python
            # Include ground-truth boxes in the set of candidate rois
            if cfg.TRAIN.USE_GT:
                zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
                all_rois = np.vstack(
                (all_rois, np.hstack((zeros, gt_boxes[:, :-1])))
                )
                # not sure if it a wise appending, but anyway i am not using it
                all_scores = np.vstack((all_scores, zeros))
            ```
            - 코드의 주석에도 알 수 있듯이, 이렇게 하는 것에 대한 이유는 없어보였고 config파일로 부터 USE_GT라는 변수를 받아 gt의 concat여부를 정할 수 있도록 짜여져있다.
        - 조교의 의견, 다른 캠퍼의 의견 그리고 내 생각을 종합해 봤을때
            - 이와 같은 trick은 기본적으로 detection task가 classification보다 hard하다는 것을 전제로 이해해야 한다. **즉, classification에서 gt label을 prediction으로 쓰는 것은 nonsense이지만 detection은 bbox를 예측해야 하기 떄문에 그럴 수 있다는 것이다.**
            - Image내에 다수의 물체가 존재하고, 그 물체의 size가 작다고 가정할때 학습 초기에 gt와 IoU가 높은 Top N개의 RoI를 신뢰할 수 있을까? 또 매 iteration 마다 예측 되는 Top N개의 roi들은 학습에 사용할 만큼 정확하다고 할 수 있는가?
            - 우리의 코드에서는 n_sample = 128, pos_ratio = 0.25이기 때문에 Top 32개의 positive sample을 뽑는다. 이 때, 이미지 내 객체의 개수가 3개라고 한다면 약 10%의 positive roi가 gt가 된다.
            - 학습 초기단계의 detection box의 uncertainty에 의한 불안정성을 고려할 때, 이 positive roi들은 **안정적인 학습을 위한 최소한의 안전장치** 역할을 해줄 것이다. **학습이 많이 진행된 시점에선 큰 의미가 없을지 몰라도, 학습 초기 단계에 positive sample의 일부로써 올바른 방향으로의 학습을 지속적으로 유도하고자 함이 concat의 목적**이라고 생각된다. 
    
---
## 멘토링

- 멘토님의 질문 : Pretrained weight를 쓰는 것이 항상 좋을까요?
    - ImageNet 도 자체적인 bias를 갖고 있다.
    - 우리가 봤을때 시각적인 점,선,면에 대한 개념이 유지되는 데이터셋이라면 ImageNet pretrained를 쓸 수 있다. 
