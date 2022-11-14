# Week 9 : Object Detection

## Contents 

- Course
    - (1강)Object Detection Overview

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
