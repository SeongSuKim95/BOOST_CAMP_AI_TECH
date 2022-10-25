# Week 6 : CV competition

## Contents 
1. About Competition
    - Overview
    - Problem Definition
    - Data description
2. Data Feeding

---

## About Competition

<p align="center"><img src="https://user-images.githubusercontent.com/62092317/197450654-18320800-6006-4626-b1e1-586b675fdf40.PNG" width= 500></p>

- Overview

    >감염자의 입, 호흡기로부터 나오는 비말, 침 등으로 인해 다른 사람에게 쉽게 전파가 될 수 있기 때문에 감염 확산 방지를 위해 무엇보다 중요한 것은 모든 사람이 마스크로 코와 입을 가려서 혹시 모를 감염자로부터의 전파 경로를 원천 차단하는 것입니다. 이를 위해 공공 장소에 있는 사람들은 **반드시 마스크를 착용해야 할 필요가 있으며, 무엇 보다도 코와 입을 완전히 가릴 수 있도록 올바르게 착용하는 것이 중요**합니다. 하지만 넓은 공공장소에서 모든 사람들의 올바른 마스크 착용 상태를 검사하기 위해서는 추가적인 인적자원이 필요할 것입니다.
    따라서, **우리는 카메라로 비춰진 사람 얼굴 이미지 만으로 이 사람이 마스크를 쓰고 있는지, 쓰지 않았는지, 정확히 쓴 것이 맞는지 자동으로 가려낼 수 있는 시스템이 필요**합니다. 이 시스템이 공공장소 입구에 갖춰져 있다면 적은 인적자원으로도 충분히 검사가 가능할 것입니다.

- Problem Definition 
    1. 내가 지금 풀어야 할 문제가 무엇인가?
        - 사람 얼굴 영상만으로 마스크를 정확히 썼느지를 판단하기
    2. 이 문제의 input과 output은 무엇인가?
        - Input : 
        - Output :
    3. 이 솔루션은 어디서 어떻게 사용되어야 하는가?
        - 이 시스템의 목적을 생각해보면 마스크를 착용했는지 여부 뿐만아니라, 어떻게 착용했는지도 중요하지 않을까? (ex: 턱스크)
        - 마스크를 어떻게 착용했는지도

- Data description

- Domain understanding

- Kaggle Discussion


## EDA 

- EDA(Exploratory Data Analysis)
    - EDA란? 
        - 이 사람이 data를 왜 줘서 나에게 어떤 것을 요구하는가? data가 어떻게 생겼는지, ,,, 내가 이 데이터를 이해하기 위한 노력. 뭐든 상관 없이 데이터에 대해 궁금한 모든 것을 알아보는 노력.
        - 주어진 데이터에 대한 본인의 생각을 자유롭게 서술해보자
            - 모든 샘플이 제대로 찍혀있을까? 중복되는 샘플이 있을까?
            - 데이터의 원천이 여러곳인지 아님 한곳인지? 여러 곳이라면 데이터의 분포엔 어떤 영향이 있을지?

## Dataset

- Pre-processing 

    - 도메인, 데이터 형식에 따라 효과적인 pre-processing 방식이 존재하기도 한다! 

- Generalization : **Data의 일반성을 어떻게 찾을 수 있을까?**
    
    - Bias & Variance trade off [[참고 LINK]](https://towardsdatascience.com/understanding-the-bias-variance-tradeoff-165e6942b229)
    - <p align="center"><img src= "https://user-images.githubusercontent.com/62092317/197497324-eb17bbfa-8e03-4d1b-a2f6-d8c7f313973a.PNG" width=600></p>

- Data Augmentation
    - 주어진 데이터가 가질 수 있는 case(경우), state(상태)의 다양성을 고려하여 augmentation을 고려하는 것이 좋다.
    - Domain의 이해에 기반하여 augmentation을 선택하지 않으면 발생할 수 없는 data를 만들어 오히려 학습에 방해가 될 수 있다.
    - albumentation library 사용하는 것이 효율적이다.
    - 전처리도구가 항상 좋은 결과를 주는 것은 아니기 때문에(not masterkey), 정의된 문제에 대해 전처리를 할 때의 당위성을 예측하고 확인하는 능력을 길러야 한다.

