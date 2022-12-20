# Week 12 : Data Annotation

## Contents 

- Course
    1. 데이터 제작의 중요성 
    2. OCR Technology & Services 
    3. Text Detection 소개 1  
    4. 데이터 소개
        - Special Mission 1 EDA 실습
    5. Annotation Guide
    6. 성능 평가 개요
    7. Annotation Tool 소개 
        - Special Mission 2  리더보드 제출 실습 
    8. Text Detection 소개 2
    9. Bag of tricks 

- About mission

- Mentoring

---
## Course

### (1,2강) 데이터 제작의 중요성 

- Lifecycle of an AI Project

    <p align="center"><img src="https://user-images.githubusercontent.com/62092317/205825958-478b57e7-ce2f-4720-ba92-ece3106a12dd.png"></p>

    - 수업/학교/연구가 **정해진** 데이터셋/평가 방식에서 더 좋은 모델을 찾는 것에 집중하는 것과 달리, AI개발 업무의 상당 부분은 데이터셋을 준비하는 작업이다.
    - 현업에서는 학습 데이터셋이 없는 경우가 훨씬 많다.
    - 서비스향 AI 모델 개발의 4단계
        
        <p align="center"><img src="https://user-images.githubusercontent.com/62092317/205826541-4f5d2260-6f27-4542-ba38-5f23cc280bb1.png"></p>
        
        - 요구사항을 충족시키는 모델을 지속적으로 확보하는 2가지 방법
            - Data Centric  : 데이터만 수정하여 모델 성능 끌어올리기
                - 사용 중인 모델의 성능 개선 및 유지 보수시 접근 방법 
            - Model Centric : 데이터는 고정시키고 모델 성능 끌어올리기
                - 처음 모델 성능을 달성하는 과정에서 사용되는 접근 방법
- Data-related tasks : 생각보다 데이터와 관련된 업무와 많다. 왜?
        
    <p align="center"><img src="https://user-images.githubusercontent.com/62092317/205830539-64240d97-cb3b-48f7-b878-aa07a5dcd261.png"></p>
        
    1. **어떻게 하면 좋을지에 대해서 알려져 있지 않다.**
        - 좋은 데이터는 많이 모으기 힘들고, 작업의 cost가 크기 때문에 학계에선 데이터를 다루기 힘들다.
    2. 데이터 라벨링 작업은 생각보다 어렵다.
        
        - 데이터가 많다고 모델 성능이 올라가는 것이 아니다.
        
        <p align="center"><img src="https://user-images.githubusercontent.com/62092317/205832440-35997ac6-271c-44b8-ad84-52138e77d06b.png"></p>
        
        - 라벨링 노이즈 : 라벨링 작업에 대해 일관되지 않음의 정도
            - 잘못 작업된 라벨링 결과의 영향을 무시하려면 정확히 라벨링된 결과가 2배 이상 필요하다.
        - **골고루, 일정하게** 라벨링된 데이터가 많아야 한다. 
            - 적은 데이터라도 class 간 balance가 맞다면 모델의 예측이 참에 근접할 수 있다.
        - 빈도가 높은 데이터에 대해서는 라벨링 노이즈가 적으나, 희귀하게 관찰 되는 데이터는 작업 가이드가 없고, 작업자간 라벨링 편차가 크기 때문에 노이즈가 크다.  
    3. 데이터 불균형을 바로 잡기가 많이 어렵다.

        <p align="center"><img src="https://user-images.githubusercontent.com/62092317/205886805-1f1256da-df38-42cd-ad55-86290cbc4b08.png"></p>

        - 특이 case에 대한 샘플을 수집하여 이를 포함한 라벨링 가이드를 만들어야 함
        - 효율적인 작업을 위해선 작업자의 충분한 도메인 지식에 대한 경험치를 통한 예외 case에 대한 인지가 중요
        - 테슬라의 경우 자율 주행에서의 예외 경우 221가지를 정의하여 세심히 관리한다.
---
## 멘토링 



---
## 프로젝트 제안서 작성
