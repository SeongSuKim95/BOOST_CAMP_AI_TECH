# Week 8 : AI 서비스 개발 기초

## Contents 

- Course
    - (1강) 강의 소개
    - (2강) MLOPs 개론
        - MLOps가 필요한 이유 이해하기
        - MLOps의 각 Component에 대해 이해하기(왜 이런 Component가 생겼는가?)
        - MLOps 관련된 자료, 논문 읽어보며 강의 내용 외에 어떤 부분이 있는지 파악해보기
        - MLOps Component 중 내가 매력적으로 생각하는 TOP3을 정해보고 왜 그렇게 생각했는지 작성해보기
    - (3강) Model Serving
        - Rules of Machine Learning: Best Practices for ML Engineering 문서 읽고 정리하기!
        - Online Serving / Batch Serving 기업들의 Use Case 찾아서 정리하기 (어떤 방식으로 되어 있는지 지금은 이해가 되지 않아도 문서를 천천히 읽고 정리하기)
    - (4강) 머신러닝 프로젝트 라이프 사이클
        - 부스트캠프 AI Tech 혹은 개인 프로젝트를 앞선 방식으로 정리해보기
        - 실제로 회사에서 한 일이 아니더라도, 특정 회사에서 활용했다고 가정하거나 아예 크게 문제 정의해서 구체화해보기
        - 이 모델이 회사에서 활용되었다면 어떤 임팩트를 낼 수 있었을까? 고민해서 정리해보기!
        - 직접 일상의 문제라도 하나씩 정의하기
    - (5강,6강) Notebook 베이스 - Voila ,웹 서비스 형태 - Streamlit 
    - (7강) Linux & Shell Command
    - (8강) Docker
    - (9강) MLflow
- 논문 스터디 : Diffusion Model
---

## Course

### (1강) 강의 소개

- AI 엔지니어로서 어떤 사람이 되어야 하는가?
    - AI 엔지니어로 출발하기 위한 시작점을 그릴 수 있는 *큰 그림을 인지*하는 사람
    - 직접 문제 정의를 하며, 필요한 도구를 찾아보는 *능동적인 자세*를 가지는 사람
    - *지속적으로 개선*하는 사람
    - 프로토타이핑부터 모델 배포, 모니터링 과정을 이해하는 사람 
- 강의를 듣는 자세
    - 강의의 전체 흐름을 기억하기. 학습하는 부분이 MLOps에서 어디서 활용할 수 있을지 생각하기
    - 외우지 말기. *스토리와 이런 기술이 왜 나왔는지 생각하며 "이건 왜 그럴까?" 고민해보기*
    - 추가 자료 찾아보기. 꾸준히 다양한 자료를 습득하기

### (2강) MLOPs 개론

- MLOps가 필요한 이유 이해하기 : ML(Machine Learning) + Ops(Operations)

    <p align="center"><img src="https://user-images.githubusercontent.com/62092317/200720553-5260dff5-5259-44f4-9ea9-471ac622e2d8.png" width = 400></p>

    - 머신러닝 모델링 코드는 머신러닝 시스템 중 일부에 불과하다.
    - 머신러닝 모델 개발(ML Dev)과 머신러닝 모델 운영(Ops)에서 사용되는 문제, 반복을 최소화하고 비즈니스 가치를 창출하는 것이 목표
    - MLOp의 목표는 빠른 시간 내에 가장 적은 위험을 부담하며 아이디어 단계부터 Production 단계까지 ML프로젝트를 진행할 수 있도록 기술적 마찰을 줄이는 것

    - Research와 Production(실제 서비스 환경)은 어떻게 다른가?

        <p align="center"><img src="https://user-images.githubusercontent.com/62092317/200721207-e7000fc5-da5a-47d8-957e-1751de7c9c67.png" width = 400></p>
    
    - Production에선 dynamic data, 빠른 Inference 속도, Explainable AI, 안정적인 운영이 중요!

- MLOps의 각 Component에 대해 이해하기(왜 이런 Component가 생겼는가?)
    - Ex) 집에서 맛있게 만들던 타코로 레스토랑 장사를 할때 필요한 모든 것...!
    - Infra
    - Serving
        - Batch serving : 많은 데이터를 일정 주기로 한꺼번에 예측
        - Online serving : 한번에 하나씩 실시간으로 예측, 동시에 여러 예측이 가능하도록 병목이 없어야 하고 확장성이 보장되어야 함
    - Experiment, Model Management
    - Feature store (이런 것도 생각해서 만들었구나...싱기방기)
    - Data Validation
        - Data Drift, Model Drift, Concept Drift
    - Continuous Training 
    - Monitoring
    - AutoML

- MLOps 관련된 자료, 논문 읽어보며 강의 내용 외에 어떤 부분이 있는지 파악해보기 (TODO)
    
- MLOps Component 중 내가 매력적으로 생각하는 TOP3을 정해보고 왜 그렇게 생각했는지 작성해보기 (TODO)
    

### (3강) MLOPs Serving

- Serving 이란??
    - Production 환경에 모델을 사용할 수 있도록 배포
    - 머신러닝 모델을 개발하고, 현실 세계에서 사용할 수 있게 만드는 행위
-  Online Serving Vs Batch Serving
    - Online Serving

        <p align="center"><img src="https://user-images.githubusercontent.com/62092317/200743936-82c42082-01b8-4ec2-82ca-77899dee558a.png" width = 400></p>

        - Web Server Basic : HTTP를 통해 웹 브라우저에서 요청하는 HTML 문서나 오브젝트를 전송해주는 서비스 프로그램
        - Server는 client로부터 request를 받아서 response! (server: 레스토랑의 웨이터, client: 손님들)
            - Ex: 우리가 사용했던 Mask classification 모델도 웹서버에서 client가 제공한 이미지를 server가 인식하고 모델에 reference하여 prediction값을 response한 것!
        - API(Application Programming Interface) :  운영체제나 프로그래밍 언어가 제공하는 기능을 제어할 수 있게 만든 인터페이스(ex : pandas,pytorch 라이브러리의 함수들)
        - 클라이언트(Application)에서 ML 모델 서버에 HTTP 요청(Request)하고, 머신러닝 모델 서버에서 실시간으로 예측한 후, 예측 값(응답)을 반환(Response)
        - 구현 방식
            1. 직접 API 웹 서버 개발: Flask,FastAPI등을 사용해 서버 구축
            2. 클라우드 서비스 활용 : AWS의 SageMaker, GCP의 Vertex AI등
            3. Serving 라이브러리 활용 : Tensorflow Serving, Torch Serve, MLFlow, BentoMl 등

                ``` python
                # Import the IrisClassifier class defined above 
                from iris_classifier import IrisClassifier
                # Create a Iris classifier service Instance
                iris_classifier_service = IrisClassifier()
                # Pack the newly trained model artifact
                iris_classifier_service.pack('model',clf)
                # Save the prediction service to dist for model serving
                saved_path = iris_classifier_service.save()
                ```
        - 구현시 고려해야하는 부분
            - 재현 가능한 코드로 만들어야 한다!(Pacakge dependency)
            - 실시간 예측을 하기 때문에 지연시간(Latency)를 최소화 해야한다! 
    
    - Batch Serving

        <p align="center"><img src="https://user-images.githubusercontent.com/62092317/200748447-31a509be-2f5d-4047-bb2e-0e017e0df457.png" width = 400></p>
    
        - 특정 주기에 대해 반복해서 학습을 하거나 예측을 하는 serving 방식
        - Ex: 추천시스템에서의 1일 전에 생성된 컨텐츠에 대한 추천 리스트 예측 (Spotify의 Discover Weekly)

- Rules of Machine Learning: Best Practices for ML Engineering 문서 읽고 정리하기!(TODO)
    
- Online Serving / Batch Serving 기업들의 Use Case 찾아서 정리하기 (어떤 방식으로 되어 있는지 지금은 이해가 되지 않아도 문서를 천천히 읽고 정리하기)(TODO)
    
### (4강)머신러닝 프로젝트 라이프 사이클

1. 문제 정의
    - 특정 현상을 파악하고, 그 현상에 있는 문제를 정의하는 과정
    - 문제를 잘 풀기(Solve) 위해선 문제 정의(Problem Definition)이 매우 중요
    - 머신러닝 알고리즘, 개발 능력도 중요하지만, 근본적인 사고 능력도 중요 (How 보다 Why에 집중)
    - 현상 파악이 선행되어야 한다!
        - 어떤 일이 발생하고 있는데, 무엇을 해결하고 싶고 무엇을 알고 싶은가?
        - 현상을 구체적으로 명확한 용어로 정리해보기
    - 문제의 해결방식은 다양하기 때문에 쪼개서 파악하자
    - 해결 방식 중에서 데이터로 해결할 수 있는 방법을 고민해보기(Rule base부터 점진적으로 실행해보기, deep learning이 만능은 아니다!)

2. 프로젝트 설계
    - 단계별 접근
        - 문제 정의 후 최적화할 Metric을 선택
        - 데이터 수집 후, 레이블 확인
        - 모델 개발
        - 모델 예측 결과를 토대로 Error Analysis. 잘못된 라벨이 왜 생기는지 확인!
        - 더 많은 데이터를 수집
        - 처음 단계부터 점검해가며 반복.. 
            > 문제 정의 후, 프로젝트의 설계를 최대한 구체적으로 해야 이후 과정의 반복을 줄일 수 있다!
    - 문제의 타당성 따져보기
        - 머신러닝 문제를 고려할 때는 얼마나 흥미로운지가 아니라 제품, 회사의 비즈니스에서 어떤 가치를 줄 수 있는지 고려해야함
        - 머신러닝이 사용되면 좋은 경우
            - 패턴 : 학습할 수 있는 패턴이 있는가?
            - 목적함수 : 학습을 위한 목적 함수를 만들 수 있어야 함
            - 복잡성 : 패턴이 복잡해야 함
            - 데이터 : 데이터가 존재하거나 수집할 수 있어야함
            - 반복 : 사람이 반복적으로 실행하는 일, 자동화 했을때 이점이 큰가?
        - 머신러닝이 사용되면 좋지 않은 경우
            - 비윤리적인 문제 : 인종차별과 같이 예민한 문제를 다룰 경우
            - 간단히 해결할 수 있는 경우 
            - 좋은 데이터를 얻기 어려울 경우
            - 한번의 예측 오류가 치명적인 결과를 발생할 경우 : 시스템에서 가격, 수량 등을 직접적으로 예측 하고 결정하는 task
            - 시스템이 내리는 모든 결정이 설명 가능해야 할 경우 : 모든 결정에 대해 충분히 설득력있는 설명이 필요한 경우
            - 비용 효율적이지 않은 경우 : 투자되는 인력, 인프라가 과한 경우
    - 목표 설정, 지표 결정 (Goal & Objectives)
        - Goal : 프로젝트의 일반적인 목적
        - Objectives : 목적을 달성하기 위한 세부 단계의 목표(구체적인 목적)
            - 현실에선 objective가 여러개인 경우가 많기 때문에 multiple objective optimization을 고려해야한다. (좋은 프로젝트를 판단하는 기준이 여러개이기 때문에)
            - Objective function(Loss function)을 weighted summation하여 모델이 여러 objective를 한번에 만족하게 하거나, 각각의 objective에 대한 학습 model을 두어 결과를 weighted summation한다.
            - 보통 objective가 여러 개인 경우 모델을 분리하는 것이 유리하다.
                - 학습 관점에서 모델이 단일 objective function을 최적화하는 것이 쉽고,
                - Weighted summation시 모델을 재학습할 필요가 없기 때문!
    - 제약 조건 (Constraint & Risk)
        - 일정, 예산, 관련된 사람(이 프로젝트로 영향을 받는 사람), Privacy, 기술적 제약(legacy), 윤리적 이슈
        - 성능 : Threshold, Performance Trade-off, 해석 가능 여부, Confidence Measurement
    - 베이스라인, 프로토타입
        - 프로토타입은 HTML에 집중하는 것 보다, 모델에 집중하는 것이 중요
        - Voila, Streamlit, Gradio 등 활용
    - Metric Evaluation
        - 모델의 성능 지표와 별개로 비즈니스 목표에 영향을 파악하는 것도 중요
        - 만든 모델이 비즈니스에 어떤 임팩트를 미쳤을지(매출 증대에 기여, 내부 구성원들의 시간 효율화 증대 등)을 고려
    - Action(모델 개발 후 배포 & 모니터링)
        - 현재 만든 모델이 어떤 결과를 내고 있는가?
        - 현실 상황에서의 error analysis를 진행
3. 비즈니스 모델
    - 비즈니스에 대한 이해도가 높을 수록 문제 정의를 잘 할 가능성이 존재!
        - 회사의 비즈니스 파악하기
        - 데이터를 활용할 수 있는 부분은 무엇인가?(Input)
        - 모델을 활용한다고 했을때 예측의 결과가 어떻게 활용되는가?(Output)

## (5강,6강)Notebook 베이스 - Voila실습, 프로토타이밍 - 웹 서비스 형태 Streamlit실습

- 웹 서비스를 만드는 것은 시간이 많이 소요되므로, 익숙한 노트북에서 프로그램을 만들자!
- ipywidget과 같이 사용하여 간단한 대시보드도 함께 사용할 수 있다.
- Voila의 강점
    1. Jupyter Notebook 결과를 쉽게 웹 형태로 띄울 수 있음
    2. Ipywidget, Ipyleaflet 등 사용 가능
    3. Jupyter Notebook의 Extension 사용 가능(=노트북에서 바로 대시보드로 변환 가능)
    4. Python, Julia, C++ 코드 지원
    5. 고유한 템플릿 생성 가능
    6. 너무 쉬운 러닝 커브

- **실습 code trouble shooting 하기**
    1. uploader를 정의하고 display(uploader)를 했는데 widget이 안뜸
        - voila 의 버전 문제가 원인이더라!
        - pip install --upgrade voila 를 하고 다시 실행하니 해결!
    2. uploader.value 값이 tuple로 감싸진 dictionary 형태로 출력
        - 내부 dictionary에 uploader.value[0]로 dictionary에 접근
        - display call back 함수를 다음과 같이 수정
            ``` python
            def on_display_click_callback(clicked_button: widgets.Button) -> None:
                global content
                content = next(iter(uploader.value))['content']
                display_image_space.value = content
            ```
    3. streamlit run app.py 오류
        > TypeError: Descriptors cannot not be created directly.
        If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0.
        - [[LINK]](https://stackoverflow.com/questions/72441758/typeerror-descriptors-cannot-not-be-created-directly)을 참고하여 pip install protobuf==3.20.* 를 통해 해결
- 실습 결과
    <p align="center"><img src="https://user-images.githubusercontent.com/62092317/201051366-8a2b4bec-f0b6-4ef6-873d-d61f2086cd2e.png" width = 300 ></p>
    
    - 강의를 들을 땐 무엇을 배우고 있는건지 개념이 모호했는데, 실제로 코드를 짜보니까 이해가 된다. 이런게 있는줄도 몰랐는데 UI가 되게 깔끔하고 좋네!
    - 이런걸 해주는 사이트들을 몇번 이용해본 적이 있는데, 배우기만하면 생각보다 어렵지 않은 일이었구나..ㅎㅎ

- **Special Mission : 대회 때 학습했던 model streamlit으로 동작시키기~!~!**

    - 위 실습 코드랑 비교할 때, 변경해줄 부분들은 많이 없다.
        1. Model.py - 대회 때 사용했던 model class와 pth file을 사용
        2. utils.py - Transform에서 Resize의 argument와 Normalize를 학습 모델과 동일하게 맞춰 주기
        3. predict.py - model_path의 경로 변경해주기
    - 실행 결과:  Webcam으로 부터 이미지를 받아, 모델이 마스크 착용 상태 / 성별 / 나이를 예측해준다.
        
        1. Wear, Male, Under 30 test 
        
            <p align="center"><img src="https://user-images.githubusercontent.com/62092317/201091637-f3f68ca1-9224-4b0d-bbfe-5b451b1fdec4.png" width = 300></p>

           - > 내가 학습한 모델이 날 어떻게 분류하는지 확인해보기! 마스크를 쓴 경우 잘 맞춘다!! (Wear, Male, under 30은 학습 당시 0번 label에 해당했으며 validation acc : 93.62%)

        2. Incorrect, Male, Under 30 test
    
            <p align="center"><img src="https://user-images.githubusercontent.com/62092317/201092197-8758779b-dd8d-42b1-a1d5-669b26b11719.png" width = 300></p>

            - > .....?
            - > Incorrect, Male, under 30은 학습 당시 6번 label이었으며 validation acc : 94.05% 였는데, 못 맞추네..??
        
        3. Wear, Female, Over 60 test

            <p align="center"><img src="https://user-images.githubusercontent.com/62092317/201093305-23d91a14-a5a2-41a1-9d6f-97b24bc33c14.png" width = 300></p>

            - > 피곤해보이는 우리 엄마..(초상권 미안)
            - > Wear, Female, Over 60은 학습 당시 5번 label이었으며 validation acc : 60.59%였다. 역시나 못 맞춘다. 우리 엄마가 동안이긴 한데..

### (7강)Linux & Shell Command

- Linux 사용하는 방법
    1. VirtualBox에 Linux 설치, Docker로 설치
    2. WSL 사용(윈도우)
    3. Notebook에서 터미널 실행
- 기초 용어
    - Shell 이란?
        - 사용자가 문자를 입력해 컴퓨터에 명령할 수 있도록 하는 프로그램
    - 터미널/콘솔
        - 쉘을 실행하기 위해 문자 입력을 받아 컴퓨터에 전달
        - 프로그램의 출력을 화면에 작성
    - sh
        - 최초의 쉘
    - bash
        - Linux 표준 쉘
    - zsh
        - Mac 카탈리나 OS 기본 쉘 
- Basic Shell command
    - man : 쉘 커맨드의 메뉴얼 문서를 보고 싶은 경우
    - mkdir : 경로 생성하기
    - ls : List segment
        - -a : .으로 시작하는 파일, 폴더를 포함해 전체 파일 출력
        - -l : 퍼미션, 소유자, 만든 날짜, 용량까지 출력
        - -h : 용량을 사람이 읽기 쉽도록 GB,MB등 표현 '-l'과 같이 사용
    - pwd : Print Working Diectory 
    - cd : Change Directory
    - echo : Python 의 print처럼 터미널에 텍스트 출력
    - vi : vim 편집기로 파일 생성
        - i: INSERT 모드에서만 수정할 수 있음
        - /문자 : 문자 탐색, 탐색한 후 n을 누르면 계속 탐색 실행
        - ESC 누른후 :wq (저장하고 나가기, write and quit)
    - bash : bash로 쉘 스크립트를 실행 (ex: bash vi-test.sh)
    - sudo(substitue user do) : 관리자 권한으로 실행
    - cp : 파일 또는 폴더 복사하기
        - -r: 디렉토리안의 파일까지 전부 복사
        - -f: 복사할때 강제로 실행
    - mv : 파일 또는 폴더 이동하기(이름 바꿀때도 사용)
    - cat : 특정 파일 내용 출력(concatenate)
        - 여러 파일을 인자로 주면 합쳐서 출력 (ex : cat vi_test2.sh vi_test3.sh)
    - clear : 터미널 창 청소
    - history : 최근 입력한 쉘 커맨드 출력
        - !숫자 입력시 그 커맨드를 다시 활용
    - find : 파일 및 디렉토리 검색 (ex : find . -name "File")
    - export : 환경 변수 설정
        - 터미널이 꺼지면 사라지게 되므로, 매번 쉘을 실행할 때마다 환경변수를 저장하고 싶으면 .bashrc, .zshrc에 저장해야 한다.
        - ex : vi ~/.bashrc 또는 vi ~/.zshrc , 즉시 적용 source ~/.bashrc 또는 source ~/.zshrc
    - alias : 별칭으로 설정
        - ex : alias ll2 = 'ls -l'
        - ll2 입력시 ls -l으로 동작
    - head, tail : 파일의 앞/뒤 n행 출력 
        - ex: head -n 3 vi-test.sh
    - sort : 행 단위 절렬
        - -r : 정렬을 내림차순으로(default : 오름차순)
        - -n : Numeric Sort
        - ex : cat fruits.tx | sort
    - uniq : 중복된 행이 연속으로 있는 경우 중복 제거
        - -c : 중복 행의 개수 출력
    - grep: 파일에 주어진 패턴 목록과 매칭되는 라인 검색
        - grep 옵션 패턴 파일명
    - cut : 파일에서 특정 필드 추출
        - -f : 잘라낼 필드 지정
        - -d : 필드를 구분하는 구분자
        - ex : vi cut_file
    - Redirection & Pipe 
        - Redirection : 프로그램의 출력(stdout)을 다른 파일이나 스트림으로 전달
            - ">" : 덮어쓰기(Overwrite)파일이 없으면 생성하고 저장
            - ">>" : 맨 아래에 추가하기(Append)
                ``` cmd
                echo "hi" > vi-test3.sh
                echo "hello" >> vi-test3.sh
                ```
        - Pipe : 프로그램의 출력(stdout)을 다른 프로그램의 입력으로 사용하고 싶은 경우
            - A의 Output을 B의 input으로 사용
            - ex: 현재 폴더에 있는 파일명 중 vi가 들어간 단어를 찾고 싶은 경우
                - ls | grep "vi"
        - 연습 문제
            - P1. test.txt파일에 "Hi!!!!"를 입력(vi 사용금지)
                 > echo 'Hi!!!!' > test.txt
            - P2. test.txt파일 맨 아래에 "kkkk"를 입력(vi 사용금지)
                > echo 'kkkk' >> test.txt
            - P3. test.txt의 라인수를 구하기(힌트: wc -l 쓰면 라인 수 구할 수 있음)
                > cat test.txt | wc -l
    - ps : Process Status 현재 실행되고 있는 프로세스 출력하기 
        - -e : 모든 프로세스
        - -f : Full Format으로 자세히 보여줌
    - curl : Client URL, Command Line기반의 data transfer command
    - df : Disk Free, 현재 사용중인 디스크 용량 확인 
    - scp : Secure Copy(Remote file copy program) SSH를 이용해 네트워크로 연결된 호스트 간 파일을 주고 받는 명령어
        - ex : scp local_path user@ip:remote_directory
    - nohup : 터미널 종료 후에도 계속 작업이 유지되도록 실행(백그라운드 실행)
        - nohup으로 실행될 파일은 Permission이 755여야 함
    - chmod : Change Mode 
         - r:Read(읽기),4
         - w:Write(쓰기),2
         - x:eXecute(실행하기),1
         - -:Denied
         - User, Group, 그외 사람들에 대한 권한을 차례대로 rwx 숫자의 합으로 표현
- **Special Mission : 카카오톡 그룹 채팅방에서 대화 내보내기로 csv로 저장 후, 쉘 커맨드 1줄로 카카오톡 대화방에서 2021년(또는 2022년)에 제일 메세지를 많이 보낸 TOP 3명 추출하기!**
    - 참고 자료 :[[Linux 문자열 검색]](https://recipes4dev.tistory.com/157)
    - 우리 팀의 카톡방 내용을 내보내기 해서 연습해보았다. 
        ```
        txt 파일 일부
        Date,User,Message
        2022-10-17 07:23:09,"김성수","ㅎㅇㅎㅇ~~"
        2022-10-17 07:35:40,"AI이우택","ㅎㅎㅎㅇ"
        2022-10-17 07:39:44,"AI강민수","이모티콘"
        2022-10-17 07:44:56,"AI 조윤재","하이하이"
        2022-10-17 08:00:17,"AI이성진","하이"
        ...
        ```
    - Steps
        1. 먼저 shell 정규식과 grep 을 이용하여 각 메세지를 말한 ID를 추출한다.
            > grep -o "\".*\"" chat.txt
            - ID가 "" 안에 들어가 있기 때문에, " "안의 문자열을 matching하여 찾았는데 메세지도 같은 패턴을 갖기 때문에 ID,message가 추출된다
                ```
                "김성수","ㅎㅇㅎㅇ~~"
                "AI이우택","ㅎㅎㅎㅇ"
                "AI강민수","이모티콘"
                "AI 조윤재","하이하이"
                "AI이성진","하이"
                ```
        2. Pipe로 추출된 것을 넘겨주고, cut command를 이용하여 ","를 기준으로 뒤 메세지 부분을 잘라준다.
            > grep -o "\".*\"" chat.txt | cut -f 1 -d ","
        3. ID만 추출된 상태에서, sorting을 한 후 uniq command를 이용하여 연속되면서 중복된 행의 개수를 세어준다.(Sorting을 하면 같은 ID끼리 연속되어 등장하기 때문에, uniq -c 를 활용하면 각 ID의 등장 빈도를 셀 수 있다.)
            > grep -o "\".*\"" chat.txt | cut -f 1 -d "," | sort | uniq -c
            ```
            출력 결과
            139 "AI 조윤재"
            175 "AI강민수"
            21 "AI이성진"
            49 "AI이우택"
            320 "김성수"
            ```
        4. Pipe로 counting 값을 넘겨주고 sort -r 을 이용해 내림차순으로 정렬 후, 맨 위 3 줄 (top 3)만을 출력한다.
            > grep -o "\".*\"" chat.txt | cut -f 1 -d "," | sort | uniq -c | sort -r| head -n 3
            ``` 
            출력 결과
            320 "김성수"
            175 "AI강민수"
            139 "AI 조윤재"
            ```
        5. 카톡방에서 내가 말을 제일 많이 했다.
    - shell command(특히, pipe)와 정규식의 합작이 텍스트 파일 처리에 이렇게 유용할 수 있는지 몰랐다.. 머리 싸매면서 했는데 신기하다.. 그리고 정규식 연습 좀 더 해야겠다..
### (8강) Docker

- 가상화란?
    - 등장 배경
        - 개발할 때, 서비스 운용에 사용하는 서버에 직접 들어가서 개발하지 않음
        - Local 환경에서 개발, 완료되면 staging, Production 서버에 배포
        - 그러나 개발을 진행한 Local 환경과 Production 서버 환경이 다르다면?
            - 운영체제가 다르기 때문에 라이브러리, 파이썬 등의 설치가 다르게 진행되어야함
            - 환경변수가 다를 수도 있다.
            - 매번 각 서버에 대한 설정을 맞춰주는건 사실 상 불가능하다!
            - 서버 환경까지도 모두 한번에 소프트웨어화 할 수 없을까? (마치,밀키트처럼)
    - 가상화의 의미
        - 특정 소프트웨어 환경을 만들고, Local, Production 서버에 그대로 사용
        - Docker 등장 전 : 가상화 기술로 VM(Virtual Machine)을 사용
            - VM은 호스트 머신이라고 하는 실제 물리적인 컴퓨터 위에, OS를 포함한 가상화 소프트웨어를 두는 방식
            - Container : VM의 무거움을 크게 덜어주면서, 가상화를 좀 더 경량화된 프로세스의 개념으로 만든 기술
- Container 기술을 쉽게 사용할 수 있도록 나온 도구가 바로 Docker다!

    <p align="center"><img src="https://user-images.githubusercontent.com/62092317/201269995-3509a5df-f37b-42b5-820d-7ff2b321c108.png" width=300></p>

    <p align="center"><img src="https://user-images.githubusercontent.com/62092317/201270071-423a2536-9536-4c18-8a07-e715c87dee2a.png" width=300></p>

    - Container들을 실은 고래의 모습..
    - Docker Image
        - 컨테이너를 실행할 때 사용할 수 있는 "템플릿"
        - Read Only
    - Docker Container
        - Docker Image를 활용해 실행된 인스턴스
        - Write 가능
- Docker로 할 수 있는 일
    - 다른 사람이 만든 소프트웨어를 가져와서 바로 사용할 수 있음!
        - ex: MySQL, Jupyter Notebook
    - 다른 사람이 만든 소프트웨어를 "Docker Image"라고 생각하면 된다.
        - OS, 설정을 포함한 실행 환경을 의미
    - 따라서, 자신만의 이미지를 만들면 다른 사람에게 공유할 수 있다.

- Docker 실습하며 배워보기 
    - 기본 명령어
        - docker pull "이미지 이름:태그": 다른 사람의 이미지를 container registry에서 땡겨오기
        - docker images : 다운 받은 이미지 확인
        - docker run "이미지 이름:태그" : 다운 받은 이미지 기반으로 Docker Container를 만들고 실행 (-d 백그라운드 모드 -p 포트 지정)
            - 로컬 호스트 포트 : 컨테이너 포트 형태로, 로컬 포트 3306으로 접근 시 컨테이너 포트 3306으로 연결되도록 설정 (로컬 호스트: 우리의 컴퓨터, 컨테이너 : 컨테이너 이미지 내부)
            - ex) docker run --name mysql-tutorial -e -d -p 3306:3306 mysql:8
        - docker ps : 실행한 컨테이너 확인
        - docker ps -a : 모든 컨테이너를 확인 / 작동을 멈춘 컨테이너도 확인할 수 있음
        - docker exec -it " 컨테이너 이름(혹은ID)" /bin/bash 
            - Compute Engine에서 SSH와 접속하는 것과 유사
        - docker rm "컨테이너 이름(ID)" : 멈춘 컨테이너를 삭제
        - Dockekhub에 공개된 모든 이미지를 다운 받을 수 있음
    - Docker Image 만들기
        - 파이썬 환경 및 어플리케이션 코드 작성
        - Dockerfile 작성
            - FROM으로 베이스 이미지를 지정
            - COPY로 로컬 내 디렉토리 및 파일을 컨테이너 내부로 복사
            - WORKDIR로 RUN, CMD등을 실행할 컨테이너 내 디렉토리 지정
            - RUN으로 어플리케이션 실행에 필요한 여러 리눅스 명령어들을 실행
            - CMD로 이미지 실행시 바로 실행할 명령어를 지정
            - EXPOSE : 컨테이너 외부에 노출할 포트 지정
            - ENTRYPOINT : 이미지를 컨테이너로 띄울 때 항상 실행하는 커맨드
        - Docker build "Dockerfile이 위치한 경로" -t "이미지 이름:태그"으로 이미지 빌드
        - docker run "이미지 이름:태그"로 빌드한 이미지를 실행
    
### (9강) MLflow

- ML flow가 해결하려고 했던 Pain point
    1. 실험을 추적하기 어렵다
    2. 코드를 재현하기 어렵다
    3. 모델을 패키징하고 배포하는 방법이 어렵다
    4. 모델을 관리하기 위한 중앙 저장소가 없다
- ML flow의 핵심 기능
    1. Experiment Management & Tracking
        - 모델 생성일, 모델 성능, 모델 메타 정보를 모두 기록할 수 있음
        - 여러 사람이 하나의 MLflow 서버 위에서 각자 자기 실험을 만들고 공유
    2. Model Registry
        - MLflow로 실행한 머신러닝 모델을 Model Registry(모델 저장소)에 등록할 수 있음
        - 모델 저장소에 모델이 저장될 대마다 해당 모델에 버전이 자동으로 올라감
    3. Model Serving   
        - Model Registry에 등록한 모델을 REST API형태의 서버로 Serving할 수 있음
- ML Component
    1. MLflow Tracking
        - 머신러닝 코드 실행, 로깅을 위한 API,UI
    2. MLflow Project
        - 머신러닝 프로젝트 코드를 패키징하기 위한 표준
    3. MLflow Model
        - 모델은 모델 파일과 코드로 저장
    4. MLflow Registry
        - MLflow Model의 전체 Lifecycle에서 사용할 수 있는 중앙 모델 저장소

## 논문 스터디 - About Diffusion model
- 다른 조 멘토님께서 매주 논문 스터디를 한다고 하셔서 몰래 잠입해보았다.
- Generative model의 3 요소 : expressiveness, inference time, resolution

    <p align="center"><img src="https://user-images.githubusercontent.com/62092317/200853843-fd36cbd3-e6c9-4081-aa4c-60acdda74619.png" width = 300></p>

    - GAN: low expressiveness
    - VAE: low resolution
    - Flow based model : both
    - Diffusion model : Slow inference time
    - How we can train p(x)?
        - GAN) Make Discriminator D(x) and generator G(z)
        - VAE) Make encoder and decoder
        - Flow based model) Make flow and train its inverse
        - Diffusion model?
            <p align="center"><img src="https://user-images.githubusercontent.com/62092317/200854696-7f63a632-4e20-4d0b-a0f9-8f1cd4ff7079.png" width = 400></p>

            - Forward process(diffusion process)
                - Add small amount of Gaussian noise to the sample in T steps
                - Data에 noise를 조금식 더해가면서 data를 완전한 noise로 만든다
            - Reverse process
                - Noise로 부터 조금씩 복원해가면서 data를 만들어낸다. 어떻게 하면 neural network가 이를 익혀나갈 수 있는 고민한 것이 diffusion model의 핵심이다.
- 멘토님께서 diffusion model의 수식적인 부분까지 파헤치시지는 않았지만 어떤 원리로 동작하고, 이것이 기존 generative model들과 비교했을 때 왜 강력한지를 설명해주셨다. 논문 스터디를 뒤늦게 알아서 처음 들어가 봤는데, 짧았지만 내용들에 관심이 생겨서 더 찾아보고자 한다.

- 참고자료 : [[yang-song의 page]](https://yang-song.net/blog/2021/score/) [[About diffusion model]](https://everyday-deeplearning.tistory.com/entry/%EC%B4%88-%EA%B0%84%EB%8B%A8-%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-Denoising-Diffusion-Probabilistic-ModelsDDPMs)
