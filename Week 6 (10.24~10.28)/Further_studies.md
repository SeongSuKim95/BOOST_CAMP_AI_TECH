# Week 6 : CV competition

## Contents 

- Course 
    1. About Competition
        - Overview
        - Problem Definition
        - Data description
    2. Data Feeding

- Team Competition
    
    - Baseline Code
        - Dataset class : Line by Line 
        - About Dataset
    
    - Code modification

- Mentoring
    - Paper review : Learning without Prejudices : Continual unbiased learning via benign and malignant forgetting
---
## Course

### About Competition

<p align="center"><img src="https://user-images.githubusercontent.com/62092317/197450654-18320800-6006-4626-b1e1-586b675fdf40.PNG" width= 500></p>

- Overview

    >감염자의 입, 호흡기로부터 나오는 비말, 침 등으로 인해 다른 사람에게 쉽게 전파가 될 수 있기 때문에 감염 확산 방지를 위해 무엇보다 중요한 것은 모든 사람이 마스크로 코와 입을 가려서 혹시 모를 감염자로부터의 전파 경로를 원천 차단하는 것입니다. 이를 위해 공공 장소에 있는 사람들은 **반드시 마스크를 착용해야 할 필요가 있으며, 무엇 보다도 코와 입을 완전히 가릴 수 있도록 올바르게 착용하는 것이 중요**합니다. 하지만 넓은 공공장소에서 모든 사람들의 올바른 마스크 착용 상태를 검사하기 위해서는 추가적인 인적자원이 필요할 것입니다.
    따라서, **우리는 카메라로 비춰진 사람 얼굴 이미지 만으로 이 사람이 마스크를 쓰고 있는지, 쓰지 않았는지, 정확히 쓴 것이 맞는지 자동으로 가려낼 수 있는 시스템이 필요**합니다. 이 시스템이 공공장소 입구에 갖춰져 있다면 적은 인적자원으로도 충분히 검사가 가능할 것입니다.

- Problem Definition 
    1. 내가 지금 풀어야 할 문제가 무엇인가?
        - 사람 얼굴 영상만으로 마스크를 정확히 썼느지를 판단하기
    2. 이 문제의 input과 output은 무엇인가?
        - Input
            - 마스크를 쓴 상태, 잘못 쓴 상태, 안 쓴 상태의 안면 이미지 
            - 전체 사람 수 : 4,500
            - 한 사람당 사진의 개수 : 7 [마스크 착용 5장, 이상하게 착용 1장, 미착용 1장]
            - 이미지 크기 : (384,512)
            - 전체 데이터셋 중 60%가 학습 데이터 셋으로 활용
        - Output
            - 해당 이미지 속 인물의 마스크 착용 상태(wear, Not wear, incorrect wear),성별(male, female),연령 그룹($ age <=30, 30<age <= 60, 60 < age$ )
            - 각 이미지를 마스크 착용여부, 성별, 나이를 기준으로 18개의 class로 나누어 예측 
    3. 이 솔루션은 어디서 어떻게 사용되어야 하는가?
        - 이 솔루션의 목적을 생각해보면 마스크를 착용했는지 여부 뿐만아니라, 누가 어떻게 착용했는지를 판단하는 것도 중요

### EDA 

- EDA(Exploratory Data Analysis)
    - EDA란? 
        - 이 사람이 data를 왜 줘서 나에게 어떤 것을 요구하는가? data가 어떻게 생겼는지..등등 내가 이 데이터를 이해하기 위한 노력. 뭐든 상관 없이 데이터에 대해 궁금한 모든 것을 알아보는 노력.
        - 주어진 데이터에 대한 본인의 생각을 자유롭게 서술해보자
            - 모든 샘플이 제대로 찍혀있을까? 중복되는 샘플이 있을까?
            - 데이터의 원천이 여러곳인지 아님 한곳인지? 여러 곳이라면 데이터의 분포엔 어떤 영향이 있을지?
        - Visualization을 통해 data의 특징을 파악하자!
### Dataset

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

### Data Generation
- Data feeding : Mdoel에게 data를 잘 만들어 주는것은 어떤 의미를 갖는가?
    - Model의 데이터 처리속도를 고려하여 data를 feed 해야한다.
        > Model의 처리량 만큼의 data를 generating 할 수 있는가??
    - Ex 1) Data generator의 augmentation순서에 따른 feeding 속도 차이

    - <p align="center"><img src="https://user-images.githubusercontent.com/62092317/197667913-e9c029dc-d3b8-4765-9b2f-9e636deaa66a.png" width = 600></p>
        
        > 같은 출력을 내는 transformation이라고 해도 순서에 따라 processing 하는 시간이 다르므로, 이를 고려하여야 한다.
    
    - Ex 2) Dataloader의 num workers에 따른 feeding 속도 차이
    
    - <p align="center"><img src="https://user-images.githubusercontent.com/62092317/197668870-7827d296-c208-44f7-97e7-c4f4ba34b123.png" width = 600></p>
        > Num workers는 무조건 크다고 좋은 것이 아니고, cpu와 gpu의 spec에 따라 최적값이 결정되어 있으므로 실험으로 최적값을 알아내는 것이 좋다! 

- Dataset class와 Dataloader는 기능적으로 구분하여 사용하여야 한다.
    - Dataset class는 해당 dataset이 요구하는 처리 방식을 적용
    - Dataloader는 어떤 dataset class든 상관 없이 **dataset class에서 처리된 data를 가져오는 방식** 에 집중하여 적용

### Model

1. modules
  
    ``` python
    import torch.nn as nn
    import torch.nn.functional as F

    class MyModel(nn.Module):
        def __init__(self):
            super(MyModel,self).__init__()
            self.conv1 = nn.Conv2d(1,20,5)
            self.conv2 = nn.Conv2d(20,20,5)

        def forward(self,x):
            x = F.relu(self.conv1(x))
            return F.relu(self.conv2(x))
    ```
- > a = MyModel()

    ``` python
    MyModel(
        (conv1): Conv2d(1,20,kernel_size=(5,5),stride=(1,1))
        (conv2): Conv2d(20,20,kernel_size=(5,5),stride=(1,1))
    )
    ```

- > list(a.modules())

    ``` python
    [MyModel(
        (conv1):Conv2d(3,20,kernel_size=(5,5),stride=(1,1))
        (conv2):Conv2d(20,20,kernel_size=(5,5), stride=(1,1))
    ),
    # Child Module
    Conv2d(3,20,kernel_size=(5,5),stride=(1,1)),
    Conv2d(20,20 kernel_size=(5,5),stride=(1,1))]

    ```
    - 모든 nn.Module은 child modules를 가질 수 있고, 내 모델을 정의하는 순간 그 모델에 연결된 모든 module을 확인할 수 있다.
    - nn.Module을 상속받은 모든 클래스의 공통된 특징은 한번에 실행된다.
        - build한 모델의 forward()를 한번만 실행한 것으로 그 모델의 forward에 정의된 모듈 각각의 forward()가 실행됨

2. Parameters

- <p align="center"><img src = "https://user-images.githubusercontent.com/62092317/197689140-cb96262a-9aa1-4315-b8c1-10b9f9ca78db.png" width = 600></p>

    - > sd     = a.state_dict()  Dictionary 형태!

        - state_dict()로부터 각 module(layer)의 이름을 확인하고, 특정 module parsing해서 사용할 수 있어야 한다.

    - > params = list(a.parameters())

        - 각 모델 파라미터들은 data,grad,requires_grad 변수를 가짐

3. Pretrained Model
    
    - Transfer learning :
        - 좋은 품질, 대용량의 데이터로 미리 학습한 모델을 바탕으로 내 목적에 맞게 다듬어서 사용
        - Pretraining 할 때 설정했던 문제와 현재 문제와의 유사성을 고려
            - Case 1. 문제를 해결하기 위한 학습 데이터가 충분할 경우
            - <p align="center"><img src= "https://user-images.githubusercontent.com/62092317/197693885-14b20d1a-11f2-40fb-9ace-808f546ffe57.png" width = 500></p>

                - Finetuning: CNN backbone이 학습한 문제가 나의 문제와 유사한 경우, 이를 전부 학습할 필요 없이 classifier(Torchvision model의 fc layer)만을 다시 학습
                - Feature extraction: 반대의 경우, 모든 layer를 재학습
            - Case 2. 학습 데이터가 충분하지 않은 경우
            - <p align="center"><img src="https://user-images.githubusercontent.com/62092317/197694702-c0445d6f-1da2-4620-a759-11d2bdde1269.png" width= 500></p>

                - 만약 학습 데이터가 충분하지도 않고, task의 연관성과 데이터 유사도가 높지 않다면 model의 overfitting을 우려하여 transfer learning을 하지 않는 것이 낫다.
       
### Training & Inference

1. Loss
    - loss도 nn.Module Family로 구현
    - Model의 input부터 loss까지 연결된 chain이 완성되므로, loss.backward() 함수를 통해 모델 parameter의 gradient 값을 update 할 수있다.

    - Custom Loss
        - Ex 1) Focal loss : Class Imbalance 문제가 있는 경우, 맞춘 확률이 높은 class는 조금의 loss를 맞춘 확률이 맍은 class는 loss를 훨씬 높게 부여 

        - Ex 2) Label smoothing : Class target label을 Onehot 표현으로 사용하기 보다는 조금 Soft하게 표현해서 일반화 성능을 높임
2. Optimizer

    - Learning rate scheduler
        - Step LR : 정해진 epoch에 도달할 때 마다 learning rate를 감쇄 
        - Consine annealing LR : Cosine 함수 형태처럼 LR을 급격히 변경 하며, 학습이 local minima에 빠지는 것을 방지
        - ReduceLROnPlateau : 성능 향상이 일어나지 않을 때 learning rate를 미세하게 조정

3. Metric
    
    - 모델의 학습 결과를 평가하기 위해 필요한 객관적인 지표
        - Classification : Accuracy, F1-score, precision, recall
        - Regression : MAE, MSE
        - Ranking : MRR, NDCG, mAP
    - 데이터 상태에 따라 적절한 metric을 선택하는 것이 필요
        - Class 별로 밸런스가 적절히 분포 --> Accuracy
        - Class 별 밸런스가 좋지 않아 각 클래스 별로 성능을 잘 낼 수 있는지 확인이 필요할때 --> F1-score

4. Process : 학습과 추론 프로세스가 어떻게 이루어지는가?

    - model.train()과 model.eval()의 차이는?

    - Training Process
        - optimizer.zero_grad()
            - 이전 batch에서 최적화했던 gradient가 그대로 남아있기 떄문에(각각의 parameter에 이전 gradient를 가지고 있는 상태) 이를 매 loop마다 초기화
            - **일반적으로** 이전 batch의 gradient값은 해당 iteration에서 parameter를 update하는데에만 쓰이고, 이 후 iteration에선 사용하지 않는다.

        - loss = criterion(outputs,labels)
            - loss와 input까지의 grad_fn chain을 생성하고, loss.backward()를 통해 backpropagation을 진행
            - loss의 역할은 gradient를 update하기만 한다!

        - optimizer.step()
            - 위 loss를 통해 update된 gradient를 optimizer가 parameter에 적용한다.

        - Gradient Accumulation : 매 batch에서 생성된 loss를 바탕으로 backward 하면서 model parameter를 update하는 것이 아니라, N번의 iteration마다 update

    - Inference Process 
        - with torch.no_grad()
            <p align="center"><img src= "https://user-images.githubusercontent.com/62092317/197746484-bf2613fb-1e19-4b28-a248-2128ee6bb1bd.png" width = 800></p>
            - model 안의 모든 tensor들의 requires_grad가 False가 된다.

        - Checkpoint
        
            ``` python
            if val_acc > best_val_acc:
                print("New best model for val accuracy! saving the model!")
                torch.save(model.state_dict(),f"results/{name}/{epoch:03}_accuracy_{val_acc:4.2%}.ckpt")
                best_val_acc = val_acc
            ```  

            - Validation 결과를 보며, model을 저장하기 !


### Ensemble

- 보통의 Deep Neural Network에서는 Low Bias, High Variance로 인한 Overfitting 현상이 발생한다. 

- Bagging, Boosting [[LINK]](https://towardsdatascience.com/ensemble-methods-bagging-boosting-and-stacking-c9214a10a205)

- Model Averaging(Voting)이 잘 동작하는 이유는 서로 다른 모델이 같은 test set에 대해 같은 error를 내는 경우는 잘 없기 때문

- Cross Validation

    <p align="center"><img src="https://user-images.githubusercontent.com/62092317/197917788-ebb51405-4480-466a-9a94-4089b3909dc0.png" width = 600></p>

    - validation set을 학습에 활용하여 validation set의 분포까지도 모델에 반영할 수는 없을까?
    - Stratified K-Fold Cross Validation 
        - 예를 들어 K=5일 경우,5개의 서로 다른 validation set(20%)을 만들고 각 model에 대해 fold를 적용하여 학습
        - **Stratified** : fold를 split하는 과정에서 각각의 validation set과 train set내의 class 분포가 균일하도록 해야 한다.

- TTA(Test Time Augmentation)

    <p align="center"><img src="https://user-images.githubusercontent.com/62092317/197918000-4e4a234d-0b08-4199-8254-b9f4d3777e41.png" width = 600></p>

    - test sample에 여러 augmentation을 적용한 후, 각각에 대해 test를 수행
    - 이렇게 수행한 결과들을 ensemble하여 test 성능을 도출할 경우, dataset에 overfit된 성능이 아닌 일반화된 test 성능을 얻을 수 있다. 

- Hyperparameter Optimization
    - Parameter를 변경할 때 마다 학습을 해야하므로, 시간/장비의 여유가 있어야 하므로 맨 마지막에 하는 것이 좋다.


### Some Tips

1. 분석 코드 보다는 설명글을 유심히 보며 필자의 생각을 읽자.
2. 코드를 볼 때는 디테일한 부분까지.
3. 최신 논문과 그 코드를 살펴보자.
4. 공유하는 것을 주저하지 말자.

---
## Team Competition

### Baseline Code 

- Dataset class : Code Line by Line 해석

    1. MaskBaseDataset(주석 표시)
        > Dataset의 각 ID folder에 들어있는 image를 전부 모은 후, random하게 train / validation set으로 나눈다. 이렇게 나눌 경우, 동일 인물에 대한 이미지가(마스크를 쓴 sample의 경우 매우 흡사한) train, validation set에 동시에 존재하게 된다. 동일 인물이기 때문에, 사실상 train때 학습 했던 sample을 validation에서 또 보여주는 꼴이 되어 정확한 validation을 할 수 없게 된다.
        ``` python
        class MaskBaseDataset(Dataset):
            num_classes = 3 * 2 * 3 # 

            _file_names = {
                "mask1": MaskLabels.MASK, # 0
                "mask2": MaskLabels.MASK, # 0 
                "mask3": MaskLabels.MASK, # 0 
                "mask4": MaskLabels.MASK, # 0 
                "mask5": MaskLabels.MASK, # 0 
                "incorrect_mask": MaskLabels.INCORRECT, # 1
                "normal": MaskLabels.NORMAL # 2
            }
            # 각 이미지의 정보를 순서대로 담을 list 생성
            image_paths = []
            mask_labels = []
            gender_labels = []
            age_labels = [] 

            def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
                self.data_dir = data_dir
                self.mean = mean
                self.std = std
                self.val_ratio = val_ratio

                self.transform = None
                self.setup() # list에 image 정보 넣기
                self.calc_statistics()

            def setup(self):
                profiles = os.listdir(self.data_dir)
                # len(profiles) = 5400 , Ex: profiles[0] = '000695_female_Asian_53'
                for profile in profiles:
                    if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                        continue

                    img_folder = os.path.join(self.data_dir, profile) # /opt/ml/input/data/train/images/~
                    for file_name in os.listdir(img_folder):
                        _file_name, ext = os.path.splitext(file_name)
                        if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                            continue

                        img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                        mask_label = self._file_names[_file_name] # 앞에서 선언한 self._file_names dictionary에서 _file_name을 key 로 하여 mask_label을 얻는다.

                        id, gender, race, age = profile.split("_")
                        gender_label = GenderLabels.from_str(gender)
                        age_label = AgeLabels.from_number(age) # 같은 방식으로 gender, age에 대한 label을 얻는다.

                        self.image_paths.append(img_path)
                        self.mask_labels.append(mask_label)
                        self.gender_labels.append(gender_label)
                        self.age_labels.append(age_label)
                        # 각 이미지에 대해 path, mask_label, gender_label, age_label을 순서대로 list에 stack
            def calc_statistics(self):
                has_statistics = self.mean is not None and self.std is not None # 만약, mean std값을 지정안했다면 3000개의 dataset에 대한 mean, std값을 계산해서 적용
                if not has_statistics:
                    print("[Warning] Calculating statistics... It can take a long time depending on your CPU machine")
                    sums = []
                    squared = []
                    for image_path in self.image_paths[:3000]:
                        image = np.array(Image.open(image_path)).astype(np.int32)
                        sums.append(image.mean(axis=(0, 1)))
                        squared.append((image ** 2).mean(axis=(0, 1)))

                    self.mean = np.mean(sums, axis=0) / 255
                    self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255

            def set_transform(self, transform):
                self.transform = transform # Dataset의 transform 방식을 self.transform에 넣고 __getitem__에서 하나씩 call할때마다 적용해서 뱉는다

            def __getitem__(self, index):
                assert self.transform is not None, 

                image = self.read_image(index)
                mask_label = self.get_mask_label(index)
                gender_label = self.get_gender_label(index)
                age_label = self.get_age_label(index)
                multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)

                image_transform = self.transform(image)
                return image_transform, multi_class_label

            def __len__(self):
                return len(self.image_paths)
        ``` 
    2. MaskSplitByProgileDataset
        > 위 dataset class가 가지고 있는 단점을 보완하기 위해, sample 단위가 아닌 ID 단위로 train과 validation set을 나눈다. MaskBaseDataset으로 부터 상속 받으며, 이미지가 아닌 2700개의 profile folder를 random하게 train / validation으로 나눈다.
        (상속은 super().init(data_dir,mean,std,val_ratio)부분을 debugging 해보면 알 수 있다.) 이렇게 할 경우 train / validation set간 인물이 겹치지 않기 때문에 정확한 validation이 가능하다.
        ``` python
        class MaskSplitByProfileDataset(MaskBaseDataset):
            """
                train / val 나누는 기준을 이미지에 대해서 random 이 아닌
                사람(profile)을 기준으로 나눕니다.
                구현은 val_ratio 에 맞게 train / val 나누는 것을 이미지 전체가 아닌 사람(profile)에 대해서 진행하여 indexing 을 합니다
                이후 `split_dataset` 에서 index 에 맞게 Subset 으로 dataset 을 분기합니다.
            """

            def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
                self.indices = defaultdict(list) 
                super().__init__(data_dir, mean, std, val_ratio) ## 중요
                # Maskdataset의 __init__ 수행, 
                # Method인 setup, calc_statistics를 수행 (MaskSplitByProfileDataset의 method)

            @staticmethod
            def _split_profile(profiles, val_ratio): 
                length = len(profiles) # 2700
                n_val = int(length * val_ratio) # 540

                val_indices = set(random.choices(range(length), k=n_val)) # random으로 validation set index 선발
                train_indices = set(range(length)) - val_indices # 전체에서 validation을 뺀 나머지
                return {
                    "train": train_indices,
                    "val": val_indices
                }

            def setup(self):
                # super().__init__을 통해 self.data_dir, self.std, self.mean, self.val_ratio...
                profiles = os.listdir(self.data_dir) # len(profiles) = 5400 , Ex: profiles[0] = '000695_female_Asian_53', 각각이 한 ID를 담은 폴더
                profiles = [profile for profile in profiles if not profile.startswith(".")] # "." 로 시작하는 파일은 무시합니다
                split_profiles = self._split_profile(profiles, self.val_ratio)
                # split_profiles[train] = train set, split_profiles[val] = val set
                cnt = 0
                for phase, indices in split_profiles.items(): # phase : train, val , indices : sample index list
                    for _idx in indices: # 각 index에 대해
                        profile = profiles[_idx]
                        img_folder = os.path.join(self.data_dir, profile)
                        for file_name in os.listdir(img_folder): # train 으로 분류된 ID의 각 폴더에 담긴 image들에 대해!
                            _file_name, ext = os.path.splitext(file_name)
                            if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시
                                continue
                            # self.data_dir : /opt/ml/input/data/train/images,  profile : 000695_female_Asian_53 , file_name : mask5.jpg
                            img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                            mask_label = self._file_names[_file_name] # MaskDataset의 _file_names

                            id, gender, race, age = profile.split("_")
                            gender_label = GenderLabels.from_str(gender)
                            age_label = AgeLabels.from_number(age)
                            # 상속받은 MaskDataset에서 정의한 list들
                            self.image_paths.append(img_path)
                            self.mask_labels.append(mask_label)
                            self.gender_labels.append(gender_label)
                            self.age_labels.append(age_label) # 각 이미지에 대해 path, mask_label, gender_label, age_label을 순서대로 list에 stack

                            self.indices[phase].append(cnt)
                            cnt += 1

            def split_dataset(self) -> List[Subset]:
                return [Subset(self, indices) for phase, indices in self.indices.items()]

        ```
- About Dataset
    1. Male / Female
        - 모든 class에 대해 여자가 남자보다 sample 수가 많다.
    2. Wear / Not wear / Incorret
        - 5:1:1의 비율로 class imbalance가 심하지만, image feature상의 차이점이 커서 예측의 난이도는 높지 않다. 
    3. Age
        - 연령 그룹($age<30,30<=age<60,60<=age$) 으로 class가 나뉘지만, 실제 data의 age label은 60세 까지 밖에 없기 때문에 60세 이상에 대한 예측이 잘 이루어지지 않는다. 
        - Class imbalance와 더불어, 마스크를 착용한 상태에서의 age 예측이 굉장히 어려운데, 얼굴의 얼마 안되는 한정된 판단 근거마저 마스크가 가려버리기 때문이다.
    > Mask를 착용한 class에 대한 undersampling과 함께, Age 60 이상의 class에 대한 oversampling이 필요하다.

- Code modification
    - parser의 입력 값을 매 학습마다 command line을 통해 바꿔주는 것이 불편해서, yaml파일로 부터 config 값을 불러오도록 수정했다.
    - Train.py
        ``` python
        from config import cfg

        if __name__ == '__main__':
        parser = argparse.ArgumentParser(description="Mask face classification")
        parser.add_argument("--config_file",default="configs/ResNet152/config.yml",help="path to config file", type = str)
        args = parser.parse_args()

        if args.config_file != "":
            cfg.merge_from_file(args.config_file)
        cfg.freeze()
        print(cfg)

        data_dir = cfg.data_dir
        model_dir = cfg.model_dir
        
        train(data_dir, model_dir, cfg)
        ```
    - defaults_config.py
        ``` python
        from yacs.config import CfgNode as CN

        # Config Definition
        _C = CN()

        # Train
        _C.train = True

        # Model Hyperparams

        _C.seed = 42
        _C.epochs = 1
        _C.dataset = "MaskSplitByProfileDataset"
        _C.augmentation = "CustomAugmentation"
        _C.cropsize = [320, 256]
        _C.resize = [128, 96]
        _C.batch_size = 64
        _C.model = "BaseModel"

        # Training
        _C.optimizer = 'SGD'
        _C.lr = 1e-3
        _C.val_ratio = 0.2
        _C.criterion = 'cross_entropy'
        _C.lr_decay_step = 20
        _C.log_interval = 20
        _C.name = 'exp'

        # Loss

        _C.sampler = 'triplet' # triplet, triplet hard
        _C.num_instance = 4
        _C.label_smooth = False
        _C.feat_norm = False

        # Validation

        _C.validation_interval = 10
        _C.valid_batch_size = 64

        # Test 
        _C.test_data_dir = "/opt/ml/input/eval"
        _C.test_batch_size = 1000
        _C.test_model = 'exp'

        # Container enviornment

        _C.data_dir = "/opt/ml/input/data/train/images"
        _C.model_dir = "./model"
        _C.output_dir = "./output"

        ```

---
## Mentoring : Paper review 

- **Learning without Prejudices : Continual unbiased learning via benign and malignant forgetting**

    - Preliminaries
        - Catastrophic forgetting : Dataset A 로 pre-train된 model을 Dataset B 로 다시 학습할 경우, 학습 후 model은 dataset A에 대한 학습 정보를 손실한다.
        - Continual Learning : 여러 dataset에 대한 sequential한 학습 후에도 각 dataset에 대한 학습 정보를 model이 전부 기억하도록 하는 학습 방법
        - Bias(spurious correlation) : data가 가지는 attribute 중에, data의 label과 natural meaning이 강하지 않음에도, correlation이 큰 attribute
    
    - Motivation
        - Continual Unbiased Learning : 서로 다른 bias를 가진 dataset들을 continual하게 학습할 때, 각 학습 단계에서 학습이 bias에 빠지지 않도록 하는 학습 방식

        - Catastrophic forgetting in continual unbiased learning  
            > 현재 학습에서 이전 학습의 정보를 기억하지 못하는 것(forgetting)이 continual learning에 방해가 되는가?
            - Forgetting 하는 attribute에 따라 다르다!
        
        - Malignant, benign forgetting

            - To discourage malignant(catastrophic) forgetting
            - To encourage benign forgetting
    
    - Methodology

        - Discouraging malignant forgetting
            - 이전에 학습한 data의 정보를 담고 있는 model의 weight로부터 GAN으로 data를 생성하고, 이를 함께 학습시키자!
            - 그러나, GAN이 만든 부정확한 image에 labeling을 부여해서 학습하는 것은 모델에게 악영향을 끼칠 수 있다.
            - GAN은 이미지를 어떻게 만드는가?
                - GAN이 feature map들의 linear summation으로 이미지를 생성하니까, 이미지가 아닌 feature map을 학습에 함께 사용
            
        - Encouraging benign forgetting
            - Bias는 label space와 attribute 간의 잘못된 correlation이므로, label이 관여하지 않는 self-supervised learning(contrastive learning)을 사용하여 unbiased attribute에만 집중하도록 유도
            - Contrastive learning을 어떻게?
                - **Feature map에 서로 다른 augmentation(dropout)을 적용한 후, 서로 같은 feature임을 학습하도록 유도**

    - 멘토님의 발표를 들었을 때 인상깊었던 부분. 
        - Feature map의 augmentation을 적용한 후 self-supervised learning을 적용한 것
            1. Label이 관여하지 않는 학습 방식이므로 불필요한 Label-attribute correlation이 생길 수 없다
            2. 서로 다른 dropout을 진행한 feature map들 간의 contrastive learning을 수행함으로써 image에 담긴 natural meaning을 포착할 수 있다.
        - DINO와 굉장히 유사한 방식인데, Image 그 자체에 적용되는 augmentation이 feature map에서 dropout이라는 개념으로 확장될 수 있음이 인상깊었다.