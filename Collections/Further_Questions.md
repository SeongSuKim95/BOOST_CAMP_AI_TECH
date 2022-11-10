# Week 1

## Course - AI Math

- 통계학 맛보기

    1. 확률과 가능도의 차이는 무엇일까요? (개념적인 차이, 수식에서의 차이, 확률밀도함수에서의 차이)
    2. 확률 대신 가능도를 사용하였을 때의 이점은 어떤 것이 있을까요?
    3. 다음의 code snippet은 어떤 확률분포를 나타내는 것일까요? 해당 확률분포에서 변수 theta가 의미할 수 있는 것은 무엇일까요?

        ``` python
        import numpy as np
        import matplotlib.pyplot as plt
        theta = np.arange(0,1,0.001)
        p = theta ** 3 * (1 - theta) ** 7
        plt.plot(theta,p)
        plt.show()
        ```

# Week 2

## Course - Pytorch

- Hyperparameter tuning
    1. 모델의 모든 layer에서 learning rate가 항상 같아야 할까요? 같이 논의해보세요!
    2. ray tune을 이용해 hyperparameter 탐색을 하려고 합니다. 아직 어떤 hyperparmeter도 탐색한적이 없지만 시간이 없어서 1개의 hyperparameter만 탐색할 수 있다면 어떤 hyperparameter를 선택할 것 같나요? 같이 논의해보세요!


# Week 3

## Course - DL Basic

- Neural Networks & Multi-layer Perceptron
    1. Regression Task와 Classification Task의 loss function이 다른 이유는 무엇인가요?
    2. Regression Task, Classification Task, Probabilistic Task의 Loss 함수(or 클래스)는 Pytorch에서 어떻게 구현이 되어있을까요?

- Optimization
    1. Cross-Validation을 하기 위해서는 어떤 방법들이 존재할까요?
    2. Time series의 경우 일반적인 K-fold CV를 사용해도 될까요? [[LINK]](https://towardsdatascience.com/time-series-nested-cross-validation-76adba623eb9)

- Modern Convolutional Neural Networks
    1. 수업에서 다룬 modern CNN network의 일부는, Pytorch 라이브러리 내에서 pre-trained 모델로 지원합니다. pytorch를 통해 어떻게 불러올 수 있을까요?

- Recurrent Neural Networks
    1. CNN 모델이 Sequantial 데이터를 처리하는 데에는 구체적으로 어떤 한계점이 있나요?
    2. LSTM 에서는 Modern CNN 내용에서 배웠던 중요한 개념이 적용되어 있습니다. 무엇일까요?
    3. RNN, LSTM, GRU 는 각각 어떤 문제를 해결할 때 강점을 가질까요?

- Transformer
    1. 앞서 소개한 RNN 기반의 모델들이 해결하기 힘든 문제들을 Transformer은 어떻게 다루나요?
    2. Transformer 의 Query, Key, Value 는 각각 어떤 기능을 하나요? NMT 문제에서의 예시를 구체적으로 생각해보세요.

# Week 4,5

## Course -  CV

- Object Detection

    1. Focal loss는 object detection에만 사용될 수 있을까요? [[참고 자료: About Focal loss]](https://gaussian37.github.io/dl-concept-focal_loss/)
    2. CornerNet/CenterNet은 어떤 형식으로 네트워크가 구성되어 있을까요? [[참고 자료: CenterNet]](https://deep-learning-study.tistory.com/622)

- CNN Visualization
    
    1. 왜 filter visualization에서 주로 첫번째 convolutional layer를 목표로할까요?
    2. Occlusion map에서 heatmap이 의미하는 바가 무엇인가요?
    3. Grad-CAM에서 linear combination의 결과를 ReLU layer를 거치는 이유가 무엇인가요? [[참고 자료: Medium]](https://medium.com/@ninads79shukla/gradcam-73a752d368be)

- Instance Panoptic Segmenetation

    1. Mask R-CNN과 Faster R-CNN은 어떤 차이점이 있을까요? (ex. 풀고자 하는 task, 네트워크 구성 등) [[참고 자료1: About Mask R-CNN]](https://blahblahlab.tistory.com/139#:~:text=MASK%20RCNN%EC%9D%80%20%EA%B8%B0%EC%A1%B4%20object,%EB%8F%99%EC%8B%9C%EC%97%90%20%EC%B2%98%EB%A6%AC%ED%95%98%EB%8A%94%20%EB%AA%A8%EB%8D%B8%EC%9E%85%EB%8B%88%EB%8B%A4.) [[참고 자료2: About ROI align]](https://firiuza.medium.com/roi-pooling-vs-roi-align-65293ab741db)
    2. Panoptic segmentation과 instance segmentation은 어떤 차이점이 있을까요?
    3. Landmark localization은 human pose estimation 이외의 어떤 도메인에 적용될 수 있을까요?

- Multi-modal

    1. Multi-modal learning에서 feature 사이의 semantic을 유지하기 위해 어떤 학습 방법을 사용했나요?
    2. Captioning task를 풀 때, attention이 어떻게 사용될 수 있었나요?
    3. Sound source localization task를 풀 때, audio 정보는 어떻게 활용되었나요?

- Recent Trends on Vision Transformer

    1. ZeroCap에서 이미지에 적합한 caption을 생성하기위해, CLIP 정보를 어떻게 활용했나요?
    2. Transformer의 어떤 특징이 Unified Model 구성을 용이하게했나요?

## 과제 2 - Training with Data Augmentation

- Some other methods to improve model performance
    Augmentation을 적용한 모델의 결과가 더 좋아졌나요? 성능이 향상되었다면 축하드리며, 성능이 좋아지지 않거나 오히려 나빠졌더라도 괜찮습니다. 중요한 것은 augmentation으로 이 과제 코드에서 성능을 얼마나 올렸는지가 아니라, 다음에 다른 dataset, 다른 model로 학습을 진행할 때 필요한 augmentation을 생각해낼 수 있는 나름의 성공/실패 경험과 이유를 만드는 것이기 때문입니다.

    데이터의 특성 및 모델의 특성을 반영할 수 있는 몇가지 추가 질문들입니다. 아래 질문들에 대해 나름의 답을 생각해보시고, 직접 실험해보면서 다음에 새로운 데이터셋과 모델을 만난다면 어떤 augmentation을 왜 적용해야 할지, 나름대로의 이유를 생각해보시고, 실험해보시고, 다른 캠퍼분들과 토의해보세요.

    1. Quickdraw dataset은 상당히 얇은 선의 이미지들로 이루어졌습니다. 이런 경우, blur augmentation을 적용해 이미지를 넓게 만들어주는 것이 성능 향상에 도움이 될 수 있습니다. 왜 blur augmentation이 모델 성능 향상에 도움이 되는지 생각해보세요. Hint) 이미지가 얇은 선으로 이루어져 있다는 것은 빈 공간이 많다는 뜻이고, 이는 곧 layer의 input으로 0이 상당히 많이 들어간다는 뜻입니다.

    2. Augmentation의 효과를 더욱 크게 하기 위해서는, 구현의 편의를 위해 (150, 150)로 image를 resize하는 대신 훨씬 작은 원본 이미지 크기를 사용하는 것이 좋을 수 있습니다. 위 blur augmentation이 (150,150)에서 효과가 적은 이유에 대해서 고민해보세요. Hint) CNN의 convolution filter 의 사이즈와 연관이 있습니다.

    3. Augmentation의 순서 역시 중요한 요소가 될 수 있습니다. Blur를 먼저 적용하고 크게 resize하는 경우와 resize를 먼저 하고 blur를 적용하는 경우, 이미지가 어떻게 달라질지 예상해보시고, 실제 augmentation package에 따라 결과가 어떻게 나타나는지도 확인해보세요. 그리고 Geometric augmentation의 경우에 대해서도 순서가 어떻게 영향을 미칠지 생각해보세요. 

    4. Augmentation을 적용했음에도, convolution layers를 고정하고 linear classifier만 새로 학습하는 fine-tuning을 수행하면 augmentation의 효과가 떨어질 수 있습니다. 그 이유에 대해 생각해보세요.

## 과제 3 - Classification to Segmentation

- 아래는 주어진 샘플 이미지에 대한 segmentation 결과를 시각화하는 과정입니다.

    첨부 그림은 예시 이미지 하나에 대해 결과를 시각화한 것입니다. 결과를 보시면 알 수 있듯이 classification 모델을 segmentation 모델로 재구성함에 따라 해당 모델이 각 영역을 어떻게 판단하고 있는지를 확인할 수 있습니다. 마스크 데이터셋을 통해 수행한 분류 문제가 해당 인물이 마스크를 어떤 타입으로 착용하고 있는지를 구분하는 것이었으므로 마스크 주위에 segmentation 결과가 집중되어 있습니다.

    추가로 정확히 마스크의 영역을 잡아내고 있지는 않은데 이는 마스크 영역에 해당하는 픽셀별 ground truth가 주어지지 않았기 때문이며 또한 입력 이미지에 비해 16분의 1 사이즈의 feature map에서 픽셀별 예측을 진행하고 단순히 bilinear interpolation을 진행했기 때문입니다. 어떻게 하면 해당 segmentation network의 성능을 더 끌어올릴 수 있을지 고민해보세요 :)
  
    <p align="center"><img src='https://drive.google.com/uc?id=1IFw0QT2zbr1txEQXaTBGuRGtgXBm8ruP'  width="224"></p>

- **want_to_check_heat_map_result** flag를 True로 설정하여 추가로 heatmap을 확인해보세요. 해당 과정을 통해 모델의 출력값을 그대로 시각화할 수 있습니다!

- Segmentation이라고 했지만, 막상 heatmap에 가깝고 정확하게 Segmentation이 안되는 결과가 나온 것 같습니다. 좀 더 Segmentation 답게 만들기 위해서는 어떤 것들을 추가적으로 해야할지 고민해보시기 바랍니다.

