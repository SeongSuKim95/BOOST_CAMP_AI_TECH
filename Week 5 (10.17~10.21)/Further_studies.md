# Week 5 : CV basic 

# Contents 

1. Studies : 교육 내용 중, 한 달 뒤에 있을 기술 면접을 고려하여 공부가 필요한 내용에 대해 심화 학습을 기록한다.

    - 나는 Vision Transformer과 CNN의 본질적인 동작 차이를 layer by layer로 설명할 수 있는가?
    - 나는 Generative model, GAN의 학습 방식을 수식으로 설명할 수 있는가?(TODO)

2. 과제 
    - 과제 1 : Conditional Generative Adverserial Network

---

# Studies


# 과제 

- 과제 1 : Conditional Generative Adverserial Network
    ``` python
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
- 50 Epoch 학습 결과
- 
<p align="center"><img src="https://user-images.githubusercontent.com/62092317/196613023-15f29fdc-91a4-48ed-9098-bfa6e58e4810.png" width = "350"></p>

- 각 input(image, label)과 output 단의 Linear layer를 하나 씩만 사용하여 학습[[LINK]](https://deep-learning-study.tistory.com/m/640)


Train Epoch: [0/50]  Step: [3700/7500]G loss: 0.83967  D loss: 0.50499 
Train Epoch: [10/50] Step: [3700/7500]G loss: 3.63719  D loss: 0.28119
Train Epoch: [20/50] Step: [3700/7500]G loss: 2.94529  D loss: 0.24260 
Train Epoch: [30/50] Step: [3700/7500]G loss: 2.63239  D loss: 0.29057 
Train Epoch: [40/50] Step: [3700/7500]G loss: 1.21396  D loss: 0.58974
Train Epoch: [49/50] Step: [3700/7500]G loss: 1.05710  D loss: 0.55929