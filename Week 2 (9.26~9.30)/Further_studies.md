# Week 2 : Pytorch

## Contents 
- Pytorch의 기본과 관련된 10개의 질문을 스스로에게 해보고 답해보자.

---
## 1. 나는 torch.Tensor.view와 torch.Tensor.reshape의 차이를 설명할 수 있는가?
- 내가 pytorch를 몇 년 쓰면서 이 질문을 대답 못하겠더라.. 참내~
- 강의에서 torch.Tensor.view와 torch.Tensor.reshape의 가장 큰 차이는 "Contiguity 보장 여부" 라는데 이 말이 무슨 뜻인지 찾아보았다.

    > 매모리 상에서 tensor의 요소들이 저장되어 있는 주소가 순차적으로 이어져 있는 경우, 이를 contiguous한 상태라고 한다.

- 예시를 들어보면,
    ``` python
    # is_contiguous는 x가 메모리에 contiguous하게 저장되어 있는지를 Boolean으로 반환해준다.
    a = torch.tensor([[1,2,3],[4,5,6]])
    print(a) 
    print(a.is_contiguous())
    # tensor([[1, 2, 3],[4, 5, 6]]) , True
    a = a.transpose(0,1)
    print(a)
    print(a.is_contiguous())
    # tensor([[1, 4],[2, 5],[3, 6]]), False
    a = a.contiguous()
    print(a)
    print(a.is_contiguous())
    # tensor([[1, 4],[2, 5],[3, 6]]), True
    ```
    위 코드에서 볼 수 있듯이, a가 transpose되는 순간 원소 4는 tensor상에서 2번째에 위치하게 되는데 실제로 a가 선언될 때 메모리 상에서는 4번째에 위치하고 있고 is_contiguous()가 False를 뱉는다. 추가로, contiguous()를 통해 tensor를 contiguous하게 바꿀 수 있다.
- 궁금하니까 실제로 그런지 확인해보자.
    ``` python
    a = torch.tensor([[1,2,3],[4,5,6]])
    b = a.t()
    for i in range(2):
        for j in range(3):
            print(a[i][j].data_ptr())
    94066690072064
    94066690072072
    94066690072080
    94066690072088
    94066690072096
    94066690072104 # 32bit float, 8 byte씩 증가
    for i in range(3):
        for j in range(2):
            print(b[i][j].data_ptr())
    94066690072064 1
    94066690072088 4
    94066690072072 2 
    94066690072096 5
    94066690072080 3 
    94066690072104 6
    b = b.contiguous()
    for i in range(3):
        for j in range(2):
            print(b[i][j].data_ptr())
    94395224670592
    94395224670600
    94395224670608
    94395224670616
    94395224670624
    94395224670632 # 8 byte씩 증가 하도록 align된 모습
    ```
    - 주소 값을 뽑아봄으로써 알게 된 한가지 사실은, contiguous()가 적용될 때엔 기존에 사용되던 메모리 공간이 아닌 새로운 주소에 값들을 정렬하여 할당한다는 것이다.
    - 그럼 기존 주소에 있는 값들은 남아있을까?
        ``` python
        import ctypes
        print(ctypes.cast(id(a), ctypes.py_object).value)
        # tensor([[1,2,3],[4,5,6]]) 남아있다.
        ```
    - **즉, contiguous()가 적용될 때엔 다른 메모리 주소로의 복사가 일어나는 것이다.**

- 다시 돌아와서, 강의자료의 torch.Tensor.view()와 torch.Tensor.reshape()의 차이를 나타내주는 코드를 살펴보자.
    ``` python
    a = torch.zeros(3,2)
    b = a.view(2,3)
    a.fill_(1)
    print(a,b) # [[1,1],[1,1],[1,1]], [[1,1,1],[1,1,1]]
    a = torch.zeros(3,2)
    b = a.t().reshape(6)
    a.fill_(1)
    print(a,b) # [[1,1],[1,1],[1,1]], [0,0,0,0,0,0]
    ```
    - view를 사용하여 변환된 tensor는 기존 텐서의 변경사항이 적용 되고, reshape을 사용한 경우엔 적용이 안되는 것을 알 수 있다.
    - 메모리 주소를 뽑아보면,
        ``` python
        a = torch.zeros(3,2)
        b = a.view(2,3)
        a.fill_(1)
        print(a,b) # [[1,1],[1,1],[1,1]], [[1,1,1],[1,1,1]]
        for i in range(3):
            for j in range(2):
                print(a[i][j].data_ptr())
        94252172893760
        94252172893764
        94252172893768
        94252172893772
        94252172893776
        94252172893780
        for i in range(2):
            for j in range(3):
                print(b[i][j].data_ptr())
        94252172893760
        94252172893764
        94252172893768
        94252172893772
        94252172893776
        94252172893780
        ```
    - a 와 b가 같은 tensor index의 element에 대해 동일한 메모리 공간을 쓰는 것을 알 수 있다. 
    - **즉 , view는 transpose후에도 기존에 사용하던 메모리 공간에 contiguous하게 element들이 저장되도록 해준다는 것이다.**
    - 그렇기 때문에, a가 할당된 메모리 주소에 1을 채워도 결국 b와 주소를 공유하기 때문에 b 또한 바뀌게 되는 것이다. 마찬가지로 b.fill_(1)을 해도 a가 함께 바뀐다.
    - 반면, reshape의 경우는 어떨까?
        ``` python
        a = torch.zeros(3,2)
        for i in range(3):
            for j in range(2):
                print(a[i][j].data_ptr())
        94725047713792
        94725047713796
        94725047713800
        94725047713804
        94725047713808
        94725047713812
        a = a.t()
        for i in range(2):
            for j in range(3):
                print(a[i][j].data_ptr())
        94725047713792
        94725047713800
        94725047713808
        94725047713796
        94725047713804
        94725047713812 # transpose로 인해 contiguous 상태가 깨진 모습
        b = a.reshape(6)
        a.fill_(1)
        print(a,b) # [[1,1,1],[1,1,1]], [0,0,0,0,0,0]
        for i in range(6):
            print(b[i].data_ptr())
        94725108856640
        94725108856644
        94725108856648
        94725108856652
        94725108856656
        94725108856660 # 메모리의 다른 공간으로 contiguous하게 저장
        ```
    - reshape은 view와 다르게 기존과 다른 메모리 공간으로 새롭게 b를 할당한 것을 알 수 있다. 따라서 a.fill()로 a의 메모리 주소에 변화를 주어도 b는 영향을 받지 않은 것이다.
    - 코드를 보다 보니 b = a.reshape((3,2))을 하면 어떻게 될지 궁금해졌다.
        ``` python
        a = torch.zeros(3,2)
        for i in range(3):
            for j in range(2):
                print(a[i][j].data_ptr())
        94768835468864
        94768835468868
        94768835468872
        94768835468876
        94768835468880
        94768835468884
        a = a.t()
        for i in range(2):
            for j in range(3):
                print(a[i][j].data_ptr())
        94768835468864
        94768835468872
        94768835468880
        94768835468868
        94768835468876
        94768835468884
        b = a.reshape((3,2))
        a.fill_(1)
        print(a,b) # [[1,1,1],[1,1,1]], [[1,1,1],[1,1,1]]
        for i in range(3):
            for j in range(2):
                print(b[i][j].data_ptr())
        94046802474176
        94046802474180
        94046802474184
        94046802474188
        94046802474192
        94046802474196
        ```
    - 예상한 대로이다. 따라서, reshape은 항상 새로운 메모리 공간으로 원소들을 복사하는 함수이다.
    - 추가로 알아낸 것은, 기존과 동일한 shape으로 reshape을 하는 경우에는 기존의 메모리공간을 사용한다.(즉, 아무일도 일어나지 않는다. 그렇게 쓸 일도 없겠지만..)

- 한가지 더 알게 된 사실, Pytorch 공식 문서를 보면 view는 contiguous()한 상태의 tensor에 대해서만 적용 가능하다고 한다. 실험해보자.
    ``` python
    a = torch.zeros(3,2)
    a = a.t()
    print(a.is_contiguous()) # False
    b = a.view(6)
    # RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
    b = a.reshape(6)
    # [0,0,0,0,0,0]
    ```
    - view는 입력값이 contiguous 하지 않을 경우 error가 나고, reshape을 사용하라는 문구가 나온다.
    - 반면에 reshape은 contiguous여부에 상관 없이 출력이 나온다.
> 내가 강의를 들었을 땐 강사님께서 view와 reshape의 차이를 설명하실 때, a.fill_()에 따른 기존 tensor의 변경 여부에 강조점을 두시면서 contiguous한 속성의 차이 때문이라고 하셨는데, 지금 다시 생각해보면 설명이 일부 틀렸다고 생각된다. view 든 reshape이든 contiguous한 tensor를 반환한다.
- 정리하자면

    1. view는 기존 tensor의 메모리 공간에 contiguous한 tensor를 return한다.
    2. reshape은 다른 메모리 공간에 contiguous한 tensor를 return한다.
    3. view는 contiguous한 tensor에만 적용할 수 있는 반면, reshape은 그렇지 않다.

> 어찌되었건 실험 결과 view든 reshape이든 contiguous한 값을 return해준다. 그렇다면 왜 contiguous하게 만들어주도록 설계되었을까?

-  Memory Misalignment에 대해..
    - 학부 마이크로프로세서 과목과 컴퓨터 아키텍쳐를 수강하면서 memory misalignment라는 개념을 배운 적이 있다.
    - 메모리의 Contiguous한 사용과 직접적인 관련이 있는 내용이다. 정리하자면 CPU와 메모리(ex: cache)간 data를 교환 할때 bus width(32bit)크기 만큼의 data를 주고 받는데, 이 때 요구되는 data 크기와 주소에 상관 없이 메모리 주소를 4(8bit x 4)씩 건너뛰어가며 data를 가져온다는 것이다. 
    -  예를 들어, 메모리 주소 01에 있는 8bit 데이터만 필요해도 00~03의 주소에 접근하여 32bit를 가져오게 되며, 03~04에 있는 16bit데이터가 필요하다면 03을 가져오기 위해 00~03 32bit를, 04를 가져오기 위해 04~07 32bit를 가져오게 된다는 것이다. 만약 이 16bit data가 00~03사이에 저장되어 있었다면 한 번의 cycle만으로 다 들고 올 수 있었던 것을 두 번의 access time으로 들고 오게 되는 셈이다.
    - 따라서, misalignment된 data들은 불필요한 data transfer cycle을 늘려 access time을 증가시키게 된다. 결국 memory에 data를 저장할때 bus크기와 data의 크기에 맞게 align하는 것이 중요하다.
    - Cache miss,hit 개념도 생각이 났는데 결국 같은 얘기라서 생략한다.

> 딥러닝의 학습과정에서 tensor의 contiguous한 성질을 어떤 영향을 줄까?
- SLACK의 질문 채널에서 다른 캠퍼분이 한 질문을 읽다 보니, 이와 연결되는 내용이 있었다.
## 2. Module Class안의 super.__init__()은 왜 써주는 걸까?
- 이게 무슨 역할인지는 알고 있었지만, 이렇게 쓰는 이유를 설명할 줄 알아야한다고 생각했다.
- 