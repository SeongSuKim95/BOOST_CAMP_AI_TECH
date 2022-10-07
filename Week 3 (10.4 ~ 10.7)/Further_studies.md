# Week 3 : DL Basic 

# Contents 

1. Studies : 교육 내용 중, 한 달 뒤에 있을 기술 면접을 고려하여 공부가 필요한 내용에 대해 심화 학습을 기록한다.

    - 나는 Vision Transformer과 CNN의 본질적인 동작 차이를 layer by layer로 설명할 수 있는가?
    - 나는 Generative model, GAN의 학습 방식을 수식으로 설명할 수 있는가?(TODO)

2. QnA with camper : 캠퍼와 주고 받은 질문들에서 얻은 insight를 기록한다.
    - 8bit 이미지 데이터를 왜 0~1로 normalize 하나요?

3. Code review : Coding study channel과 peer session에서 review한 코드에 대해 복기한다.
    - 스타트와 링크
    - 두 큐 합 같게 만들기

---

# Studies

## 1. 나는 Vision Transformer의 동작 방식을 layer by layer로 설명할 수 있는가?
- Vision Transformer가 CNN과 어떻게 다른지를 설명한 2개의 논문을 찾을 수 있었다. 내용을 정리하면서, ViT를 깊게 이해해보자.

    - *__Do vision transformers see like convolutional neural networks?__* [[LINK]](https://arxiv.org/abs/2108.08810) [[MEDIUM]](https://towardsdatascience.com/do-vision-transformers-see-like-convolutional-neural-networks-paper-explained-91b4bd5185c8)
    - *__How Do vision Transformers Work?(ICLR,2022)__*[[LINK]](https://arxiv.org/abs/2202.06709) [[CODE]](https://github.com/xxxnell/how-do-vits-work)
    
1.  Do Vision Transformers See Like Convolutional Neural Networks?
    
    - 주제 
      - Vision transformer가 "무엇"을 "어떻게" 학습하는지 CNN과 비교하여 직관적으로 이해하기 좋게 visualize한 논문
      - 논문을 읽고 알게 된 9가지 추가적인 사실을 정리하고, 생각을 적는다.
    - 핵심 내용
      1. ViT는 CNN에 비해 모든 layer에서 uniform한 representation을 학습한다.
            <p align="center"><img src= "https://user-images.githubusercontent.com/62092317/152669780-a5329c1f-a8d9-4c49-b933-a3f68315f07c.png"></p>
    
            - 왼쪽 : ViT, 오른쪽 : ResNet , ViT는 모든 layer에서의 feature가 연관성이 있는 반면, ResNet은 높은 layer와 낮은 layer에서의 feature가 확실히 구분된다.
      2. ViT가 후반부에서 보는 feature는 ResNet의 초반부의 feature와 다르다.
            <p align="center"><img src= "https://user-images.githubusercontent.com/62092317/152669782-beb6e06a-49ec-4b88-b2d6-7e546f1241ef.png"></p>

            - 또한, ViT의 초반부 layer(0-40)의 feature와 ResNet에서의 중반부 layer(0-60)의 feature가 비슷하다.
      3. ViT는 낮은 layer에서도 global한 정보와 local한 정보에 동시에 attend 할 수 있다. 높은 layer에서는 거의 모든 attention head들이 global한 정보에 attend 한다. 그러나, train data가 충분하지 않을 경우엔 ViT의 낮은 layer가 local한 정보에 대해 attend하지 못한다.(화살표로 표시)
            <p align="center"><img src= "https://user-images.githubusercontent.com/62092317/152669784-4012edb7-6932-46ab-ba03-8d7b0e55bcd8.png"></p>

      4. ViT가 보는 feature를 attention head의 mean distance에 따라 ResNet의 lower layer feature과 비교해 보았을때, ViT가 attend하는 local한 정보가 ResNet의 lower layer feature와 비슷함을 알 수 있다. (2번 그림과 연결됨, Mean distance 작음: local한 정보, Mean distance 큼 : global한 정보) 
            <p align="center"><img src= "https://user-images.githubusercontent.com/62092317/152669785-34e6ba9f-64bc-4305-af02-7ffcab264e06.png"></p>

      5. **(중요)** CLS token은 초반부 MLP layer의 skip connection을 많이 통과 하며,후반부 block 에서 main branch를 통과한다. 또한, Skip connection은 CNN보다 ViT에서 훨씬 더 effective하게 동작한다. 
            <p align="center"><img src= "https://user-images.githubusercontent.com/62092317/152669786-03ca8fa6-298d-43f4-90d4-aed157660ada.png"></p>

            - 왼쪽 그림 
                - 0번 token은 CLS token, Ratio of Norm의 크기가 클 수록 Skip connection branch를 많이 통과한 것이고 작을 수록 main branch를 많이 통과한 것
            - 오른쪽 그림
                - ResNet과의 비교로, ViT에서 전반적으로 skip connection이 많이 사용됨을 알 수 있다.
            - CLS token은 다른 patch token들과는 완전히 반대된 경향을 보인다. 즉 CLS token은 lower layer에서 skip connection을 많이 통과하는 반면, patch token들은 higher layer에서 많이 통과한다.

            <p align="center"><img src= "https://user-images.githubusercontent.com/62092317/152669778-3a8f388a-7584-4f23-901b-67d4978b67a2.png"></p>

            - (NOTE) 5-1 과 5-2 의 y axis (Block index) 방향이 반대임
            - 5-1의 결과를 MLP와 SA의 skip connection 으로 나누어 분석할 때, CLS token은 SA보단 MLP layer에 많이 영향을 받음
            - Cosine similarity graph에서 값이 1에 가까울 수록 skip connection을 통과한다고 볼수 있으며, 대응되는 부분을 빨간색 화살표로 표시. CLS token의 경우 lower layer에서 MLP, SA 모두에서 skip connection을 통과하며 higher layer에서 MLP main branch에 영향을 받는 것을 알 수 있음.
      6. **(중요)** 기존의 ViT처럼 CLS token을 통해 linear probing을 하였을 때, ViT는 higher layer까지 token의 spatial location 정보를 잘 유지한다.
            <p align="center"><img src= "https://user-images.githubusercontent.com/62092317/152669787-1e86650b-86d2-404a-a433-4ab9cd53425d.png"></p>
            <p align="center"><img src= "https://user-images.githubusercontent.com/62092317/152669788-af429bd5-898a-4e5e-b4a6-eaeb2ae02f29.png"></p>

            - 그러나 모든 token들의 GAP를 사용하여 linear probing을 할 경우 , ViT 또한 각 token의 spatial location 정보를 유지하지 못하며 모든 token들이 비슷한 location 정보를 갖게 된다.

            <p align="center"><img src= "https://user-images.githubusercontent.com/62092317/152669789-1931c803-8644-411c-8355-26323757b5a2.png"></p>

            - 왼쪽 그림
                - 각각의 token들을 이용하여 test 한후 평균 average를 구한 것. 빨간색의 경우 CLS token으로 linear probing을 한 후 각각의 token들에 대해 성능을 측정한 후 평균을 했기 때문이다. 즉, CLS token을 제외한 나머지 token 들은 CLS에 대한 정보가 아닌 각 token들의 spatial location 정보를 지닌 token들이기 때문에 test시 성능이 높지 않다. GAP로 학습할 경우 모든 token들이 비슷한 정보를 가지며 CLS를 대표하도록 학습되기 때문에 어떤 token(CLS token도 포함)으로 성능을 측정해도 성능이 잘 나온다.
            - 오른쪽 그림
                - GAP ; First token 과 GAP ; GAP 는 결과적으론 마지막 layer에서 비슷한 성능을 내지만, First token(CLS token)의 경우 그림 5의 학습과정 때문인지 후반부 layer에서 test 성능이 급격히 증가하는 것을 알 수 있다. 반면에, GAP ; GAP는 어느 구간의 layer에서도 test 성능이 꾸준히 높다. 즉, GAP로 학습을 수행해도 그림 5에서 CLS token과 patch token의 학습 차이는 존재한다. 

            <p align="center"><img src= "https://user-images.githubusercontent.com/62092317/152740247-545a0cf7-75e9-49e7-bf43-3483d5bf743e.png"></p>

            - CLS; GAP except CLS token과 CLS; GAP를 보면 전자가 미세하지만 전 구간에 대해 성능이 높게 측정됨을 알 수 있다. GAP를 사용할 때 CLS token의 역할이 다른 token들 보다 약하다는 증거로 볼 수 있다.

                - ViT를 CLS token으로 train 시킬 때, GAP로 train 시킬때에 대해서 각 token들로 classification을 했을 때 accuracy를 나타낸 것
                - 앞서 봤던 그래프에서의 결과와 동일한 양상을 보이는데(y axis 값 주의), 두 경우 모두 Layer 6까지는 비슷한 경향을 보인다는 것에 집중
                - **(중요)** 이를 통해 학습의 차이가 classifier단과 가까운 layer에서 매우 크게 존재하고, 그 이전 layer까지는 비슷하게 학습이 진행된다고 볼 수 있다.

      7. Train data가 충분하지 않을 때, ViT는 higher layer에서 충분한 representation을 학습하지 못한다.  

            <p align="center"><img src= "https://user-images.githubusercontent.com/62092317/152669775-ee928377-9323-41e3-bcfb-c88599e779cc.png"></p>

            - 반면에 data의 양에 상관 없이, lower layer에서의 representation은 유지된다.
            - Intermediate representation은 data의 양에 큰 영향을 받으며, 성능에 미치는 영향이 크다. 학습 데이터가 많을 수록 ViT는 중간 layer에서 high quality의 representation을 배우게 된다.

      8. ViT는 model size, data에 상관 없이 lower layer에서 유사한 representation을 학습한다.
        
            <p align="center"><img src= "https://user-images.githubusercontent.com/62092317/152669779-0680c71a-fa7d-4ffa-b7ab-7b8bb7bdf2af.png"></p>

      9. (중요)ViT가 모든 layer에 걸쳐 uniform한 representaion을 배우고, spatial한 location을 유지할 수 있는 것은 skip connection의 영향이 크다.

            <p align="center"><img src= "https://user-images.githubusercontent.com/62092317/152680128-d7b9182a-fce8-48db-baad-8da1574b9ce9.png"></p>

            - 특정 Block내의 skip connection을 없앨 경우, 그 이후 block에서의 representation과 이전 block에서의 representation간 괴리가 매우 큼
            
            <p align="center"><img src= "https://user-images.githubusercontent.com/62092317/152669777-d3e232a3-d6a9-4455-a992-fc698027f74a.png"></p>
        
            - Receptive field 또한 이러한 영향 때문에 center patch에 dominate하게 형성된다.
    
    - 생각 정리
      - Inductive biase가 CNN보다 훨씬 적은 ViT기 어떻게 image를 효율적으로 해석하는지에 대해 직관적으로 이해할 수 있었다.
      - __Input patch의 corresponding output patch는 각 input patch와 correlation이 가장 높다.__ 즉, CNN에선 GAP를 통해 locality 정보가 사라지는 반면, ViT에선 ClS token이 cls feature를 대표하고 나머지 output patch들은 각각의 위치에 해당하는 이미지의 정보를 담고 있다.이는 향후 detection 분야에도 유용하게 쓰일수 있다고 논문에서도 언급하고 있다.
      - Data의 size가 ViT의 intermediate representation 학습을 결정하는데, imbalance한 dataset에 대해서는 어떨지 생각해보아야 한다.

2.  How Do Vision Transformers Work?
    > 읽는 중...

## 2. 나는 Generative model, GAN의 학습 방식을 수식으로 설명할 수 있는가?  
> 여긴 다시 돌아오자. 10/16 이후..
---

# QnA with camper

- SLACK의 공통 질의 응답 채널에 다음과 같은 질문이 올라왔다.
    > Question : 8bit 이미지 데이터를 왜 0~1로 normalize 하나요?
    - 당연해 보이지만 당연하지 않은 질문이다. 
    - 흔히 데이터 정규화를 설명할 때는, 데이터의 평균과 분산으로 표준화 하는 문제를 얘기하고 이것이 학습의 수렴 속도를 높인다는 것을 이유로 든다.
    - 그러나, 이 질문은 scaling에 관한 normalization이기 떄문에 약간 다를 수 있다고 생각 하여 찾아보게 되었다. [[LINK]](https://stackoverflow.com/questions/57454271/should-i-still-normalize-image-data-divide-by-255-before-using-per-image-stand)
- **[0~255]를 [0~1]로 normalize하는 이유**ㄴ
    1. numerical 한 value의 크기가 커서 발생하는 gradient exploding을 막을 수 있고, 학습의 수렴 속도를 높인다. ( 보통 activation function의 0~1 구간의 기울기가 크기 때문에 ex: tanh, sigmoid 등)
    2. 0~1로 normalize한 후 model을 training할시 Transfer learning에 유용하다. 어떤 feature scale의 이미지 데이터에 대해서도 0~1에 대한 학습 값을 유지할 수 있기 때문에.
    3. Scaling을 하지 않을 경우 각 feature들이 loss function에 미치게 되는 영향력이 달라 지고, 이 때문에 특정 feature에 편향된 학습이 이루어질 수 있다.
---

# Code review

## [백준 14889] 스타트와 링크

> 20명 앞에서 라이브 코딩 도전 ~~~

```python
import sys; input = sys.stdin.readline

N = int(input())

_map = [list(map(int,input().split())) for i in range(N)]

visited = [False] * N
result = []
_min = 1e9

def diff():
    _start = 0
    _link = 0
    for i in range(N-1):
        for j in range(i+1,N):
            #_map 순회하기
            if visited[i] and visited[j] :
                _start += _map[i][j]
                _start += _map[j][i]
            elif not visited[i] and not visited[j]:
                _link += _map[i][j]
                _link += _map[j][i]
    return abs(_start - _link)

def dfs(depth,idx,N):
    global _min
    if depth == N//2:
        # 차이를 구하는 함수
        _min = min(_min, diff())
        # 만약 차이가 0이면 최소값이므로 바로 프로그램 종료
        if _min == 0:  
            print(_min)
            exit(0)
        return
    for i in range(idx,N):
        if not visited[i]:
            visited[i] = True
            dfs(depth+1, i+1, N)
            visited[i] = False

dfs(0,0,N) # depth, idx, N
```

- 이 코드의 중점은 List의 요소들을 양분 하되, 요소들의 성질을 고려하여 가장 공평한 이분법을 찾는 것이다. 이 때 DFS가 사용된다.
- 라이브 코딩 당시 _min 변수를 global로 선언하지 않고, _map을 탐색할 때 index i,j를 헷갈리는 바람에 오류가 떴지만 금방 잡아냈다.
- Camper 들의 피드백
    - 가장 밸런스가 잘 맞는 조합을 찾으면, 그 즉시 탐색을 종료하는 코드가 추가되면 시간을 줄일 수 있다.
    - 전체 인원을 [1,2,3,4,5,6] 이라고 할때, 123을 뽑는 경우의 수와 456을 뽑는 경우의 수는 같은 경우 이기 때문에 이를 판단하는 코드를 추가하면 탐색의 경우의 수를 줄일 수 있다.

## [프로그래머스 Lv.2] 두 큐 합 같게 만들기

- 내 코드 : 두 개의 queue 각각에 대해 append를 하면 시간초과 기준을 통과하지 못한다. 한개의 queue에만 append를 해도 된다는 사실을 파악하지 못했다.
    ```python
    from collections import deque

    def solution(queue1, queue2):
        target_sum = (sum(queue1) + sum(queue2)) / 2
        print(target_sum)
        left_sum = sum(queue1)
        queue1 = deque(queue1)
        queue2 = deque(queue2)

        answer = 0
        while queue1 and queue2:
            if left_sum < target_sum:
                tmp = queue2.popleft()
                left_sum += tmp
                queue1.append(tmp)
                answer += 1
            elif left_sum > target_sum:
                left_sum -= queue1.popleft()
                answer += 1
            else:
                return answer
            print(f"queue1:{queue1},queue2:{queue2},left_sum:{left_sum}")
        else:
            return -1
    ```
- 캠퍼 A의 코드 : Two pointer를 사용한다. 어차피 두 queue의 조합으로 만들 수 있는 배열의 순서는 정해져있다는 것이 핵심이다. 두번째 queue에 append하는 행위를 없애는 것과 동일한 효과를 얻는다.
    ```python
    from collections import deque
    def solution(queue1, queue2):      
        if sum(queue1)>sum(queue2):
            tmp=queue1
            queue1=queue2
            queue2=tmp
            
        q=queue1+queue2
        
        start=0
        end=len(queue1)-1
        cnt=0
        
        s1=sum(queue1)
        s2=sum(queue2)
        
        while start<=end and end<len(q):
            if s1==s2:
                return cnt
            elif s1<s2 and end+1<len(q):
                end+=1
                s1+=q[end]
                s2-=q[end]
            else:
                s1-=q[start]
                s2+=q[start]
                start+=1
            
            cnt+=1
        
            return -1
        ```