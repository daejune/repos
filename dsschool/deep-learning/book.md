#
# Week 01
PyTorch or Tensorflow 는 기초만

- 이론 위주의 수업을 하고 Python으로 작성
- PyTorch, Tensorflow에 적용하는 방법

Machine Learning Algorithm

- 개발자가 항상 조건을 만들게되니 유지보수가 어려워짐
- 기계가 스스로 학습
- Supervised (Data - Feature & Label, 지도학습)
    - Data 확보가 어려움
    - 연구적인 결과를 얻기 쉬움
    - Tree Base
    - Regression(Linear, Logistic, SVM) - Structure Data
    - Neural Network(Perceptron($\neq$ Singlelayer NN), Multilayer Perceptron(Multilayer NN), Convolution NN) - Unstructure Data
    - Bayesian
- Unsupervised (Data - Feature & No Label, 비지도학습)
    - Data 확보는 쉬움
    - 연구적인 결과를 얻기 힘듦
- Reinforcement(강화학습 - 쓰임세가 거의 없음)

### Deep Learning Algorithm

- Single-layer, Multi-layer NN using generated data


# Gradient Descent
- $m$ - # of data
- $X$ - features
- $y$ - label
- $W$ - weights
- $b$ - bias
- $h(X) = {W}\cdot{X} + b$
- $L(y, h(X))$ - Loss Function
    - ex) $L(y, h(X)) = {1 \over 2}(h(X)-y)^2$ - mean squared error loss
- $J(W, B) = {1 \over m}{\sum_{i=1}^{m} L(y^i, h(X^{i}))}$ - Cost Function

``` python
import numpy as np

W = W - learning_rate * X.T.dot(y_predict - y) / len(X)
b = b - learning_rate * (y_predict - y).mean()
```

Loss Function의 기울기를 이용하여 weight를 update할 크기와 방향을 결정할 수 있음

- 순간 변화율: ${{\Delta y}\over{\Delta x}} = {{f(x + \Delta x) - f(x)}\over{x + \Delta x - x}}$

- $W = W - {d \over dW}L(y, h(x))$
- ${\partial \over \partial W}L(y, h(X))$



#
## Single-layer Neural Network
- $W = W_{prev} - \lambda \cdot {d \over dW}J(W_{prev}, B)$

Classification | 
- 암환자 구분 (Binary)
- 스팸 (Binary)
- 상품 분류 (Multi Classes)
- $y$ = 0 or 1
- Categorical

Regression
- 부동산
- 주가 예측
- $y$ = $0$ ~ $\infty$
- Continuous

Classification | Regression
---------------|-----------
암환자 구분 (Binary) | 부동산
스팸 (Binary) | 주가 예측
상품 분류 (Multi Classes) |
$y$ = 0 or 1 | $y$ = $0$ ~ $\infty$
Categorical | Continuous

Function
- $z(X)~=~X \cdot W~+~B$
- $h(X)~=~f(z(X)$)


Sigmoid
- ## ${1\over{1+e^{-x}}}$

https://www.wolframalpha.com

Loss Function
- ## $-\log({1\over{1+e^{-x}}})$

Cross Entropy
- $L(h(x), y) = -y \cdot \log h(x) - (1-y)\log(1-h(x))$
- $y=0 \rightarrow -\log(1-h(x)), h(x)=0 \rightarrow 0, h(x) = 1 \rightarrow \infty$
- $y=1 \rightarrow -\log h(x), h(x)=0 \rightarrow \infty, h(x) = 1 \rightarrow 0$

## ${\partial L(h(x), y)} \over {\partial W}$
$$f(x) = {{g(x)} \over {h(x)}}$$
$$~$$
$$f'(x) = {{g'(x)h(x) - g(x)h'(x)} \over {(h(x))^2}}$$
$$~$$
$$...$$
$$~$$
$$sigmoid~'(x)~~=sigmoid(x)(1-sigmoid(x))$$


### Define Sigmoid Function
```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```

### Binary Classification
``` python
for epoch in range(epoch_num):
    y_predict = sigmoid(X.dot(W) + b)
    predict = (y_predict > threshold).astype('int')

    accuracy = (predict == y).mean()
    if accuracy > 0.95: # 95% Over
        break
    W = W - learning_rate * (X.T.dot(y_predict - y)) / len(X)
    b = b - learning_rate * (y_predict - y).mean()
```


### Multi Classes Classification
- One Hot Encoding


```python
import numpy as np
np.array.mean(axis = 0) # Columns mean
np.array.mean(axis = 1) # Rows mean
```

```python
W = np.random.uniform(low=-1.0, high=1.0, size=(features, classes))
b = np.random.uniform(low=-1.0, high=1.0, size=(1, classes))

for epoch in range(num_epoch):
    y_predict_hot = X.dot(W) + b
    y_predict_hot = sigmoid(y_predict_hot)

    y_predict = y_predict_hot.argmax(axis = 1)

    accuracy = (y_predict == y).mean()
    if accuracy > 0.95: # 95% Over
        break

    W = W - learning_rate * X.T.dot(y_predict_hot - y_hot) / len(X)
    b = b - learning_rate * (y_predict_hot - y_hot).mean(axis = 0)
```


#
# Week 02

## AND / OR
- Dicision Boundery가 보이면 Single Layer로 풀 수 있다.

## XOR
- Dicision Boundery가 보이지 않으면 Single Layer로 풀 수 없다.

## Polynominal
- feature 끼리 곱셈 or 제곱 등을 통해 추가적인 feature를 만들어 문제를 풀 수 있다.

## Multi Layer Neural Network (Perceptron)
- input-layer - hidden-layer - output-layer
- {hidden-layer - output-layer} -> Classification
- {input-layer - hidden-layer} -> Feture Extractor

----
hidden-layer 는 이론상 무한히 넣을 수 있다.

forward propagation / back propagation

chain-rule을 사용하면 아무리 어려운 deep learning 공식을 쉽게 풀 수 있다.

공식이 복잡하더라도 잘게 쪼개면 그 공식은 풀기 쉽다.

back propagation은 caching효과를 만들어낼 수 있다.

Learning Representations by Backpropagating Error

Convolution Layer...

Many configurations....

가로x세로는 3x3이 제일 좋음

## vggnet
- small convolutions == lagre convolution

9*(49 25 9 1) + 9*9 + 9
25 * (25 + 1)

overlaping maxpooling

```python
## (heigth - 1) * (width - 1) * 3 * 3
cov_height = height - 1
cov_width = width - 1
for h in range(1, height - 1):
    for w in range(1, width - 1):
        sum = 0
        for mr in range(0, 3):
            for mc in range(0, 3):
               sum += img[(h + mr - 1) * width + (w + mc - 1)] * cov1[mr * 3 + mc]
        cov_img[(h - 1) * con_width + (w - 1)] = sum

## (cov_height - 1) * (cov_width - 1) * 3 * 3
cov2_height = cov_height - 1
cov2_width = cov_width - 1
for h in range(1, cov_height - 1):
    for w in range(1, cov_width - 1):
        sum = 0
        for mr in range(0, 3):
            for mc in range(0, 3):
               sum += cov_img[(h + mr - 1) * cov_width + (w + mc - 1)] * cov2[mr * 3 + mc]
        cov2_img[(h - 1) * cov2_width + (w - 1)] = sum
```

```python
## (heigth - 2) * (width - 2) * 5 * 5
cov_height = height - 2
cov_width = width - 2
for h in range(2, height - 2):
    for w in range(2, width - 2):
        sum = 0
        for mr in range(0, 5):
            for mc in range(0, 5):
               sum += img[(h + mr - 2) * width + (w + mc - 2)] * cov[mr * 5 + mc]
        cov_img[(h - 2) * cov_width + (w - 2)] = sum
```