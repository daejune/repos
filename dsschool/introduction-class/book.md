#
# Week 01
기본적으로 자기 전공을 살리면서 부족한 부분을 채워 나간다.<br>
data science domain이 부족하면 하기 힘들다.<br>
팀을 만들어서 부족한 부분을 그떄 그떄 채워 나간다.<br>
단점을 극복하는 수업이 아니다.<br>
kaggle: data scientist의 경진대회<br>
스폰서: 문제 + 데이터 + 상금<br>
ds 그렇게 어렵지 않다.<br>
열과 필터를 버린다!!!<br>

어린아이 (18세 미만)

1. 다죽는다.
2. 여성이면 살고 남성이면 죽는다.
3. 여성이면 사는데 Pclass가 3이면 죽는다.
4. 여성이면 사는데 Pclass가 3일 떄 S출신이면 죽는다.
5. 4조건 + S출신이 아닐 떄, SbiSp이 3이상이면 죽는다.
6. 4조건 + S출신 10세 미만 남자는 산다.
7. 가족수가 4이상 + 5조건
7. 4조건 + S출신이 아닐 떄, 가족수가 4이상이면 죽는다.
9. 3조건 + 가족수가 4이상이면 죽는다.


## Pandas & Seaborn
pandas 색인화

low_fare = train[train["Fare"] < 500]

decisiion tree
Feature - Pclass, Sex, Fare ... <br>
Label - Survived

feature_name = "Pclass"<br>
label_name = "Survived"<br>
x = train[feature_name]<br>
y = train[label_name]<br>

# scikit-learn (sklearn)
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()<br>
1) fit (train)
2) predict (test)

## Fitting
model.fit(x, y)

# Visualize

from sklearn.tree import export_graphviz<br>
dot_tree = export_graphviz(mode, feature_names = feature_name, class_names = ["Perish", "Survived"], out_file = None)<br>
graphviz.Source(dot_tree)

## Predict
x_test = test[feature_name]<br>
model.predict(x_test)

## Submit
submit = pd.read_csv("data/gender_submission.csv")<br>
submit = <br>
submit.to_csv("decision_tree.csv")<br>

## Preprocessiong

#### DS
DS는 범용적으로 사용할 수 있다.!!<br>
이는 전문 지식이 없어도 어느정도는 가능하다.<br>
모든 데이터는 예측을 할 수 있는거지 대변하는 것은 아니다. (81~85%)

datascience@dsschool.co.kr

email: https://goo.gl/EtMfQY<br>
notebook: https://goo.gl/6MS78n<br>
html: https://goo.gl/JzwhHc<br>

#
# Week 02
Python Programming 기초

python 사칙연산 및 비교연산자

변수 사용의 기초

문자열<br>
'Shayne's weight' > Syntax error!<br>
"Shayne's weight" > Ok!<br>
Indexing([]), Slicing([:]), Include (in)<br>

Error >> Must See Error Message. <br>
아래서부터 보면서 Error 위치를 파악!! >> Stack Trace를 봐라!!


## 10 minutes pandas

https://pandas.pydata.org/pandas-docs/stable/10min.html

### Preprocessing을 진행한 후에는 반드시 shape를 이용해서 크기를 확인하고
### head() method를 통해 정상적으로 변경되었는지도 같이 확인한다.!!!

python beginners notebook: https://goo.gl/AEMv3E<br>
python beginners html: https://goo.gl/YMWfY2<br>
pandas beginners notebook: https://goo.gl/9fqzN4<br>
padnas beginners html: https://goo.gl/uS8B3S<br>
titanic solution notebook: https://goo.gl/FyuqNL<br>
titanic solution html: https://goo.gl/Gq87Zm<br>


#
# Week03

## Visualize => Explore
Data Science에서는 시각화과정을 탐험하는 과정이라고 한다<br>
시각화를 통해서 데이터를 전체적으로 살펴보기 때문

Data 분석 기본자들은 distplot에서 히스토그램을 보는 것을 추천하지 않음<br>
Data는 연속적인 것 보다 이산적인 분포가 많아서 이상점이 존재함<br>
Gaussian 분포를 확인 할 수 있음



### Classification vs Regression
맞춰야하는 정답이 Categorical하면 Classification<br>
맞춰야하는 정답이 Continuous하면 Regression

Classification => DecisionTreeClassifier
Regression => DecisionTreeRegressor

## Random State
model에서 Random 값을 고정하여 결과를 동일하게 만드는 과정

## datetime parse in pandas

pd.to_datetime(train["datetime"]) <br>
train["datetime"].dt.year


## Exploratory Data Analysis (EDA)
IDEA!!! <br>
이론화 되어 있는게 아님!! => 특정 분야의 전문가들이 있으면 그 Gap을 줄일 수 있다.<br>
이론보다 경험이 더 중요함!!