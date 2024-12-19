import pandas as pd
from matplotlib.pyplot import imshow
# pip install scikit-learn
from sklearn.linear_model import LinearRegression  # 선형회귀모델
from sklearn.model_selection import train_test_split  # 훈련셋, 평가셋 나누기
from sklearn.metrics import mean_squared_error, r2_score  # MSE, R Squared
import seaborn as sns
import matplotlib.pyplot as plt



data_df = pd.read_csv("data/auto-mpg.csv", header=0, engine="python")
# print(data_df)
pd.set_option("display.max_rows", None)  # 모든 행 출력(생략 해제)
pd.set_option("display.max_columns",None)  # 모든 열 출력(생략 해제)
# print(data_df.head(6))  # 상위 6개만 출력

data_df = data_df.drop(["horsepower", "origin", "car_name"], axis=1, inplace=False)
# 연비에 영향을 미치지 않는 3개의 열 삭제 후 재저장
print(data_df.head(6))

# 종속변수 Y,독립변수 X 로 분리
Y = data_df["mpg"] # 종속변수
X = data_df.drop(["mpg"], axis=1, inplace=False)  # 독립변수

print(Y.head(6))
print(X.head(6))

# 훈련용 데이터셋과 평가용 데이터셋으로 분리
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# 선형회귀분석 모델 생성
lr = LinearRegression()

# X_train, Y_train -> 학습용 데이터
lr.fit(X_train, Y_train)  # 모델 훈련

# 평가셋(X_test)으로 예측 수행->Y_predict
Y_predict = lr.predict(X_test)

mse = mean_squared_error(Y_test, Y_predict)
r2Score = r2_score(Y_test, Y_predict)

print(f"MSE : {mse:.3f}")  # 소수 3자리까지 출력
print(f"R^2 Score : {r2Score:.3f}")  # 소수 3자리까지만 출력

intercept = lr.intercept_  # 절편 값
coef = lr.coef_  # 회귀계수 5개(독립변수의 수)

print(f"Y 절편 : {intercept:.3f}")
print(f"회귀계수 : {coef}")
print(f"회귀식 : y = {intercept:.3f}+{coef[0]:.3f}CYL+{coef[1]:.3f}DIS+{coef[2]:.3f}WEI+{coef[3]:.3f}ACC+{coef[4]:.3f}YEAR")

coef_series = pd.Series(data=coef, index=X.columns)
print(coef_series)
coef_series = coef_series.sort_values(ascending=False)  # 회귀계수의 내림차순으로 정렬
print(coef_series)
# 시각화
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(16,16))  # 3행 2열 총 6개의 그래프 공간 확보

x_features = ["model_year","accerleration","displacement","weight","cylinders"]
plot_color =["r","g","b","y","r"]

for i, feature in enumerate(x_features):
    row = int(i/3)  # i=0 1 2 3 4 -> row=0 0 0 1 1
    col = i%3  # i=0 1 2 3 4 -> col=0 1 2 0 1
    sns.regplot(x=feature, y="mpg", data=data_df, ax=axs[row][col], color=plot_color[i])

plt.show()
