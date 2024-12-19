import pandas as pd
from matplotlib.pyplot import imshow
# pip install scikit-learn
from sklearn.linear_model import LinearRegression  # 선형회귀모델
from sklearn.model_selection import train_test_split  # 훈련셋, 평가셋 나누기
from sklearn.metrics import mean_squared_error, r2_score  # MSE, R Squared

data_df = pd.read_csv("data/auto-mpg.csv", header=0, engine="python")
# print(data_df)
pd.set_option("display.max_rows", None)  # 모든 행 출력(생략 해제)
pd.set_option("display.max_columns",None)  # 모든 열 출력(생략 해제)
# print(data_df.head(6))  # 상위 6개만 출력

data_df = data_df.drop(["horsepower", "origin", "car_name"], axis=1, inplace=False)
# 연비에 영향을 미치지 않는 3개의 열 삭제 후 재저장
print(data_df.head(6))




