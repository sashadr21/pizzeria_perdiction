import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, SGDRegressor, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error, mean_squared_error, mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,3)

# Подготавливаем и разделяем данные
used_params =['T','U','Ff','P','CountryId','unitId','day_of_week', 'is_pandepic','is_weekend']
df = pd.read_csv('data.csv')
X = df[used_params]
y = df['NumberOfOrders']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Обучаем модели
# Линейная регрессия
model = LinearRegression()
model.fit(X_train,y_train)
predict = model.predict(X_test)
rmse = np.sqrt(mean_squared_log_error(predict,y_test))
mse = mean_squared_error(predict,y_test)
mae = mean_absolute_error(predict,y_test)
print("Linear model")
print(f"rmse: {rmse}")
print(f"mse: {mse}")
print(f"mae: {mae}")


# Просто берем среднее значение заказов и проверяем эту модель
mean_y = [np.mean(y) for i in range(len(predict))]
rmse = np.sqrt(mean_squared_log_error(mean_y,y_test))
mse = mean_squared_error(mean_y,y_test)
mae = mean_absolute_error(mean_y,y_test)
print("Mean Model")
print(f"rmse: {rmse}")
print(f"mse: {mse}")
print(f"mae: {mae}")

# Модель на основе Случайного леса
model = RandomForestRegressor()
model.fit(X_train,y_train)
predict = model.predict(X_test)
rmse = np.sqrt(mean_squared_log_error(predict,y_test))
mse = mean_squared_error(predict,y_test)
mae = mean_absolute_error(predict,y_test)
print("Random Forest model")
print(f"rmse: {rmse}")
print(f"mse: {mse}")
print(f"mae: {mae}")

