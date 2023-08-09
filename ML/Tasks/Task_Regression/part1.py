import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

"""

PART1
Star this repository on GitHub if you like it (or use it))))

docs: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

"""

# Читаем исходные данные
df = pd.read_csv("data1.csv")
print(df)

# Обучите модель линейной регрессии для прогнозирования и введите указанные параметры.

# Определите выборочное среднее X_mean:
x_mean = df.X.mean()
print("Mean X is", x_mean)

# Определите выборочное среднее Y_mean:
y_mean = df.Y.mean()
print("Mean Y is", y_mean)

# Обучим модель:
X = np.expand_dims(df.X, axis=1)  # or X = df[["X"]] can be used instead
Y = df.Y
linear_model = LinearRegression().fit(X, Y)

# Найдите коэффициент Theta1:
print("Theta1 is", linear_model.coef_[0])


# Найдите коэффициент Theta0:
print("Theta0 is", linear_model.intercept_)


# Оцените точность модели, вычислив R2 статистику:
y = linear_model.predict(X)
print("R2 score is", r2_score(Y, y))
