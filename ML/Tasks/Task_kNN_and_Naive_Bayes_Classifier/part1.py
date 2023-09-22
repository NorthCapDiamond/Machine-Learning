import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors

"""

PART1
Star this repository on GitHub if you like it (or use it))))

docs: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

"""

# Считаем данные:
data = pd.read_csv("data1.csv")
df = data.copy()
print(data)

# Введите расстояние от нового объекта с координатами (52, 18)
# до ближайшего соседа, используя евклидову метрику.

# Change this to yours
x_wanted = 52
# change this to yours
y_wanted = 18
# change this to yours 
k = 3


# Используем библиотеку для решения:
x_y_df = pd.DataFrame([[x_wanted, y_wanted]], columns=["X", "Y"])
neigh = NearestNeighbors(n_neighbors=k).fit(data.drop(["Class", "id"], axis=1))
nearest_neighbor_array = neigh.kneighbors(x_y_df)
print("Nearest neighbor is", nearest_neighbor_array[1][0][0],
      "Distance is", nearest_neighbor_array[0][0][0], "|Method used sklearn lib")

# Попробуем сделать это сами))
# Мы знаем, что расстояние Минковского --
# это общий вид евклидова расстояния и расстояния городских кварталов.
# Значит, мы можем использовать эту формулу с разными параметрами!


def distancer(df0, df1, p=2):
    return (abs(df1["X"] - df0.X)**p + abs(df1["Y"] - df0.Y)**p)**(1 / p)


x_y_dict = dict(X=x_wanted, Y=y_wanted)

data["Euclid"] = distancer(data, x_y_dict)
print("Without using sklearn: \n", data.sort_values(by="Euclid").head(1))


# Введите идентификатры трех ближайших точек при к = 3  для евклидовой метрики.
# sklearn
print("k nearest neighbors are",
      nearest_neighbor_array[1], "|Method used sklearn lib")


# без sklearn
print("Without using sklearn: 3 nearest neighbors are\n",
      data.sort_values(by="Euclid").head(k))


# Введите класс для нового объекта при k = 3 и евклидовой метрике.
# sklearn
X = df.drop(["Class", "id"], axis=1)
y = df.Class
kneighbors = KNeighborsClassifier(n_neighbors=k).fit(X, y)
print("Class of our object is ", kneighbors.predict(
    x_y_df), "|Method used sklearn lib")
print("Without using sklearn: Class of our object is",
      1 if data.sort_values(by="Euclid").head(3).Class.sum() > 1 else 0)


# Введите расстояние от нового объекта до ближайшего соседа, используя метрику городских кварталов (Манхеттенское расстояние).
# sklearn
manhattan = NearestNeighbors(n_neighbors=k, p=1).fit(
    df.drop(["Class", "id"], axis=1))
nearest_neighbor_array_manhattan = manhattan.kneighbors(x_y_df)
print("Distance from new object to the nearest is",
      nearest_neighbor_array_manhattan[0][0][0], "|Method used sklearn lib")

# без sklearn
data["Manhattan"] = distancer(data, x_y_dict, p=1)
print("Without using sklearn: Distance from new object to the nearest is\n",
      data.sort_values("Manhattan").head(1))

# Введите идентификатры трех ближайших точек для метрики городских кварталов.
# sklearn:
print("k nearest neighbors are",
      nearest_neighbor_array_manhattan[1], "|Method used sklearn lib")

# без sklearn:
print("Without using sklearn: k nearest neighbors are\n",
      data.sort_values("Manhattan").head(k))


# Введите класс для нового объекта при k = 3 и метрике городских кварталов.

kneighbors_manhattan = KNeighborsClassifier(n_neighbors=k, p=1).fit(X, y)
print("Class of our object is ", kneighbors_manhattan.predict(
    x_y_df), "|Method used sklearn lib")
print("Without using sklearn: Class of our object is",
      1 if data.sort_values(by="Manhattan").head(k).Class.sum() > 1 else 0)
