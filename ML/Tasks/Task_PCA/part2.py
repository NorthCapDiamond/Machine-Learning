import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

"""

PART2
Star this repository on GitHub if you like it (or use it))))

docs: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

"""
score_matrix = pd.read_csv(
    "https://courses.openedu.ru/assets/courseware/v1/ee3345b3500b7571a589d3caa49ec743/asset-v1:ITMOUniversity+MLDATAN+spring_2023_ITMO_bac+type@asset+block/X_reduced_417.csv", header=None, delimiter=';')
weight_matrix = pd.read_csv(
    "https://courses.openedu.ru/assets/courseware/v1/ad22c96aa75c298f81ec21de6f334865/asset-v1:ITMOUniversity+MLDATAN+spring_2023_ITMO_bac+type@asset+block/X_loadings_417.csv", header=None, delimiter=';')
''' Восстановление признаков по главным компонентам

Пусть Ф - матрица, столбцы которой отвечают координатам 
нормированных собственных векторов – векторов весов.

Тогда: 

Z_[n x p] = F_[n x p] * Ф_[p x p]

и матрица счетов имеет размерность [𝑛 × 𝑝]. Тогда старые центрированные
координаты восстанавливаются без всяких потерь домножением всего равенства 
справа на ФT , откуда:

Z*ФT = F*Ф*ФT = F*E = F 

так как, в силу ортогональности Ф,
Ф*ФT = E – единичная матрица. 

Однако обычно количество ГК, которые мы оставляем, 
меньше, чем размерность исходного пространства. 
Оставив их 𝑘 штук, получим матрицу Φ размера [p x k] и вектор счётов

Z_[n x k] = F_[n x p] * Ф_[p x k]

размера [n x k]. Домножим справа на  ФT и заметим, что теперь, хотя произведение
Ф*ФT и будет иметь размер [p x p],оно не
будет единичной матрицей. Поэтому

𝑍 * ФT = F * Ф * ФT = new_F,
new_F - матрица с координатами приближенно восстановленных центрированных исходных объектов.


Это значит, что нужно прибавить среднее: 

new_F' = Z * ФT + X_mean

В этой задаче достаточно взять формулу: 
new_F = Z * ФT

'''

print("Weight matrix : ")
print(weight_matrix)
print("Score matrix, transposed : ")
print(score_matrix.T)
new_F = np.dot(score_matrix, weight_matrix.T)
print("Resulting Matrix : ")
print(new_F)
plt.imshow(new_F, cmap="Blues")
plt.show()
