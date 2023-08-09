import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

"""

PART1
Star this repository on GitHub if you like it (or use it))))

docs: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

"""
df = pd.read_csv("https://courses.openedu.ru/assets/courseware/v1/537f50b9da15bbb433272426febf974a/asset-v1:ITMOUniversity+MLDATAN+spring_2023_ITMO_bac+type@asset+block/34_25.csv", header=None)

# Введите координату первого объекта относительно первой главной компоненты. :

pca_new_df = PCA(n_components=2, svd_solver='full').fit(df)

variance_for_two = np.sum(pca_new_df.explained_variance_ratio_)

pca_new_df = pca_new_df.transform(df)

coordinates = pca_new_df[0]
print()
print("The first coordinate is " + str(coordinates[0]))

# Введите координату первого объекта относительно второй главной компоненты.
print("The second coordinate is " + str(coordinates[1]))

# Введите долю объясненной дисперсии при использовании первых двух главных компонент.
print("Explained variance is " + str(variance_for_two))

# Какое минимальное количество главных компонент необходимо использовать, чтобы доля объясненной дисперсии превышала 0.85
expected_variance = 0.85
current_variance = 0
current_number_of_components = 2

pca_n = PCA(n_components=current_number_of_components,
            svd_solver='full').fit(df)


# Stupid way to do this task, but no need to understand ML

if(variance_for_two > expected_variance):
    print("Amount of components is 1 | easy way")
else:
    while(current_variance <= expected_variance):
        current_number_of_components += 1
        pca_n = PCA(n_components=current_number_of_components).fit(df)
        current_variance = np.sum(pca_n.explained_variance_ratio_)
    print("Amount of components is " +
          str(current_number_of_components) + " | easy way")

# Better way:

pca_var = PCA().fit(df)

# np.cumsum( A ) возвращает совокупную сумму A запуск в начале первого измерения массива в A чей размер не равняется 1.
# np.argmax() , позволяет найти индекс максимального значения в массиве

expected_variance_cumsum = np.cumsum(pca_var.explained_variance_ratio_)
print("Amount of components is", np.argmax(
    expected_variance_cumsum > expected_variance) + 1, "| fast way")


plt.scatter(pca_new_df[:, 0], pca_new_df[:, 1])
plt.show()
