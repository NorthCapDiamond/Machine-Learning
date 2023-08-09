import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


"""

PART3
Star this repository on GitHub if you like it (or use it))))

docs: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
docs: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split


TASK:

Используя весь датасет из предыдущего задания как обучающий, 
выполните предсказания для следующего набора данных. 
Для успешного выполнения задания необходимо, чтобы ваш результат
превысил пороговое значение, равное 0.98. В качестве метрики используется 
r2_score.

"""
df_train = pd.read_csv(
    "https://courses.openedu.ru/assets/courseware/v1/fed9823d73d2b53f5591d98b87c20b8a/asset-v1:ITMOUniversity+MLDATAN+spring_2023_ITMO_bac+type@asset+block/fish_train.csv")
df_test = pd.read_csv(
    "https://courses.openedu.ru/assets/courseware/v1/33b24e589714e963ea7081912668c93d/asset-v1:ITMOUniversity+MLDATAN+spring_2023_ITMO_bac+type@asset+block/fish_reserved.csv")
print("Train", df_train)
print("Test", df_test)


# Так как dataset не очень большой, мы не будем использовать PCA... Кроме того,
# возведение в куб R2_score значительно вырос. Попробуем это использовать.
# Категориальные признаки тоже снизили точность модели, поэтому их не будем прибавлять
X_train_species = df_train.Species
X_test_species = df_test.Species
new_df_train = df_train.drop(["Species"], axis=1)
new_df_test = df_test.drop(["Species"], axis=1)
X_train = new_df_train.drop(["Weight"], axis=1)
X_test = new_df_test
y_train = new_df_train.Weight

X_train_in_cube = X_train**3
X_test_in_cube = X_test**3

linear_regression = LinearRegression().fit(X_train_in_cube, y_train)
demo_predict = linear_regression.predict(X_train_in_cube)
print("Probably R2 score is", r2_score(y_train, demo_predict))
y_pred = linear_regression.predict(X_test_in_cube)
print("Our predictions : \n", y_pred.tolist())
print("OpenEdu returns: Your result is 0.983460168371")
