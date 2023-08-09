import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

"""

PART2
Star this repository on GitHub if you like it (or use it))))

docs: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
docs: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split


TASK:

Представленный набор данных — это набор данных о семи различных типах рыб, 
продаваемых в некоторой рыбной лавке. Цель заключается в том,
чтобы предсказать массу рыбы по каким-то косвенным признакам, 
известным о рыбе. Сами признаки, быть может, нужно синтезировать из тех, 
что известны.


ВАЖНО: Задания с подробными пояснениями представлены в блокноте. 
Выполняя задания в блокноте, следует вводить полученные ответы в 
соответствующие поля ввода ниже. Рекомендуем вводить ответы поэтапно, 
для этого специально предусмотрено достаточное количество попыток.

При помощи train_test_split() разбейте набор данных на обучающую и 
тестовую выборки с параметрами test_size=0.2, random_state=11. 
Используйте стратификацию по колонке Species.
Стратификация позволит сохранить доли представленных объектов 
(по представителям типов рыб) в тренировочной и тестовой выборках.
"""


# Считаем исходные данные
df = pd.read_csv("https://courses.openedu.ru/assets/courseware/v1/fed9823d73d2b53f5591d98b87c20b8a/asset-v1:ITMOUniversity+MLDATAN+spring_2023_ITMO_bac+type@asset+block/fish_train.csv")
print(df)

random_state_value = 11
test_size_value = 0.2

# Вычислите выборочное среднее колонки Width полученной тренировочной выборки.
new_df = df.drop(["Species"], axis=1)
X = new_df.drop(["Weight"], axis=1)
y = new_df.Weight

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size_value, random_state=random_state_value, stratify=df.Species)
print("Mean Width is", X_train.mean())

# 1. Построение базовой модели

# Избавьтесь от категориальных признаков и обучите модель линейной регрессии
#(LinearRegression()) на тренировочном наборе данных.
# Выполните предсказания для тестового набора данных.
# Оцените модель при помощи метрики r2_score().

linear_regression = LinearRegression().fit(X_train, y_train)
y_pred = linear_regression.predict(X_test)
print(r2_score(y_test, y_pred))

# 2. Добавление предварительной обработки признаков.

# Использование PCA.

# Перечислите через запятую и пробел тройку отбрасываемых наиболее коррелированных признаков.
new_df = df.drop(["Species"], axis=1)
print("Correlation matrix: \n", new_df.corr())

# Создадим Heatmap корреляции :
data_plot = sns.heatmap(new_df.corr(), annot=True, fmt=".1f")
plt.show()

# My answer is :
correlated_features = ["Length1", "Length2", "Length3"]


# Обучите модель PCA для трех наиболее коррелированных признаков.
# Введите долю объясненной дисперсии при использовании только первой главной компоненты.
pca_model = PCA(svd_solver="full", n_components=1).fit(
    X_train[correlated_features])
print("Explained variance is", pca_model.explained_variance_ratio_[0])

# Замените тройку наиболее коррелированных признаков на полученный признак Lengths,
# значения которого совпадают со счетами первой главной компоненты.
# Обучите модель линейной регрессии. Введите r2_score() полученной модели.
X_train["Lengths"] = pca_model.transform(X_train[correlated_features])
X_test["Lengths"] = pca_model.transform(X_test[correlated_features])
X_train = X_train.drop(correlated_features, axis=1)
X_test = X_test.drop(correlated_features, axis=1)

linear_regression_model_after_pca = LinearRegression().fit(X_train, y_train)
y_pred_after_pca = linear_regression_model_after_pca.predict(X_test)
print("R2 score is", r2_score(y_test, y_pred_after_pca))

# Используя полученный на предыдущем этапе набор данных,
# возведите в куб значения признаков Width, Height, Lengths.
# Введите выборочное среднее колонки Width тренировочного набора данных
# после возведения в куб.
X_train = X_train**3
X_test = X_test**3

print("Mean Width after X_train**3 is", X_train.Width.mean())

# Построим график зависимости Weight от Width
sns.scatterplot(x=X_train.Width, y=y_train, hue=df.Species)
plt.show()
# В моем случае ответ: 2. Вам нужно выбрать свой


# Обучите модель линейнной регрессии. Введите r2_score() полученной модели.
linear_regression_model_after_cube = LinearRegression().fit(X_train, y_train)
y_pred_after_cube = linear_regression_model_after_cube.predict(X_test)
print(r2_score(y_test, y_pred_after_cube))


# Добавление категориальных признаков.

# Добавьте к набору данных, полученному на предыдущем этапе,
# ранее исключенные категориальные признаки, предварительно
# произведя one-hot кодирование при помощи pd.get_dummies().
# Обучите модель регрессии. Введите r2_score() полученной модели.


X_cat = df.drop(["Weight"], axis=1)
y_cat = df.Weight


X_cat_train, X_cat_test, y_cat_train, y_cat_test = train_test_split(
    X_cat, y_cat, test_size=test_size_value, random_state=random_state_value, stratify=df.Species)
X_cat_train_species = X_cat_train.Species
X_cat_test_species = X_cat_test.Species


X_train_in_cube_one_hot_coding = X_train.join(
    pd.get_dummies(X_cat_train_species))
X_test_in_cube_one_hot_coding = X_test.join(pd.get_dummies(X_cat_test_species))

linear_regression_model_after_one_hot_coding = LinearRegression().fit(
    X_train_in_cube_one_hot_coding, y_cat_train)
y_pred_after_one_hot_coding = linear_regression_model_after_one_hot_coding.predict(
    X_test_in_cube_one_hot_coding)

print("R2 score after get_dummies is", r2_score(
    y_cat_test, y_pred_after_one_hot_coding))

# Закодируйте категориальные признаки при помощи pd.get_dummies(drop_first=True).
X_train_in_cube_one_hot_coding_drop_first = X_train.join(
    pd.get_dummies(X_cat_train_species, drop_first=True))
X_test_in_cube_one_hot_coding_drop_first = X_test.join(
    pd.get_dummies(X_cat_test_species, drop_first=True))

linear_regression_model_after_one_hot_coding_drop_first = LinearRegression(
).fit(X_train_in_cube_one_hot_coding_drop_first, y_cat_train)
y_pred_after_one_hot_coding_drop_first = linear_regression_model_after_one_hot_coding_drop_first.predict(
    X_test_in_cube_one_hot_coding_drop_first)

print("R2 score after get_dummies and drop_first is", r2_score(
    y_cat_test, y_pred_after_one_hot_coding_drop_first))
