import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import category_encoders as ce
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

"""

PART3
Star this repository on GitHub if you like it (or use it))))

docs: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

TASK:
    Используя весь датасет из предыдущего задания как обучающий, 
    выполните предсказания для следующего набора данных. Для успешного
     выполнения задания необходимо, чтобы ваш результат превысил пороговое 
     значение, равное 0.645. В качестве метрики используется f1_score.

"""

# Считаем данные
df_test = pd.read_csv(
    "https://courses.openedu.ru/assets/courseware/v1/81d9cf5671cf3576fd7776f5165d9cc5/asset-v1:ITMOUniversity+MLDATAN+spring_2023_ITMO_bac+type@asset+block/adult_data_reserved.csv", na_values=["?"])
data = pd.read_csv(
    "https://courses.openedu.ru/assets/courseware/v1/3b5e697be14f493785e3d21577f9fcb3/asset-v1:ITMOUniversity+MLDATAN+spring_2023_ITMO_bac+type@asset+block/adult_data_train.csv", na_values=["?"])
test_size = 0.2
random_state = 11


# Для начала поработаем с train data
print("\nTrain data shape is", data.shape)
print("\nNaN values:")
print(data.isna().sum())


# Сначала выбросим те строки, в которых пропущена целевая переменная
drop_idx = data[data.label.isna()].index
data = data.drop(index=drop_idx, axis=1)
df_train = data.copy()
X = data.drop(["label"], axis=1)
y = data.label


# Разделим на категориальные и нет
cat_mask = X.dtypes.values == object
X_cat = X[X.columns[cat_mask]]
X_noncat = X[X.columns[~cat_mask]]

# Заполним пропуски:

mis_replacer = SimpleImputer(strategy="most_frequent")
cat_replacer = SimpleImputer(strategy='constant', fill_value='')

X_noncat = pd.DataFrame(mis_replacer.fit_transform(
    X_noncat), columns=X_noncat.columns)
X_cat = pd.DataFrame(cat_replacer.fit_transform(X_cat), columns=X_cat.columns)


# one-hot довольно слабый способ борьбы с количеством категориальных признаков, поэтому используем ce:
te = ce.target_encoder.TargetEncoder(cols=X_cat.columns)
X_cat_ce = te.fit_transform(X_cat, y)
X_prepared = pd.concat([X_noncat, X_cat_ce], axis=1)

print("\nClean data:")
print(X_prepared)
print("\nCheck NaN values:")
print(X_prepared.isna().sum())


# Разделяем данные
X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
    X_prepared, y, test_size=test_size, random_state=random_state)
scaler_t = MinMaxScaler().fit(X_train_t)

# Масштабируем числовые признаки
X_train_t = scaler_t.transform(X_train_t)
X_test_t = scaler_t.transform(X_test_t)

# MLP круто работает, поэтому берем его)))
classifier = MLPClassifier().fit(X_train_t, y_train_t)
print("MLP f1_score is", f1_score(y_test_t, classifier.predict(X_test_t)))

"""
# А есть ли шанс у KNN??
parameter_grid = {
    'n_neighbors': np.arange(2, 20, 1),
    'p': [1, 2, 3]
}
grid_searcher = GridSearchCV(estimator=KNeighborsClassifier(),
                             param_grid=parameter_grid,
                             cv=5,
                             scoring='accuracy',
                             n_jobs=-1
                             )
grid_searcher.fit(X_train_t, y_train_t)
best_params = grid_searcher.best_params_
print("KNN wants:\n")
print("KNN Best params:", best_params)
print("KNN best score:", grid_searcher.best_score_)

neigh_classifier = KNeighborsClassifier(
    n_neighbors=best_params["n_neighbors"], p=best_params["p"]).fit(X_train_t, y_train_t)
print("KNN f1 score is", f1_score(y_test_t, neigh_classifier.predict(X_test_t)))
"""


X_train = df_train.drop(["label"], axis=1)
X_test = df_test


X_train_mask = X_train.dtypes == object
X_test_mask = X_test.dtypes == object


new_te = ce.target_encoder.TargetEncoder(cols=X_train.columns)
X_cat_train = X_train[X_train.columns[X_train_mask]]
X_noncat_train = X_train[X_train.columns[~X_train_mask]]
X_cat_test = X_test[X_test.columns[X_test_mask]]
X_noncat_test = X_test[X_test.columns[~X_test_mask]]

X_cat_train = pd.DataFrame(cat_replacer.fit_transform(
    X_cat_train), columns=X_cat_train.columns)
X_cat_test = pd.DataFrame(cat_replacer.transform(
    X_cat_test), columns=X_cat_test.columns)
X_noncat_train = pd.DataFrame(mis_replacer.fit_transform(
    X_noncat_train), columns=X_noncat_train.columns)
X_noncat_test = pd.DataFrame(mis_replacer.transform(
    X_noncat_test), columns=X_noncat_test.columns)

new_te = ce.target_encoder.TargetEncoder(
    cols=X_cat_train.columns).fit(X_cat_train, y)
X_cat_train = new_te.transform(X_cat_train)
X_cat_test = new_te.transform(X_cat_test)

X_train_ce = pd.concat([X_cat_train, X_noncat_train], axis=1)
X_test_ce = pd.concat([X_cat_test, X_noncat_test], axis=1)


scaler = MinMaxScaler().fit(X_train_ce)
X_train = scaler.transform(X_train_ce)
X_test = scaler.transform(X_test_ce)

mlp_class = MLPClassifier().fit(X_train, y)
print("MLP TRY:\n", mlp_class.predict(X_test).tolist())


"""BONUS
# А Что там KNN????
knn_class = KNeighborsClassifier(
    n_neighbors=best_params["n_neighbors"], p=best_params["p"]).fit(X_train, y)
print("KNN TRY:\n", knn_class.predict(X_test).tolist())
# Очень близко, но нет.......
"""
