import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

"""

PART2
Star this repository on GitHub if you like it (or use it))))

docs: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

TASK:
Представленный набор данных — Набор данных получен в 
результате переписи населения  года и содержит 
информацию о некотором количестве людей, проживающих в США.
 Задача состоит в том, чтобы предсказать, зарабатывает человек более 
к в год или нет.

"""

# считаем данные
data_frame = pd.read_csv(
    "https://courses.openedu.ru/assets/courseware/v1/3b5e697be14f493785e3d21577f9fcb3/asset-v1:ITMOUniversity+MLDATAN+spring_2023_ITMO_bac+type@asset+block/adult_data_train.csv", na_values=["?"])
print(data_frame)

# Избавьтесь от признаков education и marital-status.
# Удалите соответствующие колонки из набора данных.
# Определите количество числовых и нечисловых признаков.
df = data_frame.drop(["education", "marital-status"], axis=1)
extra_df = df.copy()
print(df.dtypes)
object_typed_data = sum(df.dtypes == object)
print("Type = object %d times" % object_typed_data)
int_typed_data = len(df.select_dtypes(np.number).columns)
print("Type is int %d times" % int_typed_data)

df = df.drop(df.select_dtypes(include=object), axis=1)


# Постройте гистограмму распределения объектов по классам.
# Постройте гистограмму распределения объектов по классам.
# Вычислите долю объектов класса .

df["label"].value_counts().plot.pie(autopct="%.3f")
plt.title("Histogram")
plt.show()
X = df.drop("label", axis=1)
y = df.label


# 1. Построение базовой модели
# Первое приближение.

# Change this to yours
random_state = 41
test_size = 0.2

# Отберите из набора данных только числовые признаки.
# При помощи train_test_split() разбейте набор данных на
# обучающую и тестовую выборки с параметрами test_size=0.2,
# random_state=41. Используйте стратификацию по колонке label.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=df.label)


# Вычислите выборочное среднее колонки fnlwgt тренировочного набора данных.
print("fnlwgt mean is", round(X_train.fnlwgt.mean(), 3))


# Обучите модель KNeighborsClassifier() с параметрами по умолчанию на тренировочных данных.
# Оцените на тестовых данных.
# Вычислите f1_score() для тестового набора данных.

kneighbors = KNeighborsClassifier().fit(X_train, y_train)
y_pred = kneighbors.predict(X_test)

print("f1_score is", f1_score(y_test, y_pred))

# Обучите преобразование MinMaxScaler() на
# тренировочном наборе данных и примените его для тренировочных и тестовых данных.

# Вычислите выборочное среднее колонки fnlwgt полученного тренировочного набора данных.

scaler = MinMaxScaler().fit(X_train)

X_train = pd.DataFrame(scaler.transform(
    X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
print("Mean fnlwgt is", X_train.fnlwgt.mean())

# Обучите модель KNeighborsClassifier() с параметрами по умолчанию на тренировочных данных.
# Оцените на тестовых данных.
# Вычислите f1_score() для тестового набора данных.

new_kneighbors = KNeighborsClassifier().fit(X_train, y_train)
y_new_pred = new_kneighbors.predict(X_test)

print("f1_score is", f1_score(y_test, y_new_pred))


# 2. Работа с нечисловыми признаками

# Визуализация

# Верните в рассмотрение нечисловые признаки.
# Используйте исходный датасет (без колонок education и marital-status).
# Постройте гистограммы, иллюстрирующие частоту того или иного значения
# по каждому нечисловому признаку, например,
# при помощи sns.barplot().


data_frame = data_frame.drop(["education", "marital-status"], axis=1)
# Гистограмма частот уникальных значений в рамках некоторого признака.

non_int_df = data_frame.drop(df.select_dtypes(np.number).columns, axis=1)

columns_non_int = non_int_df.columns
columns_amount = len(columns_non_int)

sidea = 4
sideb = columns_amount // sidea
other = columns_amount % sidea
if(other > 0):
    sideb += 1


fig, axs = plt.subplots(sideb, sidea, figsize=(
    sidea * 4, sideb * 3), squeeze=False)
for r in range(sideb):
    for c in range(sidea):
        col = r * sidea + c
        if col < columns_amount:
            axs[r][c].bar(non_int_df[columns_non_int[col]].value_counts(
            ).index, non_int_df[columns_non_int[col]].value_counts().to_numpy())
            axs[r][c].set_title(columns_non_int[col])
plt.tight_layout()
plt.show()


# Определите число строк исходного набора данных
# (без колонок education и marital-status), в которых присутствует хотя бы одно пропущенное значение.

print("Amount of bad rows is",
      data_frame.shape[0] - data_frame.dropna().shape[0])
# Удалите строки, содеражащие пропуски.
# Произведите one-hot кодировние нечисловых признаков, например,
# с помощью pd.get_dummies(drop_first=True).

data_frame = data_frame.dropna()

one_hot_coding_data = pd.get_dummies(data_frame, drop_first=True)
print("after one hot\n", one_hot_coding_data)

print("Amount of parameters is", one_hot_coding_data.shape[1] - 1)

# Используя полученный набор данных,
# обучите модель классификации аналогично тому,
# как это было сделано для базовой модели (split, scaling).

# Произведите предсказания для тестовых данных. Вычислите f1_score() модели.
X_last = one_hot_coding_data.drop(["label"], axis=1)
y_last = one_hot_coding_data.label
X_train_last, X_test_last, y_train_last, y_test_last = train_test_split(
    X_last, y_last, test_size=test_size, random_state=random_state, stratify=data_frame.label)

scala = MinMaxScaler().fit(X_train_last)


X_train_last = scala.transform(X_train_last)
X_test_last = scala.transform(X_test_last)


classifier = KNeighborsClassifier().fit(X_train_last, y_train_last)
y_pred_last = classifier.predict(X_test_last)

print("f1_score is", f1_score(y_test_last, y_pred_last))


# Заполнение пропущенных значений

# Используя исходный датасет (без колонок education и marital-status),
# заполните пропуски самым часто встречающимся значением в рамках столбца.
# Далее -- аналогично предыдущему случаю: one-hot, split, scaling,
# обучение и оценка. Вычислите f1_score() модели.


print(extra_df)


# Сначала выбросим те строки, в которых пропущена целевая переменная

drop_idx = extra_df[extra_df.label.isna()].index
extra_df = extra_df.drop(index=drop_idx, axis=1)
print(extra_df.isna().sum())


cat_mask = (extra_df.dtypes.values == object)
print("Cat Mask : ", cat_mask)

X_cat = extra_df[extra_df.columns[cat_mask]]
X_noncat = extra_df[extra_df.columns[~cat_mask]]

mis_replacer = SimpleImputer(strategy="most_frequent")
X_noncat = pd.DataFrame(mis_replacer.fit_transform(
    X_noncat), columns=X_noncat.columns)

cat_replacer = SimpleImputer(strategy='most_frequent', fill_value='')
X_cat = pd.DataFrame(cat_replacer.fit_transform(X_cat), columns=X_cat.columns)

X_clean = pd.concat([X_cat, X_noncat], axis=1)
print(X_clean.isna().sum())


y_clean = X_clean.label
X_clean = X_clean.drop(["label"], axis=1)
X_clean = pd.get_dummies(X_clean, drop_first=True)

X_train_clean, X_test_clean, y_train_clean, y_test_clean = train_test_split(
    X_clean, y_clean, test_size=test_size, random_state=random_state, stratify=extra_df.label)

min_max_scaler = MinMaxScaler().fit(X_train_clean)


X_train_clean = min_max_scaler.transform(X_train_clean)
X_test_clean = min_max_scaler.transform(X_test_clean)


model_class = KNeighborsClassifier().fit(X_train_clean, y_train_clean)

print("f1_score is", f1_score(y_test_clean, model_class.predict(X_test_clean)))
