#!/usr/bin/env python
# coding: utf-8

# **Обобщённые линейные модели**
# 
# Обобщённые линейные модели - семейство регрессионных моделей вида
# $$
# \hat{Y} = g^{-1}(f(X)\cdot\beta + \varepsilon)
# $$
# 
# Несмотря на то, что $f$ и $g$ могут быть нелинейными функциями и $Y$ в результате может весьма нелинейно зависеть от $Х$, модель все равно остается *линейной* относительно параметров $\beta$.
# 
# В этой модели зависимая переменная принадлежит экспоненциальному семейству, а монотонная дифференцируемая функция $g$ называется функцией связи. $f(X)$ - некоторое преобразование над признаками объектов.  Плотность распределения в экспоненциальном семействе определяется соотношением
# 
# $$
# f(y,\theta, \varphi) = exp\left(\frac{y\cdot\theta-b(\theta)}{\alpha(\varphi)}+c(y,\varphi)\right)
# $$
# 
# Примеры распределений из экспоненциального семейства: нормальное, гамма, бета, Бернулли, Дирихле и многие другие.
# 
# **Мотивация**
# 
# Простейшая линейная модель - линейная регрессия. Более сложные регрессионные модели применяются в тех случаях, когда диагностика показала
# несостоятельность простых регрессионных моделей. Вот несколько наиболее распространенных
# случаев, когда применяются сложные регрессионные модели:
# 1. Нелинейность модели
# 2. Ненормальное распределение остатков
# 3. Неодинаковое распределение остатков
# 4. Зависимость остатков
# 
# **Пример -  линейная регрессия**
# 
# Модель линейной регрессии представляет собой линейную функцию от аргументов, где каждый аргумент - числовое значение признака из признакового описания объекта.
# 
# $$
# \hat{y} = w_0 + w_1x_1 + \ldots + w_1x_n = \sum_{i=0}^{n}w_ix_i=\vec{w}^{T}\cdot \vec{X}
# $$
# 
# 
# мы умножаем каждый признак $x_i$ на соответствующий ему вес $w_i$
# 
# В реальном мире на каждое наблюдение накладывается ошибки (шумовая компонента) $\varepsilon$, тогда для каждого индивидуального набдюдения $y_i$ получаем следующую модель:
# 
# $$
# y_i = \vec{w}^{T}\cdot \vec{X}+ \varepsilon
# $$
# 
# В линейной регрессии на ошибки накладываются ограничения
# 
# 1. матожидание случайных ошибок равно нулю: $\forall i:E[\varepsilon_i]=0$;
# 2. дисперсия случайных ошибок одинакова и конечна, это свойство называется гомоскедастичностью: $\forall  i: Var(\varepsilon_i)=\sigma^2< \inf$;
# 3. случайные ошибки не скоррелированы: $\forall i\neq j: Cov(\varepsilon_i, \varepsilon_j)=0$ .
# 
# *Обучить модель* - значит, найти веса $w_0,\ldots,w_n$ используя матрицу объекты-признаки $X$ и известные значения целевой переменной $y$. Мы хотим чтобы модель максимально хорошо определяла, какое значение целевой переменной $y_i$ соответствует объекту $X_i$. Формально понятие "хорошести" определяет функция потерь.
# 
# *Функция потерь* - количественное выражение того, сколько мы теряем в случе неправильного решения. Для линейной регрессии функция потерь (Loss functioon) представляет собой среднее значение квадратов отклонения прогнозных и реальных значений (mean squared error).
# 
# $$
# L(X,y,w) = \frac{1}{2n} \sum_{i=0}^{n}(y - \hat{y})^2 = \frac{1}{2n} \sum_{i=0}^{n}(y - w^{T}\cdot X)^2
# $$
# 
# Задача обучения - найти такие веса $w$, при которых функция потерь $L$ принимает минимальное значение на обучающей выборке.
# 
# **Важное примечание:** В реальных задачах многими условиями можно пренебречь - например, выбирать другие функции потерь (сумма модулей ). Ошибки $\varepsilon$ могут не подчиняться условиям (1,2,3) - в этом случае наши оценки весов $\hat{w}$ перестанут быть лучшими среди линейных и несмещенных.
# 
# Построим простую линейную регрессию на данных "House sale prices for King County"

# In[ ]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
warnings.filterwarnings('ignore') # отключаем сообщения об ошибках
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
get_ipython().run_line_magic('matplotli', '')


# In[ ]:


raw_data = pd.read_csv("../input/kc_house_data.csv")
print((raw_data.columns))
raw_data.head(3)


# Удаляем некоторые столбцы - год постройки и координаты. Price назначим целевой переменной. Удалим поле id (вопрос - почему?)

# In[ ]:


# удаляем неинтересные колонки
drop_cols = ['zipcode','lat','long', 'yr_built', 'yr_renovated', 'date', 'id']
data = raw_data.drop(drop_cols, axis = 1)
target = data['price']
data = data.drop(['price'], axis = 1)
data.head(1)


# In[ ]:


data.describe(percentiles=[])


# Разбиваем данные на тест и валидацию

# In[ ]:


X_train, X_test, y_train, y_test =     train_test_split(data, target, random_state=42, train_size=0.8, shuffle=True)
print(("train size={}, test_size={}, total_size={}".format(
    X_train.shape[0], X_test.shape[0], data.shape[0])
))


# Обучаем модель линейной регрессии

# In[ ]:


from sklearn.linear_model import LinearRegression

model = LinearRegression(normalize=False)
# обучаем линейную модель на обучающей выборке
model.fit(X_train, y_train)
print(("num_ftrs = {}, num_coeff = {} ".format(X_train.shape[1], len(model.coef_))))
reg_coeff = dict(list(zip(data.columns, model.coef_)))
print(reg_coeff)


# In[ ]:


y_pred_train = model.predict(X_train)
print(("Качество на тесте {}".format(mean_squared_error(y_train, y_pred_train))))
y_pred = model.predict(X_test)
print(("Качество на контроле {}".format(mean_squared_error(y_test, y_pred))))

sns.barplot(x = X_train.columns, y=model.coef_)
plt.xticks(rotation=90)


# В силу того, что у коэффициентов очень разные масштабы, получили неадекватные коэффициенты регрессии
# 
# Из коробки можно применить нормализацию данных, где для каждого значения признака $x_j$ выполняем
# 
# $$
# \overline{\mu_j} = \frac{1}{n}\sum_{i=0}^{n}x_{ij}, \overline{\sigma_j} = \frac{1}{n}\sqrt{\sum_{i=0}^{n}(x_{ij} - \mu_j )^2}
# $$
# Получаем новые признаки - нормализованные
# $$
# x_{new} = \frac{\overline{\mu_j}} { \overline{\sigma_j}}
# $$
# 
# В sklearn такое преобразование делает StandartScaler

# In[ ]:


from sklearn.preprocessing import StandardScaler
X_train_scale = pd.DataFrame(StandardScaler().fit_transform(X_train), columns = X_train.columns)
X_train_scale.set_index(X_train.index, inplace = True)
X_test_scale = pd.DataFrame(StandardScaler().fit_transform(X_test), columns = X_test.columns)
X_test_scale.set_index(X_test.index, inplace = True)


# In[ ]:


X_train_scale.mean()


# In[ ]:


model_norm = LinearRegression(normalize=False)
model_norm.fit(X_train_scale, y_train)
reg_coeff_norm = dict(list(zip(data.columns, model_norm.coef_)))
print(reg_coeff_norm)


# Изменение масштаба помогло, коэффициенты начали выравниваться, но качество модель улучшилась слабо - качество осталось отвратительным.

# In[ ]:


y_pred_train = model_norm.predict(X_train_scale)
print(("Качество на тесте {}".format(mean_squared_error(y_train, y_pred_train))))
y_pred = model_norm.predict(X_test_scale)
print(("Качество на контроле {}".format(mean_squared_error(y_test, y_pred))))

sns.barplot(x = X_train_scale.columns, y=model_norm.coef_)
plt.xticks(rotation=90)


# Способ ограничить коэффициенты регрессии - регуляризация
# 
# L2-регуляризация применяется в гребневой регрессии

# In[ ]:


from sklearn.linear_model import Ridge

ridge_model = Ridge(alpha=0.5)
ridge_model.fit(X_train_scale, y_train)
reg_coeff_ridge = dict(list(zip(data.columns, ridge_model.coef_)))
print(reg_coeff_ridge)


# In[ ]:


y_pred_train = ridge_model.predict(X_train_scale)
print(("Качество на тесте {}".format(mean_squared_error(y_train, y_pred_train))))
y_pred = ridge_model.predict(X_test_scale)
print(("Качество на контроле {}".format(mean_squared_error(y_test, y_pred))))

sns.barplot(x = X_train_scale.columns, y=ridge_model.coef_)
plt.xticks(rotation=90)


# RMSE метрика принимает огромные значения, что бы мы не делали. С признаками сделали всё, что можно - видимо беда с целевой переменной, проверим распределение target.

# In[ ]:


target.plot.hist(bins=100)


# Мы видим, что у распределения очень тяжёлый хвост - целевая переменная имеет распределение, далёкое от нормального. Из-за этого модель плохо работает
# 
# Проведём resudal analysis - посмотрим ошибки по всем объектам выборки

# In[ ]:


resudal_vec = y_pred_train - y_train
resudal_vec.plot.hist(bins=100)


# 1% самых экстремальных значений - это выбросы.  Они слишком сильно превышают среднее значение по выборке - из-за этого распределение целевой переменной не похоже на распределение экспоненциального семейства.
# 
# Выкинем выбросы и снова посмотрим на остатки.

# In[ ]:


filter_outlier = resudal_vec > resudal_vec.quantile(q=0.01)
resudal_vec[filter_outlier].hist(bins=100)


# Распределение стало более симметричным. Можем удалить из обучающей выборки и посмотреть, как изменилось качество.

# In[ ]:


model_norm = LinearRegression(normalize=False)
model_norm.fit(X_train_scale[filter_outlier], y_train[filter_outlier])
reg_coeff_norm = dict(list(zip(data.columns, model_norm.coef_)))
print(reg_coeff_norm)


# Коэффициенты стали выглядеть более осмысленно - теперь они как минимум в одном масштабе

# In[ ]:


y_pred_train = model_norm.predict(X_train_scale[filter_outlier])
print(("Качество на тесте {}".format(mean_squared_error(y_train[filter_outlier], y_pred_train))))
y_pred = model_norm.predict(X_test_scale)
print(("Качество на контроле {}".format(mean_squared_error(y_test, y_pred))))

sns.barplot(x = X_train_scale.columns, y=model_norm.coef_)
plt.xticks(rotation=90)


# Качество на тесте сильно улучшилось, на контрол

# In[ ]:





# In[ ]:





# In[ ]:




