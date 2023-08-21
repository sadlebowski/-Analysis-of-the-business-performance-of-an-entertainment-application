#!/usr/bin/env python
# coding: utf-8

# **Анализ бизнес-показателей развлекательного приложения Procrastinate Pro+**
# 
# Цель проекта
# 
# На основе данных, предоставленных компанией, необходимо провести анализ и ответить на вопросы:
# 
# * откуда приходят пользователи и какими устройствами они пользуются,
# * сколько стоит привлечение пользователей из различных рекламных каналов;
# * сколько денег приносит каждый клиент,
# * когда расходы на привлечение клиента окупаются,
# * какие факторы мешают привлечению клиентов.
# 
# Ход исследования
# 
# Исследование пройдёт в четыре этапа:
# 
# * Обзор и предобработка данных;
# * Исследовательский анализ данных;
# * Анализ маркетинговых расходов;
# * Оценка окупаемости рекламы.

# ### Загрузите данные и подготовьте их к анализу

# Загрузите данные о визитах, заказах и рекламных расходах из CSV-файлов в переменные.
# 
# **Пути к файлам**
# 
# - визиты: `/datasets/visits_info_short.csv`. [Скачать датасет](https://code.s3.yandex.net/datasets/visits_info_short.csv);
# - заказы: `/datasets/orders_info_short.csv`. [Скачать датасет](https://code.s3.yandex.net/datasets/orders_info_short.csv);
# - расходы: `/datasets/costs_info_short.csv`. [Скачать датасет](https://code.s3.yandex.net/datasets/costs_info_short.csv).
# 
# Изучите данные и выполните предобработку. Есть ли в данных пропуски и дубликаты? Убедитесь, что типы данных во всех колонках соответствуют сохранённым в них значениям. Обратите внимание на столбцы с датой и временем.

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from datetime import datetime, timedelta

visits = pd.read_csv('/datasets/visits_info_short.csv')
orders = pd.read_csv('/datasets/orders_info_short.csv')
costs = pd.read_csv('/datasets/costs_info_short.csv')


# In[2]:


#предобработка visits
visits.head()


# In[3]:


visits.info()


# In[4]:


visits.columns.to_list()


# In[5]:


visits.columns = ['user_id', 'region', 'device', 'channel', 'session_start', 'session_end']


# In[6]:


visits['session_end'] = pd.to_datetime(visits['session_end'])
visits['session_start'] = pd.to_datetime(visits['session_start'])


# In[7]:


visits.info()


# In[8]:


visits.duplicated().sum()


# In[9]:


#предобработка orders
orders.head()


# In[10]:


orders.info()


# In[11]:


orders.columns.to_list()


# In[12]:


orders.columns = ['user_id', 'event_dt', 'revenue']


# In[13]:


orders['event_dt'] = pd.to_datetime(orders['event_dt'])


# In[14]:


orders.info()


# In[15]:


orders.duplicated().sum()


# In[16]:


#предобработка costs
costs.head()


# In[17]:


costs.info()


# In[18]:


costs.columns.to_list()


# In[19]:


costs.columns = ['dt', 'channel', 'costs']


# In[20]:


costs['dt'] = pd.to_datetime(costs['dt']).dt.date 


# In[21]:


costs.info()


# In[22]:


costs.duplicated().sum()


# **Вывод:**
# 
# Ознакомились с таблицами. Пропусков и дупликатов не обнаружилось. Исправили тип данных в столбцах с датами и временем и привели стобцы к единому стилю

# ### Задайте функции для расчёта и анализа LTV, ROI, удержания и конверсии.
# 
# Разрешается использовать функции, с которыми вы познакомились в теоретических уроках.
# 
# Это функции для вычисления значений метрик:
# 
# - `get_profiles()` — для создания профилей пользователей,
# - `get_retention()` — для подсчёта Retention Rate,
# - `get_conversion()` — для подсчёта конверсии,
# - `get_ltv()` — для подсчёта LTV.
# 
# А также функции для построения графиков:
# 
# - `filter_data()` — для сглаживания данных,
# - `plot_retention()` — для построения графика Retention Rate,
# - `plot_conversion()` — для построения графика конверсии,
# - `plot_ltv_roi` — для визуализации LTV и ROI.

# In[23]:


# функция для создания пользовательских профилей

def get_profiles(sessions, orders, ad_costs):

    # находим параметры первых посещений
    profiles = (
        sessions.sort_values(by=['user_id', 'session_start'])
        .groupby('user_id')
        .agg(
            {
                'session_start': 'first',
                'channel': 'first',
                'device': 'first',
                'region': 'first',
            }
        )
        .rename(columns={'session_start': 'first_ts'})
        .reset_index()
    )

    # для когортного анализа определяем дату первого посещения
    # и первый день месяца, в который это посещение произошло
    profiles['dt'] = profiles['first_ts'].dt.date
    profiles['month'] = profiles['first_ts'].astype('datetime64[M]')
    profiles['week']= profiles['first_ts'].dt.isocalendar().week

    # добавляем признак платящих пользователей
    profiles['payer'] = profiles['user_id'].isin(orders['user_id'].unique())

    # считаем количество уникальных пользователей
    # с одинаковыми источником и датой привлечения
    new_users = (
        profiles.groupby(['dt', 'channel'])
        .agg({'user_id': 'nunique'})
        .rename(columns={'user_id': 'unique_users'})
        .reset_index()
    )

    # объединяем траты на рекламу и число привлечённых пользователей
    ad_costs = ad_costs.merge(new_users, on=['dt', 'channel'], how='left')

    # делим рекламные расходы на число привлечённых пользователей
    ad_costs['acquisition_cost'] = ad_costs['costs'] / ad_costs['unique_users']

    # добавляем стоимость привлечения в профили
    profiles = profiles.merge(
        ad_costs[['dt', 'channel', 'acquisition_cost']],
        on=['dt', 'channel'],
        how='left',
    )

    # стоимость привлечения органических пользователей равна нулю
    profiles['acquisition_cost'] = profiles['acquisition_cost'].fillna(0)

    return profiles


# In[24]:


# функция для расчёта удержания

def get_retention(
    profiles,
    sessions,
    observation_date,
    horizon_days,
    dimensions=[],
    ignore_horizon=False,
):

    # добавляем столбец payer в передаваемый dimensions список
    dimensions = ['payer'] + dimensions

    # исключаем пользователей, не «доживших» до горизонта анализа
    last_suitable_acquisition_date = observation_date
    if not ignore_horizon:
        last_suitable_acquisition_date = observation_date - timedelta(
            days=horizon_days - 1
        )
    result_raw = profiles.query('dt <= @last_suitable_acquisition_date')

    # собираем «сырые» данные для расчёта удержания
    result_raw = result_raw.merge(
        sessions[['user_id', 'session_start']], on='user_id', how='left'
    )
    result_raw['lifetime'] = (
        result_raw['session_start'] - result_raw['first_ts']
    ).dt.days

    # функция для группировки таблицы по желаемым признакам
    def group_by_dimensions(df, dims, horizon_days):
        result = df.pivot_table(
            index=dims, columns='lifetime', values='user_id', aggfunc='nunique'
        )
        cohort_sizes = (
            df.groupby(dims)
            .agg({'user_id': 'nunique'})
            .rename(columns={'user_id': 'cohort_size'})
        )
        result = cohort_sizes.merge(result, on=dims, how='left').fillna(0)
        result = result.div(result['cohort_size'], axis=0)
        result = result[['cohort_size'] + list(range(horizon_days))]
        result['cohort_size'] = cohort_sizes
        return result

    # получаем таблицу удержания
    result_grouped = group_by_dimensions(result_raw, dimensions, horizon_days)

    # получаем таблицу динамики удержания
    result_in_time = group_by_dimensions(
        result_raw, dimensions + ['dt'], horizon_days
    )

    # возвращаем обе таблицы и сырые данные
    return result_raw, result_grouped, result_in_time


# In[25]:


# функция для расчёта конверсии

def get_conversion(
    profiles,
    purchases,
    observation_date,
    horizon_days,
    dimensions=[],
    ignore_horizon=False,
):

    # исключаем пользователей, не «доживших» до горизонта анализа
    last_suitable_acquisition_date = observation_date
    if not ignore_horizon:
        last_suitable_acquisition_date = observation_date - timedelta(
            days=horizon_days - 1
        )
    result_raw = profiles.query('dt <= @last_suitable_acquisition_date')

    # определяем дату и время первой покупки для каждого пользователя
    first_purchases = (
        purchases.sort_values(by=['user_id', 'event_dt'])
        .groupby('user_id')
        .agg({'event_dt': 'first'})
        .reset_index()
    )

    # добавляем данные о покупках в профили
    result_raw = result_raw.merge(
        first_purchases[['user_id', 'event_dt']], on='user_id', how='left'
    )

    # рассчитываем лайфтайм для каждой покупки
    result_raw['lifetime'] = (
        result_raw['event_dt'] - result_raw['first_ts']
    ).dt.days

    # группируем по cohort, если в dimensions ничего нет
    if len(dimensions) == 0:
        result_raw['cohort'] = 'All users' 
        dimensions = dimensions + ['cohort']

    # функция для группировки таблицы по желаемым признакам
    def group_by_dimensions(df, dims, horizon_days):
        result = df.pivot_table(
            index=dims, columns='lifetime', values='user_id', aggfunc='nunique'
        )
        result = result.fillna(0).cumsum(axis = 1)
        cohort_sizes = (
            df.groupby(dims)
            .agg({'user_id': 'nunique'})
            .rename(columns={'user_id': 'cohort_size'})
        )
        result = cohort_sizes.merge(result, on=dims, how='left').fillna(0)
        # делим каждую «ячейку» в строке на размер когорты
        # и получаем conversion rate
        result = result.div(result['cohort_size'], axis=0)
        result = result[['cohort_size'] + list(range(horizon_days))]
        result['cohort_size'] = cohort_sizes
        return result

    # получаем таблицу конверсии
    result_grouped = group_by_dimensions(result_raw, dimensions, horizon_days)

    # для таблицы динамики конверсии убираем 'cohort' из dimensions
    if 'cohort' in dimensions: 
        dimensions = []

    # получаем таблицу динамики конверсии
    result_in_time = group_by_dimensions(
        result_raw, dimensions + ['dt'], horizon_days
    )

    # возвращаем обе таблицы и сырые данные
    return result_raw, result_grouped, result_in_time


# In[26]:


# функция для расчёта LTV и ROI

def get_ltv(
    profiles,
    purchases,
    observation_date,
    horizon_days,
    dimensions=[],
    ignore_horizon=False,
):

    # исключаем пользователей, не «доживших» до горизонта анализа
    last_suitable_acquisition_date = observation_date
    if not ignore_horizon:
        last_suitable_acquisition_date = observation_date - timedelta(
            days=horizon_days - 1
        )
    result_raw = profiles.query('dt <= @last_suitable_acquisition_date')
  
    # добавляем данные о покупках в профили
    result_raw = result_raw.merge(
        purchases[['user_id', 'event_dt', 'revenue']], on='user_id', how='left'
    )
    
    # рассчитываем лайфтайм пользователя для каждой покупки
    result_raw['lifetime'] = (
        result_raw['event_dt'] - result_raw['first_ts']
    ).dt.days
    
    # группируем по cohort, если в dimensions ничего нет
    if len(dimensions) == 0:
        result_raw['cohort'] = 'All users'
        dimensions = dimensions + ['cohort']

    # функция группировки по желаемым признакам
    def group_by_dimensions(df, dims, horizon_days):
        # строим «треугольную» таблицу выручки
        result = df.pivot_table(
            index=dims, columns='lifetime', values='revenue', aggfunc='sum'
        )
        # находим сумму выручки с накоплением
        result = result.fillna(0).cumsum(axis=1)
        # вычисляем размеры когорт
        cohort_sizes = (
            df.groupby(dims)
            .agg({'user_id': 'nunique'})
            .rename(columns={'user_id': 'cohort_size'})
        )
        # объединяем размеры когорт и таблицу выручки
        result = cohort_sizes.merge(result, on=dims, how='left').fillna(0)
        # считаем LTV: делим каждую «ячейку» в строке на размер когорты
        result = result.div(result['cohort_size'], axis=0)
        # исключаем все лайфтаймы, превышающие горизонт анализа
        result = result[['cohort_size'] + list(range(horizon_days))]
        # восстанавливаем размеры когорт
        result['cohort_size'] = cohort_sizes

        # собираем датафрейм с данными пользователей и значениями CAC, 
        # добавляя параметры из dimensions
        cac = df[['user_id', 'acquisition_cost'] + dims].drop_duplicates()

        # считаем средний CAC по параметрам из dimensions
        cac = (
            cac.groupby(dims)
            .agg({'acquisition_cost': 'mean'})
            .rename(columns={'acquisition_cost': 'cac'})
        )

        # считаем ROI: делим LTV на CAC
        roi = result.div(cac['cac'], axis=0)

        # удаляем строки с бесконечным ROI
        roi = roi[~roi['cohort_size'].isin([np.inf])]

        # восстанавливаем размеры когорт в таблице ROI
        roi['cohort_size'] = cohort_sizes

        # добавляем CAC в таблицу ROI
        roi['cac'] = cac['cac']

        # в финальной таблице оставляем размеры когорт, CAC
        # и ROI в лайфтаймы, не превышающие горизонт анализа
        roi = roi[['cohort_size', 'cac'] + list(range(horizon_days))]

        # возвращаем таблицы LTV и ROI
        return result, roi

    # получаем таблицы LTV и ROI
    result_grouped, roi_grouped = group_by_dimensions(
        result_raw, dimensions, horizon_days
    )

    # для таблиц динамики убираем 'cohort' из dimensions
    if 'cohort' in dimensions:
        dimensions = []

    # получаем таблицы динамики LTV и ROI
    result_in_time, roi_in_time = group_by_dimensions(
        result_raw, dimensions + ['dt'], horizon_days
    )

    return (
        result_raw,  # сырые данные
        result_grouped,  # таблица LTV
        result_in_time,  # таблица динамики LTV
        roi_grouped,  # таблица ROI
        roi_in_time,  # таблица динамики ROI
    )


# In[27]:


# функция для сглаживания фрейма

def filter_data(df, window):
    # для каждого столбца применяем скользящее среднее
    for column in df.columns.values:
        df[column] = df[column].rolling(window).mean() 
    return df


# In[28]:


# функция для визуализации удержания

def plot_retention(retention, retention_history, horizon, window=7):

    # задаём размер сетки для графиков
    plt.figure(figsize=(15, 10))

    # исключаем размеры когорт и удержание первого дня
    retention = retention.drop(columns=['cohort_size', 0])
    # в таблице динамики оставляем только нужный лайфтайм
    retention_history = retention_history.drop(columns=['cohort_size'])[
        [horizon - 1]
    ]

    # если в индексах таблицы удержания только payer,
    # добавляем второй признак — cohort
    if retention.index.nlevels == 1:
        retention['cohort'] = 'All users'
        retention = retention.reset_index().set_index(['cohort', 'payer'])

    # в таблице графиков — два столбца и две строки, четыре ячейки
    # в первой строим кривые удержания платящих пользователей
    ax1 = plt.subplot(2, 2, 1)
    retention.query('payer == True').droplevel('payer').T.plot(
        grid=True, ax=ax1
    )
    plt.legend()
    plt.xlabel('Лайфтайм')
    plt.title('Удержание платящих пользователей')

    # во второй ячейке строим кривые удержания неплатящих
    # вертикальная ось — от графика из первой ячейки
    ax2 = plt.subplot(2, 2, 2, sharey=ax1)
    retention.query('payer == False').droplevel('payer').T.plot(
        grid=True, ax=ax2
    )
    plt.legend()
    plt.xlabel('Лайфтайм')
    plt.title('Удержание неплатящих пользователей')

    # в третьей ячейке — динамика удержания платящих
    ax3 = plt.subplot(2, 2, 3)
    # получаем названия столбцов для сводной таблицы
    columns = [
        name
        for name in retention_history.index.names
        if name not in ['dt', 'payer']
    ]
    # фильтруем данные и строим график
    filtered_data = retention_history.query('payer == True').pivot_table(
        index='dt', columns=columns, values=horizon - 1, aggfunc='mean'
    )
    filter_data(filtered_data, window).plot(grid=True, ax=ax3)
    plt.xlabel('Дата привлечения')
    plt.title(
        'Динамика удержания платящих пользователей на {}-й день'.format(
            horizon
        )
    )

    # в чётвертой ячейке — динамика удержания неплатящих
    ax4 = plt.subplot(2, 2, 4, sharey=ax3)
    # фильтруем данные и строим график
    filtered_data = retention_history.query('payer == False').pivot_table(
        index='dt', columns=columns, values=horizon - 1, aggfunc='mean'
    )
    filter_data(filtered_data, window).plot(grid=True, ax=ax4)
    plt.xlabel('Дата привлечения')
    plt.title(
        'Динамика удержания неплатящих пользователей на {}-й день'.format(
            horizon
        )
    )
    
    plt.tight_layout()
    plt.show() 


# In[29]:


def plot_conversion(conversion, conversion_history, horizon, window=7):

    # задаём размер сетки для графиков
    plt.figure(figsize=(15, 5))

    # исключаем размеры когорт
    conversion = conversion.drop(columns=['cohort_size'])
    # в таблице динамики оставляем только нужный лайфтайм
    conversion_history = conversion_history.drop(columns=['cohort_size'])[
        [horizon - 1]
    ]

    # первый график — кривые конверсии
    ax1 = plt.subplot(1, 2, 1)
    conversion.T.plot(grid=True, ax=ax1)
    plt.legend()
    plt.xlabel('Лайфтайм')
    plt.title('Конверсия пользователей')

    # второй график — динамика конверсии
    ax2 = plt.subplot(1, 2, 2, sharey=ax1)
    columns = [
        # столбцами сводной таблицы станут все столбцы индекса, кроме даты
        name for name in conversion_history.index.names if name not in ['dt']
    ]
    filtered_data = conversion_history.pivot_table(
        index='dt', columns=columns, values=horizon - 1, aggfunc='mean'
    )
    filter_data(filtered_data, window).plot(grid=True, ax=ax2)
    plt.xlabel('Дата привлечения')
    plt.title('Динамика конверсии пользователей на {}-й день'.format(horizon))

    plt.tight_layout()
    plt.show() 


# In[30]:


# функция для визуализации LTV и ROI

def plot_ltv_roi(ltv, ltv_history, roi, roi_history, horizon, window=7):

    # задаём сетку отрисовки графиков
    plt.figure(figsize=(20, 20))

    # из таблицы ltv исключаем размеры когорт
    ltv = ltv.drop(columns=['cohort_size'])
    # в таблице динамики ltv оставляем только нужный лайфтайм
    ltv_history = ltv_history.drop(columns=['cohort_size'])[[horizon - 1]]

    # стоимость привлечения запишем в отдельный фрейм
    cac_history = roi_history[['cac']]

    # из таблицы roi исключаем размеры когорт и cac
    roi = roi.drop(columns=['cohort_size', 'cac'])
    # в таблице динамики roi оставляем только нужный лайфтайм
    roi_history = roi_history.drop(columns=['cohort_size', 'cac'])[
        [horizon - 1]
    ]

    # первый график — кривые ltv
    ax1 = plt.subplot(3, 2, 1)
    ltv.T.plot(grid=True, ax=ax1)
    plt.legend()
    plt.xlabel('Лайфтайм')
    plt.title('LTV')

    # второй график — динамика ltv
    ax2 = plt.subplot(3, 2, 2, sharey=ax1)
    # столбцами сводной таблицы станут все столбцы индекса, кроме даты
    columns = [name for name in ltv_history.index.names if name not in ['dt']]
    filtered_data = ltv_history.pivot_table(
        index='dt', columns=columns, values=horizon - 1, aggfunc='mean'
    )
    filter_data(filtered_data, window).plot(grid=True, ax=ax2)
    plt.xlabel('Дата привлечения')
    plt.title('Динамика LTV пользователей на {}-й день'.format(horizon))

    # третий график — динамика cac
    ax3 = plt.subplot(3, 2, 3, sharey=ax1)
    # столбцами сводной таблицы станут все столбцы индекса, кроме даты
    columns = [name for name in cac_history.index.names if name not in ['dt']]
    filtered_data = cac_history.pivot_table(
        index='dt', columns=columns, values='cac', aggfunc='mean'
    )
    filter_data(filtered_data, window).plot(grid=True, ax=ax3)
    plt.xlabel('Дата привлечения')
    plt.title('Динамика стоимости привлечения пользователей')

    # четвёртый график — кривые roi
    ax4 = plt.subplot(3, 2, 4)
    roi.T.plot(grid=True, ax=ax4)
    plt.axhline(y=1, color='red', linestyle='--', label='Уровень окупаемости')
    plt.legend()
    plt.xlabel('Лайфтайм')
    plt.title('ROI')

    # пятый график — динамика roi
    ax5 = plt.subplot(3, 2, 5, sharey=ax4)
    # столбцами сводной таблицы станут все столбцы индекса, кроме даты
    columns = [name for name in roi_history.index.names if name not in ['dt']]
    filtered_data = roi_history.pivot_table(
        index='dt', columns=columns, values=horizon - 1, aggfunc='mean'
    )
    filter_data(filtered_data, window).plot(grid=True, ax=ax5)
    plt.axhline(y=1, color='red', linestyle='--', label='Уровень окупаемости')
    plt.xlabel('Дата привлечения')
    plt.title('Динамика ROI пользователей на {}-й день'.format(horizon))

    plt.tight_layout()
    plt.show()


# ### Исследовательский анализ данных
# 
# - Составьте профили пользователей. Определите минимальную и максимальную даты привлечения пользователей.
# - Выясните, из каких стран пользователи приходят в приложение и на какую страну приходится больше всего платящих пользователей. Постройте таблицу, отражающую количество пользователей и долю платящих из каждой страны.
# - Узнайте, какими устройствами пользуются клиенты и какие устройства предпочитают платящие пользователи. Постройте таблицу, отражающую количество пользователей и долю платящих для каждого устройства.
# - Изучите рекламные источники привлечения и определите каналы, из которых пришло больше всего платящих пользователей. Постройте таблицу, отражающую количество пользователей и долю платящих для каждого канала привлечения.
# 
# После каждого пункта сформулируйте выводы.

# In[31]:


#Составьте профили пользователей. Определите минимальную и максимальную даты привлечения пользователей.
profiles = get_profiles(visits, orders, costs)
profiles.head(10)


# In[32]:


observation_date = datetime(2019, 11, 1).date()  # момент анализа
analysis_horizon = 14  # горизонт анализа
# определяем минимальную дату привлечения
min_analysis_date = profiles['dt'].min()
# считаем максимальную дату привлечения
max_analysis_date =  profiles['dt'].max()
print(f'Минимальная дата привлечения пользователей: {min_analysis_date}')
print(f'Максимальная дата привлечения пользователей: {max_analysis_date}')


# In[33]:


#Выясните, из каких стран пользователи приходят в приложение и на какую страну приходится больше всего платящих пользователей. 
#Постройте таблицу, отражающую количество пользователей и долю платящих из каждой страны.
region = (
    profiles
    .pivot_table(index='region',
                columns='payer',
                values='user_id',
                aggfunc='count')
    .rename(columns={True: 'payer', False: 'not_payer'})
    .sort_values(by='payer', ascending=False)
)
region['payer_share'] = (region.payer / (region.not_payer + region.payer) * 100).round(2)
region


# Можно заметить что самое большое количество привлечённых пользователей находится в США. Так же эта страна с подавляющим преимуществом обходит другие страны в количество платящих пользователей. Великобритания с Францией схожи, а Германия отстаёт(при этом имеет большую долю платящих)

# In[34]:


#Узнайте, какими устройствами пользуются клиенты и какие устройства предпочитают платящие пользователи. 
#Постройте таблицу, отражающую количество пользователей и долю платящих для каждого устройства.
device = (
    profiles
    .pivot_table(index='device',
                columns='payer',
                values='user_id',
                aggfunc='count')
    .rename(columns={True: 'payer', False: 'not_payer'})
    .sort_values(by='payer', ascending=False)
)
device['payer_share'] = (device.payer / (device.not_payer + device.payer) * 100).round(2)
device


# Клиенты предпочитают пользоваться Iphone, на втором месте Android, делее с малым отставанием идёт Mac, который при этом имеет большую долю платящих, на последнем месте PC

# In[35]:


#Изучите рекламные источники привлечения и определите каналы, из которых пришло больше всего платящих пользователей. 
#Постройте таблицу, отражающую количество пользователей и долю платящих для каждого канала привлечения.
channel = (
    profiles
    .pivot_table(index='channel',
                columns='payer',
                values='user_id',
                aggfunc='count')
    .rename(columns={True: 'payer', False: 'not_payer'})
    .sort_values(by='payer', ascending=False)
)
channel['payer_share'] = (channel.payer / (channel.not_payer + channel.payer) * 100).round(2)
channel


# Органические пользователи являются самой большой группой, однако доля платящих органических клиентов самая низкая. Два самых больших рекламных источника - FaceBoom и TipTop. AdNonSense и lambdaMediaAds имеют большую конверсию.	

# ### Маркетинг
# 
# - Посчитайте общую сумму расходов на маркетинг.
# - Выясните, как траты распределены по рекламным источникам, то есть сколько денег потратили на каждый источник.
# - Постройте визуализацию динамики изменения расходов во времени (по неделям и месяцам) по каждому источнику. Постарайтесь отразить это на одном графике.
# - Узнайте, сколько в среднем стоило привлечение одного пользователя (CAC) из каждого источника. Используйте профили пользователей.
# 
# Напишите промежуточные выводы.

# In[36]:


#Посчитайте общую сумму расходов на маркетинг.
print(f'Общая сумма затрат на рекламу: {costs.costs.sum().round()}')


# In[37]:


#Выясните, как траты распределены по рекламным источникам, то есть сколько денег потратили на каждый источник.
costs.pivot_table(
    index='channel',
    values='costs',
    aggfunc='sum').sort_values(by='costs', ascending=False)


# Больше всего ыло потрачено на такие источники как - TipTop и FaceBoom. Меньше всего на - MediaTornado и YRabbit

# In[38]:


#Постройте визуализацию динамики изменения расходов во времени (по неделям и месяцам) по каждому источнику
plt.figure(figsize=(20,8))
ax1 = plt.subplot(1, 2, 1)

(profiles
 .pivot_table(
    index='month',
    values='acquisition_cost',
    aggfunc='sum',
    columns='channel'
)
 .plot(ax=ax1, grid=True)
)

plt.legend()
plt.title('Расходы за месяц')

ax2 = plt.subplot(1, 2, 2, sharey=ax1)
(profiles
 .pivot_table(
    index='week',
    values='acquisition_cost',
    aggfunc='sum',
    columns='channel'
)
 .plot(ax=ax2, grid=True)
)
plt.title('Расходы за неделю')


# * Ближе к середине осени расходы на рекламу повышались, так же есть незначительный рост расходов по неделям
# * Каналы TipTop и FaceBoom обходятся дороже остальных, причём стоимость расходов на TipTop стабильно росла до сентября 2019

# In[39]:


#Узнайте, сколько в среднем стоило привлечение одного пользователя (CAC) из каждого источника. 
#Используйте профили пользователей.
cac_channel = (profiles
       .pivot_table(index = 'channel',
                    values = 'acquisition_cost',
                    aggfunc='mean')
       .sort_values(by='acquisition_cost', ascending=False)
       .rename(columns={'acquisition_cost': 'cac'})
      )
cac_channel


# Привлечение одного пользователя дороже всего обходится в источнике - TipTop

# ### Оцените окупаемость рекламы
# 
# Используя графики LTV, ROI и CAC, проанализируйте окупаемость рекламы. Считайте, что на календаре 1 ноября 2019 года, а в бизнес-плане заложено, что пользователи должны окупаться не позднее чем через две недели после привлечения. Необходимость включения в анализ органических пользователей определите самостоятельно.
# 
# - Проанализируйте окупаемость рекламы c помощью графиков LTV и ROI, а также графики динамики LTV, CAC и ROI.
# - Проверьте конверсию пользователей и динамику её изменения. То же самое сделайте с удержанием пользователей. Постройте и изучите графики конверсии и удержания.
# - Проанализируйте окупаемость рекламы с разбивкой по устройствам. Постройте графики LTV и ROI, а также графики динамики LTV, CAC и ROI.
# - Проанализируйте окупаемость рекламы с разбивкой по странам. Постройте графики LTV и ROI, а также графики динамики LTV, CAC и ROI.
# - Проанализируйте окупаемость рекламы с разбивкой по рекламным каналам. Постройте графики LTV и ROI, а также графики динамики LTV, CAC и ROI.
# - Ответьте на такие вопросы:
#     - Окупается ли реклама, направленная на привлечение пользователей в целом?
#     - Какие устройства, страны и рекламные каналы могут оказывать негативное влияние на окупаемость рекламы?
#     - Чем могут быть вызваны проблемы окупаемости?
# 
# Напишите вывод, опишите возможные причины обнаруженных проблем и промежуточные рекомендации для рекламного отдела.

# In[40]:


#исключим органически привлечённых пользователей для оценки окупаемости рекламы
profiles = profiles.query('channel != "organic"')


# In[41]:


#Проанализируйте окупаемость рекламы c помощью графиков LTV и ROI, а также графики динамики LTV, CAC и ROI.
# считаем LTV и ROI
ltv_raw, ltv_grouped, ltv_history, roi_grouped, roi_history = get_ltv(
    profiles, orders, observation_date, analysis_horizon
)

# строим графики
plot_ltv_roi(ltv_grouped, ltv_history, roi_grouped, roi_history, analysis_horizon)


# Выводы:
#      
#     * САС увеоличивается, следовательно расходы на рекламу увеличиваются
#     * привлеченные клиенты перестают окупаться, начиная с июня
#     * реклама к 14-му дню и далее не окупается

# In[42]:


#Проверьте конверсию пользователей и динамику её изменения. То же самое сделайте с удержанием пользователей. 
#Постройте и изучите графики конверсии и удержания
# смотрим конверсию с разбивкой по устройствам
dimensions = ['device']

conversion_raw, conversion_grouped, conversion_history = get_conversion(
    profiles, orders, observation_date, analysis_horizon, dimensions=dimensions
)

plot_conversion(conversion_grouped, conversion_history, analysis_horizon)


# У всех устройств достаточно хорошая конверсия, Iphone и Mac идут на первых позициях

# In[43]:


# смотрим удержание с разбивкой по устройствам

retention_raw, retention_grouped, retention_history = get_retention(
    profiles, visits, observation_date, analysis_horizon, dimensions=dimensions
)

plot_retention(retention_grouped, retention_history, analysis_horizon)


# Больших отличий по удержанию не видно, Pc удерживает платящих клиентов чуть лучше

# In[44]:


# смотрим конверсию с разбивкой по странам
dimensions = ['region']

conversion_raw, conversion_grouped, conversion_history = get_conversion(
    profiles, orders, observation_date, analysis_horizon, dimensions=dimensions
)

plot_conversion(conversion_grouped, conversion_history, analysis_horizon)


# Видно огромное преимущество в конверсии пользователей в США

# In[45]:


# смотрим удержание с разбивкой по странам

retention_raw, retention_grouped, retention_history = get_retention(
    profiles, visits, observation_date, analysis_horizon, dimensions=dimensions
)

plot_retention(retention_grouped, retention_history, analysis_horizon)


# Однако с удержанием платящих пользователей США хуже всех. Среди неплатящих нет особых отличий

# In[46]:


# смотрим конверсию с разбивкой по каналам 
dimensions = ['channel']

conversion_raw, conversion_grouped, conversion_history = get_conversion(
    profiles, orders, observation_date, analysis_horizon, dimensions=dimensions
)

plot_conversion(conversion_grouped, conversion_history, analysis_horizon)


# Лучшая конверсия у источника - FaceBoom

# In[47]:


# смотрим удержание с разбивкой по каналам

retention_raw, retention_grouped, retention_history = get_retention(
    profiles, visits, observation_date, analysis_horizon, dimensions=dimensions
)

plot_retention(retention_grouped, retention_history, analysis_horizon)


# Хуже всего платящих клиентов удерживает - FaceBoom и AdNonSense

# In[48]:


#Проанализируйте окупаемость рекламы с разбивкой по устройствам. 
#Постройте графики LTV и ROI, а также графики динамики LTV, CAC и ROI.
dimensions = ['device']

ltv_raw, ltv_grouped, ltv_history, roi_grouped, roi_history = get_ltv(
    profiles, orders, observation_date, analysis_horizon, dimensions=dimensions
)

plot_ltv_roi(
    ltv_grouped, ltv_history, roi_grouped, roi_history, analysis_horizon, window=14
)


# * Стоимость привлечения клиентов на продукции Apple увеличивается, однако инвестиции не окупаются
# * Видим что LTV у каждого девайса примерно одинаковый, но засчёт того что САС у PC значительно ниже остальных, то можно увидеть что PC это единственный девайс, который начинает окупаться к 14 дню. Однако и PC начинает уходить в убыток ближе к сентябрю

# In[49]:


#Проанализируйте окупаемость рекламы с разбивкой по странам. 
#Постройте графики LTV и ROI, а также графики динамики LTV, CAC и ROI.
dimensions = ['region']

ltv_raw, ltv_grouped, ltv_history, roi_grouped, roi_history = get_ltv(
    profiles, orders, observation_date, analysis_horizon, dimensions=dimensions
)

plot_ltv_roi(
    ltv_grouped, ltv_history, roi_grouped, roi_history, analysis_horizon, window=14
)


# * Затраты на рекламу в Европе остаются неизменными и окупаются, а вот в США растут и перестали окупаться
# * LTV в страназ Европы примерно одинаковый, а США занимает первое место по этому параметру с значительным отрывом. Однако если посмотреть на рекламные затраты (САС), то можно увидеть огромную пропасть между двумя сторонами земного шара. За счёт огромных затрат на рекламу в США, уровень окупаемости пересёк черту в негативную сторону ещё в июне и стабильно ползёт вниз. В странах европы затраты окупаются уже к 5 дню лайфтайма и на протяжении всех месяцев

# In[50]:


#Проанализируйте окупаемость рекламы с разбивкой по рекламным каналам. 
#Постройте графики LTV и ROI, а также графики динамики LTV, CAC и ROI.
dimensions = ['channel']

ltv_raw, ltv_grouped, ltv_history, roi_grouped, roi_history = get_ltv(
    profiles, orders, observation_date, analysis_horizon, dimensions=dimensions
)

plot_ltv_roi(
    ltv_grouped, ltv_history, roi_grouped, roi_history, analysis_horizon, window=14
)


# * Стоимость привлечения пользователей на канале TipTop значительно выше чем у других и постоянно увеличивается, однако прибыль эти инвестиции перестали приносить с июня
# * По LTV выделяются такие каналы как TipTop и lambdaMediaAds, но при этом только один из этих каналов имеет чрезмерно высокий САС (TipTop). В итоге мы видим что TipTop становится аутсайдером по возвратам инвестиций. Вместе с ним FaceBoom и AdNonSense

# #Ответьте на такие вопросы:
# 
# #Окупается ли реклама, направленная на привлечение пользователей в целом?
# 
# #Какие устройства, страны и рекламные каналы могут оказывать негативное влияние на окупаемость рекламы?
# 
# #Чем могут быть вызваны проблемы окупаемости?
# 
# * В целом реклама не окупается, всё изза низкой окупаемости в США. Пользователи из США конверсируются больше, но есть проблемы с удержанием платящих пользователей
# * Один из самых проблемных каналов - FaceBoom, не окупается имея второе место по сумме затрат на рекламу, низкое удержание платящих пользователей. TipTop - высокий рост САС, в следствие чего он через месяц перестал окупаться. Так же AdNonSense имеет 3 место по САС и низкий LTV, за счёт чего имеет нестабильный уровень окупаемости
# * Почти все устройства перестали приносить доход уже к июлю, PC смогло продержаться дольше за счёт низкого уровня САС
# * Почему при высокой конверсии у FaceBoom низкое удержание? Низкий Churn Rate, низкие затраты на удержание
# 

# ### Напишите выводы
# 
# - Выделите причины неэффективности привлечения пользователей.
# - Сформулируйте рекомендации для отдела маркетинга.

# Компания представлена на рынке Европы и в США. Несмотря на огромные вложения в рекламу, последние несколько месяцев компания терпит убытки.
# В результате анализа было выявлено, что основной причиной финансовых проблем являются рекламные траты на привлечение пользователей посредством FaceBoom, TipTop в США и AdNonSence в Европе
# * Расходы на привлечение в TipTop за полгода выросли в три раза
# * Платящие пользователи FaceBoom и AdNonSence очень плохо удерживаются
# * Высокая стоимость привлечения у этих каналов
# 
# Как рекомендации предлагаю следующее:
# * Подавляющее большинство пользователей находится в США, поэтому уходить из региона нецелесообразно, рекомендую проработать отношения с здешниями каналами
# * Выявить причину роста САС канала TipTop, возможно отказаться от инвестиций на этом канале
# * Обратить внимание на YRabbit
# * Дать больше внимания европейскому рынку
# * Снизить расходы на рекламу в AdNonSense

# In[ ]:




