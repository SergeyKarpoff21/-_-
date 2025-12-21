# -*- coding: utf-8 -*-
"""
Проект: Demand Planning ARM (Automated Risk Management)
Курс: Информационные системы прогнозирования и планирования цепи поставок

Описание:
Прототип системы поддержки принятия решений для планировщика спроса.
Реализует полный цикл прогнозирования на данных M5 (Walmart):
1. ETL и оптимизация памяти (float32, category).
2. Сегментация спроса (ADI/CV2).
3. Прогнозирование (AutoARIMA vs XGBoost).
4. Иерархическое согласование (Bottom-Up, Top-Down).
5. Управление по отклонениям (Exception Management).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import xgboost as xgb
import warnings
import logging
import gc

# --- Библиотеки Nixtla (State-of-the-Art для временных рядов) ---
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, SeasonalNaive
from mlforecast import MLForecast
from mlforecast.target_transforms import Differences
from window_ops.rolling import rolling_mean, rolling_std, rolling_max

warnings.filterwarnings("ignore")
logging.getLogger('statsforecast').setLevel(logging.ERROR)

# ==========================================
# 1. ПОДГОТОВКА ДАННЫХ (DATA ENGINEERING)
# ==========================================

st.set_page_config(page_title="Demand Planning ARM", layout="wide")


def get_season(month):
    """Определяет сезон по месяцу для анализа сезонности."""
    if month in [12, 1, 2]:
        return 'Зима'
    elif month in [3, 4, 5]:
        return 'Весна'
    elif month in [6, 7, 8]:
        return 'Лето'
    else:
        return 'Осень'


@st.cache_data(show_spinner=False)
def load_raw_files():
    """
    Загрузка сырых данных.
    Кэшируется, чтобы не читать диск при каждом обновлении фильтров.
    ОПТИМИЗАЦИЯ: Сразу фильтруем по штату CA и категориям, чтобы не тянуть лишнее в память.
    """
    try:
        sales = pd.read_csv('sales_train_evaluation.csv')
        sales = sales[(sales['state_id'] == 'CA') & (sales['cat_id'].isin(['FOODS', 'HOBBIES']))]

        calendar = pd.read_csv('calendar.csv')
        prices = pd.read_csv('sell_prices.csv')

        # Препроцессинг календаря
        calendar['date'] = pd.to_datetime(calendar['date'])
        calendar['event_name_1_raw'] = calendar['event_name_1'].fillna('Нет события')
        calendar['event_name_1'] = calendar['event_name_1'].fillna('No_event')
        calendar['event_code'] = calendar['event_name_1'].astype('category').cat.codes
        # Downcasting для экономии памяти
        calendar['snap_CA'] = calendar['snap_CA'].fillna(0).astype('int8')
        calendar['month'] = calendar['date'].dt.month
        calendar['weekday_name'] = calendar['date'].dt.day_name()

        # Важно: season создаем здесь, чтобы он был доступен во всех вкладках
        calendar['season'] = calendar['month'].apply(get_season)

        prices['sell_price'] = prices['sell_price'].astype('float32')

        return sales, calendar, prices
    except FileNotFoundError:
        st.error("Ошибка: Файлы данных не найдены.")
        st.stop()


@st.cache_data(show_spinner=False)
def process_data(sales_df, calendar_df, prices_df, limit_skus):
    """
    Трансформация данных (Wide to Long).
    Выполняется при изменении слайдера 'Объем SKU'.
    ОПТИМИЗАЦИЯ: Используем float32 и сборщик мусора (gc) для предотвращения OOM (Out Of Memory).
    """
    # 1. Сэмплирование (для MVP работаем с подмножеством)
    if len(sales_df) > limit_skus:
        subset_sales = sales_df.sample(n=limit_skus, random_state=42).copy()
    else:
        subset_sales = sales_df.copy()

    # 2. MELT (Unpivot) - превращаем 1900 колонок дней в строки
    day_cols = [c for c in subset_sales.columns if c.startswith('d_')]
    id_cols = [c for c in subset_sales.columns if c not in day_cols]

    long_sales = subset_sales.melt(id_vars=id_cols, value_vars=day_cols, var_name='d', value_name='sales')

    # Сразу приводим типы
    long_sales['sales'] = long_sales['sales'].fillna(0).astype('float32')

    # 3. Обогащение (Merge)
    data = long_sales.merge(calendar_df[['d', 'date', 'wm_yr_wk', 'event_code', 'snap_CA']], on='d', how='left')
    data = data.merge(prices_df, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')

    # 4. Feature Engineering
    data = data.sort_values(['item_id', 'store_id', 'date'])
    data['sell_price'] = data.groupby(['item_id', 'store_id'])['sell_price'].ffill().fillna(0).astype('float32')
    # Изменение цены - важный признак эластичности спроса
    data['price_change'] = data.groupby('id')['sell_price'].pct_change(periods=7).replace([np.inf, -np.inf], 0).fillna(
        0).astype('float32')

    # Переименование для Nixtla (требует колонки ds, y, unique_id)
    data = data.rename(columns={'date': 'ds', 'sales': 'y'})
    data['unique_id'] = data['id']

    del long_sales
    gc.collect()

    return data


@st.cache_data
def get_future_features(train_df, calendar, prices, horizon=28, start_date=None):
    """Генерация фичей (цены, календарь) для будущего периода."""
    last_date = start_date if start_date else train_df['ds'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon)
    unique_ids = train_df['unique_id'].unique()

    future_df = pd.DataFrame([(uid, date) for uid in unique_ids for date in future_dates], columns=['unique_id', 'ds'])
    future_df = future_df.merge(calendar, left_on='ds', right_on='date', how='left').drop(columns='date')

    ref_sku = train_df[['unique_id', 'store_id', 'item_id']].drop_duplicates()
    future_df = future_df.merge(ref_sku, on='unique_id', how='left')
    future_df = future_df.merge(prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')

    future_df = future_df.sort_values(['unique_id', 'ds'])
    # Предполагаем, что цена в будущем = последней известной цене
    future_df['sell_price'] = future_df.groupby('unique_id')['sell_price'].ffill().fillna(0).astype('float32')
    future_df['price_change'] = 0.0

    return future_df


@st.cache_data
def segment_demand(df):
    """
    Сегментация временных рядов (Smooth, Intermittent, Lumpy, Erratic).
    Используется ADI (интервал между продажами) и CV2 (вариативность).
    """
    stats = df.groupby('unique_id')['y'].agg(
        mean='mean', std='std',
        nonzero_count=lambda x: (x > 0).sum(), total_count='count'
    ).reset_index()

    stats = stats[stats['total_count'] > 0]
    stats['adi'] = stats['total_count'] / stats['nonzero_count'].replace(0, 1)
    stats['cv2'] = (stats['std'] / stats['mean'].replace(0, np.nan)) ** 2

    def classify(row):
        if row['mean'] == 0: return 'Отсутствует (Cold Start)'
        if row['adi'] < 1.32 and row['cv2'] < 0.49: return 'Стабильный'
        if row['adi'] >= 1.32 and row['cv2'] < 0.49: return 'Эпизодический'
        if row['adi'] < 1.32 and row['cv2'] >= 0.49: return 'Вариативный'
        return 'Спорадический'

    stats['demand_type'] = stats.apply(classify, axis=1)
    return stats


# ==========================================
# 2. МОДЕЛИРОВАНИЕ (FORECASTING ENGINE)
# ==========================================

@st.cache_resource
def train_forecast_models(train_df, future_exog_df, horizon=28, model_type='Машинное обучение (XGBoost)'):
    """
    Bottom-Up прогнозирование.
    Строит прогноз для каждого SKU отдельно.
    """
    active_ids = train_df.groupby('unique_id')['y'].sum().loc[lambda s: s > 0].index
    train_clean = train_df[train_df['unique_id'].isin(active_ids)].copy()
    if train_clean.empty: return pd.DataFrame()

    train_cols = ['unique_id', 'ds', 'y', 'sell_price', 'price_change', 'event_code', 'snap_CA']
    df_nixtla = train_clean[train_cols].copy()

    forecasts = None

    if model_type == 'Статистический (AutoARIMA)':
        # Классика: AutoARIMA с учетом недельной сезонности
        models = [SeasonalNaive(season_length=7), AutoARIMA(season_length=7, approximation=True)]
        sf = StatsForecast(models=models, freq='D', n_jobs=1)
        sf.fit(df_nixtla[['unique_id', 'ds', 'y']])
        forecasts = sf.predict(h=horizon).rename(columns={'AutoARIMA': 'y_pred'})

    elif model_type == 'Машинное обучение (XGBoost)':
        # ML: XGBoost с лаговыми признаками
        fcst = MLForecast(
            models=[xgb.XGBRegressor(
                verbosity=0,
                n_estimators=200,
                learning_rate=0.04,
                max_depth=9,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=1
            )],
            freq='D',
            lags=[1, 2, 3, 4, 5, 6, 7, 14, 28, 365],
            lag_transforms={7: [(rolling_mean, 7)], 28: [(rolling_mean, 28)]},
            date_features=['dayofweek', 'month', 'day'],
            target_transforms=[Differences([7])]  # Убираем тренд/сезонность
        )
        fcst.fit(df_nixtla, static_features=[])

        future_subset = future_exog_df[future_exog_df['unique_id'].isin(active_ids)]
        pred_cols = ['unique_id', 'ds', 'sell_price', 'price_change', 'event_code', 'snap_CA']
        forecasts = fcst.predict(horizon, X_df=future_subset[pred_cols]).rename(columns={'XGBRegressor': 'y_pred'})

    return forecasts


@st.cache_resource
def run_top_down(train_df, horizon=28):
    """
    Top-Down прогнозирование.
    1. Агрегируем продажи всех товаров.
    2. Прогнозируем общий ряд.
    3. Распределяем (дезагрегируем) по историческим долям.
    ВАЖНО: Учтена проблема Data Leakage - веса считаются только по прошлому.
    """
    agg_train = train_df.groupby('ds')['y'].sum().reset_index()
    agg_train['unique_id'] = 'Total_Category'
    sf = StatsForecast(models=[AutoARIMA(season_length=7, approximation=True)], freq='D', n_jobs=1)
    sf.fit(agg_train)
    agg_forecast = sf.predict(h=horizon)

    # Расчет весов по последним 28 дням ИЗ ОБУЧАЮЩЕЙ выборки (исключаем заглядывание в будущее)
    max_train_date = train_df['ds'].max()
    start_weight_calc = max_train_date - pd.Timedelta(days=28)
    recent_sales = train_df[train_df['ds'] > start_weight_calc]

    total_sales = recent_sales['y'].sum()
    if total_sales == 0: return pd.DataFrame()

    # Доля каждого SKU в общих продажах
    sku_weights = recent_sales.groupby('unique_id')['y'].sum() / total_sales
    sku_weights = sku_weights.reset_index(name='weight')

    agg_forecast_clean = agg_forecast[['ds', 'AutoARIMA']]
    distributed_forecast = agg_forecast_clean.merge(sku_weights, how='cross')
    distributed_forecast['y_pred'] = distributed_forecast['AutoARIMA'] * distributed_forecast['weight']
    return distributed_forecast[['unique_id', 'ds', 'y_pred']]


def calculate_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = np.sum(np.abs(y_true))
    # WAPE (Weighted Absolute Percentage Error) - бизнес-метрика
    wape = np.sum(np.abs(y_true - y_pred)) / denom if denom != 0 else 0
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return wape, mae, rmse


# ==========================================
# 3. БИЗНЕС-ЛОГИКА (ARM / Управление рисками)
# ==========================================

def determine_reason(row):
    """Эвристика для определения причины отклонения."""
    if pd.notna(row['event_name_1_raw']) and row['event_name_1_raw'] != 'Нет события':
        return 'Событие'
    elif row['weekday_name'] in ['Saturday', 'Sunday']:
        return 'Сезонность'
    return 'Не выявлено'


def generate_recommendation(row):
    """Автоматическая генерация действия для планера."""
    diff = row['diff']
    if diff > 5:
        return f"Снизить закупку (-{diff:.0f} шт.)"
    elif diff < -5:
        return f"Увеличить закупку (+{abs(diff):.0f} шт.)"
    return "В норме"


def get_status(diff):
    return "Дефицит" if diff < 0 else "Профицит"


def recommend_model(demand_type):
    # Для стабильного спроса классика часто лучше и быстрее
    if demand_type in ['Стабильный', 'Эпизодический']: return 'AutoARIMA'
    return 'XGBoost'


def recommend_method(demand_type):
    # Top-Down сглаживает шум, хорошо для стабильных товаров
    if demand_type in ['Стабильный', 'Эпизодический']: return 'Top-Down'
    return 'Bottom-Up'


# ==========================================
# 4. ГЛАВНЫЙ ИНТЕРФЕЙС (UI/UX)
# ==========================================

def main():
    st.title("Demand Planning ARM (Automated Risk Management)")

    # --- 1. ЗАГРУЗКА И КОНФИГУРАЦИЯ ---
    st.sidebar.header("1. Конфигурация")

    with st.spinner('Чтение файлов данных...'):
        raw_sales, calendar, prices = load_raw_files()

    limit_selection = st.sidebar.slider("Объем SKU:", 100, 3000, 500, 100)

    with st.spinner(f'Обработка {limit_selection} SKU...'):
        full_data = process_data(raw_sales, calendar, prices, limit_selection)
        min_date, max_available_date = full_data['ds'].min(), full_data['ds'].max()

        mode = st.sidebar.radio("Режим работы:", ["Бэктестинг (Тест)", "Прогноз (Future)"])
        test_horizon = st.sidebar.slider("Горизонт (дней):", 7, 28, 28)

        if mode == "Бэктестинг (Тест)":
            default_start = max_available_date - pd.Timedelta(days=test_horizon - 1)
            min_selectable = min_date + pd.Timedelta(days=90)
            st.sidebar.subheader("Параметры теста")
            user_start_date = st.sidebar.date_input("Дата начала теста:", value=default_start, min_value=min_selectable,
                                                    max_value=default_start)

            test_start = pd.to_datetime(user_start_date)
            test_end = test_start + pd.Timedelta(days=test_horizon - 1)

            # Разделение на Train и Test (Holdout)
            train = full_data[full_data['ds'] < test_start].copy()
            test = full_data[(full_data['ds'] >= test_start) & (full_data['ds'] <= test_end)].copy()
            future_exog = get_future_features(train, calendar, prices, horizon=test_horizon)

        else:
            st.sidebar.success("Режим реального прогноза")
            train = full_data.copy()
            test = pd.DataFrame()
            future_exog = get_future_features(train, calendar, prices, horizon=test_horizon,
                                              start_date=max_available_date)

        segmentation_df = segment_demand(train)
        train = train.merge(segmentation_df[['unique_id', 'demand_type', 'std']], on='unique_id', how='left')

    # --- 2. ФИЛЬТРЫ ---
    st.sidebar.header("2. Фильтры")
    available_stores = sorted(list(train['store_id'].unique()))
    if 'selected_store_idx' not in st.session_state: st.session_state.selected_store_idx = 0

    all_stores = ['Все магазины'] + available_stores
    selected_store = st.sidebar.selectbox("Магазин:", all_stores)

    selected_cat = st.sidebar.selectbox("Категория:", train['cat_id'].unique())
    available_segments = sorted(train['demand_type'].unique())
    selected_segments = st.sidebar.multiselect("Сегменты:", available_segments, default=available_segments)

    # Фильтрация датасета
    train_subset = train[(train['cat_id'] == selected_cat) & (train['demand_type'].isin(selected_segments))]

    if selected_store != 'Все магазины':
        train_subset = train_subset[train_subset['store_id'] == selected_store]
        if not test.empty: test = test[test['store_id'] == selected_store]
        future_exog = future_exog[
            future_exog['store_id'] == selected_store] if 'store_id' in future_exog.columns else future_exog

    if train_subset.empty:
        st.error("Нет данных для выбранных фильтров.")
        st.stop()

    valid_ids = train_subset['unique_id'].unique()
    if not test.empty:
        test_subset = test[test['unique_id'].isin(valid_ids)]
    else:
        test_subset = pd.DataFrame()
    future_exog_subset = future_exog[future_exog['unique_id'].isin(valid_ids)]

    st.sidebar.markdown("---")
    model_option = st.sidebar.radio("Алгоритм:", ('Машинное обучение (XGBoost)', 'Статистический (AutoARIMA)'))

    if mode == "Бэктестинг (Тест)":
        tabs = st.tabs(["📊 Аналитика", "📈 Прогноз vs Факт", "🚨 ARM (Аномалии)"])
    else:
        tabs = st.tabs(["📊 Аналитика", "🔮 Будущий Прогноз"])

    # --- TAB 1: АНАЛИТИКА ---
    with tabs[0]:
        st.subheader("Здоровье портфеля")
        high_risk_count = len(segmentation_df[(segmentation_df['unique_id'].isin(valid_ids)) & (
            segmentation_df['demand_type'].isin(['Спорадический', 'Вариативный']))])
        c1, c2 = st.columns(2)
        c1.metric("SKU в работе", len(valid_ids))
        c2.metric("Сложные SKU", high_risk_count)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.pie(segmentation_df[segmentation_df['unique_id'].isin(valid_ids)], names='demand_type',
                                   title="Сегментация", hole=0.4), use_container_width=True)
        with col2:
            train_subset['weekday'] = pd.to_datetime(train_subset['ds']).dt.day_name()
            daily = train_subset.groupby('weekday')['y'].mean().reindex(
                ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']).reset_index()
            st.plotly_chart(px.bar(daily, x='weekday', y='y', title="Сезонность"), use_container_width=True)

    # --- РАСЧЕТ ПРОГНОЗА ---
    hierarchy_method = st.sidebar.radio("Метод:", ["Bottom-Up (SKU)", "Top-Down (Агрегат)"], horizontal=True)
    forecast_df = pd.DataFrame()

    with st.spinner("Расчет прогноза..."):
        if hierarchy_method == "Bottom-Up (SKU)":
            forecast_df = train_forecast_models(train_subset, future_exog_subset, horizon=test_horizon,
                                                model_type=model_option)
        else:
            forecast_df = run_top_down(train_subset, horizon=test_horizon)

    if not forecast_df.empty:
        # --- TAB 2: ВИЗУАЛИЗАЦИЯ ---
        with tabs[1]:
            st.subheader("Визуализация")

            # Подготовка данных
            if mode == "Бэктестинг (Тест)":
                merged_res = test_subset.merge(forecast_df, on=['unique_id', 'ds'], how='left')
                merged_res['y_pred'] = merged_res['y_pred'].fillna(0).clip(lower=0)
            else:
                merged_res = forecast_df.copy()

            # Обогащение календарем для фильтров
            merged_res = merged_res.merge(calendar[['date', 'event_name_1_raw', 'season', 'weekday_name']],
                                          left_on='ds', right_on='date', how='left')

            # --- Фильтры отображения ---
            with st.expander("Фильтры отображения"):
                f1, f2, f3 = st.columns(3)
                avail_seasons = sorted([x for x in merged_res['season'].unique() if pd.notna(x)])
                avail_wdays = [x for x in merged_res['weekday_name'].unique() if pd.notna(x)]
                avail_events = [x for x in merged_res['event_name_1_raw'].unique() if pd.notna(x)]

                sel_season = f1.multiselect("Сезон", avail_seasons, default=avail_seasons)
                sel_wday = f2.multiselect("День недели", avail_wdays, default=avail_wdays)
                sel_event = f3.multiselect("События", avail_events, default=avail_events)

            final_season = sel_season if sel_season else avail_seasons
            final_wday = sel_wday if sel_wday else avail_wdays
            final_event = sel_event if sel_event else avail_events

            filtered_res = merged_res[
                (merged_res['season'].isin(final_season)) &
                (merged_res['weekday_name'].isin(final_wday)) &
                (merged_res['event_name_1_raw'].isin(final_event))
                ]

            # --- Выбор уровня агрегации ---
            level_agg = f"{selected_store} -> {selected_cat} (Итого)"
            level_sku = f"{selected_store} -> {selected_cat} -> SKU"
            view_level = st.selectbox("Уровень:", [level_agg, level_sku])

            if view_level == level_agg:
                if not filtered_res.empty:
                    agg_pred = filtered_res.groupby('ds')['y_pred'].sum()

                    if mode == "Бэктестинг (Тест)":
                        agg_fact = filtered_res.groupby('ds')['y'].sum()
                        wape, mae, rmse = calculate_metrics(agg_fact, agg_pred)
                        c1, c2, c3 = st.columns(3)
                        c1.metric("WAPE", f"{wape:.2%}")
                        c2.metric("MAE", f"{mae:.0f}")
                        c3.metric("RMSE", f"{rmse:.0f}")

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=agg_fact.index, y=agg_fact, name='Факт', line=dict(color='green')))
                        fig.add_trace(go.Scatter(x=agg_pred.index, y=agg_pred, name='Прогноз',
                                                 line=dict(color='blue', dash='dot')))
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        fig = go.Figure()
                        hist_agg = \
                        train_subset[train_subset['ds'] > train_subset['ds'].max() - pd.Timedelta(days=60)].groupby(
                            'ds')['y'].sum()
                        fig.add_trace(go.Scatter(x=hist_agg.index, y=hist_agg, name='История', line=dict(color='gray')))
                        fig.add_trace(go.Scatter(x=agg_pred.index, y=agg_pred, name='Будущий Прогноз',
                                                 line=dict(color='blue', dash='dot')))
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Нет данных для выбранных фильтров")

            else:  # SKU Level
                if not filtered_res.empty:
                    sku_list = filtered_res['unique_id'].unique()
                    sel_sku = st.selectbox("SKU:", sku_list)
                    sku_data = filtered_res[filtered_res['unique_id'] == sel_sku]

                    if mode == "Бэктестинг (Тест)":
                        swape, smae, srmse = calculate_metrics(sku_data['y'], sku_data['y_pred'])
                        c1, c2, c3 = st.columns(3)
                        c1.metric("WAPE", f"{swape:.2%}")
                        c2.metric("MAE", f"{smae:.2f}")
                        c3.metric("RMSE", f"{srmse:.2f}")

                        fig = go.Figure()
                        fig.add_trace(
                            go.Scatter(x=sku_data['ds'], y=sku_data['y'], name='Факт', line=dict(color='green')))
                        fig.add_trace(go.Scatter(x=sku_data['ds'], y=sku_data['y_pred'], name='Прогноз',
                                                 line=dict(color='blue', dash='dot')))
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        fig = go.Figure()
                        sku_hist = train_subset[(train_subset['unique_id'] == sel_sku) & (
                                    train_subset['ds'] > train_subset['ds'].max() - pd.Timedelta(days=60))]
                        fig.add_trace(
                            go.Scatter(x=sku_hist['ds'], y=sku_hist['y'], name='История', line=dict(color='gray')))
                        fig.add_trace(go.Scatter(x=sku_data['ds'], y=sku_data['y_pred'], name='Будущий Прогноз',
                                                 line=dict(color='blue', dash='dot')))
                        st.plotly_chart(fig, use_container_width=True)

        # --- TAB 3: ARM (Exception Management) ---
        if mode == "Бэктестинг (Тест)":
            with tabs[2]:
                st.subheader("Монитор отклонений")

                # Добавляем сегментацию для логики рекомендаций
                merged_res = merged_res.merge(segmentation_df[['unique_id', 'demand_type', 'std']], on='unique_id',
                                              how='left')

                merged_res['diff'] = merged_res['y_pred'] - merged_res['y']
                merged_res['abs_error'] = merged_res['diff'].abs()

                # Правило 3 сигм для определения критичности
                merged_res['is_critical'] = (merged_res['abs_error'] > 3 * merged_res['std']) & (
                        merged_res['abs_error'] > 2)

                critical = merged_res[merged_res['is_critical']].copy()
                non_critical = merged_res[~merged_res['is_critical'] & (merged_res['abs_error'] > 2)].copy()

                for df in [critical, non_critical]:
                    if not df.empty:
                        df['store_id'] = df['unique_id'].str.extract(r'_(CA_\d)_')
                        df['status'] = df['diff'].apply(get_status)
                        df['reason'] = df.apply(determine_reason, axis=1)
                        df['Рекомендация'] = df.apply(generate_recommendation, axis=1)
                        # Экспертные советы системы на основе сегментации
                        df['Совет (Модель)'] = df['demand_type'].apply(recommend_model)
                        df['Совет (Метод)'] = df['demand_type'].apply(recommend_method)
                        df['Тек. Модель'] = model_option.split(' ')[0]
                        df['Тек. Метод'] = hierarchy_method.split(' ')[0]

                unique_crit_skus = critical['unique_id'].nunique() if not critical.empty else 0
                st.error(
                    f"🔴 Критические отклонения:")

                if not critical.empty:
                    # Интерактивная таблица с возможностью редактирования причины
                    cols = ['unique_id', 'ds', 'y', 'y_pred', 'abs_error', 'reason', 'event_name_1_raw', 'Рекомендация',
                            'Совет (Модель)',
                            'Совет (Метод)']
                    edited_critical = st.data_editor(
                        critical[cols].sort_values('abs_error', ascending=False).head(50),
                        column_config={
                            "reason": st.column_config.SelectboxColumn("Причина",
                                                                       options=["Сезонность", "Событие", "Аномалия",
                                                                                "Out-of-Stock", "Не выявлено"],
                                                                       required=True),
                            "event_name_1_raw": st.column_config.TextColumn("Событие (Инфо)", disabled=True),
                            "y_pred": st.column_config.NumberColumn("Прогноз", format="%.2f"),
                            "abs_error": st.column_config.NumberColumn("Ошибка", format="%.2f")
                        },
                        disabled=['unique_id', 'ds', 'y', 'Рекомендация', 'Совет (Модель)', 'Совет (Метод)',
                                  'event_name_1_raw'],
                        use_container_width=True,
                        key="critical_editor"
                    )
                else:
                    st.success("Критических сбоев не обнаружено.")

                unique_nc_skus = non_critical['unique_id'].nunique() if not non_critical.empty else 0
                st.warning(f"🟠 Некритические отклонения:")

                if not non_critical.empty:
                    cols_nc = ['unique_id', 'ds', 'y', 'y_pred', 'abs_error', 'reason', 'event_name_1_raw',
                               'Совет (Модель)',
                               'Совет (Метод)']
                    st.dataframe(
                        non_critical[cols_nc].sort_values('abs_error', ascending=False).head(50),
                        use_container_width=True
                    )


if __name__ == "__main__":
    main()