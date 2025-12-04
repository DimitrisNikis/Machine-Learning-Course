import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import requests
import io


st.set_page_config(page_title="Car Price ML App", layout="wide")

@st.cache_resource
def load_artifacts_from_url():
    url = "https://raw.githubusercontent.com/DimitrisNikis/Machine-Learning-Course/main/HW/HW01/car_price_model.pkl"
    response = requests.get(url)
    response.raise_for_status()
    buffer = io.BytesIO(response.content)
    artifacts = pickle.load(buffer)
    return artifacts


@st.cache_data
def load_train_data():
    url = "https://raw.githubusercontent.com/DimitrisNikis/Machine-Learning-Course/main/HW/HW01/train_EDA.csv"
    df = pd.read_csv(url)
    return df


artifacts = load_artifacts_from_url()
model = artifacts["model"]
ohe = artifacts["ohe"]
scaler = artifacts["scaler"]
num_cols = artifacts["num_cols"]
cat_cols = artifacts["cat_cols"]

df_raw = load_train_data()

st.title("Предсказание стоимости автомобилей")

st.markdown(
    """
- EDA по данным автомобилей  
- Предсказание цены по загруженному CSV или ручному вводу  
- Визуализация весов обученной модели (Ridge)
"""
)


def full_preprocess(df: pd.DataFrame):
    df = df.copy()

    drop_cols = ["name", "torque"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    expected_cols = set(num_cols + cat_cols)
    missing = expected_cols - set(df.columns)

    if missing:
        raise ValueError(f"Отсутствуют нужные колонки: {missing}")

    if df["mileage"].dtype == "object":
        df["mileage"] = df["mileage"].str.extract(r"(\d+\.?\d*)").astype(float)

    if df["engine"].dtype == "object":
        df["engine"] = df["engine"].str.extract(r"(\d+\.?\d*)").astype(float)

    if df["max_power"].dtype == "object":
        df["max_power"] = df["max_power"].str.extract(r"(\d+\.?\d*)").astype(float)

    df["seats"] = df["seats"].astype(int)

    X_num = df[num_cols].astype(float)
    
    X_cat = df[cat_cols].copy()

    for i, col in enumerate(cat_cols):
        trained_types = type(ohe.categories_[i][0])

        if np.issubdtype(trained_types, np.integer):
            X_cat[col] = X_cat[col].astype(int)
        else:
            X_cat[col] = X_cat[col].astype(str)

    X_cat_ohe = ohe.transform(X_cat)
    ohe_feature_names = ohe.get_feature_names_out(cat_cols)
    X_cat_ohe = pd.DataFrame(X_cat_ohe, index=df.index, columns=ohe_feature_names)

    X_final = pd.concat([X_num, X_cat_ohe], axis=1)

    X_scaled = scaler.transform(X_final)

    return X_scaled, X_final.columns


def predict_df(df_features: pd.DataFrame) -> pd.Series:
    X_scaled, feature_names = full_preprocess(df_features)
    y_pred = model.predict(X_scaled)
    return pd.Series(y_pred, index=df_features.index)


def get_coefficients():
    ohe_feature_names = ohe.get_feature_names_out(cat_cols)
    feature_names = list(num_cols) + list(ohe_feature_names)
    coefs = model.coef_
    coef_df = pd.DataFrame(
        {"feature": feature_names, "coef": coefs, "abs_coef": np.abs(coefs)}
    ).sort_values("abs_coef", ascending=False)
    return coef_df


tab_eda, tab_predict, tab_weights = st.tabs(["EDA", "Предсказание", "Веса модели"])

with tab_eda:
    st.header("Исследовательский анализ данных")

    st.subheader("Общая информация")
    st.write(df_raw.head())

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Распределение цены**")
        fig, ax = plt.subplots()
        sns.histplot(df_raw["selling_price"], bins=30, ax=ax)
        ax.set_xlabel("Selling price")
        st.pyplot(fig)

        st.markdown("**Boxplot цены по типу топлива**")
        fig, ax = plt.subplots()
        sns.boxplot(data=df_raw, x="fuel", y="selling_price", ax=ax)
        st.pyplot(fig)

    with col2:
        st.markdown("**Распределение года выпуска**")
        fig, ax = plt.subplots()
        sns.histplot(df_raw["year"], bins=20, ax=ax)
        ax.set_xlabel("Year")
        st.pyplot(fig)

        st.markdown("**Цена по типу коробки передач по годам**")
        fig, ax = plt.subplots()
        sns.scatterplot(
            data=df_raw, x="year", y="selling_price", hue="transmission", alpha=0.4
        )
        ax.set_xlabel("Year")
        st.pyplot(fig)

with tab_predict:
    st.header("Предсказание цены")

    mode = st.radio(
        "Выберите режим ввода данных:", ("Загрузить CSV", "Ручной ввод признаков")
    )

    if mode == "Загрузить CSV":
        st.write("CSV должен содержать столбцы:", ", ".join(num_cols + cat_cols))

        uploaded_file = st.file_uploader(
            "Загрузите CSV с признаками автомобилей", type="csv"
        )

        if uploaded_file is not None:
            try:
                df_new = pd.read_csv(uploaded_file)
            except Exception:
                st.error("Ошибка: файл не является корректным CSV!")
                st.stop()
                
            st.write("Загруженные данные:")
            st.dataframe(df_new.head())
            
            if df_new.isna().any().any():
                st.warning("Обнаружены пропуски в данных — строки с NaN будут удалены.")
                df_new = df_new.dropna()
            
            required_cols = set(num_cols + cat_cols)
            missing = required_cols - set(df_new.columns)

            if missing:
                st.error(f"В CSV не хватает колонок: {missing}")
                st.stop()

            if st.button("Сделать предсказания"):
                preds = predict_df(df_new)
                result = df_new.copy()
                result["predicted_price"] = preds
                st.success("Предсказания готовы!")
                st.dataframe(result.head())
    else:
        st.subheader("Ручной ввод признаков")

        col1, col2 = st.columns(2)

        with col1:
            year = st.number_input(
                "Год выпуска (year)", min_value=1983, max_value=2020, value=2015
            )
            km_driven = st.number_input(
                "Пробег (km_driven)", min_value=1, max_value=236000, value=60000
            )
            mileage = st.number_input(
                "Расход (mileage, kmpl)", min_value=0.0, max_value=42.0, value=18.0
            )
        with col2:
            engine = st.number_input(
                "Объем двигателя (engine, cc)",
                min_value=625,
                max_value=3600,
                value=1200,
            )
            max_power = st.number_input(
                "Max power (bhp)", min_value=0.0, max_value=400.0, value=80.0
            )
            seats = st.number_input(
                "Количество мест (seats)", min_value=2, max_value=14, value=4
            )

        fuel = st.selectbox("Тип топлива (fuel)", ["Petrol", "Diesel", "CNG", "LPG"])
        seller_type = st.selectbox(
            "Тип продавца (seller_type)", ["Individual", "Dealer", "Trustmark Dealer"]
        )
        transmission = st.selectbox(
            "Коробка передач (transmission)", ["Manual", "Automatic"]
        )
        owner = st.selectbox(
            "Владельцы (owner)",
            [
                "First Owner",
                "Second Owner",
                "Third Owner",
                "Fourth & Above Owner",
                "Test Drive Car",
            ],
        )

        if st.button("Предсказать цену"):
            df_one = pd.DataFrame(
                [
                    {
                        "year": year,
                        "km_driven": km_driven,
                        "mileage": mileage,
                        "engine": engine,
                        "max_power": max_power,
                        "seats": seats,
                        "fuel": fuel,
                        "seller_type": seller_type,
                        "transmission": transmission,
                        "owner": owner,
                    }
                ]
            )

            pred = predict_df(df_one)[0]
            st.success(f"Предсказанная цена: {pred:,.0f} рублей")

with tab_weights:
    st.header("Веса модели Ridge")

    coef_df = get_coefficients()

    st.write("Топ признаков по важности:")
    st.dataframe(coef_df)

    st.markdown("**Барплот по абсолютным значениям коэффициентов:**")
    top_n = st.slider(
        "Сколько признаков показать на графике?",
        min_value=5,
        max_value=coef_df.shape[0],
        value=15,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=coef_df.head(top_n), x="abs_coef", y="feature", ax=ax)
    ax.set_xlabel("|коэффициент|")
    ax.set_ylabel("Признак")
    st.pyplot(fig)
