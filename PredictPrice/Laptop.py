import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

st.header("Laptop Fiyat Tahmin Ekranı")

# Sidebar menüsü
rad = st.sidebar.radio("Menu", ["Home", "RandomForestRegressor", "LinearRegression", "DecisionTreeRegressor", "GradientBoostingRegressor"])

if rad == "Home":
    st.subheader("Ana Sayfa")
    st.write("Lütfen sol taraftaki menüden bir model seçiniz.")
else:
    st.subheader(f"{rad} ile Fiyat Tahmini")

    laptop = pd.read_csv("C:/Users/emree/Desktop/PredictPrice/Laptop.csv")



    # Brand sütununu kategorikten sayısala çevirme
    brand_mapping = {"Asus": 0, "Acer": 1, "Lenovo": 2, "HP": 3, "Dell": 4}
    laptop["Brand"] = laptop["Brand"].map(brand_mapping)

    # Özellikler ve hedef değişken
    X = laptop.drop(["Price"], axis=1)
    y = laptop["Price"]

    # Veri setini eğitim ve test setlerine ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modelleri eğitme
    models = {
        "RandomForestRegressor": RandomForestRegressor(random_state=42),
        "LinearRegression": LinearRegression(),
        "DecisionTreeRegressor": DecisionTreeRegressor(random_state=42),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42)
    }

    model = models[rad]
    model.fit(X_train, y_train)

    # Kullanıcı girdileri
    brand = st.selectbox("Marka:", ('Asus', 'Acer', 'Lenovo', 'HP', 'Dell'))
    processor_speed = st.number_input("İşlemci Hızı (GHz):", min_value=1.5, max_value=4.0, step=0.1)
    ram_size = st.number_input("RAM Boyutu (GB):", min_value=4, max_value=32, step=1)
    storage_capacity = st.number_input("Depolama Kapasitesi (GB):", min_value=256, max_value=1000, step=1)
    screen_size = st.number_input("Ekran Boyutu (inç):", min_value=11.0, max_value=17.0, step=0.1)
    weight = st.number_input("Ağırlık (kg):", min_value=2.0, max_value=5.0, step=0.1)

    # Markayı sayısala çevirme
    brand = brand_mapping[brand]

    # Tahmin yapma
    if st.button("Fiyat Tahmini Yap"):
        input_data = [[brand, processor_speed, ram_size, storage_capacity, screen_size, weight]]
        prediction = model.predict(input_data)
        st.write(f"Tahmini Fiyat: {prediction[0]:.2f} TL")
