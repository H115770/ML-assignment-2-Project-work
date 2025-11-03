
import streamlit as st
#  HEADER

st.set_page_config(page_title="Bruktbilpris-prediksjon", layout="centered")

import numpy as np
import pandas as pd
import joblib
import re
import json


# MODELL OG LISTE

@st.cache_resource
def load_model():
    return joblib.load("models/2.11_model.pkl")

@st.cache_data
def load_brand_list():
    with open("data/brand_list.json", "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_info():
    return joblib.load("data/app_info.pkl")

model = load_model()
info = load_info()
brand_models = load_brand_list()




st.title(" Bruktbilpris-prediksjon med XGBoost")
st.write(
    "Velg bilmodell, drivstofftype og spesifikasjoner for å estimere markedspris. "
    "Modellen er trent på amerikanske bruktbildata (USD), og prisen konverteres automatisk til NOK."
)


# INPUT-FELT

brand_model = st.selectbox(
    "Bilmerke og modell",
    brand_models,
    index=None,
    placeholder="Søk eller velg bilmodell (f.eks. Toyota_Corolla)..."
)


fuel_map = {
    "Bensin": "Gasoline",
    "Diesel": "Diesel",
    "Elektrisk": "Electric",
    "Hybrid": "Hybrid",
    "Annet": "Other"
}
fuel_display = st.selectbox("Drivstofftype", list(fuel_map.keys()))
fuel_type = fuel_map[fuel_display]

col1, col2 = st.columns(2)
with col1:
    model_year = st.slider("Årsmodell", 1990, 2025, 2018)
with col2:
    milage = st.slider("Kilometerstand (km)", 0, 500000, 100000, step=5000, format="%d")


engine_size = st.slider("Motorvolum (liter)", 0.5, 8.0, 2.0, step=0.1)


# PREDIKSJON

if st.button("🔮 Estimer pris"):

    #  Avledede og faste features
    car_age = 2025 - model_year
    km_per_year = milage / car_age if car_age > 0 else milage
    clean_title_val = 1
    accident_val = 0
    horsepower = 150
    transmission = "automatic"

    # Rens og tilpass brand_model
    brand_model = str(brand_model).strip().replace("_", " ")

    clean_name = re.sub(r'_+', ' ', brand_model).strip() if brand_model else ""
    parts = clean_name.split(" ", 1)
    brand = parts[0].strip() if len(parts) > 0 else ""
    model_name = parts[1].strip() if len(parts) > 1 else ""

    # Opprett DataFrame med ALLE features
    input_df = pd.DataFrame({
        "model_year": [model_year],
        "milage": [milage],
        "accident": [accident_val],
        "clean_title": [clean_title_val],
        "engine_size": [engine_size],
        "horsepower": [horsepower],
        "car_age": [car_age],
        "km_per_year": [km_per_year],
        "brand": [brand],
        "model": [model_name],
        "fuel_type": [fuel_type],
        "transmission": [transmission],
        "brand_model": [brand_model]
    })

    # Prediksjon log-pris → pris
    try:
        pred_log = model.predict(input_df)[0]
        prediction_usd = np.expm1(pred_log)
        usd_to_nok = 11.0
        prediction_nok = prediction_usd * usd_to_nok

        st.success(
            f" Estimert bruktbilpris: **{prediction_nok:,.0f} kr**  "
            f"(≈ {prediction_usd:,.0f} USD)"
        )
        st.caption("Prisen vises i norske kroner. Modellen er trent på amerikanske data (USD).")
    except Exception as e:
        st.error(f"Feil ved prediksjon: {e}")

    #Debugpanel
    with st.expander("📋 Data sendt til modellen"):
        st.write("Valgt bilmodell (etter formatering):", repr(brand_model))
        st.write("Drivstofftype sendt til modellen:", fuel_type)
        st.dataframe(input_df)



# FOOTER

st.markdown("---")
st.caption(

)