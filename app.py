# app.py
import sys
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# 0) Streamlit-setup
# -----------------------------
st.set_page_config(page_title="Bilpris-prediktor (motorstørrelse)", page_icon="🚗", layout="centered")
st.title("🚗 Bilpris-prediktor — motorstørrelse som nøkkelfeature")
st.caption(f"Python: {sys.version.split()[0]}  •  Venv: {Path(sys.executable).parent}")

# ------------------------------------------------------------
# 1) Egendefinerte trafo-funksjoner (må finnes ved unpickling)
#    Disse ble (trolig) brukt i treningen via FunctionTransformer.
#    IKKE bruk lambda i treningsnotatbok neste gang – bruk navngitte funksjoner.
# ------------------------------------------------------------
def is_accident_reported(X):
    """
    Returnerer (n_samples, 1) array med 1 hvis verdi != 'None reported', ellers 0.

    """
    if hasattr(X, "astype"):  # pandas Series/DataFrame
        x = pd.DataFrame(X).astype(str).apply(lambda s: s.str.lower()).to_numpy().reshape(-1)
    else:
        x = np.array(X).astype(str).reshape(-1)
        x = np.char.lower(x)
    return (x != "none reported").astype(int).reshape(-1, 1)

def clean_title_yes(X):
    """
    Returnerer (n_samples, 1) array med 1 hvis verdi == 'Yes', ellers 0.
    """
    if hasattr(X, "astype"):
        x = pd.DataFrame(X).astype(str).apply(lambda s: s.str.lower()).to_numpy().reshape(-1)
    else:
        x = np.array(X).astype(str).reshape(-1)
        x = np.char.lower(x)
    return (x == "yes").astype(int).reshape(-1, 1)

# ------------------------------------------------------------
# 2) Finn og last modellen robust
# ------------------------------------------------------------
HERE = Path(__file__).resolve().parent
candidates = [
    HERE / "models" / "engine_ridge.pkl",   # din lagrede modell fra notatboka
    HERE / "models" / "motorstrelse.pkl",   # fallback hvis du har brukt dette navnet
]
MODEL_PATH = next((p for p in candidates if p.exists()), None)

if MODEL_PATH is None:
    st.error(
        "Fant ikke modellfil. Legg modellen i `models/engine_ridge.pkl` "
        "eller oppdater filbanen i appen."
    )
    st.stop()

st.caption(f"Modellfil: {MODEL_PATH}")

# Prøv å laste (fanger manglende avhengigheter pent)
try:
    loaded = joblib.load(MODEL_PATH.open("rb"))
except ModuleNotFoundError as e:
    st.error(
        f"Avhengighet mangler ved lasting av modellen: {e}\n\n"
        r"Kjør i terminal: .\.venv\Scripts\python -m pip install scikit-learn numpy scipy joblib"
    )
    st.stop()
except Exception as e:
    st.error(f"Kunne ikke laste modellen: {e}")
    st.stop()

# Støtt både bundle {'pipeline': ..., 'anchor_year': ...} og ren pipeline
if isinstance(loaded, dict) and "pipeline" in loaded:
    pipe = loaded["pipeline"]
    ANCHOR_YEAR = loaded.get("anchor_year", 2025)
else:
    pipe = loaded
    ANCHOR_YEAR = 2025  # fallback hvis ikke lagret i bundle



# ------------------------------------------------------------
# 3) Inndata (må matche feature-navn fra treningen)
#    Numeriske: engine_size, model_year, milage, age
#    Kategoriske: fuel_type, transmission, ext_base, int_base
#    Tekst (binariseres i pipeline): accident, clean_title
# ------------------------------------------------------------
with st.sidebar:
    st.header("🔧 Inndata")
    engine_size = st.number_input("Motorstørrelse (Liter)", min_value=0.5, max_value=10.0, value=2.0, step=0.1)
    model_year = st.number_input("Modellår", min_value=1980, max_value=int(ANCHOR_YEAR), value=2018, step=1)
    milage = st.number_input("Kilometerstand (milage)", min_value=0, max_value=1_500_000, value=60000, step=1000)

    fuel_type = st.selectbox("Drivstoff", ["Gasoline", "Diesel", "Hybrid", "Electric", "E85 Flex Fuel", "Other"], index=0)
    transmission = st.selectbox("Girkasse", ["Automatic", "Manual", "CVT", "A/T", "8-Speed A/T", "10-Speed Automatic", "Other"], index=0)
    ext_base = st.selectbox("Utvendig farge (basis)", ["Black","White","Gray","Silver","Blue","Red","Green","Brown","Beige","Yellow","Other"], index=2)
    int_base = st.selectbox("Innvendig farge (basis)", ["Black","White","Gray","Silver","Blue","Red","Green","Brown","Beige","Yellow","Other"], index=0)
    accident = st.selectbox("Skadehistorikk", ["None reported", "At least 1 accident or damage reported", "Other"], index=0)
    clean_title = st.selectbox("Clean title?", ["Yes", "No"], index=0)

age = int(ANCHOR_YEAR) - int(model_year)

row = pd.DataFrame([{
    "engine_size": float(engine_size),
    "model_year": int(model_year),
    "milage": int(milage),
    "fuel_type": fuel_type,
    "transmission": transmission,
    "ext_base": ext_base,
    "int_base": int_base,
    "age": age,
    "accident": accident,
    "clean_title": clean_title,
}])

st.subheader("📦 Inndata brukt for prediksjon")
st.dataframe(row, use_container_width=True)

# ------------------------------------------------------------
# 4) Prediksjon
# ------------------------------------------------------------
st.divider()
if st.button("🔮 Estimer pris"):
    try:
        yhat = float(pipe.predict(row)[0])
        yhat = max(0.0, yhat)  # liten sikring mot negative estimat
        st.success(f"Estimert pris: **${yhat:,.0f}**")
        st.caption("Merk:"
                   " Bruk estimatet veiledende.")
    except Exception as e:
        st.error(f"Prediksjon feilet: {e}")
        st.info("Sjekk at kolonnenavn matcher det modellen ble trent med, og at pipeline håndterer tekst/encoding som forventet.")

st.divider()
st.caption("Tips: Endre motorstørrelse og sammenlign estimatene.")
