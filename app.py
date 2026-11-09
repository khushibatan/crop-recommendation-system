import sqlite3
import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# PAGE SETTINGS
# ----------------------------

st.set_page_config(
    page_title="Smart Crop Recommendation",
    page_icon="🌱",
    layout="centered"
)

DATABASE = "farm_history.db"
# ----------------------------
# LOAD MACHINE LEARNING MODEL
# ----------------------------

saved_model = joblib.load("crop_model.pkl")

model = saved_model["model"]
features = saved_model["features"]
# ----------------------------
# CROP PREDICTION FUNCTION
# ----------------------------

def predict_crops(n, p, k, temperature, humidity, ph, rainfall):

    input_data = pd.DataFrame(
        [[n, p, k, temperature, humidity, ph, rainfall]],
        columns=features
    )

    probabilities = model.predict_proba(input_data)[0]

    results = sorted(
        zip(model.classes_, probabilities),
        key=lambda x: x[1],
        reverse=True
    )

    return [result for result in results if result[1] > 0][:3]

# ----------------------------
# DATABASE FUNCTIONS
# ----------------------------

def create_table():
    connection = sqlite3.connect(DATABASE)

    connection.execute("""
        CREATE TABLE IF NOT EXISTS farm_submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            soil TEXT,
            season TEXT,
            rainfall REAL,
            water TEXT
        )
    """)

    connection.execute("""
        CREATE TABLE IF NOT EXISTS crop_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nitrogen REAL NOT NULL,
            phosphorus REAL NOT NULL,
            potassium REAL NOT NULL,
            temperature REAL NOT NULL,
            humidity REAL NOT NULL,
            ph REAL NOT NULL,
            rainfall REAL NOT NULL,
            top_crop TEXT NOT NULL,
            second_crop TEXT NOT NULL,
            third_crop TEXT NOT NULL
        )
    """)

    connection.commit()
    connection.close()


def save_prediction(n, p, k, temperature, humidity, ph, rainfall, recommendations):

    connection = sqlite3.connect(DATABASE)

    crops = [item[0] for item in recommendations]

    while len(crops) < 3:
        crops.append("Not available")

    connection.execute("""
        INSERT INTO crop_predictions (
            nitrogen, phosphorus, potassium,
            temperature, humidity, ph, rainfall,
            top_crop, second_crop, third_crop
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        n, p, k, temperature, humidity, ph, rainfall,
        crops[0], crops[1], crops[2]
    ))

    connection.commit()
    connection.close()


def get_history():
    connection = sqlite3.connect(DATABASE)

    rows = connection.execute("""
        SELECT id, nitrogen, phosphorus, potassium,
               temperature, humidity, ph, rainfall,
               top_crop, second_crop, third_crop
        FROM crop_predictions
        ORDER BY id DESC
    """).fetchall()

    connection.close()
    return rows


create_table()

# ----------------------------
# APP INTERFACE
# ----------------------------

st.title("🌱 Smart Crop Recommendation System")

st.write(
    "Helping farmers explore suitable crops "
    "based on their soil and weather conditions."
)

st.info(
    "Enter your soil nutrient values and environmental "
    "conditions to receive crop recommendations."
)

st.subheader("Enter Your Farm Details")

with st.form("farm_details_form"):

    st.write("### Soil Nutrients")

    col1, col2, col3 = st.columns(3)

    with col1:
        n = st.number_input("Nitrogen (N)", min_value=0.0, max_value=140.0, value=90.0)

    with col2:
        p = st.number_input("Phosphorus (P)", min_value=5.0, max_value=145.0, value=42.0)

    with col3:
        k = st.number_input("Potassium (K)", min_value=5.0, max_value=205.0, value=43.0)

    st.write("### Environmental Conditions")

    col1, col2 = st.columns(2)

    with col1:
        temperature = st.number_input(
            "Temperature (°C)", min_value=8.83, max_value=43.68, value=20.88
        )

        humidity = st.number_input(
            "Humidity (%)", min_value=14.26, max_value=99.98, value=82.0
        )

    with col2:
        ph = st.number_input(
            "Soil pH", min_value=3.50, max_value=9.94, value=6.50
        )

        rainfall = st.number_input(
            "Rainfall (mm)", min_value=20.21, max_value=298.56, value=202.94
        )

    submitted = st.form_submit_button("Get Crop Recommendations")


if submitted:

    recommendations = predict_crops(
        n, p, k, temperature, humidity, ph, rainfall
    )
    save_prediction(
        n, p, k, temperature, humidity, ph, rainfall,
        recommendations
    )

    st.success("Crop recommendations generated successfully!")

    st.subheader("🌾 Top 3 Recommended Crops")

    for rank, (crop, probability) in enumerate(recommendations, start=1):

        st.write(f"**{rank}. {crop.title()}**")

        st.progress(float(probability))

        st.caption(f"Model probability score: {probability * 100:.2f}%")

# ----------------------------
# SUBMISSION HISTORY
# ----------------------------

st.divider()
st.subheader("📋 Previous Crop Recommendations")

history = get_history()

if history:
    for record in history:

        with st.expander(f"Prediction #{record[0]}"):

            st.write("**Soil Nutrients**")
            st.write("Nitrogen:", record[1])
            st.write("Phosphorus:", record[2])
            st.write("Potassium:", record[3])

            st.write("**Environmental Conditions**")
            st.write("Temperature:", record[4], "°C")
            st.write("Humidity:", record[5], "%")
            st.write("pH:", record[6])
            st.write("Rainfall:", record[7], "mm")

            st.write("**Recommended Crops**")
            for rank, crop in enumerate(record[8:11], start=1):
                if crop != "Not available":
                    st.write(f"{rank}. {crop.title()}")
else:
    st.write("No crop predictions yet.")