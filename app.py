import sqlite3
import streamlit as st

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
# DATABASE FUNCTIONS
# ----------------------------

def create_table():
    connection = sqlite3.connect(DATABASE)

    connection.execute("""
        CREATE TABLE IF NOT EXISTS farm_submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            soil TEXT NOT NULL,
            season TEXT NOT NULL,
            rainfall REAL NOT NULL,
            water TEXT NOT NULL
        )
    """)

    connection.commit()
    connection.close()


def save_submission(soil, season, rainfall, water):
    connection = sqlite3.connect(DATABASE)

    connection.execute("""
        INSERT INTO farm_submissions
        (soil, season, rainfall, water)
        VALUES (?, ?, ?, ?)
    """, (soil, season, rainfall, water))

    connection.commit()
    connection.close()


def get_history():
    connection = sqlite3.connect(DATABASE)

    rows = connection.execute("""
        SELECT id, soil, season, rainfall, water
        FROM farm_submissions
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
    "based on their farm conditions."
)

st.info(
    "Crop predictions will be enabled after "
    "the machine learning model is integrated."
)

st.subheader("Enter Your Farm Details")

with st.form("farm_details_form"):

    soil = st.selectbox(
        "Soil Type",
        ["Red Soil", "Black Soil", "Sandy Soil",
         "Loamy Soil", "Clay Soil"]
    )

    season = st.selectbox(
        "Growing Season",
        ["Kharif", "Rabi", "Summer"]
    )

    rainfall = st.number_input(
        "Expected Rainfall (mm)",
        min_value=0.0,
        value=500.0
    )

    water = st.selectbox(
        "Water Availability",
        ["Low", "Medium", "High"]
    )

    submitted = st.form_submit_button("Save Farm Details")


if submitted:
    save_submission(soil, season, rainfall, water)

    st.success("Farm details saved successfully!")

    st.write("**Soil type:**", soil)
    st.write("**Season:**", season)
    st.write("**Expected rainfall:**", rainfall, "mm")
    st.write("**Water availability:**", water)


# ----------------------------
# SUBMISSION HISTORY
# ----------------------------

st.divider()
st.subheader("📋 Previous Farm Submissions")

history = get_history()

if history:
    for record in history:
        with st.expander(f"Submission #{record[0]}"):
            st.write("Soil type:", record[1])
            st.write("Season:", record[2])
            st.write("Expected rainfall:", record[3], "mm")
            st.write("Water availability:", record[4])
else:
    st.write("No farm submissions yet.")
