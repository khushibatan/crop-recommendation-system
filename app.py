import html
import sqlite3
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

import weather

# ----------------------------
# PAGE SETTINGS
# ----------------------------

st.set_page_config(
    page_title="Crop Recommendation",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATABASE = str(BASE_DIR / "farm_history.db")
MODEL_PATH = BASE_DIR / "crop_model.pkl"
DATA_PATH = BASE_DIR / "Crop_recommendation.csv"


@st.cache_data(ttl=1800, show_spinner=False)
def cached_geocode(query):
    return weather.geocode_village(query)


@st.cache_data(ttl=900, show_spinner=False)
def cached_weather(latitude, longitude):
    return weather.fetch_weather(latitude, longitude)


# ----------------------------
# LOAD MACHINE LEARNING MODEL
# ----------------------------

saved_model = joblib.load(MODEL_PATH)

model = saved_model["model"]
features = saved_model["features"]

TRAIN_RANGES = pd.read_csv(DATA_PATH, usecols=features).agg(["min", "max"])

FEATURE_LABELS = {
    "N": "Nitrogen",
    "P": "Phosphorus",
    "K": "Potassium",
    "temperature": "Temperature",
    "humidity": "Humidity",
    "ph": "pH",
    "rainfall": "Rainfall",
}


def out_of_range_inputs(values):
    """Return (feature, value, min, max) tuples the model has not seen."""
    flagged = []
    for name, value in zip(features, values):
        low = TRAIN_RANGES.loc["min", name]
        high = TRAIN_RANGES.loc["max", name]
        if value < low or value > high:
            flagged.append((name, value, low, high))
    return flagged


def predict_crops(n, p, k, temperature, humidity, ph, rainfall):
    input_data = pd.DataFrame(
        [[n, p, k, temperature, humidity, ph, rainfall]],
        columns=features,
    )

    probabilities = model.predict_proba(input_data)[0]

    results = sorted(
        zip(model.classes_, probabilities),
        key=lambda item: item[1],
        reverse=True,
    )

    return results[:3]


# ----------------------------
# DATABASE
# ----------------------------

def create_table():
    connection = sqlite3.connect(DATABASE)

    connection.execute("DROP TABLE IF EXISTS farm_submissions")

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
            third_crop TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
            location TEXT,
            latitude REAL,
            longitude REAL,
            weather_source TEXT,
            weather_fetched_at TEXT,
            rainfall_window TEXT
        )
    """)

    columns = [
        row[1]
        for row in connection.execute("PRAGMA table_info(crop_predictions)")
    ]

    if "created_at" not in columns:
        connection.execute(
            "ALTER TABLE crop_predictions ADD COLUMN created_at TEXT"
        )

    connection.execute("""
        UPDATE crop_predictions
        SET created_at = datetime('now', 'localtime')
        WHERE created_at IS NULL
    """)

    extra_columns = {
        "location": "TEXT",
        "latitude": "REAL",
        "longitude": "REAL",
        "weather_source": "TEXT",
        "weather_fetched_at": "TEXT",
        "rainfall_window": "TEXT",
    }

    for name, column_type in extra_columns.items():
        if name not in columns:
            connection.execute(
                f"ALTER TABLE crop_predictions ADD COLUMN {name} {column_type}"
            )

    connection.commit()
    connection.close()


def save_prediction(
    n, p, k, temperature, humidity, ph, rainfall, recommendations,
    location=None, latitude=None, longitude=None,
    weather_source=None, weather_fetched_at=None, rainfall_window=None,
):
    connection = sqlite3.connect(DATABASE)

    crops = [item[0] for item in recommendations]

    while len(crops) < 3:
        crops.append("Not available")

    connection.execute("""
        INSERT INTO crop_predictions (
            nitrogen, phosphorus, potassium,
            temperature, humidity, ph, rainfall,
            top_crop, second_crop, third_crop, created_at,
            location, latitude, longitude,
            weather_source, weather_fetched_at, rainfall_window
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now', 'localtime'),
                ?, ?, ?, ?, ?, ?)
    """, (
        n, p, k, temperature, humidity, ph, rainfall,
        crops[0], crops[1], crops[2],
        location, latitude, longitude,
        weather_source, weather_fetched_at, rainfall_window,
    ))

    connection.commit()
    connection.close()


def get_history(limit=8):
    connection = sqlite3.connect(DATABASE)

    rows = connection.execute("""
        SELECT id, nitrogen, phosphorus, potassium,
               temperature, humidity, ph, rainfall,
               top_crop, second_crop, third_crop, created_at,
               location, latitude, longitude,
               weather_source, weather_fetched_at, rainfall_window
        FROM crop_predictions
        ORDER BY id DESC
        LIMIT ?
    """, (limit,)).fetchall()

    connection.close()
    return rows


def clear_history():
    connection = sqlite3.connect(DATABASE)

    connection.execute("DELETE FROM crop_predictions")

    connection.commit()
    connection.close()


create_table()

# ----------------------------
# SESSION STATE
# ----------------------------

INPUT_DEFAULTS = {
    "in_n": 90.0,
    "in_p": 42.0,
    "in_k": 43.0,
    "in_temp": 20.88,
    "in_humidity": 82.0,
    "in_ph": 6.5,
    "in_rainfall": 202.94,
}

for key, value in INPUT_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

for key, value in {
    "geo_results": [],
    "weather_result": None,
    "last_result": None,
    "dark_mode": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = value


def clamp(value, low, high):
    return max(low, min(high, float(value)))


def apply_weather_to_form(result):
    """Write the three Open-Meteo values into the form widgets."""
    st.session_state["in_temp"] = clamp(result["temperature"], -50.0, 60.0)
    st.session_state["in_humidity"] = clamp(result["humidity"], 0.0, 100.0)
    st.session_state["in_rainfall"] = clamp(
        result["rainfall_30d"], 0.0, 3000.0
    )


def fetch_and_store_weather(latitude, longitude, location_label):
    result = cached_weather(latitude, longitude)
    result["location_label"] = location_label
    st.session_state["weather_result"] = result
    apply_weather_to_form(result)
    return result


# ----------------------------
# PRESENTATION HELPERS
# ----------------------------

APP_CSS = """
<style>
:root {
    --page: #F7F6F2;
    --surface: #FFFFFF;
    --ink: #1C2321;
    --muted: #646B63;
    --line: #E2DFD6;
    --accent: #2E5D4B;
    --accent-hover: #244A3C;
    --accent-soft: #EAF0EB;
    --button: #2E5D4B;
    --button-hover: #244A3C;
}

html, body, .stApp {
    background: var(--page);
    color: var(--ink);
}

.stApp {
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
}

header[data-testid="stHeader"],
[data-testid="stToolbar"],
#MainMenu,
footer {
    display: none;
}

.block-container {
    max-width: 860px !important;
    padding-top: 2.75rem !important;
    padding-bottom: 4rem;
}

::selection {
    background: #DDE7DF;
    color: var(--ink);
}

* {
    scrollbar-width: thin;
    scrollbar-color: #C9C5B8 transparent;
}

h1, h2, h3, h4 {
    color: var(--ink);
}

.app-header {
    margin-bottom: 0;
}

.app-title {
    font-size: 2.15rem !important;
    line-height: 1.1 !important;
    font-weight: 650 !important;
    letter-spacing: -0.035em !important;
    padding: 0 !important;
    margin: 0 0 0.5rem !important;
}

.app-subtitle {
    color: var(--muted);
    font-size: 1rem;
    line-height: 1.5;
    margin: 0;
    max-width: 62ch;
}

.section-heading {
    font-size: 1.15rem !important;
    font-weight: 600 !important;
    line-height: 1.35 !important;
    letter-spacing: -0.015em !important;
    color: var(--ink) !important;
    padding: 0 !important;
    margin: 2rem 0 0.75rem !important;
}

.section-heading.first {
    margin-top: 0 !important;
}

.group-label {
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--ink);
    margin: 1.5rem 0 0.35rem;
}

.detail-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--muted);
    margin: 0 0 0.5rem;
}

[data-testid="stCaptionContainer"] p {
    color: var(--muted);
    font-size: 0.82rem;
    line-height: 1.45;
}

[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label {
    font-size: 0.875rem;
    font-weight: 500;
    color: var(--ink);
}

[data-testid="stCheckbox"] {
    display: flex;
    justify-content: flex-end;
}

[data-testid="stCheckbox"] label {
    margin-left: auto;
}

[data-testid="stCheckbox"] p {
    color: var(--ink) !important;
    font-size: 0.82rem !important;
}

input::placeholder {
    color: #8B918A !important;
}

div[data-baseweb="input"],
div[data-baseweb="select"] > div {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 8px;
    transition: border-color 150ms ease, box-shadow 150ms ease;
}

div[data-baseweb="input"]:focus-within,
div[data-baseweb="select"]:focus-within {
    border-color: var(--accent);
    box-shadow: 0 0 0 1px var(--accent);
}

div[data-baseweb="input"] input {
    background: transparent;
    color: var(--ink);
    font-size: 0.95rem;
}

[data-testid="stNumberInput"] button {
    border: none;
    background: transparent;
    color: var(--muted);
}

.stButton > button,
[data-testid="stFormSubmitButton"] > button {
    border-radius: 8px;
    border: 1px solid var(--line);
    background: var(--surface);
    color: var(--ink);
    font-size: 0.9rem;
    font-weight: 550;
    padding: 0.5rem 0.9rem;
    box-shadow: none;
    transition: background-color 150ms ease, border-color 150ms ease,
        color 150ms ease;
}

.stButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover {
    border-color: #C9C5B8;
    background: #FCFBF8;
    color: var(--ink);
}

.stButton > button[kind="primary"],
.stButton > button[kind="primaryFormSubmit"],
[data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"] {
    background: var(--button);
    border-color: var(--button);
    color: #FFFFFF;
}

.stButton > button[kind="primary"]:hover,
.stButton > button[kind="primaryFormSubmit"]:hover,
[data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"]:hover {
    background: var(--button-hover);
    border-color: var(--button-hover);
    color: #FFFFFF;
}

.stButton > button:focus-visible,
[data-testid="stFormSubmitButton"] > button:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
}

[data-testid="stForm"] {
    border: none;
    padding: 0;
    background: transparent;
}

[data-testid="stAlert"] {
    border-radius: 8px;
}

[data-testid="stExpander"] {
    border: 1px solid var(--line);
    border-radius: 8px;
    background: var(--surface);
    margin-bottom: 0.5rem;
}

[data-testid="stExpander"] summary {
    padding: 0.8rem 1rem;
    font-size: 0.9rem;
    font-weight: 500;
}

[data-testid="stExpander"] summary:hover {
    color: var(--accent);
}

[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    padding: 0 1rem 1rem;
}

.results-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 0.5rem;
}

.results-title {
    font-size: 1.15rem !important;
    font-weight: 600 !important;
    line-height: 1.35 !important;
    letter-spacing: -0.015em !important;
    padding: 0 !important;
    margin: 0 !important;
}

.confidence {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    border-radius: 999px;
    padding: 0.28rem 0.6rem;
    font-size: 0.76rem;
    font-weight: 600;
    white-space: nowrap;
}

.confidence::before {
    content: "";
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: currentColor;
}

.confidence.high {
    background: var(--accent-soft);
    color: var(--accent);
}

.confidence.moderate {
    background: #F6EFDF;
    color: #7A5A16;
}

.confidence.low {
    background: #F0EEE8;
    color: var(--muted);
}

.result-list {
    border-top: 1px solid var(--line);
}

.result-row {
    display: flex;
    gap: 0.9rem;
    align-items: flex-start;
    padding: 1rem 0;
    border-bottom: 1px solid var(--line);
}

.rank {
    width: 1.6rem;
    height: 1.6rem;
    border-radius: 50%;
    background: #EFEDE6;
    color: var(--muted);
    font-size: 0.75rem;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    flex: none;
    margin-top: 0.1rem;
}

.result-row.top .rank {
    background: var(--accent);
    color: #FFFFFF;
}

.result-body {
    flex: 1;
    min-width: 0;
}

.result-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 1rem;
}

.result-name {
    font-size: 1.02rem;
    font-weight: 600;
}

.result-pct {
    color: var(--muted);
    font-size: 0.9rem;
    font-variant-numeric: tabular-nums;
}

.result-row.top .result-pct {
    color: var(--accent);
    font-weight: 600;
}

.bar {
    height: 5px;
    background: #EDEAE2;
    border-radius: 999px;
    overflow: hidden;
    margin-top: 0.55rem;
}

.bar-fill {
    height: 100%;
    background: #B9C6BD;
    border-radius: 999px;
}

.result-row.top .bar-fill {
    background: var(--accent);
}

.weather-panel {
    border-top: 1px solid var(--line);
    border-bottom: 1px solid var(--line);
    padding: 1.1rem 0;
    margin-top: 1.25rem;
}

.weather-location {
    font-size: 0.92rem;
    font-weight: 600;
    margin-bottom: 0.85rem;
}

.weather-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 1.25rem;
}

.weather-label {
    font-size: 0.75rem;
    color: var(--muted);
    margin-bottom: 0.25rem;
}

.weather-value {
    font-size: 1.3rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    font-variant-numeric: tabular-nums;
}

.weather-meta {
    color: var(--muted);
    font-size: 0.76rem;
    line-height: 1.5;
    margin-top: 0.9rem;
}

.detail-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.55rem 1.25rem;
}

.detail-item span {
    display: block;
    color: var(--muted);
    font-size: 0.74rem;
    margin-bottom: 0.1rem;
}

.detail-item strong {
    display: block;
    color: var(--ink);
    font-size: 0.86rem;
    font-weight: 500;
    overflow-wrap: anywhere;
}

.crop-line {
    color: var(--ink);
    font-size: 0.9rem;
    margin: 0;
}

.footer-note {
    color: var(--muted);
    font-size: 0.76rem;
    margin-top: 1.5rem;
}

@media (max-width: 640px) {
    .block-container {
        padding-top: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .app-title {
        font-size: 1.8rem;
    }

    .weather-grid,
    .detail-grid {
        grid-template-columns: 1fr;
    }

    .results-head {
        align-items: flex-start;
        flex-direction: column;
        gap: 0.5rem;
    }
}
</style>
"""

DARK_CSS = """
<style>
:root {
    --page: #151917;
    --surface: #1D2320;
    --ink: #E9EDE8;
    --muted: #A0A9A1;
    --line: #333B36;
    --accent: #7FB89D;
    --accent-hover: #95C9AF;
    --accent-soft: #22312B;
    --button: #346B51;
    --button-hover: #3D7D5F;
}

::selection {
    background: #2C4036;
    color: var(--ink);
}

* {
    scrollbar-color: #4A544D transparent;
}

[data-testid="stSelectbox"] [data-baseweb="select"] * {
    color: var(--ink);
}

[data-testid="stTextInputRootElement"],
[data-testid="stNumberInputContainer"],
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background: var(--surface) !important;
    border-color: var(--line) !important;
}

[data-testid="stTextInputRootElement"] input,
[data-testid="stNumberInputContainer"] input {
    color: var(--ink) !important;
    -webkit-text-fill-color: var(--ink) !important;
}

[data-testid="stSelectbox"] svg {
    color: var(--muted) !important;
    fill: var(--muted) !important;
}

[data-testid="stNumberInput"] button:hover {
    background: #262C28;
}

.stButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover {
    border-color: #4A544D;
    background: #242B27;
    color: var(--ink);
}

.result-row.top .rank {
    background: var(--button);
}

.rank {
    background: #262C28;
}

.bar {
    background: #2B312D;
}

.bar-fill {
    background: #506057;
}

.result-row.top .bar-fill {
    background: var(--accent);
}

.confidence.high {
    background: var(--accent-soft);
    color: var(--accent);
}

.confidence.moderate {
    background: #3A2F1A;
    color: #E0BE7A;
}

.confidence.low {
    background: #262C28;
    color: var(--muted);
}

[data-testid="stAlert"] {
    background: var(--surface);
    border: 1px solid var(--line);
    color: var(--ink);
}

[data-testid="stAlert"] p {
    color: var(--ink) !important;
}

div[data-baseweb="popover"] div[data-baseweb="menu"],
div[data-baseweb="popover"] ul {
    background: var(--surface) !important;
}

div[data-baseweb="popover"] li {
    background: var(--surface);
    color: var(--ink) !important;
}

div[data-baseweb="popover"] li:hover {
    background: #262C28 !important;
}

input::placeholder {
    color: #7E877F !important;
}

[data-testid="stCheckbox"] label > div:first-of-type {
    background: #3D4640 !important;
}

[data-testid="stCheckbox"] label:has(input:checked) > div:first-of-type {
    background: var(--accent) !important;
}

[data-testid="stCheckbox"] label > div:first-of-type > div {
    background: #E9EDE8 !important;
}
</style>
"""

st.markdown(APP_CSS, unsafe_allow_html=True)

if st.session_state.get("dark_mode", False):
    st.markdown(DARK_CSS, unsafe_allow_html=True)

def esc(value):
    return html.escape("" if value is None else str(value))


def section_heading(text, first=False):
    first_class = " first" if first else ""
    st.markdown(
        f'<h2 class="section-heading{first_class}">{esc(text)}</h2>',
        unsafe_allow_html=True,
    )


def detail_grid(items):
    cells = "".join(
        f'<div class="detail-item"><span>{esc(label)}</span>'
        f"<strong>{esc(value)}</strong></div>"
        for label, value in items
    )
    return f'<div class="detail-grid">{cells}</div>'


def format_timestamp(value):
    if not value:
        return "Unknown time"

    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.strftime("%d %b %Y, %H:%M")
    except ValueError:
        return str(value)


def format_day(value):
    try:
        return datetime.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def format_date_range(start, end):
    start_date = format_day(start)
    end_date = format_day(end)

    if start_date is None or end_date is None:
        return f"{start} to {end}"

    if start_date.year != end_date.year:
        return (
            f"{start_date.strftime('%d %b %Y')} – "
            f"{end_date.strftime('%d %b %Y')}"
        )

    if start_date.month == end_date.month:
        return f"{start_date.day}–{end_date.strftime('%d %b %Y')}"

    return (
        f"{start_date.strftime('%d %b')} – "
        f"{end_date.strftime('%d %b %Y')}"
    )


def render_weather(result):
    location = esc(result.get("location_label") or "Selected location")
    temperature = f"{float(result['temperature']):.1f}"
    humidity = f"{float(result['humidity']):.0f}"
    rainfall = f"{float(result['rainfall_30d']):.1f}"
    start = esc(result.get("rainfall_start"))
    end = esc(result.get("rainfall_end"))
    window = format_date_range(start, end)
    fetched = esc(format_timestamp(result.get("fetched_at")))
    timezone = esc(result.get("timezone") or "local time")

    st.markdown(
        f'<div class="weather-panel">'
        f'<div class="weather-location">{location}</div>'
        f'<div class="weather-grid">'
        f'<div class="weather-item">'
        f'<div class="weather-label">Temperature</div>'
        f'<div class="weather-value">{temperature} °C</div>'
        f"</div>"
        f'<div class="weather-item">'
        f'<div class="weather-label">Humidity</div>'
        f'<div class="weather-value">{humidity}%</div>'
        f"</div>"
        f'<div class="weather-item">'
        f'<div class="weather-label">Rainfall, 30 days</div>'
        f'<div class="weather-value">{rainfall} mm</div>'
        f"</div>"
        f"</div>"
        f'<div class="weather-meta">'
        f"Open-Meteo · {window} · updated {fetched} ({timezone})"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_recommendations(recommendations):
    top_probability = float(recommendations[0][1])
    margin = top_probability - float(recommendations[1][1])

    if top_probability >= 0.70 and margin >= 0.15:
        label, css_class = "High confidence", "high"
    elif top_probability >= 0.40 and margin >= 0.08:
        label, css_class = "Moderate confidence", "moderate"
    else:
        label, css_class = "Low confidence", "low"

    rows = []

    for rank, (crop, probability) in enumerate(recommendations, start=1):
        percentage = float(probability) * 100
        top_class = " top" if rank == 1 else ""
        rows.append(
            f'<div class="result-row{top_class}">'
            f'<div class="rank">{rank}</div>'
            f'<div class="result-body">'
            f'<div class="result-head">'
            f'<span class="result-name">{esc(crop.title())}</span>'
            f'<span class="result-pct">{percentage:.2f}%</span>'
            f"</div>"
            f'<div class="bar"><div class="bar-fill" '
            f'style="width:{percentage:.2f}%"></div></div>'
            f"</div></div>"
        )

    st.markdown(
        f'<div class="results-head">'
        f'<h2 class="results-title">Recommendations</h2>'
        f'<span class="confidence {css_class}">{label}</span>'
        f"</div>"
        f'<div class="result-list">{"".join(rows)}</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Scores are model probabilities on the training distribution, "
        "not field guarantees."
    )


def render_history(history):
    for record in history:
        top_crop = str(record[8]).title()
        location = record[12] or "Manual location"
        saved_at = format_timestamp(record[11])

        with st.expander(
            f"{top_crop} · {location} · {saved_at}"
        ):
            left, right = st.columns(2)

            with left:
                st.markdown(
                    '<p class="detail-label">Inputs</p>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    detail_grid([
                        ("Nitrogen", f"{record[1]:g}"),
                        ("Phosphorus", f"{record[2]:g}"),
                        ("Potassium", f"{record[3]:g}"),
                        ("Temperature", f"{record[4]:g} °C"),
                        ("Humidity", f"{record[5]:g}%"),
                        ("pH", f"{record[6]:g}"),
                        ("Rainfall", f"{record[7]:g} mm"),
                    ]),
                    unsafe_allow_html=True,
                )

            with right:
                st.markdown(
                    '<p class="detail-label">Source</p>',
                    unsafe_allow_html=True,
                )

                coordinates = (
                    f"{record[13]:.4f}, {record[14]:.4f}"
                    if record[13] is not None and record[14] is not None
                    else "Not set"
                )

                st.markdown(
                    detail_grid([
                        ("Location", location),
                        ("Weather", record[15] or "Manual"),
                        ("Coordinates", coordinates),
                        (
                            "Rainfall window",
                            record[17] or "Not set",
                        ),
                        (
                            "Fetched",
                            format_timestamp(record[16])
                            if record[16]
                            else "Not set",
                        ),
                    ]),
                    unsafe_allow_html=True,
                )

            st.markdown(
                '<p class="detail-label" style="margin-top:1rem">'
                "Ranked crops</p>",
                unsafe_allow_html=True,
            )
            st.markdown(
                '<p class="crop-line">'
                + esc(" · ".join(
                    f"{rank}. {crop.title()}"
                    for rank, crop in enumerate(record[8:11], start=1)
                    if crop != "Not available"
                ))
                + "</p>",
                unsafe_allow_html=True,
            )


# ----------------------------
# INTERFACE
# ----------------------------

header_col, toggle_col = st.columns(
    [5, 1],
    vertical_alignment="top",
)

with header_col:
    st.markdown(
        '<div class="app-header">'
        '<h1 class="app-title">Crop Recommendation</h1>'
        '<p class="app-subtitle">Match soil and weather conditions against '
        "twenty-two crops, ranked by a trained Random Forest model.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

with toggle_col:
    st.toggle(
        "Dark",
        key="dark_mode",
    )

# ----------------------------
# LOCATION
# ----------------------------

section_heading("Location")

search_col, search_button_col = st.columns(
    [4, 1],
    vertical_alignment="bottom",
)

with search_col:
    village_query = st.text_input(
        "Village or town",
        placeholder="e.g. Pokhara, Nepal",
        key="village_query",
    )

with search_button_col:
    search_clicked = st.button(
        "Search",
        key="village_search",
        use_container_width=True,
    )

if search_clicked:
    st.session_state["geo_results"] = []
    st.session_state["weather_result"] = None

    if not (village_query or "").strip():
        st.caption("Enter a village or town to search.")
    else:
        try:
            with st.spinner("Searching"):
                results = cached_geocode(village_query)
        except weather.WeatherError as exc:
            st.error(f"Location search failed: {exc}")
        else:
            if results:
                st.session_state["geo_results"] = results
                st.session_state["geo_index"] = 0
            else:
                st.caption(
                    "No matching place found. Try a nearby town or use "
                    "coordinates."
                )

geo_results = st.session_state["geo_results"]

if geo_results:
    labels = [item["label"] for item in geo_results]

    select_col, load_button_col = st.columns(
        [4, 1],
        vertical_alignment="bottom",
    )

    with select_col:
        selected_index = st.selectbox(
            "Matching location",
            options=range(len(labels)),
            format_func=lambda index: labels[index],
            key="geo_index",
        )

    with load_button_col:
        load_clicked = st.button(
            "Load weather",
            key="location_load",
            type="primary",
            use_container_width=True,
        )

    if load_clicked:
        st.session_state["weather_result"] = None
        selected = geo_results[selected_index]

        try:
            with st.spinner("Loading weather"):
                fetch_and_store_weather(
                    selected["latitude"],
                    selected["longitude"],
                    selected["label"],
                )
        except weather.WeatherError as exc:
            st.error(f"Weather could not be loaded: {exc}")

with st.expander("Use coordinates instead"):
    lat_col, lon_col, coordinate_button_col = st.columns(
        [2, 2, 1.3],
        vertical_alignment="bottom",
    )

    with lat_col:
        manual_lat = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=27.7172,
            step=0.0001,
            format="%.4f",
            key="manual_lat",
        )

    with lon_col:
        manual_lon = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=85.3240,
            step=0.0001,
            format="%.4f",
            key="manual_lon",
        )

    with coordinate_button_col:
        coordinate_clicked = st.button(
            "Load weather",
            key="coordinate_load",
            use_container_width=True,
        )

    if coordinate_clicked:
        st.session_state["weather_result"] = None

        try:
            with st.spinner("Loading weather"):
                fetch_and_store_weather(
                    manual_lat,
                    manual_lon,
                    (
                        f"Custom location "
                        f"({manual_lat:.4f}, {manual_lon:.4f})"
                    ),
                )
        except weather.WeatherError as exc:
            st.error(f"Weather could not be loaded: {exc}")

weather_result = st.session_state["weather_result"]

if weather_result:
    render_weather(weather_result)

# ----------------------------
# FARM DETAILS
# ----------------------------

section_heading("Farm details")

with st.form("farm_details_form", border=False):
    st.markdown(
        '<p class="group-label">Soil</p>',
        unsafe_allow_html=True,
    )
    st.caption("Use soil-test or sensor values.")

    soil_col1, soil_col2, soil_col3, soil_col4 = st.columns(4)

    with soil_col1:
        n = st.number_input(
            "Nitrogen (N)",
            min_value=0.0,
            max_value=500.0,
            step=1.0,
            key="in_n",
        )

    with soil_col2:
        p = st.number_input(
            "Phosphorus (P)",
            min_value=0.0,
            max_value=500.0,
            step=1.0,
            key="in_p",
        )

    with soil_col3:
        k = st.number_input(
            "Potassium (K)",
            min_value=0.0,
            max_value=500.0,
            step=1.0,
            key="in_k",
        )

    with soil_col4:
        ph = st.number_input(
            "Soil pH",
            min_value=0.0,
            max_value=14.0,
            step=0.01,
            format="%.2f",
            key="in_ph",
        )

    st.markdown(
        '<p class="group-label">Weather</p>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Auto-filled from Open-Meteo when a location is loaded. "
        "Edit if needed."
    )

    weather_col1, weather_col2, weather_col3 = st.columns(3)

    with weather_col1:
        temperature = st.number_input(
            "Temperature (°C)",
            min_value=-50.0,
            max_value=60.0,
            step=0.1,
            key="in_temp",
        )

    with weather_col2:
        humidity = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            step=1.0,
            key="in_humidity",
        )

    with weather_col3:
        rainfall = st.number_input(
            "Rainfall, last 30 days (mm)",
            min_value=0.0,
            max_value=3000.0,
            step=0.1,
            key="in_rainfall",
        )

    submitted = st.form_submit_button(
        "Get recommendation",
        type="primary",
        use_container_width=True,
    )

if submitted:
    recommendations = predict_crops(
        n, p, k, temperature, humidity, ph, rainfall
    )

    weather_data = st.session_state.get("weather_result") or {}

    if weather_data:
        location = weather_data["location_label"]
        latitude = weather_data["latitude"]
        longitude = weather_data["longitude"]
        weather_source = weather_data["source"]
        weather_fetched_at = weather_data["fetched_at"]
        rainfall_window = (
            f"{weather_data['rainfall_start']} to "
            f"{weather_data['rainfall_end']}"
        )

        edited = any(
            abs(float(entered) - float(fetched)) > tolerance
            for entered, fetched, tolerance in (
                (temperature, weather_data["temperature"], 0.05),
                (humidity, weather_data["humidity"], 0.5),
                (rainfall, weather_data["rainfall_30d"], 0.05),
            )
        )

        if edited:
            weather_source += " (values edited manually)"
    else:
        location = "Manual location"
        latitude = None
        longitude = None
        weather_source = "Manual"
        weather_fetched_at = None
        rainfall_window = None

    save_prediction(
        n, p, k, temperature, humidity, ph, rainfall,
        recommendations,
        location=location,
        latitude=latitude,
        longitude=longitude,
        weather_source=weather_source,
        weather_fetched_at=weather_fetched_at,
        rainfall_window=rainfall_window,
    )

    st.session_state["last_result"] = {
        "recommendations": recommendations,
        "flagged": out_of_range_inputs(
            [n, p, k, temperature, humidity, ph, rainfall]
        ),
    }

last_result = st.session_state.get("last_result")

if last_result:
    st.divider()
    flagged = last_result["flagged"]

    if flagged:
        st.warning(
            "Some inputs are outside the model's training range. "
            "Treat this ranking as indicative."
        )
        details = "; ".join(
            f"{FEATURE_LABELS.get(name, name)} = {value:g} "
            f"(expected {low:g}–{high:g})"
            for name, value, low, high in flagged
        )
        st.caption(f"Outside range: {details}")

    render_recommendations(last_result["recommendations"])

# ----------------------------
# HISTORY
# ----------------------------

history = get_history(limit=8)

if history:
    st.divider()

    heading_col, clear_col = st.columns(
        [4, 1],
        vertical_alignment="center",
    )

    with heading_col:
        section_heading("Recent predictions", first=True)

    with clear_col:
        if st.button("Clear history", use_container_width=True):
            clear_history()
            st.rerun()

    render_history(history)

# ----------------------------
# NOTES
# ----------------------------

st.divider()

with st.expander("Model notes"):
    st.markdown(
        "- Trained on the Kaggle Crop Recommendation Dataset "
        "(2,200 rows, 22 crops). The ~99% score is dataset accuracy, "
        "not field accuracy.\n"
        "- Weather is village-level Open-Meteo data. Rainfall is the "
        "previous 30 complete days, not today's rain.\n"
        "- N, P, K and pH should come from a soil test or the 7-in-1 "
        "sensor.\n"
        "- The model was trained on Indian crop data; recommendations "
        "elsewhere are indicative."
    )

st.markdown(
    '<p class="footer-note">Weather data by Open-Meteo.</p>',
    unsafe_allow_html=True,
)
