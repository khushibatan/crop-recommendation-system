"""Open-Meteo helpers for village-level weather inputs.

The crop model needs air temperature, relative humidity and rainfall.  This
module turns a village name into coordinates (Open-Meteo geocoding) and fetches
current conditions plus the rainfall total for the previous 30 complete days.

No API key is required.  Data: https://open-meteo.com/ (CC BY 4.0).
"""

from __future__ import annotations

from datetime import date

import requests


class WeatherError(Exception):
    """Raised when Open-Meteo cannot provide usable weather data."""


GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT_SECONDS = 12
RAINFALL_DAYS = 30
HEADERS = {"User-Agent": "SmartCropRecommendation/1.0 (college project)"}

__all__ = ["WeatherError", "geocode_village", "fetch_weather"]


def _get_json(url: str, params: dict) -> dict:
    """Call Open-Meteo and turn transport/API errors into WeatherError."""
    try:
        response = requests.get(
            url, params=params, timeout=TIMEOUT_SECONDS, headers=HEADERS
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        raise WeatherError(f"Could not reach Open-Meteo: {exc}") from exc
    except ValueError as exc:
        raise WeatherError("Open-Meteo returned an invalid response") from exc

    if not isinstance(data, dict):
        raise WeatherError("Open-Meteo returned an unexpected response")

    if data.get("error"):
        raise WeatherError(data.get("reason", "Open-Meteo returned an error"))

    return data


def _location_label(item: dict) -> str:
    """Build a readable label such as 'Pokhara, Gandaki Pradesh, Nepal'."""
    parts = [item.get("name")]

    admin2 = item.get("admin2")
    admin1 = item.get("admin1")

    if admin2 and admin2 != admin1:
        parts.append(admin2)
    if admin1:
        parts.append(admin1)
    if item.get("country"):
        parts.append(item["country"])

    label = ", ".join(str(part) for part in parts if part)
    return label or f"{item.get('latitude')}, {item.get('longitude')}"


def geocode_village(query: str, count: int = 8, language: str = "en") -> list[dict]:
    """Search Open-Meteo's geocoder for a village, town or city.

    Returns a list of normalised location dictionaries.  An empty list means
    the name was not found; the caller can offer coordinates as a fallback.
    """
    query = (query or "").strip()
    if not query:
        return []

    data = _get_json(
        GEOCODE_URL,
        {
            "name": query,
            "count": max(1, min(int(count), 100)),
            "language": language,
            "format": "json",
        },
    )

    locations = []

    for item in data.get("results") or []:
        latitude = item.get("latitude")
        longitude = item.get("longitude")

        if latitude is None or longitude is None:
            continue

        locations.append(
            {
                "name": item.get("name"),
                "admin1": item.get("admin1"),
                "admin2": item.get("admin2"),
                "country": item.get("country"),
                "country_code": item.get("country_code"),
                "latitude": float(latitude),
                "longitude": float(longitude),
                "timezone": item.get("timezone"),
                "elevation": item.get("elevation"),
                "population": item.get("population"),
                "feature_code": item.get("feature_code"),
                "label": _location_label(item),
            }
        )

    return locations


def _check_coordinates(latitude, longitude) -> tuple[float, float]:
    try:
        latitude = float(latitude)
        longitude = float(longitude)
    except (TypeError, ValueError) as exc:
        raise WeatherError("Latitude and longitude must be numbers") from exc

    if not -90.0 <= latitude <= 90.0:
        raise WeatherError("Latitude must be between -90 and 90")
    if not -180.0 <= longitude <= 180.0:
        raise WeatherError("Longitude must be between -180 and 180")

    return latitude, longitude


def fetch_weather(latitude, longitude) -> dict:
    """Fetch current weather and previous-30-day rainfall for a location.

    Returns a dictionary with the three model inputs plus provenance fields.
    """
    latitude, longitude = _check_coordinates(latitude, longitude)

    data = _get_json(
        FORECAST_URL,
        {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,precipitation",
            "daily": "precipitation_sum",
            # Ask for 30 past days + today, then keep the 30 complete days
            # before today.  This matches the monthly-ish scale of the
            # training data better than "rain in the last hour".
            "past_days": RAINFALL_DAYS,
            "forecast_days": 1,
            "temperature_unit": "celsius",
            "precipitation_unit": "mm",
            "timezone": "auto",
        },
    )

    current = data.get("current") or {}
    temperature = current.get("temperature_2m")
    humidity = current.get("relative_humidity_2m")

    if temperature is None or humidity is None:
        raise WeatherError(
            "Open-Meteo did not return temperature and humidity for this location"
        )

    daily = data.get("daily") or {}
    times = daily.get("time") or []
    precipitation = daily.get("precipitation_sum") or []

    if not times or not precipitation:
        raise WeatherError("Open-Meteo did not return rainfall data")

    today = (current.get("time") or "")[:10] or date.today().isoformat()

    # Keep only complete days before today, then the most recent 30.
    pairs = [
        (day, amount)
        for day, amount in zip(times, precipitation)
        if day < today
    ][-RAINFALL_DAYS:]

    if not pairs:
        raise WeatherError("No completed rainfall days were returned")

    amounts = [amount for _, amount in pairs if amount is not None]

    if not amounts:
        raise WeatherError("Rainfall values are missing")

    rainfall_total = sum(float(amount) for amount in amounts)

    return {
        "temperature": round(float(temperature), 2),
        "humidity": round(float(humidity), 1),
        "current_precipitation": round(
            float(current.get("precipitation") or 0.0), 2
        ),
        "rainfall_30d": round(rainfall_total, 1),
        "rainfall_start": pairs[0][0],
        "rainfall_end": pairs[-1][0],
        "days_used": len(pairs),
        "timezone": data.get("timezone"),
        "timezone_abbreviation": data.get("timezone_abbreviation"),
        "fetched_at": current.get("time"),
        "latitude": float(data.get("latitude", latitude)),
        "longitude": float(data.get("longitude", longitude)),
        "elevation": data.get("elevation"),
        "source": "Open-Meteo",
        "source_url": "https://open-meteo.com/",
    }
