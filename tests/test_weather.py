"""Unit tests for the Open-Meteo weather helper.

These tests never touch the network: requests.get is replaced with a fake.
Run from the project root with:

    python -m unittest discover -s tests -v
"""

import unittest
from datetime import date, timedelta
from unittest.mock import patch

import requests

import weather


class FakeResponse:
    def __init__(self, payload=None, status_code=200, json_error=False):
        self.payload = payload
        self.status_code = status_code
        self.json_error = json_error

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        if self.json_error:
            raise ValueError("not JSON")
        return self.payload


def forecast_payload(days=32, current_date="2026-11-16"):
    """Build a forecast response with dated rainfall entries."""
    today = date.fromisoformat(current_date)
    dates = [
        (today - timedelta(days=days - 1 - offset)).isoformat()
        for offset in range(days)
    ]
    amounts = [1.0] * days

    return {
        "latitude": 28.26689,
        "longitude": 83.96851,
        "timezone": "Asia/Kathmandu",
        "timezone_abbreviation": "+0545",
        "elevation": 820.0,
        "current": {
            "time": f"{current_date}T10:30",
            "interval": 900,
            "temperature_2m": 21.4,
            "relative_humidity_2m": 63,
            "precipitation": 0.2,
        },
        "daily": {
            "time": dates,
            "precipitation_sum": amounts,
        },
    }


class GeocodeTests(unittest.TestCase):
    @patch("weather.requests.get")
    def test_normalises_results_and_builds_label(self, mock_get):
        mock_get.return_value = FakeResponse(
            {
                "results": [
                    {
                        "name": "Pokhara",
                        "admin1": "Gandaki Pradesh",
                        "admin2": "Kaski",
                        "country": "Nepal",
                        "country_code": "NP",
                        "latitude": 28.26689,
                        "longitude": 83.96851,
                        "timezone": "Asia/Kathmandu",
                        "elevation": 820.0,
                        "population": 200000,
                        "feature_code": "PPLA",
                    }
                ]
            }
        )

        results = weather.geocode_village("Pokhara")

        self.assertEqual(len(results), 1)
        self.assertEqual(
            results[0]["label"],
            "Pokhara, Kaski, Gandaki Pradesh, Nepal",
        )
        self.assertAlmostEqual(results[0]["latitude"], 28.26689)
        self.assertEqual(results[0]["timezone"], "Asia/Kathmandu")

        _, kwargs = mock_get.call_args
        self.assertEqual(kwargs["params"]["name"], "Pokhara")
        self.assertEqual(kwargs["params"]["format"], "json")
        self.assertEqual(kwargs["timeout"], weather.TIMEOUT_SECONDS)

    @patch("weather.requests.get")
    def test_no_results_returns_empty_list(self, mock_get):
        mock_get.return_value = FakeResponse({"generationtime_ms": 0.1})

        self.assertEqual(weather.geocode_village("Nowhere At All"), [])

    @patch("weather.requests.get")
    def test_blank_query_skips_network(self, mock_get):
        self.assertEqual(weather.geocode_village("   "), [])
        mock_get.assert_not_called()

    @patch("weather.requests.get")
    def test_http_error_becomes_weather_error(self, mock_get):
        mock_get.return_value = FakeResponse(status_code=503)

        with self.assertRaises(weather.WeatherError):
            weather.geocode_village("Pokhara")

    @patch("weather.requests.get")
    def test_timeout_becomes_weather_error(self, mock_get):
        mock_get.side_effect = requests.Timeout("timed out")

        with self.assertRaises(weather.WeatherError):
            weather.geocode_village("Pokhara")

    @patch("weather.requests.get")
    def test_api_error_body_becomes_weather_error(self, mock_get):
        mock_get.return_value = FakeResponse(
            {"error": True, "reason": "Parameter 'name' is required"}
        )

        with self.assertRaises(weather.WeatherError):
            weather.geocode_village("Pokhara")


class FetchWeatherTests(unittest.TestCase):
    @patch("weather.requests.get")
    def test_sums_last_30_complete_days_excluding_today(self, mock_get):
        mock_get.return_value = FakeResponse(forecast_payload(days=32))

        result = weather.fetch_weather(28.26689, 83.96851)

        self.assertEqual(result["days_used"], 30)
        self.assertEqual(result["rainfall_30d"], 30.0)
        self.assertEqual(result["rainfall_start"], "2026-10-17")
        self.assertEqual(result["rainfall_end"], "2026-11-15")
        self.assertEqual(result["temperature"], 21.4)
        self.assertEqual(result["humidity"], 63.0)
        self.assertEqual(result["current_precipitation"], 0.2)
        self.assertEqual(result["source"], "Open-Meteo")
        self.assertEqual(result["timezone"], "Asia/Kathmandu")

    @patch("weather.requests.get")
    def test_missing_rainfall_days_are_skipped(self, mock_get):
        payload = forecast_payload(days=31)
        payload["daily"]["precipitation_sum"][5] = None
        payload["daily"]["precipitation_sum"][10] = None
        mock_get.return_value = FakeResponse(payload)

        result = weather.fetch_weather(28.26689, 83.96851)

        self.assertEqual(result["days_used"], 30)
        self.assertEqual(result["rainfall_30d"], 28.0)

    @patch("weather.requests.get")
    def test_uses_only_most_recent_30_complete_days(self, mock_get):
        payload = forecast_payload(days=45)
        mock_get.return_value = FakeResponse(payload)

        result = weather.fetch_weather(28.26689, 83.96851)

        self.assertEqual(result["days_used"], 30)
        self.assertEqual(result["rainfall_30d"], 30.0)
        self.assertEqual(result["rainfall_end"], "2026-11-15")

    @patch("weather.requests.get")
    def test_missing_temperature_raises(self, mock_get):
        payload = forecast_payload(days=31)
        payload["current"].pop("temperature_2m")
        mock_get.return_value = FakeResponse(payload)

        with self.assertRaises(weather.WeatherError):
            weather.fetch_weather(28.26689, 83.96851)

    @patch("weather.requests.get")
    def test_missing_daily_rainfall_raises(self, mock_get):
        payload = forecast_payload(days=31)
        payload["daily"] = {}
        mock_get.return_value = FakeResponse(payload)

        with self.assertRaises(weather.WeatherError):
            weather.fetch_weather(28.26689, 83.96851)

    @patch("weather.requests.get")
    def test_all_null_rainfall_raises(self, mock_get):
        payload = forecast_payload(days=31)
        payload["daily"]["precipitation_sum"] = [None] * 31
        mock_get.return_value = FakeResponse(payload)

        with self.assertRaises(weather.WeatherError):
            weather.fetch_weather(28.26689, 83.96851)

    @patch("weather.requests.get")
    def test_invalid_coordinates_raise_without_network(self, mock_get):
        for latitude, longitude in [(91, 0), (-91, 0), (0, 181), (0, -181)]:
            with self.assertRaises(weather.WeatherError):
                weather.fetch_weather(latitude, longitude)

        with self.assertRaises(weather.WeatherError):
            weather.fetch_weather("north", "east")

        mock_get.assert_not_called()

    @patch("weather.requests.get")
    def test_non_json_response_raises_weather_error(self, mock_get):
        mock_get.return_value = FakeResponse(json_error=True)

        with self.assertRaises(weather.WeatherError):
            weather.fetch_weather(28.26689, 83.96851)

    @patch("weather.requests.get")
    def test_forecast_request_uses_expected_params(self, mock_get):
        mock_get.return_value = FakeResponse(forecast_payload(days=31))

        weather.fetch_weather(28.26689, 83.96851)

        _, kwargs = mock_get.call_args
        params = kwargs["params"]
        self.assertEqual(params["latitude"], 28.26689)
        self.assertEqual(params["longitude"], 83.96851)
        self.assertEqual(params["past_days"], 30)
        self.assertEqual(params["forecast_days"], 1)
        self.assertEqual(params["timezone"], "auto")
        self.assertEqual(params["temperature_unit"], "celsius")
        self.assertEqual(params["precipitation_unit"], "mm")
        self.assertIn("temperature_2m", params["current"])
        self.assertIn("relative_humidity_2m", params["current"])


if __name__ == "__main__":
    unittest.main()
