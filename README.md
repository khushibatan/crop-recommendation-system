# Smart Crop Recommendation System

ML-based crop recommendation with low-cost IoT soil sensing. A Streamlit app takes soil and weather values and returns the top 3 suitable crops.

## Run

Requires Python 3.11+ (the pinned pandas, scikit-learn and Streamlit releases need it).

The app starts in light mode. Use the **Dark** switch in the header to change theme.

```bash
pip install -r requirements.txt
python train_model.py     # trains 3 models, writes model_comparison.csv and crop_model.pkl
streamlit run app.py
```

## Model

- Inputs (7): N, P, K, temperature, humidity, pH, rainfall
- Output: top 3 crops with model probability scores (22 crops)
- Models compared: Decision Tree, Random Forest, KNN. The best by 5-fold cross-validation is saved.
- Results: see `model_comparison.csv` after running `train_model.py`.

## Weather (Open-Meteo)

Temperature, humidity and rainfall are fetched from [Open-Meteo](https://open-meteo.com/). No API key is needed.

- **Village search:** type a village or town, click **Search**, pick the matching place, then click **Load weather**. Geocoding is global.
- **Temperature and humidity:** current conditions at the selected location, shown with its local timezone.
- **Rainfall:** total precipitation for the **previous 30 complete days** (today is excluded). The app shows the exact date window. The model's rainfall feature is in the 20–299 mm range, so today's rain would be the wrong scale and would trigger an out-of-range warning.
- **Manual override:** the three weather fields stay editable after a fetch; edits are noted in the saved history entry.
- **Village not found:** enter latitude and longitude under **Use coordinates instead**. Weather works for any coordinate worldwide.
- **Offline/API failure:** the app shows an error and manual entry continues to work.
- **Attribution:** weather and geocoding data by [Open-Meteo.com](https://open-meteo.com/) (CC BY 4.0).

## Tests

```bash
python -m unittest discover -s tests -v
```

The weather tests use mocked Open-Meteo responses and do not need a network connection.

## Data and limitations (read before quoting accuracy)

- `Crop_recommendation.csv` is a **published dataset**: [Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) by Atharva Ingle (Apache 2.0), 2,200 rows, exactly 100 per crop, no duplicates. It is **not** data from our local farmer interviews.
- The ~99% accuracy is on this clean, balanced dataset. It is **not** a measure of accuracy on real farms. Field accuracy can only be measured with our own survey data.
- Rainfall in the training data spans about 20 to 299 mm. Inputs outside the trained range show a warning in the app because the model is extrapolating.
- Farmers usually do not know their N, P and K values. These come from a soil test or the sensor.
- The weather API provides **village-level** data, not on-farm measurements. Current temperature/humidity are a snapshot; the training data represents climate ranges.
- The model was trained on Indian crop data and covers 22 crops (no wheat, potato or barley). Recommendations for other regions are indicative only.
- The training dataset's exact rainfall period is undocumented; this app defines the input as the last 30 complete days and labels it in the UI.

## Demo flow

1. `streamlit run app.py`
2. Enter a village → **Search** → select the correct match → **Load weather**.
3. Check the temperature, humidity and 30-day rainfall panel above the form.
4. Enter N, P, K and pH (soil test or sensor). The weather values are filled automatically and stay editable.
5. **Get recommendation** → show the ranked top 3, then open the latest entry under **Recent predictions** for the saved inputs and weather source.

Backup if the venue network fails: use **Use coordinates instead**, or type the weather values directly.

The app keeps the latest 8 predictions and marks weather values that were edited after a fetch.

## Next steps

- Add local survey records as a separate source (`published` vs `local_survey`) and report accuracy for each.
- Connect the 7-in-1 NPK/pH sensor (USB serial first, so the laptop stays online for weather).
- Optional: fall back to a monthly rainfall normal from Open-Meteo's archive when the 30-day total is below the model's range.
- Add a 4-input fallback (temperature, humidity, pH, rainfall) for users with no NPK values.
