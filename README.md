# Smart Crop Recommendation System

ML-based crop recommendation with low-cost IoT soil sensing. A Streamlit app takes soil and weather values and returns the top 3 suitable crops.

## Run

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

## Data and limitations (read before quoting accuracy)

- `Crop_recommendation.csv` is a **published dataset** (2,200 rows, exactly 100 per crop, no duplicates). It is **not** data from our local farmer interviews.
- The ~99% accuracy is on this clean, balanced dataset. It is **not** a measure of accuracy on real farms. Field accuracy can only be measured with our own survey data.
- Rainfall in the training data spans about 20 to 299 mm. Inputs outside the trained range show a warning in the app because the model is extrapolating.
- Farmers usually do not know their N, P and K values. These come from a soil test or the sensor.

## Next steps

- Add local survey records as a separate source (`published` vs `local_survey`) and report accuracy for each.
- Confirm what period the dataset's rainfall figure covers and align the app's units with it.
- Add a 4-input fallback (temperature, humidity, pH, rainfall) for users with no NPK values.
- Connect the sensor serial reader to the form.
