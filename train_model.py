import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report


# 1. Load the dataset
df = pd.read_csv("Crop_recommendation.csv")

# 2. Separate inputs and output
X = df.drop("label", axis=1)
y = df["label"]

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Define the models
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "K-Nearest Neighbors": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)
    )
}

# 5. Train and evaluate each model
#    Hold-out accuracy comes from one split; 5-fold cross-validation is a
#    steadier number, so the best model is chosen on cross-validation.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for name, model in models.items():
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    holdout_accuracy = accuracy_score(y_test, predictions)
    cv_accuracy = cross_val_score(model, X, y, cv=cv).mean()

    results.append({
        "model": name,
        "holdout_accuracy_pct": round(holdout_accuracy * 100, 2),
        "cv_accuracy_pct": round(cv_accuracy * 100, 2),
    })

    print(f"\n{name}")
    print("-" * 40)
    print(f"Hold-out accuracy: {holdout_accuracy * 100:.2f}%")
    print(f"5-fold CV accuracy: {cv_accuracy * 100:.2f}%")
    print(classification_report(y_test, predictions))

# 6. Save the comparison table (use it directly in the project report)
results_df = pd.DataFrame(results)
results_df.to_csv("model_comparison.csv", index=False)
print("\nModel comparison")
print(results_df.to_string(index=False))

# 7. Save the model that scored best in cross-validation
best_name = results_df.loc[results_df["cv_accuracy_pct"].idxmax(), "model"]
best_model = models[best_name]

joblib.dump(
    {
        "model": best_model,
        "features": list(X.columns),
        "classes": list(best_model.classes_)
    },
    "crop_model.pkl"
)

print(f"\nBest model by cross-validation: {best_name}. Saved successfully!")
