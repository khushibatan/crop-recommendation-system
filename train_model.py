import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
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
for name, model in models.items():
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"\n{name}")
    print("-" * 40)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_test, predictions))

# 6. Save the best-performing model
best_model = models["Random Forest"]

joblib.dump(
    {
        "model": best_model,
        "features": list(X.columns),
        "classes": list(best_model.classes_)
    },
    "crop_model.pkl"
)

print("\nRandom Forest model saved successfully!")