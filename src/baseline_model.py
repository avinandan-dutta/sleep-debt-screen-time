
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# 1. Load the dataset
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "bedtime_screentime_sleep_debt.csv"

df = pd.read_csv(DATA_PATH)

# 2. Define target and input features
TARGET = "sleep_debt_category"

X = df.drop(
    columns=["user_id", TARGET, "next_day_fatigue_score"],
    errors="ignore",
)
y = df[TARGET]

# 3. Identify numerical and categorical columns
numeric_columns = X.select_dtypes(include="number").columns.tolist()
categorical_columns = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

# 4. Prepare numerical features
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# 5. Prepare categorical features
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

# 6. Combine preprocessing steps
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns),
])

# 7. Split the data once for a fair comparison
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# 8. Define both classification models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    ),
}

results = []

# 9. Train and evaluate both models
for name, classifier in models.items():
    model = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("classifier", classifier),
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Macro F1": macro_f1,
    })

    print(f"\n{'=' * 50}")
    print(f"Model: {name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        zero_division=0,
    ))

# 10. Compare model results
results_df = pd.DataFrame(results).sort_values(
    by="Macro F1",
    ascending=False,
)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

print(f"\nTraining rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")
print(f"Number of classes: {y.nunique()}")