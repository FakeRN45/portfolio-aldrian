import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from tqdm import tqdm

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

test_passenger_id = test["PassengerId"]

drop_cols = ["PassengerId", "Name", "Ticket", "Cabin"]
X = train.drop(columns=drop_cols + ["Survived"])
y = train["Survived"]
X_test = test.drop(columns=drop_cols)

for col in ["Sex", "Embarked"]:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

X = X.fillna(X.median(numeric_only=True))
X_test = X_test.fillna(X_test.median(numeric_only=True))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X_scaled)
X_test_poly = poly.transform(X_test_scaled)

X_train, X_val, y_train, y_val = train_test_split(
    X_poly, y, test_size=0.2, random_state=42, stratify=y
)

xgb = XGBClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=7,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
    random_state=42
)

cat = CatBoostClassifier(
    iterations=500,
    learning_rate=0.03,
    depth=7,
    eval_metric="Accuracy",
    verbose=0,
    random_state=42
)

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=12,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

meta_model = LogisticRegression(max_iter=2000, random_state=42)

stacking = StackingClassifier(
    estimators=[("xgb", xgb), ("cat", cat), ("rf", rf)],
    final_estimator=meta_model,
    cv=5,
    n_jobs=-1
)

print("Training models...")
for _ in tqdm(range(1), desc="Training Progress"):
    stacking.fit(X_train, y_train)

y_pred = stacking.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print(f"\nValidation Accuracy: {acc:.5f}")

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(stacking, X_poly, y, cv=cv, scoring="accuracy", n_jobs=-1)
print(f"Cross-Validation Mean Score (10-fold): {cv_scores.mean():.5f}")
print(f"Cross-Validation Std Dev             : {cv_scores.std():.5f}")

print("Predictions...")
y_test_pred = stacking.predict(X_test_poly)

submission = pd.DataFrame({
    "PassengerId": test_passenger_id,
    "Survived": y_test_pred
})
submission.to_csv("submission.csv", index=False)
print("submission.csv saved!")