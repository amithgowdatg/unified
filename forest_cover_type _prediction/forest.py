!pip install -q xgboost catboost seaborn

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

df = pd.read_csv("train.csv")

X = df.drop(columns=["Cover_Type"])
y = df["Cover_Type"] - 1

for col in [
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Horizontal_Distance_To_Fire_Points",
]:
    X[col] = X[col].clip(lower=0)
    X[col] = np.log1p(X[col])

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=7,
    random_state=42,
    eval_metric="mlogloss",
    n_jobs=-1,
    use_label_encoder=False,
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=40,
    verbose=False
)

y_pred = np.argmax(xgb_model.predict_proba(X_val), axis=1)

print("XGBoost Accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred, target_names=[
    "Spruce/Fir", "Lodgepole Pine", "Ponderosa Pine", "Cottonwood/Willow",
    "Aspen", "Douglas-fir", "Krummholz"
]))

cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

joblib.dump(xgb_model, "forest_cover_xgb.pkl")

cat = CatBoostClassifier(
    depth=10,
    learning_rate=0.05,
    iterations=500,
    loss_function="MultiClass",
    random_state=42,
    verbose=False,
)

train_pool = Pool(X_train, y_train)
val_pool   = Pool(X_val, y_val)

cat.fit(train_pool, eval_set=val_pool, early_stopping_rounds=40)

y_pred_cat = np.argmax(cat.predict_proba(val_pool), axis=1)
print("CatBoost Accuracy:", accuracy_score(y_val, y_pred_cat))
