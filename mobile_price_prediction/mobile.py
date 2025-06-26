!pip install -q catboost seaborn

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from catboost import CatBoostClassifier, Pool

df = pd.read_csv("liver_cirrhosis.csv")

df = df.dropna(subset=["Stage"])

df["Status"] = df["Status"].map({"C": 0, "CL": 1, "D": 2})
df["Drug"] = df["Drug"].map({"D-penicillamine": 0, "placebo": 1})
df["Sex"] = df["Sex"].map({"M": 0, "F": 1})
df["Ascites"] = df["Ascites"].map({"N": 0, "Y": 1})
df["Hepatomegaly"] = df["Hepatomegaly"].map({"N": 0, "Y": 1})
df["Spiders"] = df["Spiders"].map({"N": 0, "Y": 1})
df["Edema"] = df["Edema"].map({"N": 0, "S": 1, "Y": 2})

df = df.dropna()

X = df.drop(columns=["Stage"])
y = df["Stage"] - 1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = CatBoostClassifier(
    iterations=800,
    learning_rate=0.05,
    depth=6,
    loss_function="MultiClass",
    eval_metric="Accuracy",
    verbose=False,
    random_state=42
)

model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=40)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Liver Cirrhosis Stage Detection")
plt.show()

model.save_model("liver_cirrhosis_stage_catboost_model.cbm")
