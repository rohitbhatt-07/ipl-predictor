import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =========================================================
# 1. LOAD DATA
# =========================================================
data = pd.read_csv("matches.csv")

print("Original shape:", data.shape)
print("\nMissing values before cleaning:")
print(data.isnull().sum())

# =========================================================
# 2. DROP HIGH-NULL COLUMN
# =========================================================
if "umpire3" in data.columns:
    data.drop("umpire3", axis=1, inplace=True)

# =========================================================
# 3. DROP ONLY IMPORTANT NULLS
# =========================================================
important_cols = ["winner", "player_of_match", "umpire1", "umpire2", "city"]
existing_important_cols = [col for col in important_cols if col in data.columns]

data.dropna(subset=existing_important_cols, inplace=True)

print("\nShape after cleaning:", data.shape)

# =========================================================
# 4. STANDARDIZE TEAM NAMES
# =========================================================
for col in ["team1", "team2", "winner"]:
    if col in data.columns:
        data[col] = data[col].replace("Delhi Daredevils", "Delhi Capitals")
        data[col] = data[col].replace("Deccan Chargers", "Sunrisers Hyderabad")
        data[col] = data[col].replace("Rising Pune Supergiants", "Rising Pune Supergiant")

# =========================================================
# 5. DROP UNNECESSARY COLUMNS
# =========================================================
drop_cols = [
    "id", "season", "city", "date",
    "player_of_match", "venue", "umpire1", "umpire2"
]
existing_drop_cols = [col for col in drop_cols if col in data.columns]
data.drop(existing_drop_cols, axis=1, inplace=True)

print("\nColumns after dropping unnecessary ones:")
print(data.columns.tolist())

# =========================================================
# 6. FEATURES AND TARGET
# =========================================================
X = data.drop("winner", axis=1)
y = data["winner"]

# =========================================================
# 7. ENCODE FEATURES
# =========================================================
categorical_cols = ["team1", "team2", "toss_winner", "toss_decision", "result"]
existing_categorical_cols = [col for col in categorical_cols if col in X.columns]

X = pd.get_dummies(X, columns=existing_categorical_cols, drop_first=True)

# =========================================================
# 8. ENCODE TARGET
# =========================================================
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print("\nFinal X shape:", X.shape)
print("Final y length:", len(y))

# Safety check
if len(X) == 0:
    raise ValueError("Dataset is empty after cleaning. Please check your cleaning steps.")

# =========================================================
# 9. TRAIN-TEST SPLIT
# =========================================================
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================================================
# 10. TRAIN MODEL
# =========================================================
model = RandomForestClassifier(
    n_estimators=200,
    min_samples_split=3,
    max_features="sqrt",
    random_state=42
)

model.fit(x_train, y_train)

# =========================================================
# 11. PREDICT
# =========================================================
y_pred = model.predict(x_test)

# =========================================================
# 12. EVALUATION
# =========================================================
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# =========================================================
# 13. SAVE FILES
# =========================================================
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("\nModel files created successfully")