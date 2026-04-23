import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load data
df = pd.read_csv("matches.csv")

# Clean team names
df.replace({
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings"
}, inplace=True)

# Drop rows without winner
df = df.dropna(subset=["winner"])

# Feature engineering
df["is_toss_winner_batting"] = (df["toss_winner"] == df["team1"]).astype(int)

# Features to use
features = [
    "team1", "team2",
    "toss_winner", "toss_decision",
    "venue", "city", "season",
    "is_toss_winner_batting"
]

X = df[features]
y = df["winner"]

# One-hot encoding
X_encoded = pd.get_dummies(X)

# Save columns
with open("columns.pkl", "wb") as f:
    pickle.dump(X_encoded.columns, f)

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save label encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_encoded, y_encoded)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained successfully")