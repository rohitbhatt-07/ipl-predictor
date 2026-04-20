import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
data = pd.read_csv("C:\\Users\\rohit\\Desktop\\ML Presentation\\matches.csv")

# Drop useless column and null rows
if "umpire3" in data.columns:
    data.drop("umpire3", axis=1, inplace=True)
data.dropna(inplace=True)

# Replace old team names
data["team1"] = data["team1"].replace("Delhi Daredevils", "Delhi Capitals")
data["team2"] = data["team2"].replace("Delhi Daredevils", "Delhi Capitals")
data["winner"] = data["winner"].replace("Delhi Daredevils", "Delhi Capitals")

data["team1"] = data["team1"].replace("Deccan Chargers", "Sunrisers Hyderabad")
data["team2"] = data["team2"].replace("Deccan Chargers", "Sunrisers Hyderabad")
data["winner"] = data["winner"].replace("Deccan Chargers", "Sunrisers Hyderabad")

# Drop unnecessary columns
drop_cols = ["id", "season", "city", "date", "player_of_match", "venue", "umpire1", "umpire2"]
drop_cols = [col for col in drop_cols if col in data.columns]
data.drop(drop_cols, axis=1, inplace=True)

# Features and target
X = data.drop("winner", axis=1)
y = data["winner"]

# Encoding
X = pd.get_dummies(X, columns=["team1", "team2", "toss_winner", "toss_decision", "result"], drop_first=True)
le = LabelEncoder()
y = le.fit_transform(y)

# Split data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=200, min_samples_split=3, max_features="sqrt", random_state=42)
model.fit(x_train, y_train)

# Predict
y_pred = model.predict(x_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))