import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="IPL Predictor", layout="centered")

st.title("🏏 IPL Match Winner Predictor")
st.info("Predict winner using teams, toss, venue, city, and season.")

# Load saved files
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("columns.pkl", "rb") as f:
    columns = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Load dataset
data = pd.read_csv("matches.csv")

# Clean names
data.replace({
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings"
}, inplace=True)

# Dropdown values
teams = sorted(set(data["team1"]).union(set(data["team2"])))
venues = sorted(data["venue"].dropna().unique())
cities = sorted(data["city"].dropna().unique())
seasons = sorted(data["season"].dropna().unique())

# UI
team1 = st.selectbox("Team 1", teams)
team2 = st.selectbox("Team 2", [t for t in teams if t != team1])
toss_winner = st.selectbox("Toss Winner", [team1, team2])
toss_decision = st.selectbox("Toss Decision", ["bat", "field"])
venue = st.selectbox("Venue", venues)
city = st.selectbox("City", cities)
season = st.selectbox("Season", seasons)

if st.button("Predict Winner"):

    # Feature engineering
    is_toss_winner_batting = 1 if toss_winner == team1 else 0

    # Create input dataframe
    input_df = pd.DataFrame([{
        "team1": team1,
        "team2": team2,
        "toss_winner": toss_winner,
        "toss_decision": toss_decision,
        "venue": venue,
        "city": city,
        "season": season,
        "is_toss_winner_batting": is_toss_winner_batting
    }])

    # One-hot encode input
    input_encoded = pd.get_dummies(input_df)

    # Match training columns
    input_encoded = input_encoded.reindex(columns=columns, fill_value=0)

    # Predict
    prediction = model.predict(input_encoded)

    # Decode predicted winner
    winner = label_encoder.inverse_transform(prediction)[0]

    st.success(f"🏆 Predicted Winner: {winner}")