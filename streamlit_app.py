import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="IPL Predictor", layout="centered")

st.title("🏏 IPL Match Winner Predictor")
st.info("This model uses enhanced features like venue, city, season, and toss impact.")

# Load model & columns
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# Load dataset
data = pd.read_csv("matches.csv")

# Clean team names
data.replace({
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings"
}, inplace=True)

teams = sorted(set(data["team1"]).union(set(data["team2"])))
venues = sorted(data["venue"].dropna().unique())
cities = sorted(data["city"].dropna().unique())
seasons = sorted(data["season"].dropna().unique())

# UI Inputs
team1 = st.selectbox("Team 1", teams)
team2 = st.selectbox("Team 2", [t for t in teams if t != team1])

toss_winner = st.selectbox("Toss Winner", [team1, team2])
toss_decision = st.selectbox("Toss Decision", ["bat", "field"])

venue = st.selectbox("Venue", venues)
city = st.selectbox("City", cities)
season = st.selectbox("Season", seasons)

# Prediction
if st.button("Predict Winner"):

    is_toss_winner_batting = int(toss_winner == team1)

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

    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=columns, fill_value=0)

    prediction = model.predict(input_encoded)[0]

    st.success(f"🏆 Predicted Winner: {prediction}")