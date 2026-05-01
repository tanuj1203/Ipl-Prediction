from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# =========================
# LOAD MODEL
# =========================
with open('advanced_model.pkl', 'rb') as f:
    pipe = pickle.load(f)

# =========================
# LOAD DATA (for features)
# =========================
batting = pd.read_csv('data/batting_stats.csv')
bowling = pd.read_csv('data/bowling_stats.csv')
venues = pd.read_csv('data/venues.csv')
points = pd.read_csv('data/points_table.csv')

# Clean names
batting['team'] = batting['team'].str.lower().str.strip()
bowling['team'] = bowling['team'].str.lower().str.strip()
points['team'] = points['team'].str.lower().str.strip()

# Dictionaries
batting_strength = batting.groupby('team')['runs'].mean().to_dict()
bowling_strength = bowling.groupby('team')['wickets'].mean().to_dict()
points_dict = dict(zip(points['team'], points['points']))

# Teams list
teams = sorted(list(set(batting['team'])))
cities = sorted(list(set(venues['city'])))

# =========================
# HOME
# =========================
@app.route('/')
def home():
    return render_template('index.html', teams=teams, cities=cities)

# =========================
# PREDICT
# =========================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        team1 = request.form['team1'].lower().strip()
        team2 = request.form['team2'].lower().strip()
        city = request.form['city'].lower().strip()

        # =========================
        # REAL FEATURE CALCULATION
        # =========================
        team1_points = points_dict.get(team1, 0)
        team2_points = points_dict.get(team2, 0)
        team_form_diff = team1_points - team2_points

        bat_strength = batting_strength.get(team1, 150)
        bowl_strength = bowling_strength.get(team2, 7)

        # Venue score
        if 'avg_score' in venues.columns:
            venue_row = venues[venues['city'].str.lower() == city]
            avg_score = venue_row['avg_score'].values[0] if not venue_row.empty else 160
        else:
            avg_score = 160

        # =========================
        # CREATE INPUT
        # =========================
        input_df = pd.DataFrame({
            'team1': [team1],
            'team2': [team2],
            'city': [city],
            'team_form_diff': [team_form_diff],
            'batting_strength': [bat_strength],
            'bowling_strength': [bowl_strength],
            'avg_score': [avg_score]
        })

        # =========================
        # PREDICTION
        # =========================
        result = pipe.predict_proba(input_df)

        team1_prob = result[0][1] * 100
        team2_prob = result[0][0] * 100

        if team1_prob > team2_prob:
            winner = team1.title()
            prob = round(team1_prob, 2)
        else:
            winner = team2.title()
            prob = round(team2_prob, 2)

        prediction_text = f"{winner} is likely to win ({prob}% probability)"

        return render_template(
            'index.html',
            prediction_text=prediction_text,
            teams=teams,
            cities=cities
        )

    except Exception as e:
        return f"Error: {str(e)}"

# =========================
# RUN
# =========================
if __name__ == '__main__':
    app.run(debug=True, port=5500)