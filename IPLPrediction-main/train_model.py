import pandas as pd
import pickle

# =========================
# LOAD DATA
# =========================
matches = pd.read_csv('data/matches.csv')
deliveries = pd.read_csv('data/deliveries.csv')
batting = pd.read_csv('data/batting_stats.csv')
bowling = pd.read_csv('data/bowling_stats.csv')
venues = pd.read_csv('data/venues.csv')
points = pd.read_csv('data/points_table.csv')

print("✅ Data loaded...")

# =========================
# CREATE TOTAL RUNS
# =========================
deliveries['runs_of_bat'] = pd.to_numeric(deliveries['runs_of_bat'], errors='coerce').fillna(0)
deliveries['extras'] = pd.to_numeric(deliveries['extras'], errors='coerce').fillna(0)

deliveries['total_runs'] = deliveries['runs_of_bat'] + deliveries['extras']

# total runs per match
total_score = deliveries.groupby('match_no')['total_runs'].sum().reset_index()

# =========================
# MERGE MATCH DATA
# =========================
df = matches.merge(total_score, left_on='match_id', right_on='match_no', how='inner')

print("✅ Merged matches + deliveries")

# =========================
# CLEAN TEXT DATA
# =========================
df['team1'] = df['team1'].str.lower().str.strip()
df['team2'] = df['team2'].str.lower().str.strip()
df['match_winner'] = df['match_winner'].str.lower().str.strip()

points['team'] = points['team'].str.lower().str.strip()
batting['team'] = batting['team'].str.lower().str.strip()
bowling['team'] = bowling['team'].str.lower().str.strip()

# =========================
# TEAM FORM
# =========================
points_dict = dict(zip(points['team'], points['points']))

df['team1_points'] = df['team1'].map(points_dict)
df['team2_points'] = df['team2'].map(points_dict)
df['team_form_diff'] = df['team1_points'] - df['team2_points']

# =========================
# VENUE FEATURE
# =========================
df.rename(columns={'venue': 'city'}, inplace=True)

if 'avg_score' in venues.columns:
    venue_avg = venues[['city', 'avg_score']]
else:
    venue_avg = venues[['city']].copy()
    venue_avg['avg_score'] = 160

df = df.merge(venue_avg, on='city', how='left')

# =========================
# PLAYER STRENGTH
# =========================
batting_strength = batting.groupby('team')['runs'].mean().to_dict()
bowling_strength = bowling.groupby('team')['wickets'].mean().to_dict()

df['batting_strength'] = df['team1'].map(batting_strength)
df['bowling_strength'] = df['team2'].map(bowling_strength)

# =========================
# HANDLE MISSING VALUES
# =========================
df['batting_strength'] = df['batting_strength'].fillna(df['batting_strength'].mean())
df['bowling_strength'] = df['bowling_strength'].fillna(df['bowling_strength'].mean())
df['team_form_diff'] = df['team_form_diff'].fillna(0)
df['avg_score'] = df['avg_score'].fillna(160)

# Drop only critical missing values
df = df.dropna(subset=['team1', 'team2', 'city', 'match_winner'])

print("Final dataset shape:", df.shape)

# =========================
# FINAL FEATURES
# =========================
features = [
    'team1', 'team2', 'city',
    'team_form_diff',
    'batting_strength',
    'bowling_strength',
    'avg_score'
]

df['result'] = (df['match_winner'] == df['team1']).astype(int)

X = df[features]
y = df['result']

print("✅ Features ready...")

# =========================
# MODEL TRAINING
# =========================
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

trf = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'),
     ['team1', 'team2', 'city'])
], remainder='passthrough')

pipe = Pipeline([
    ('transform', trf),
    ('model', RandomForestClassifier(n_estimators=150, random_state=42))
])

pipe.fit(X, y)

# =========================
# SAVE MODEL
# =========================
pickle.dump(pipe, open('advanced_model.pkl', 'wb'))

print("🎉 Model trained successfully and saved!")