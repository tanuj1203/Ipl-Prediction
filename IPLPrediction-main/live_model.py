import pandas as pd
import pickle

deliveries = pd.read_csv('data/deliveries.csv')
matches = pd.read_csv('data/matches.csv')

# Merge
df = deliveries.merge(matches, left_on='match_no', right_on='match_id')

# Create features
df['total_runs'] = df['runs_of_bat'] + df['extras']

df['current_score'] = df.groupby('match_no')['total_runs'].cumsum()
df['balls_bowled'] = df.groupby('match_no').cumcount() + 1

df['runs_left'] = df['first_ings_score'] - df['current_score']
df['balls_left'] = 120 - df['balls_bowled']

df['wickets'] = df.groupby('match_no')['player_dismissed'].apply(lambda x: x.notna().cumsum())

df['crr'] = df['current_score'] / (df['balls_bowled'] / 6)
df['rrr'] = (df['runs_left'] * 6) / df['balls_left']

df = df.dropna()

# Target
df['result'] = (df['match_winner'] == df['batting_team']).astype(int)

X = df[['batting_team','bowling_team','venue','runs_left','balls_left','wickets','crr','rrr']]
y = df['result']

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

trf = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['batting_team','bowling_team','venue'])
], remainder='passthrough')

pipe = Pipeline([
    ('trf', trf),
    ('model', RandomForestClassifier(n_estimators=200))
])

pipe.fit(X, y)

pickle.dump(pipe, open('live_model.pkl','wb'))

print("🔥 Live model ready!")