)

🏏 IPL Live Match Predictor (Cricbuzz-Style)

A machine learning-powered web application that predicts the winning probability of IPL teams in real-time based on match situation — inspired by platforms like Cricbuzz.

🚀 Features
📊 Live Match Prediction
Predicts winning team based on current match situation
⚡ Ball-by-Ball Logic
Uses runs left, balls left, wickets, CRR & RRR
🤖 Machine Learning Model
Trained on historical IPL ball-by-ball data
🌐 Flask Web App
Simple and interactive UI
🎯 Dynamic Probability Output
Shows winning chances in percentage
🧠 How It Works
🔹 Input Parameters
Batting Team
Bowling Team
City (Venue)
Target Score
Current Score
Overs Completed
Wickets Fallen
🔹 Feature Engineering

The system calculates:

runs_left = target - current_score
balls_left = 120 - balls_bowled
current_run_rate (CRR)
required_run_rate (RRR)
🔹 Model Prediction

A Random Forest Classifier predicts:

Win Probability (%)
Losing Probability (%)
🏗️ Tech Stack
Layer	Technology
Frontend	HTML, CSS
Backend	Python (Flask)
ML Model	Scikit-learn
Data Processing	Pandas
Model Type	Random Forest
📁 Project Structure
IPLPrediction/
│
├── app.py                # Flask backend
├── live_model.py         # Model training script
├── live_model.pkl        # Trained ML model
│
├── data/
│   ├── matches.csv
│   ├── deliveries.csv
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
└── README.md
⚙️ Installation & Setup
1️⃣ Clone Repository
git clone https://github.com/your-username/IPL-Predictor.git
cd IPL-Predictor
2️⃣ Install Dependencies
pip install pandas scikit-learn flask
3️⃣ Train Model
python live_model.py
4️⃣ Run Application
python app.py
5️⃣ Open in Browser
http://127.0.0.1:5000
📸 Screenshots

(Add screenshots here for better GitHub visibility)

🔥 Future Enhancements
📡 Live API integration (real IPL matches)
📊 Win probability graph (over-by-over)
🧑‍🤝‍🧑 Player performance impact
📱 Responsive mobile UI
☁️ Deployment (Render / AWS / Vercel)
💡 Use Cases
Cricket analytics
Sports prediction systems
Machine learning portfolio project
Data science practice
⚠️ Disclaimer

This project is for educational purposes only.
Predictions are based on historical data and may not reflect real match outcomes.

👨‍💻 Author

Tanuj Kumar Mishra
📍 Prayagraj, India
