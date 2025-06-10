from flask import Flask, request, render_template
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# Load and prepare the data
data = pd.read_csv('data.csv')

# Normalize input for matching
data['name_lower'] = data['name'].str.lower().str.strip()

# Features used for recommendation
features = ['danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'valence', 'tempo']

# Scale features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[features])

# Fit Nearest Neighbors model
model = NearestNeighbors(n_neighbors=6, metric='cosine')
model.fit(scaled_features)

@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    song_name = ""

    if request.method == "POST":
        song_name = request.form["song"].strip().lower()

        if song_name in data['name_lower'].values:
            idx = data[data['name_lower'] == song_name].index[0]
            distances, indices = model.kneighbors([scaled_features[idx]])

            for i in indices[0][1:]:
                recommendations.append({
                    "name": data.iloc[i]["name"],
                    "artist": data.iloc[i].get("artists", "Unknown Artist")
                })
        else:
            recommendations.append({"name": "Song not found", "artist": ""})

    return render_template("index.html", recommendations=recommendations, song=song_name.title())

if __name__ == "__main__":
    app.run(debug=True)
