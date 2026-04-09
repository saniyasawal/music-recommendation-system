import joblib
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

# ==============================
# PATH SETUP 
# ==============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONTENT_PATH = os.path.join(BASE_DIR, "models", "content")
COLLAB_PATH = os.path.join(BASE_DIR, "models", "collaborative")


# ==============================
# CONTENT BASED RECOMMENDATION
# ==============================

def predict_content(song_name, top_n=5):

    model_name = joblib.load(
        os.path.join(CONTENT_PATH, "model_name.pkl")
    )

    df = joblib.load(
        os.path.join(CONTENT_PATH, "data.pkl")
    )

    song_name = song_name.lower().strip()

    df["Track_lower"] = df["Track"].str.lower().str.strip()

    if song_name not in df["Track_lower"].values:
        return ["Song not found"]

    index = df[
        df["Track_lower"] == song_name
    ].index[0]

    # ==========================
    # TF-IDF & COSINE
    # ==========================

    if model_name in ["tfidf", "cosine"]:

        similarity = joblib.load(
            os.path.join(CONTENT_PATH, "similarity.pkl")
        )

        scores = list(enumerate(similarity[index]))

        scores = sorted(
            scores,
            key=lambda x: x[1],
            reverse=True
        )

        # remove same song
        scores = [s for s in scores if s[0] != index]

        #REMOVE DUPLICATES
        seen = set()
        recommendations = []

        for i in scores:
            track = df.iloc[i[0]]["Track"]

            track_clean = track.lower().strip()

            if track_clean not in seen:
                recommendations.append(track)
                seen.add(track_clean)

            if len(recommendations) == top_n:
                break

        return recommendations


    # ==========================
    # KNN MODEL
    # ==========================

    elif model_name == "knn":

        knn = joblib.load(
            os.path.join(CONTENT_PATH, "model.pkl")
        )

        scaler = joblib.load(
            os.path.join(CONTENT_PATH, "scaler.pkl")
        )

        features = [
            'Danceability','Energy','Loudness','Speechiness',
            'Acousticness','Instrumentalness','Liveness',
            'Valence','Tempo','Popularity','mood_score','intensity'
        ]

        scaled = scaler.transform(df[features])

        distances, indices = knn.kneighbors(
            scaled[index].reshape(1, -1),
            n_neighbors=top_n + 10
        )

        indices = indices.flatten()[1:]

        #  REMOVE DUPLICATES
        seen = set()
        recommendations = []

        for i in indices:
            track = df.iloc[i]["Track"]
            track_clean = track.lower().strip()

            if track_clean not in seen:
                recommendations.append(track)
                seen.add(track_clean)

            if len(recommendations) == top_n:
                break

        return recommendations


# ==============================
# COLLABORATIVE FILTERING
# ==============================

def predict_collaborative(artist_name, top_n=5):

    model_name = joblib.load(
        os.path.join(COLLAB_PATH, "model_name.pkl")
    )

    matrix = joblib.load(
        os.path.join(COLLAB_PATH, "matrix.pkl")
    )

    if not hasattr(matrix, "columns"):
        return ["Matrix format error"]

    artists = list(matrix.columns)

    artist_name = artist_name.lower().strip()

    artists_lower = [
        str(a).lower().strip()
        for a in artists
    ]

    if artist_name not in artists_lower:
        return ["Artist not found"]

    index = artists_lower.index(artist_name)

    matrix_values = np.array(matrix)
    matrix_values = np.nan_to_num(matrix_values)


    # ==========================
    # USER-USER / ITEM-ITEM
    # ==========================

    if model_name in ["user_user", "item_item"]:

        similarity = joblib.load(
            os.path.join(COLLAB_PATH, "similarity.pkl")
        )

        similarity = np.nan_to_num(np.array(similarity))

        if index >= similarity.shape[0]:
            return ["Not enough data for this artist"]

        scores = list(enumerate(similarity[index]))


    # ==========================
    # SVD 
    # ==========================

    elif model_name == "svd":

        item_matrix = matrix_values.T

        if index >= item_matrix.shape[0]:
            return ["Not enough data for this artist"]

        target_vector = item_matrix[index]

        if np.all(target_vector == 0):
            return ["Not enough data for this artist"]

        similarities = cosine_similarity(
            [target_vector],
            item_matrix
        )[0]

        scores = list(enumerate(similarities))


    # ==========================
    # SORT + CLEAN
    # ==========================

    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )

    # remove same artist
    scores = [s for s in scores if s[0] != index]

    if len(scores) == 0:
        return ["No recommendations available"]

    # REMOVE DUPLICATES
    seen = set()
    recommendations = []

    for i in scores:
        artist = artists[i[0]]
        artist_clean = str(artist).lower().strip()

        if artist_clean not in seen:
            recommendations.append(artist)
            seen.add(artist_clean)

        if len(recommendations) == top_n:
            break

    return recommendations