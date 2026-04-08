import joblib
import numpy as np
import os

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


    # TF-IDF MODEL
    if model_name == "tfidf":

        similarity = joblib.load(
            os.path.join(CONTENT_PATH, "similarity.pkl")
        )

        scores = list(
            enumerate(similarity[index])
        )


    # COSINE NUMERIC MODEL
    elif model_name == "cosine":

        similarity = joblib.load(
            os.path.join(CONTENT_PATH, "similarity.pkl")
        )

        scores = list(
            enumerate(similarity[index])
        )


    # KNN MODEL
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

        scaled = scaler.transform(
            df[features]
        )

        distances, indices = knn.kneighbors(
            scaled[index].reshape(1, -1),
            n_neighbors=top_n+1
        )

        indices = indices.flatten()[1:]

        return df.iloc[
            indices
        ]["Track"].tolist()


    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )[1:top_n+1]


    recommendations = [

        df.iloc[i[0]]["Track"]

        for i in scores

    ]


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

    # ensure dataframe
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


    # convert matrix safely
    matrix_values = np.array(matrix)

    # handle NaN
    matrix_values = np.nan_to_num(matrix_values)


    # USER-USER or ITEM-ITEM
    if model_name in ["user_user", "item_item"]:

        similarity = joblib.load(
            os.path.join(COLLAB_PATH, "similarity.pkl")
        )

        similarity = np.nan_to_num(
            np.array(similarity)
        )

        if index >= similarity.shape[0]:
            return ["Not enough data for this artist"]

        scores = list(
            enumerate(similarity[index])
        )


    # SVD MODEL
    elif model_name == "svd":

        if index >= matrix_values.shape[0]:
            return ["Not enough data for this artist"]

        target_vector = matrix_values[index]

        # if vector all zeros
        if np.all(target_vector == 0):
            return ["Not enough data for this artist"]

        similarities = np.dot(
            matrix_values,
            target_vector
        )

        scores = list(
            enumerate(similarities)
        )


    # sort safely
    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )


    # remove same artist
    scores = [
        s for s in scores
        if s[0] != index
    ]


    if len(scores) == 0:
        return ["No recommendations available"]


    scores = scores[:top_n]


    recommendations = [

        artists[i[0]]

        for i in scores

    ]


    return recommendations