import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.title("🎵 Music Recommendation System")


# CONTENT BASED
st.header("Content Based Recommendation")

song = st.text_input(
    "Enter Song Name"
)

if st.button("Recommend songs"):

    r = requests.post(
        f"{API_URL}/recommend/content",
        json={"name": song}
    )

    if r.status_code == 200:

        st.write(
            r.json()["recommendations"]
        )

    else:

        st.error(r.text)



# COLLABORATIVE
st.header("Collaborative Filtering")

artist = st.text_input(
    "Enter Artist Name"
)

if st.button("Recommend artists"):

    r = requests.post(
        f"{API_URL}/recommend/collaborative",
        json={"name": artist}
    )

    if r.status_code == 200:

        st.write(
            r.json()["recommendations"]
        )

    else:

        st.error(r.text)