"""
metrics_clean_data.py
----------------------
Loads lyrics and inferred genre labels from .csv files in the data/ directory.
Each .csv file should have a column named 'lyrics'. The genre is inferred from the filename.

Returns:
    - all_songs: List of lyrics strings
    - all_genres: List of genre labels (aligned with each song)
"""

import os
import pandas as pd

def collect_cleaned_lyrics(folder_path="data"):
    all_songs = []
    all_genres = []

    for filename in os.listdir(folder_path):
        if not filename.endswith(".csv"):
            continue  # Skip non-csv files

        genre = os.path.splitext(filename)[0].lower()
        file_path = os.path.join(folder_path, filename)

        try:
            df = pd.read_csv(file_path, encoding="utf-8", on_bad_lines="skip")
            if "lyrics" not in df.columns:
                print(f"**Skipping** {filename}: no 'lyrics' column found.")
                continue

            # Drop NaN or empty lyrics rows
            df = df.dropna(subset=["lyrics"])
            df = df[df["lyrics"].str.strip().astype(bool)]

            for lyric in df["lyrics"]:
                all_songs.append(lyric.strip())
                all_genres.append(genre)

        except Exception as e:
            print(f" !!Error!! reading {filename}: {e}")

    return all_songs, all_genres
