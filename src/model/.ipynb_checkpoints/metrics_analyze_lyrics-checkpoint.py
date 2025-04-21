"""
This file is designed to load all songs and genres from
metrics_analyze_lyrics.py then compute the metrics 
using the methods provided in metrics.py.

Next, we will generate visualizations with matplotlib.pyplot.
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from src.metrics import (
    average_line_length, song_word_variation, genre_word_variation, i_vs_you_score, count_word_repetitions
)
from src.metrics_clean_data import collect_cleaned_lyrics

# Step 1: Load cleaned data
songs, genres = collect_cleaned_lyrics("data")

# Step 2: Organize by genre
genre_data = defaultdict(list)
for lyric, genre in zip(songs, genres):
    genre_data[genre].append(lyric)

# Step 3: Compute metrics for each song
results = []
for genre, lyric_list in genre_data.items():
    for lyrics in lyric_list:
        results.append({
            "genre": genre,
            "avg_line_length": average_line_length(lyrics),
            "word_variation": song_word_variation(lyrics),
            "i_vs_you": i_vs_you_score(lyrics),
            "repetitions": count_word_repetitions(lyrics)
        })

df = pd.DataFrame(results)

# Step 4: Visualiza Metrics
def plot_metric_by_genre(metric_name, ylabel):
    grouped = df.groupby("genre")[metric_name].mean().sort_values()
    plt.figure(figsize=(10,5))
    grouped.plot(kind="bar", color="skyblue", edgecolor="black")
    plt.title(f"Average {metric_name.replace('_','').title()} by Genre")
    plt.ylabel(ylabel)
    plt.xlabel("Genre")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    plt.close()

# Calls to visualization method
plot_metric_by_genre("avg_line_length", "Avg Words per Line")
plot_metric_by_genre("word_variation", "Unique/Total Words")
plot_metric_by_genre("i_vs_you", "'I' vs. 'You' Count")
plot_metric_by_genre("repetitions", "Word Repetitions")