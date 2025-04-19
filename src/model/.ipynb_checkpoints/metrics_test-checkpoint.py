import metrics

sample_lyrics = """
I walk this empty street
On the Boulevard of Broken Dreams
Where the city sleeps
And I'm the only one and I walk alone
I walk alone I walk alone
"""

genre_dict = {
    "rock": [sample_lyrics],
    "pop": ["Baby baby baby oh\nYou know you love me\nYou know you care"],
}

print("+===== Metric Report =====+")
print("Average line length:", metrics.average_line_length(sample_lyrics))
print("Song word variation:", metrics.song_word_variation(sample_lyrics))
print("Genre word variation:", metrics.genre_word_variation(genre_dict))
print("I vs You score:", metrics.i_vs_you_score(sample_lyrics))
print("Word repetition count:", metrics.count_word_repetitions(sample_lyrics))
print("+=========================+")
