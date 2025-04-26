import ssl
import re
import os
import pickle
import string
import nltk
from nltk.corpus import words

## from https://stackoverflow.com/questions/38916452/nltk-download-ssl-certificate-verify-failed
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('words')

english_vocab = set(words.words())

id_pattern = re.compile(r'^[a-zA-Z0-9]{22},')

current_song_lyrics = []
first_id = False
first_line = True

english_check = True
non_english_count = 0

all_songs = []
all_genres = []

folder_path = '../../data/'
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    genre_file_name = os.path.splitext(filename)[0]
    genre = genre_file_name.split('_')[0]
    if genre != "my" and genre != "test" and genre != "test2":
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if first_line == True:
                    first_line = False
                    continue
                else:
                    if id_pattern.match(line):
                        if first_id == False:
                            first_id = True
                        else:
                            print(current_song_lyrics)
                            if english_check == True and len(current_song_lyrics) > 0:
                                lyrics_in_song = ''.join(current_song_lyrics).strip()
                                all_songs.append(lyrics_in_song)
                                all_genres.append(genre) 
                            current_song_lyrics = []
                            english_check = True
                            non_english_count = 0
                    else:
                        no_punct = line.translate(str.maketrans('', '', string.punctuation))
                        line_words = no_punct.split()
                        for word in line_words:
                            if word.lower() not in english_vocab:
                                non_english_count += 1
                                if non_english_count >= 20:
                                    english_check = False
                        current_song_lyrics.append(line)  

            if current_song_lyrics:
                if english_check == True and len(current_song_lyrics) > 0:
                    lyrics_in_song = ''.join(current_song_lyrics).strip()
                    all_songs.append(lyrics_in_song)   
                    all_genres.append(genre)  