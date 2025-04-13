import re
import os

id_pattern = re.compile(r'^[a-zA-Z0-9]{22},')

current_song_lyrics = []
first_id = False
all_songs_in_genre = []
first_line = True

all_songs = []
all_genres = []

folder_path = 'data/'
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    genre = os.path.splitext(filename)[0]
    all_songs_in_genre = []
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
                        lyrics_in_song = ''.join(current_song_lyrics).strip()
                        all_songs_in_genre.append(lyrics_in_song)
                        current_song_lyrics = []
                else:
                    current_song_lyrics.append(line)  

        if current_song_lyrics:
            lyrics_in_song = ''.join(current_song_lyrics).strip()
            all_songs_in_genre.append(lyrics_in_song)     

    all_genres.append(genre)
    all_songs.append(all_songs_in_genre)  
    