# CSCI3832 - Natural Language Processing - FinalProject

## Note: This repository was migrated from my former academic GitHub account @gaha4495

# NLP-Based Analysis and Generation of Song Lyrics Using Spotify Data

## Project Overview

This project aims to implement and experiment with Natural Language Processing (NLP) models to classify songs into their respective genres and generate new lyrics based on genres. We utilize accuracy, precision, and F-1 to evaluate the classifier, and utilize human evaluation and metric comparisons to evaluate the quality of generated lyrics. The project leverages the Spotify API and Genius API to extract lyrics and metadata. The models used to complete the task are the Uncased Base BERT model for Sequence classification and LSTM for lyric generation. 


## Index

- [Key Tasks](#key-tasks)
  - [Data Collection](#data-collection)
  - [Text Classification](#text-classification)
  - [Lyric Generation](#lyric-generation)
- [Project Goals](#project-goals)
- [Team Structure](#team-structure)
- [Data Collection Process](#data-collection-process)
  - [Spotify API Integration](#spotify-api-integration)
  - [Genius API Integration](#genius-api-integration)
  - [Data Pipeline](#data-pipeline)
- [Technologies and Libraries](#technologies-and-libraries)
  - [Data Collection](#data-collection)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Feature Extraction](#feature-extraction)
- [Text Classification](#text-classification)
   - [About the Model](#about-the-model)
   - [Data](#data)
   - [Final Results for Bert](#final-results-for-bert)
- [Lyric Generation](#lyric-generation)
   - [Model Selection](#model-selection)
   - [Model Implementation](#model-implementation)
      - [Libraries Used](#libraries-used)
      - [SRC](#src)
         - [LSTM.py](#lstmpy)
         - [LSTMtraining.pynb](#lstmtrainingpynb)
         - [lyric_generator.pynb](#lyric_generatorpynb)
   - [Usage](#usage)
   - [Final Results](#final-results)
      - [Metrics](#metrics)
      - [Human Evaluation](#human-evaluation)
   - [Future Steps](#future-steps)
- [Acknowledgments](#acknowledgments)


## Key Tasks

1. **Data Collection:**
   - Collect songs and genres
2. **Text Classification:**
   - Classify song lyrics by genre (e.g., rock, pop, hiphop).
3. **Lyric Generation:**
   - Generate lyrics by genre (e.g., rock, pop, hiphop).


## Project Goals

- Provide insights into how NLP can enhance music-related applications.
- Address challenges such as dataset bias and computational efficiency.
- Identify challenges in lyric generation, experiment with various solutions

## Team Structure

The project is divided into three sub-teams:

- **Data Preprocessing and Classification:** Ima Mervin
- **NLP Model Task Teams:**
  - **Team BERT for Classification:** Mia Ray and Mariana Vadas-Arendt.
  - **Team LSTM for Generation:** Gavin Hanville and Chloe Circenis.

# Data Collection Process

### Spotify API Integration
This project collects song metadata and audio features using the Spotify Web API:

**Track Metadata Collection**: We use the Spotify API to collect basic information about tracks including name, artists, album, release date, and popularity.

### Genius API Integration

**Automatic Matching**: The system attempts to match Spotify tracks with their corresponding entries on Genius using artist name and track title.

**Lyrics Scraping**: Once a match is found, the lyrics are scraped from the Genius page using web scraping techniques with BeautifulSoup.

### Data Pipeline
The complete data collection process follows these steps:

1. Collect track IDs through playlists, searches, or genre recommendations
2. Fetch track metadata and audio features from Spotify
3. Match tracks to Genius entries
4. Scrape and process lyrics
5. Combine all data into a structured dataset

***Note:*** Due to API rate limits, the collection process includes small delays between requests to avoid being blocked by either service.

## Technologies and Libraries

### Data Collection
- **Spotipy**: Python library for the Spotify Web API
  - Handles authentication via SpotifyClientCredentials
  - Manages API requests for track metadata and audio features
  
- **BeautifulSoup4**: For web scraping lyrics from Genius pages

- **Pandas**: Used for data manipulation and CSV export

- **Requests**: Handles HTTP requests to the Genius API

- **Argparse**: Provides command-line argument parsing for flexible data collection options

- **Time**: Implements rate limiting to avoid API request throttling

- **OS/Sys**: Manages file paths and environment variables

## Project Structure
```bash

project/
├── config/
│   └── config.py         # Configuration settings and API credentials
├── data/                 # Data storage directory
│   └── spotify_dataset.csv   # Default output location
├── src/
│   ├── collector.py      # Spotify data collection module
├── venv/                 # Virtual environment folder
│   ├──                   # Any additional files generated by venv
├── requirements.txt      # Project dependencies
├── .env                  # Environment variables (not in version control)
├── .env.example          # Environment variables example structure
├── .gitignore            
└── README.md
```

## Getting Started

1. **Clone the Repository:**
   ```bash
   git clone git@github.com:ima-mervin/CSCI3832_FinalProject.git
   ```
2. **Navigate to your project directory:**
   ```bash
   cd path/to/CSCI3832_FinalProject
   ```
3. **Environment Setup:**
   Create a ```.env``` file in the project root:
   ```bash touch.env```
4. **Setting up API credentials:**
   You can follow the instructions in ```.env.example``` to make your own ```.env```
   ```bash
   # Spotify API Credentials
   # Create these at https://developer.spotify.com/dashboard/
   SPOTIFY_CLIENT_ID=your_client_id_here
   SPOTIFY_CLIENT_SECRET=your_client_secret_here
   
   # Genius API Credentials
   # Create this at https://genius.com/api-clients/new
   GENIUS_ACCESS_TOKEN =your_genius_access_token
   
   # Default output file
   DEFAULT_OUTPUT_FILE = "your_path_to_data/data_file_name.csv"
   ```
5. **Create a virtual environment (this keeps your project dependencies isolated):**
   ```bash
   python -m venv <name_of_environment>
   ```
6. **Activate the virtual environment:**

   *On Windows:*
      ```bash
      <name_of_environment>\Scripts\activate
      ```
   *On macOS/Linux:*
      ```bash
      source <name_of_environment>/bin/activate
      ```
7. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```
8. **Running Scripts:**
   
  *Make sure your virtual environment is activated (you should see (venv) at the beginning of your command line)*
  
 *The script can be run in several different ways depending on what you want to accomplish, here are examples for a few data collection tasks you can run:*

   -**By Playlist**: To collect data from a specific playlist:```python -m src.collector --playlist "spotify_playlist"```
   
   -**By Search**: Search for playlists matching a term and collect data from the top result:```python -m src.collector --search "pop hits"```
   
   -**By Genre**: Collect recommended tracks based on specific genres:```python -m src.collector --genres "rock,pop,hip hop,jazz,country"```
   
   -**By Sentiment**:Collect recommended tracks based on specific genres:```python -m src.collector --sentiments "happy,sad,energetic,relaxed,angry"```
   
   -**Additional Options**: 
      - Limit the number of tracks::```--limit 50```
      - Specify output location: ```--output "data/custom_filename.csv"```

   ```bash
      #Example Commands
      # Collect from a specific playlist
      python -m src.collector --playlist "37i9dQZF1DXcBWIGoYBM5M"
      
      # Search for an indie playlist and limit to 30 tracks
      python -m src.collector --search "indie essentials" --limit 30
      
      # Get recommendations for dance music with custom output file
      python -m src.collector --genres "dance,electronic" --output "data/dance_tracks.csv"
      
   ```

9. **Viewing the Data:**
  Once the script has been successfully run, the data will be loaded into ```your_path/CSCI3832_FinalProject/data/spotify_dataset.csv``` or a custom path if that has been specified.

## Feature Extraction

The collector extracts the following data for each track:

| Column Name | Description |
|-------------|-------------|
| track_id | Spotify unique identifier for the track |
| track_name | Title of the song |
| track_number | Position of the track on its album |
| disc_number | Disc number for multi-disc albums |
| duration_ms | Duration of the track in milliseconds |
| explicit | Flag indicating explicit content |
| popularity | Spotify popularity score |
| isrc | International Standard Recording Code |
| preview_url | URL to a 30-second preview of the track |
| track_url | Spotify URL for the track |
| playlist_id | ID of the source playlist |
| playlist_name | Name of the source playlist |
| added_at | Timestamp when track was added to playlist |
| added_by | User who added the track to playlist |
| primary_artist | Main artist of the track |
| all_artists | List of all artists featured on the track |
| artist_id | Spotify ID for the primary artist |
| artist_genre | Genres associated with the artist |
| album_name | Name of the album |
| album_id | Spotify ID for the album |
| album_type | Type of album (single, album, compilation) |
| album_release_date | Release date of the album |
| album_image_url | URL to the album cover art |
| lyrics | Full lyrics of the track |

# Text Classification

## About the Model

We used the 12 layer uncased Bert model from huggingface.


      HYPERPARAMETERS:

      batch_size = 32
      learning_rate = 2e-5
      warmup_steps = 0
      epochs = 5
      max_allowed_unknowns = 100
      num_genres = 14
      limit_rb_disco = True
      limit_number = 600

      Versions of Libraries:
      spotipy: 2.23.0
      pandas: 2.2.2
      python-dotenv: 1.0.0
      transformers: 4.51.0
      torch: 2.6.0
      numpy: 2.2.5
      tensorflow: 2.19.0
      sklearn: 0.0.post12
      ssl: 1.16
      nltk: 3.9.1
      matplotlib: 3.10.1
      seaborn: 0.13.2
      time (built in python module, no version)
      datetime (built in python module, no version)
      random (built in python module, no version)
      re (built in python module, no version)
      os (built in python module, no version)
      string (built in python module, no version)
      collections (built in python module, no version)

   
We used Google Colab as our coding enviornment. The only set up needed is having a Google account!

Below is a link that goes to our shared file in Google Colab that produced the results
[https://colab.research.google.com/drive/1YSSxpP3xkHEFEB76VOeBX_qoyj1Ly5Gm#scrollTo=ZUkoeLBOlAf2](https://colab.research.google.com/drive/19NjW9EvF9The_Skhi_ayVrWxzToWpfer?usp=sharing)

We ran this model using Python version 3.11.12 (default for Google Colab)

## Data
Our pool of usable data contains 5131 songs across 14 genres. We filtered out songs that had more than 100 unknown words in the nltk word corpus. We filtered these songs because we wanted only english songs, but were worried that some words in these songs could have typos or would not exist in the word corpus so we added a buffer to not filter out any songs that we didn't want to. For every song, there is a genre associated with it. For some of our data, songs had multiple genres so we chose to classify the song as the first (hopefully primary) genre before handing it to the model. Our data was not spread evenly across genres, and we had around 2800 songs from two categories (rb and disco), which produced a heavy bias toward those two genres in the models trained from this version of the dataset (a 10 genre and 14 genre version). To solve this, we implemented a hyperparameter that allows the user to decide if they want to cap the number of songs from these two genres and what that number should be. We trained several models using a smaller, more evenly spread, dataset by capping these two genres at 600, and evaluating at 14 genres, 10 genres, and 6 genres. These models had various performances, discussed below.

The input to train the model must have two arrays, one being the songs and the other being the genre associated with each song (both strings). For example, song at index 2 in the songs array with correspond with the genre at index 2 in the genres array. For validation and testing, the output is the predicted label of a song.

Per batch evaluations are outputted into training_eval_data.csv file. We store training loss, validation loss, validation accuracy, validation precision, validation recall, validation F1 score, training time, validation time, predicted labels, and true labels. Through this, we should be able to identify the optimal training time for each version of the dataset we try, and compute confusion matrices for any epoch we chose.

Checkpoints are stored in the .ipynb_checkpoints folder.

## Final Results for Bert
- Top 14 Genres, Unlimited
  - Average Test Accuracy: 0.599
  - Average Test F-1 Score: 0.579
  - This model, along with these metrics, is highly focused on the two categories rb and disco. Several genres were virtually unrepresented in the dataset and were almost absent from the predictions the model made.  
- Top 10 Genres, Unlimited
  - Average Test Accuracy: 0.563
  - Average Test F-1 Score: 0.545 
  - We thought that reducing the number of genres the model was predicting would improve the model's metrics, but unfortunately this wasn't the case. We changed our approach after this round of training.
- Top 14 Genres, Max Count = 600
  - Average Test Accuracy: 0.430
  - Average Test F-1 Score: 0.409
  - This model performed the worst out of any model we ran, and we attribute this to the smaller number of songs. We lost almost 1000 songs from limiting the dataset, and our performance takes a hit because of that. Additionally, we're trying to get the model to learn more about different genres, so it makes more misclassifications. The most common misclassification was labelling rock music as metal, which makes sense considering how similar the genres are, especially lyrically.
- Top 10 Genres, Max Count = 600
  - Average Test Accuracy: 0.454
  - Average Test F-1 Score: 0.431
  - This model improves on its metrics marginally, but doesn't make a massive jump in productivity. When comparing pre- and post-finetuned confusion matrices, it's clear that the model is learning, it's just limited by the amount of data we have.
- Top 6 Genres, Max Count = 600
  - Average Test Accuracy: 0.493
  - Average Test F-1 Score: 0.480
  - This is our last, and smallest model. It has the smallest number of songs and genres to choose from (genres we have at least a couple hundred songs for each). This model continues to frequently misclassify rock as metal, but I'd be satisfied drawing a conclusion about the two genres from that, instead of a conclusion about our model.

All of our results are available at https://drive.google.com/drive/folders/1g28x-4IAeSjNeTKNv3d_eMEw10Ggol1E?usp=sharing

# Lyric Generation

## Model Selection

We chose to implement an LSTM model for the lyric generation task based on the promising results in Gill et al. (2020). LSTMs are relatively lightweight in terms of computation, making them suitable for training within our project’s timeframe. Initially, we planned to improve upon Gill et al.'s design by initializing the hidden state with a learned embedding from a fine-tuned BERT model. However, during experimentation, we found that this approach required significantly more computational resources than justified by the marginal improvements in output quality.

Instead, we opted to initialize the hidden and cell states using the target genre embedding, expanded to the appropriate size. Preliminary testing indicated that this method improved model output noticeably compared to zero initialization.

Aside from this change, our model remains largely similar to that of Gill et al. (2020). It uses a genre-embedded initial state, consists of 2 LSTM layers with a dropout rate of 0.3, and employs a teacher forcing ratio of 0.5. The output of the final linear layer is passed through a softmax to generate a probability distribution over the vocabulary. In the generation loop, we allow for temperature adjustment; lower temperatures reduce creativity and lead to repetitive output, while higher temperatures increase diversity at the cost of coherence. We use top-p (nucleus) sampling for token selection during generation.

---

## Model Implementation

### Libraries Used

- `torch`: Core PyTorch library
- `torch.nn as nn`: Provides neural network layers and modules
- `torch.nn.functional as F`: Functional interface (e.g., for softmax)
- `torch.utils.data.Dataset`: For building custom datasets
- `torch.utils.data.DataLoader`: For batching and shuffling data during training
- `random`: Used in teacher forcing for randomly choosing between predicted and ground truth tokens
- `nltk` (`from nltk.tokenize import word_tokenize`): Used for word-level tokenization during vocabulary creation
- `collections.Counter`: Used to count word frequencies when building the vocabulary
- `tqdm`: For training progress visualization with progress bars

---

## Source Code

The implementation is organized into the following files:

### `LSTM.py`

#### Class `LyricsTokenizer`
- `__init__`: Initializes vocabulary dictionaries, sets special tokens, and defines minimum word frequency and maximum sequence length
- `build_vocab`: Tokenizes all lyrics using NLTK and creates `word2id` and `id2word` mappings
- `tokenize`: Converts tokenized lyrics into lists of word IDs using the vocabulary
- `vocab_size`: Returns the size of the vocabulary
- `untokenize`: Not used in the current code, but can convert token ID sequences back to readable lyrics

#### Class `LyricsGenreDataset`
- `__init__`: Tokenizes lyrics and converts genres to numerical IDs
- `__len__`: Returns the number of lyric samples
- `__getitem__`: Retrieves a tokenized lyric sample and its corresponding genre ID

#### Function `collate_fn`
- Batches and stacks lyric tensors and genre IDs
- Splits lyrics into input and target sequences for teacher forcing

#### Class `LSTMLyrics_by_Genre`
- Defines an LSTM decoder conditioned on genre embeddings
- `forward`: Runs a forward pass with teacher forcing
- `train_model`: Trains the model for one epoch using backpropagation and an optimizer (Adam with epsilon of 1e-3)
- `generate_lyrics`: Generates lyrics step-by-step using the genre and a beginning-of-song <BOS> token

---

### `LSTMtraining.ipynb`

- **Data Processing**: The dataset is augmented by extracting the first and last 100 words from each song. This increases dataset size while keeping sequence lengths manageable. Training uses data stored in `LSTM_data`.
- **Tokenizer**: The tokenizer from `LSTM.py` builds a vocabulary of the most frequent words and tokenizes input lyrics with padding and truncation
- **Dataset & Dataloader**: Lyrics and genre labels are wrapped in a PyTorch `Dataset` and loaded in batches using a `collate_fn`
- **Model Architecture**: The LSTM takes tokenized lyrics and a genre embedding. It uses an embedding size of 256, hidden size of 256, genre embedding size of 32, two LSTM layers, and dropout of 0.3
- **Training**: The model is trained using teacher forcing and cross-entropy loss. Best-performing weights (based on training loss) are saved. The training set includes ~10,000 songs across 5 genres, with a teacher forcing ratio of 0.5

---

### `lyric_generator.ipynb`

- **Data Processing**: Follows the same tokenization and formatting steps as during training
- **Generation**: Lyrics are generated by loading the trained LSTM model and using a `<BOS>` (beginning-of-song) token as input. Generation continues for up to 100 tokens (this can be changed to be more or less) or until an `<EOS>` token is produced. Output randomness is controlled by a temperature parameter (current results used a temperature of 1, less resulted in too much repitition, more resulted in a lack of creativity). Genre embeddings guide both style and vocabulary during generation. Top-p sampling is used for token selection
- **Bulk Generation**: Includes functionality for generating 100 songs per genre and filtering results to include only ASCII characters. Outputs are saved to a CSV file (e.g., `rb_generated.csv`)

---

## Usage

To use the notebooks:

1. Download and install all required libraries
2. Adjust file paths according to your environment and the dataset you're using
3. Run the notebooks to preprocess the data, train the model, or generate lyrics
4. Alternatively, use the pre-trained model stored in `LSTM_final.pth` to skip training and immediately begin generating lyrics

## Final Results
### Metrics

We implemented a comprehensive metrics suite to evaluate both the real lyrics (collected from Spotify + Genius) and the generated lyrics from our models.

The metrics implemented are inspired by Gill et al.’s 2020 research and include:

- Song Word Variation → Number of unique words divided by total words in a song across all songs of a genre
  - Dataset: We see the most variation in country, R&B, and hip-hop. But overall, the variance across all the genres is not vastly different with the lowest being rock at .31 and the highest being hip-hop at 0.44.
  - Generated: We see much less difference between genres, and a much higher variation in general, with all the genres hovering around 0.7. This is likely due to the smaller data set, less than 500 songs with 100 words each.

- I vs. You Point-of-View → Difference between the count of lines starting with “I” vs. “You”
  - Dataset: We see pretty consistent I vs You across all genres except hip-hop, which is much higher. 
  - Generated: Because the model doesn’t generate endlines, we cannot count the number of lines starting with I vs you, can resorted to counting the overall use of I and you. We see the highest I usage in hip-hop, which is consistent with the input data. On the other hand, we see a huge ‘you’ usage in rock and pop that may not be in the dataset or may not be line starting ‘you’s.

- Word Repetition Count → Number of immediate repeated words in a song (e.g., “good good”)
  - Dataset: There is very little repetition in country, a moderate amount in pop and R&B, and a more than moderate amount in hip-hop and rock. These values range from 2 to 16. 
  - Generated: We see very little repetition in country, more in hip-hop and pop, and the most in rock and R&B. Overall, the range is between .8 and 1.4. The small range of low repetition values can be explained by the small song length, and by the fact we trained the model on only the first and last 100 words of each song, thus missing the chorus in most songs, which would likely have more repetition. 

- Cosine Similarity → Quantitative similarity score between generated lyrics and real training lyrics using a bag-of-words vectorization

![image](https://github.com/user-attachments/assets/a7beaeb7-ac26-4a07-851f-622692bae28a)

Overall, we see the greatest cosine similarity in the country genre, and the lowest in rock. The other genres all have similar variance. This alignes with our other metrics of evaluation. 

These metrics are implemented in metrics.py and applied using ```metrics_tests.ipynb,``` where we visualize distributions and averages using ```matplotlib``` and ```seaborn```. The visualizations are saved in CSCI3832_FinalProject/SupplementaryMaterials/metrics_images

The results provide a quantitative foundation to assess how closely generated lyrics match the stylistic and linguistic patterns of real-world lyrics.


### Human Evaluation
Ten human reviewers evaluated a random sample of the generated lyrics, 2 from each genre, 10 songs total. They were prompted with three questions: first, to guess the genre, second, to rate the creativity of the lyrics on a scale of 10 stars, and third, to rate the coherence of the lyrics on a scale of 10 stars. Coherence was defined for them as “how logically and semantically consistent the lines of a song are with each other in regard to thematic consistency, logical flow, grammatical and syntactical structure.” Creativity was defined for them as “how original, imaginative, and emotionally or intellectually engaging the lyrics are.”

![human_eval](https://github.com/user-attachments/assets/988cb43a-d845-4692-95f1-4b1a662a92e2)

Overall, we see the highest accuracy in hip-hop, country, and R&B. This makes sense, as hip-hop and country both contain common themes and vocabulary specific to them while rock and pop are very general. In terms of creativity, rock, pop, and hip-hop are all rated the highest, while country and R&B are rated very low. For coherence, all the genres are rated around 3.5-4 excepting hip-hop, which is slightly lower. Regardless of these trends, all the ratings are below 5, which suggests that the model still has a long way to go, and reinforces the value of having human evaluators. 

## Future Steps
As the model currently stands, it is effectively a first verse lyric generator, and not a very good one. Steps that could be taken to improve and further experiment with this model would be to train it with more data for more epochs. This would result in general improved behavior. Another thing that would be interesting to explor would be to train the model on music from only one artist and then generate songs that mimic that artist. Lastly, to get a really good song, you would likely want a different model for each sections of a song (verse, chorus, bridge) that can take in the previous models work and add the next section to it. All of these would require more time and computational power, except perhaps training the model on one artists work - which may end up being a summer project. Overall, we are happy with the current results and excited to see where this could go further. 
 
## Acknowledgments

- This project uses the Spotify API for metadata collection and Genius API to scrape lyrics.
- Team members: Ima Mervin, Mia Ray, Mariana Vadas-Arendt, Gavin Hanville, Chloe Circenis.
