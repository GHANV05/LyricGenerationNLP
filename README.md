# CSCI3832_FinalProject

# NLP-Based Analysis and Generation of Song Lyrics Using Spotify Data

## Project Overview

This project aims to implement Natural Language Processing (NLP) models to classify songs into their respective genres and sentiments. We compare the accuracy of different models to determine which performs best and analyze the relationships between sentiment and genre. The project leverages the Spotify API to extract lyrics, metadata, and audio features.

## Key Tasks

1. **Text Classification:**
   - Classify song lyrics by sentiment (e.g., happy, sad, energetic).
   - Classify song lyrics by genre (e.g., rock, pop, rap).

2. **Model Comparison:**
   - Evaluate the performance of different NLP models (Encoder/Decoder, BERT, n-gram, LSTM).
   - Compare model accuracy and efficiency.

## Team Structure

The project is divided into three sub-teams:

- **Data Preprocessing and Classification:** Ima Mervin
- **Language Modeling Approaches:**
  - **Team Finetuning BERT:** Mia Ray and Mariana.
  - **Team Encoder/Decoder:** Gavin Hanville and Chloe.

## Dataset

- **Source:** Spotify API.
- **Data Types:** Lyrics, metadata, and audio features.
- **Rate Limiting:** The Spotify API has a rate limit of approximately 180 calls per 30 seconds. We implement a backoff request system to manage this limit.

## Project Goals

- Provide insights into how NLP can enhance music-related applications.
- Address challenges such as dataset bias and computational efficiency.
- Compare the performance of advanced models with basic models.

## Getting Started

1. **Clone the Repository:**
   ```bash
   git clone git@github.com:ima-mervin/CSCI3832_FinalProject.git
   ```

2. **Install Required Libraries:**
   - See `requirements.txt` for a list of dependencies.
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Spotify API:**
   - Register your application on the Spotify Developer Dashboard.
   - Replace placeholders in `spotify_api_setup.py` with your client ID and client secret.

4. **Run Scripts:**
   - Follow instructions in each script for data preprocessing, model training, and evaluation.

## Acknowledgments

- This project uses the Spotify API for data collection.
- Team members: Ima, Mia Ray, Mariana, Gavin Hanville, Chloe.

## Running the Script

# Navigate to your project directory
```bash
cd path/to/CSCI3832_FinalProject
```

# Create a virtual environment (this keeps your project dependencies isolated)
```bash
python -m venv venv
```
# Activate the virtual environment
# On Windows:
```bash
venv\Scripts\activate
```
# On macOS/Linux:
```bash
source venv/bin/activate
```
# Install the required packages
```bash
pip install -r requirements.txt
```
# Make sure your virtual environment is activated (you should see (venv) at the beginning of your command line)

# To search for playlists and collect data:
```bash
python -m src.collector --search "electronic dance music"
```

# Or to use a specific playlist ID:
```bash
python -m src.collector --playlist 37i9dQZF1DXcBWIGoYBM5M
```