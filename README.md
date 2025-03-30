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
2. **Navigate to your project directory:**
   ```bash
   cd path/to/CSCI3832_FinalProject
   ```
3. **Create a virtual environment (this keeps your project dependencies isolated):**
   ```bash
   python -m venv venv
   ```
4. **Activate the virtual environment:**

   *On Windows:*
      ```bash
      venv\Scripts\activate
      ```
   *On macOS/Linux:*
      ```bash
      source venv/bin/activate
      ```
6. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```
7. **Running Scripts:**
   
  *Make sure your virtual environment is activated (you should see (venv) at the beginning of your command line)*
  
 *The script can be run in several different ways depending on what you want to accomplish, here are examples for a few data collection tasks you can run:*
 
 
   
   





## Acknowledgments

- This project uses the Spotify API for data collection.
- Team members: Ima, Mia Ray, Mariana, Gavin Hanville, Chloe.



# To search for playlists and collect data:
```bash
python -m src.collector --search "electronic dance music"
```

# Or to use a specific playlist ID:
```bash
python -m src.collector --playlist 37i9dQZF1DXcBWIGoYBM5M
```
