import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Spotify API credentials
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

# Output settings
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DEFAULT_OUTPUT_FILE = os.path.join(DEFAULT_OUTPUT_DIR, "spotify_dataset.csv")

# Create output directory if it doesn't exist
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)