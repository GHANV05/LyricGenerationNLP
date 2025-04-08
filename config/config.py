import os
from dotenv import load_dotenv

# load environment variables from .env file
load_dotenv()

# spotify API credentials
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

# output settings
DEFAULT_OUTPUT_DIR = os.path.join(os.getcwd(), "data")  # Use current working directory
DEFAULT_OUTPUT_FILE = os.path.join(DEFAULT_OUTPUT_DIR, "spotify_dataset.csv")

# create output directory if it doesn't exist
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

print("Configuration loaded successfully.")
