# Audio Playlist Generator

This project generates personalized audio playlists based on user prompts. It uses AI models to analyze prompts, generate scripts, and convert them into audio files.

## How to Run

1. Ensure you have all the required dependencies installed.
2. Run the main file `app.py`:
   ```bash
   python app.py
   ```
3. Once the application starts, a weblink will be displayed in the console (e.g., `http://127.0.0.1:5000`).
4. Open the link in your browser to access the UI and interact with the model.

## Features

- Generate audio playlists based on user prompts.
- View and listen to generated audio segments.
- Simple and intuitive web-based interface.

## Folder Structure

- `app.py`: Main application file.
- `model_wt_audio_2.py`: Contains the core logic for processing user prompts and generating audio.
- `templates/`: HTML templates for the web interface.
- `static/`: CSS styles for the web interface.
- `.env`: Environment variables for API keys.

## Requirements

- Python 3.7 or higher
- Flask
- Required API keys for Google Generative AI, Tavily, and ElevenLabs (set in `.env` file).

## Notes

- Ensure the `.env` file is properly configured with valid API keys before running the application.
- Generated playlists are stored in the `generated_playlists` directory.