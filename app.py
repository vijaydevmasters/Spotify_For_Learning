import os
import flask
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, flash
import json


try:

    from model_wt_audio_2 import process_single_request 
except ImportError as e:
    print(f"ERROR: Could not import 'process_single_request' from model_wt_audio.py: {e}")
    print("Make sure model_wt_audio.py exists and the function is defined correctly.")
    exit()


app = Flask(__name__)
app.secret_key = os.urandom(24) 
PLAYLIST_BASE_DIR = "generated_playlists"

os.makedirs(PLAYLIST_BASE_DIR, exist_ok=True)


#NOTE
SESSION_HISTORY = []
# Store {'name': 'folder_name_relative_to_base', 'title': 'Playlist Title'}
GENERATED_FOLDERS_INFO = []



@app.route('/', methods=['GET', 'POST'])
def index():
    global SESSION_HISTORY 

    if request.method == 'POST':
        input_text = request.form.get('text_input', '').strip()

        if not input_text:
            flash('Please enter some text.', 'error')
            return redirect(url_for('index'))

        print(f"Received POST request with prompt: '{input_text}'")
        print(f"Current session history before processing: {SESSION_HISTORY}")

        try:
           
            result = process_single_request(input_text, SESSION_HISTORY)
            # ---------------------------------------------

            if result and result.get("folder_path") and result.get("folder_name"):
                 folder_name = result["folder_name"]
                 folder_path = result["folder_path"] 
                 new_topics = result.get("generated_topics", [])

                 # Update app-level session history
                 SESSION_HISTORY.extend(new_topics)
                 print(f"Session history updated: {SESSION_HISTORY}")

                 # Prepare info for display list
                 folder_info = {'name': folder_name, 'title': f"Playlist: {folder_name}"}

                 # Avoid adding duplicates 
                 if not any(f['name'] == folder_name for f in GENERATED_FOLDERS_INFO):
                      GENERATED_FOLDERS_INFO.insert(0, folder_info) # Add newest first
                      flash(f'Successfully generated playlist: {folder_name}', 'success')
                 else:
                      flash(f'Playlist folder already generated: {folder_name}', 'info')

            elif result is None:
                 # Function indicated failure explicitly
                 flash('Playlist generation failed during processing.', 'error')
            else:
                 raise ValueError(f"Processing function returned unexpected data: {result}")


        except Exception as e:
            print(f"Error during processing call: {e}") 
            flash(f'An error occurred during processing: {e}', 'error')


        # Redirect back to the GET request AFTER processing is complete
        return redirect(url_for('index'))

    # --- GET Request ---
    # Render the template with the current list of generated folders info
    # Pass a copy to avoid potential modification issues if needed
    return render_template('index.html', folders=list(GENERATED_FOLDERS_INFO))


@app.route('/view/<folder_name>')
def view_folder(folder_name):
    # Construct the expected absolute path
    folder_path = os.path.abspath(os.path.join(PLAYLIST_BASE_DIR, folder_name))

    if not os.path.isdir(folder_path):
        flash(f"Folder '{folder_name}' not found or is inaccessible.", "error")
        return redirect(url_for('index'))

    audio_files = []
    summary_data = None
    error_message = None

    try:
        # Load summary data
        summary_path = os.path.join(folder_path, "playlist_summary.json")
        if os.path.exists(summary_path):
             with open(summary_path, 'r', encoding='utf-8') as f:
                 summary_data = json.load(f)
                 # Extract audio filenames from summary if available
                 if summary_data and 'segments' in summary_data:
                      for segment in summary_data['segments']:
                           if 'audio_file' in segment:
                                # Get only the filename part from the relative path
                                audio_files.append(os.path.basename(segment['audio_file']))
        else:
             # Fallback: list directory if summary is missing
             for filename in os.listdir(folder_path):
                 if filename.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
                      audio_files.append(filename)
        audio_files.sort() # Sort alphabetically or by segment number if possible

    except Exception as e:
        print(f"Error reading folder or summary {folder_path}: {e}")
        error_message = f"Could not read contents of the folder. Error: {e}"

    playlist_title = summary_data.get('playlist_title', folder_name) if summary_data else folder_name

    return render_template(
        'view_folder.html',
        folder_name=folder_name,
        playlist_title=playlist_title,
        audio_files=audio_files, 
        error=error_message
    )


@app.route('/audio/<path:filepath>')
def serve_audio(filepath):
    """Serves audio files directly from the base playlist directory."""

    base_dir = os.path.abspath(PLAYLIST_BASE_DIR)
    # Create absolute path
    abs_filepath = os.path.abspath(os.path.join(base_dir, filepath))

    print(f"Attempting to serve: {abs_filepath}")
    print(f"Base directory:      {base_dir}")

    # Prevent path traversal attacks
    if not abs_filepath.startswith(base_dir + os.sep):
        print("Forbidden path traversal attempt.")
        flask.abort(403) 

    # Check if the directory part exists 
    directory = os.path.dirname(abs_filepath)
    filename = os.path.basename(abs_filepath)

    if not os.path.isdir(directory):
         print(f"Directory not found: {directory}")
         flask.abort(404)

    try:

        return send_from_directory(directory, filename, as_attachment=False)
    except FileNotFoundError:
         print(f"File not found: {filename} in {directory}")
         flask.abort(404)
    except Exception as e:
         print(f"Error serving file {filename} from {directory}: {e}")
         flask.abort(500)


if __name__ == '__main__':
  
    app.run(host='0.0.0.0', port=5000, debug=True)