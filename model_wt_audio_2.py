import google.generativeai as genai
from tavily import TavilyClient
import os
import json
from dotenv import load_dotenv
import math
import time 
import requests 
import re 

# --- Configuration --- 
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")


if not GOOGLE_API_KEY: raise ValueError("GOOGLE_API_KEY not set.")
if not TAVILY_API_KEY: raise ValueError("TAVILY_API_KEY not set.")
if not ELEVENLABS_API_KEY: raise ValueError("ELEVENLABS_API_KEY not set.")

# Configure Gemini 
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# Configure Tavily Search 
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)


SEGMENT_DURATION_MINUTES = 5
WORDS_PER_MINUTE = 160
TARGET_WORD_COUNT = SEGMENT_DURATION_MINUTES * WORDS_PER_MINUTE
SEARCH_RESULT_COUNT = 3

ELEVENLABS_VOICE_ID = "EXAVITQu4vr4xnSDxMaL"
ELEVENLABS_API_URL = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
ELEVENLABS_HEADERS = {
    "xi-api-key": ELEVENLABS_API_KEY,
    "Content-Type": "application/json"
}


def analyze_user_prompt(user_prompt):
    """
    Uses Gemini to analyze the user's prompt. Focuses on extracting explicit
    time and core topic concepts. Calculates segments based PRIMARILY on time if provided.
    Adds a flag 'segments_based_on_time'.
    """
    print("Analyzing user prompt...")
    
    prompt = f"""
    Analyze the following user request for generating personalized learning audio snippets. Extract the following information and provide it ONLY as a JSON object:

    1.  `total_time_minutes`: The total available time explicitly mentioned (integer). Output 0 if not mentioned. **Prioritize this value if provided.**
    2.  `requested_topics`: A list of strings representing the CORE concepts or questions the user is explicitly interested in. If the user mentions a broad topic (e.g., "cars", "AI"), list just that core concept. Do not break it down into sub-topics at this stage unless the user explicitly listed multiple distinct items. Output [] if none are explicitly stated. Do not include any sound effects or special characters- provide only plin text.
    3.  `requires_suggestion`: Set to true if the user explicitly asks for suggestions (e.g., "suggest something", "what else?") OR uses phrases implying they want topics chosen for them (e.g., "surprise me", "teach me something interesting"). Otherwise, set to false.

    User Request: "{user_prompt}"

    Example Output Format for "surprise me":
    {{
      "total_time_minutes": 0,
      "requested_topics": [],
      "requires_suggestion": true
    }}

    Example Output Format for "20 mins on cars":
    {{
      "total_time_minutes": 20,
      "requested_topics": ["Cars"],
      "requires_suggestion": false
    }}

    Example Output Format for "10 mins on AI and photosynthesis":
    {{
      "total_time_minutes": 10,
      "requested_topics": ["AI", "Photosynthesis"],
      "requires_suggestion": false
    }}

     Example Output Format for "Tell me about space exploration for 15 minutes and suggest something else":
    {{
      "total_time_minutes": 15,
      "requested_topics": ["Space Exploration"],
      "requires_suggestion": true
    }}

    Provide ONLY the JSON object as the response.
    """
    try:
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        response = gemini_model.generate_content(prompt, safety_settings=safety_settings)
        try:
             json_text = response.text.strip().replace('```json', '').replace('```', '').strip()
             analysis = json.loads(json_text)
        except ValueError:
            print("Warning: Gemini response did not contain valid text content. Checking parts...")
            if response.parts:
                 json_text = response.parts[0].text.strip().replace('```json', '').replace('```', '').strip()
                 analysis = json.loads(json_text)
            else:
                raise ValueError("Gemini response blocked or empty.")

        # --- Refined Post-processing and Defaulting ---
        if "error" in analysis:
             return analysis
        if not analysis.get("requested_topics"):
             analysis["requested_topics"] = []

        extracted_time = analysis.get("total_time_minutes", 0)
        num_extracted_topics = len(analysis.get("requested_topics", []))

        final_time = 0
        segments_needed = 0
        segments_based_on_time = False 

        if extracted_time > 0:
            final_time = extracted_time
            segments_needed = math.ceil(final_time / SEGMENT_DURATION_MINUTES)
            segments_based_on_time = True
            print(f"Time specified ({final_time} mins). Calculated segments needed: {segments_needed}. Segments based on time: {segments_based_on_time}")

        elif num_extracted_topics > 0:
            final_time = num_extracted_topics * SEGMENT_DURATION_MINUTES
            segments_needed = num_extracted_topics
            analysis["total_time_minutes"] = final_time
            print(f"Time estimated from {num_extracted_topics} topics. Calculated segments needed: {segments_needed}. Segments based on time: {segments_based_on_time}")

        elif analysis.get("requires_suggestion"):
            final_time = SEGMENT_DURATION_MINUTES
            segments_needed = 1
            analysis["total_time_minutes"] = final_time
            print(f"Suggestion required, no time/topic. Defaulting. Segments needed: {segments_needed}. Segments based on time: {segments_based_on_time}")

        else:
             final_time = SEGMENT_DURATION_MINUTES
             segments_needed = 1
             analysis["total_time_minutes"] = final_time
             analysis["requires_suggestion"] = True
             print(f"Ambiguous input. Defaulting to suggestion. Segments needed: {segments_needed}. Segments based on time: {segments_based_on_time}")

        if final_time > 0 and segments_needed == 0:
             segments_needed = 1

        analysis["segments_needed"] = segments_needed
        analysis["segments_based_on_time"] = segments_based_on_time 

        print("Prompt Analysis Complete:", analysis)
        return analysis

    except json.JSONDecodeError as e:
        print(f"Error: Gemini did not return valid JSON for prompt analysis. JSONDecodeError: {e}")
        print("Raw Response Text:", json_text if 'json_text' in locals() else 'N/A')
        return {"error": f"JSON Decode Error during analysis: {e}"}
    except ValueError as e:
         print(f"Error: Gemini response issue (likely blocked or empty). Error: {e}")
         print("Full Response object:", response)
         return {"error": f"Gemini response issue: {e}"}
    except Exception as e:
        print(f"Error during prompt analysis: {e}")
        return {"error": f"General Error during analysis: {e}"}


def suggest_single_topic(user_prompt_for_context, topic_history):
    """
    Uses Gemini to infer user intent from the original prompt AND past history
    to suggest exactly ONE relevant new topic.
    """
    print("Inferring intent and suggesting a single topic based on history...")
    history_string = ", ".join(topic_history) if topic_history else 'None provided yet.'

    prompt = f"""
    The user provided the following request: "{user_prompt_for_context}"
    This current request is vague, but they want to learn something new.

    To understand their potential interests, consider their learning history from this session:
    Past topics learned: {history_string}

    Analyze their current vague request AND their past topics (if any) to infer underlying interests.
    Suggest exactly ONE concise, engaging NEW learning topic that aligns with these inferred interests but is DIFFERENT from the topics already learned in the history list.
    The topic should be suitable for a single 5-minute audio explanation.

    Examples (if history was ["Basics of black holes", "How stars are formed"]):
    - Current request: "teach me something cool" -> Suggest: "The concept of gravitational waves"
    - Current request: "surprise me" -> Suggest: "What is dark matter?"

    Respond with ONLY the suggested topic as a plain string, without quotes or labels.
    """
    try:
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        response = gemini_model.generate_content(prompt, safety_settings=safety_settings)
        suggested_topic = ""
        try:
            suggested_topic = response.text.strip().strip('"')
        except ValueError:
            if response.parts:
                suggested_topic = response.parts[0].text.strip().strip('"')
            else:
                 raise ValueError("Gemini response blocked or empty for single topic suggestion.")

        if not suggested_topic:
            print("Warning: Gemini did not suggest a topic. Defaulting.")
            return "A random interesting fact"
        elif topic_history and suggested_topic.lower() == topic_history[-1].lower():
             print(f"Warning: Suggested topic '{suggested_topic}' is same as last history item. Asking for another.")
             return "A different interesting science fact"

        print(f"Suggested single topic (considering history): '{suggested_topic}'")
        return suggested_topic
    except Exception as e:
        print(f"Error during single topic suggestion with history: {e}")
        print("Full Response object (if available):", response if 'response' in locals() else 'N/A')
        return "Error suggesting topic"


def suggest_multiple_topics(existing_topics, num_suggestions):
    """
    Uses Gemini to suggest multiple *additional* topics based on existing ones.
    """
    print(f"Generating {num_suggestions} additional topic suggestions based on: {existing_topics}")
    if not existing_topics:
        print("Warning: Cannot suggest multiple topics without initial topics.")
        return []

    prompt = f"""
    Based on the user's interest in these topics: {', '.join(existing_topics)}
    Suggest {num_suggestions} additional, distinct learning topics they might find curious. The topics should be suitable for a 5-minute audio explanation and different from the initial list.
    Provide ONLY a JSON list of strings as the response. Example: ["Topic A", "Topic B"]
    """
    try:
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        response = gemini_model.generate_content(prompt, safety_settings=safety_settings)
        json_text = ""
        try:
             json_text = response.text.strip().replace('```json', '').replace('```', '').strip()
             suggestions = json.loads(json_text)
        except ValueError:
            if response.parts:
                 json_text = response.parts[0].text.strip().replace('```json', '').replace('```', '').strip()
                 suggestions = json.loads(json_text)
            else:
                raise ValueError("Gemini response blocked or empty for multiple topic suggestion.")

        print("Suggestions received:", suggestions)
        return suggestions if isinstance(suggestions, list) else []
    except json.JSONDecodeError:
        print("Error: Gemini did not return valid JSON for multiple topic suggestions.")
        print("Raw Response Text:", json_text if 'json_text' in locals() else 'N/A')
        return []
    except Exception as e:
        print(f"Error during multiple topic suggestion: {e}")
        print("Full Response object (if available):", response if 'response' in locals() else 'N/A')
        return []



def expand_or_suggest_topics(initial_topics, num_needed):
    """
    Expands a broad topic into sub-topics OR suggests related topics
    to reach the target number ('num_needed').
    Returns a list of topic strings.
    """
    print(f"Expanding/Suggesting topics to reach {num_needed} segments based on: {initial_topics}")

    if not initial_topics:
        print("Warning: expand_or_suggest_topics called with no initial topics. Suggesting general topics.")
        return [f"Interesting Topic {i+1}" for i in range(num_needed)]

    context_topics = ", ".join(initial_topics)

    prompt = f"""
    The user wants to learn about "{context_topics}" for a duration requiring {num_needed} distinct 5-minute segments.
    Their initial request might be broad.

    Your task is to generate a list of exactly {num_needed} specific, engaging topic titles suitable for individual 5-minute audio snippets.
    These topics should logically cover the core subject(s) "{context_topics}".
    - If "{context_topics}" is broad (like "Cars", "AI", "History of Rome"), break it down into logical sub-topics (e.g., "History of Cars", "How Car Engines Work", "Future of Electric Cars", "Self-Driving Technology").
    - If "{context_topics}" already contains multiple related items, suggest further related topics to reach the total of {num_needed}.
    - Ensure the topics are distinct from each other.
    - Make the titles concise and appealing for a learning playlist.

    Provide ONLY a JSON list of exactly {num_needed} topic strings as the response.

    Example (initial_topics=["Cars"], num_needed=4):
    ["History of the Automobile", "How Internal Combustion Engines Work", "The Rise of Electric Vehicles", "Future Trends in Automotive Tech"]

    Example (initial_topics=["AI", "Photosynthesis"], num_needed=3):
    ["What is Artificial Intelligence?", "How Plants Create Energy (Photosynthesis)", "Machine Learning Basics"]
    """

    try:
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        response = gemini_model.generate_content(prompt, safety_settings=safety_settings)
        json_text = ""
        try:
             json_text = response.text.strip().replace('```json', '').replace('```', '').strip()
             expanded_topics = json.loads(json_text)
        except ValueError:
            if response.parts:
                 json_text = response.parts[0].text.strip().replace('```json', '').replace('```', '').strip()
                 expanded_topics = json.loads(json_text)
            else:
                raise ValueError("Gemini response blocked or empty for topic expansion.")

        if isinstance(expanded_topics, list) and len(expanded_topics) == num_needed:
            print(f"Expanded/Suggested topics: {expanded_topics}")
            return expanded_topics
        else:
            print(f"Warning: Gemini returned an unexpected format or wrong number of topics for expansion. Expected {num_needed}, Got: {expanded_topics}")
            result = list(initial_topics)
            while len(result) < num_needed:
                 result.append(f"{initial_topics[0]} - Aspect {len(result)}")
            return result[:num_needed]

    except json.JSONDecodeError:
        print("Error: Gemini did not return valid JSON for topic expansion.")
        print("Raw Response Text:", json_text if 'json_text' in locals() else 'N/A')
        result = list(initial_topics)
        while len(result) < num_needed:
             result.append(f"{initial_topics[0]} - Aspect {len(result)}")
        return result[:num_needed]
    except Exception as e:
        print(f"Error during topic expansion/suggestion: {e}")
        print("Full Response object (if available):", response if 'response' in locals() else 'N/A')
        result = list(initial_topics)
        while len(result) < num_needed:
             result.append(f"{initial_topics[0]} - Aspect {len(result)}")
        return result[:num_needed]



def search_web_for_topic(topic):
    """
    Uses Tavily to search the web for a given topic and returns context.
    """
    print(f"Searching web for: '{topic}'...")
    try:
        response = tavily_client.search(
            query=f"Comprehensive overview of {topic} for a 5-minute explanation",
            search_depth="basic",
            max_results=SEARCH_RESULT_COUNT,
            include_answer=False
        )
        context = f"Topic: {topic}\n\nSearch Results Context:\n"
        if response.get('results'):
             for result in response['results']:
                 context += f"- Source: {result.get('url', 'N/A')}\n  Snippet: {result.get('content', 'N/A')}\n\n"
        else:
             print(f"Warning: No search results found for '{topic}'. Summary might be less informative.")
             context += "No specific search results found."
        return context
    except Exception as e:
        print(f"Error during web search for '{topic}': {e}")
        return f"Topic: {topic}\n\nError: Could not fetch search results."



def generate_learning_script(topic, context):
    """
    Uses Gemini to generate a ~5-minute script based on the topic and context.
    """
    print(f"Generating script for: '{topic}'...")

    prompt = f"""
    You are an AI assistant creating an engaging audio learning script.
    The target audience wants a clear, concise, and interesting explanation suitable for listening (like a mini-podcast episode).

    Topic: {topic}
    Available Context from Web Search:
    ---
    {context}
    ---

    Task: Generate a script for a 5-minute audio snippet (approximately {TARGET_WORD_COUNT} words) covering the key aspects of the topic.
    - Start with a brief, engaging hook.
    - Explain the core concepts clearly.
    - Structure the information logically.
    - Use simple language, avoid excessive jargon or explain it if necessary.
    - Conclude with a brief summary or a thought-provoking point.
    - The tone should be informative but conversational and easy to follow.
    - Output ONLY the script text, ready for text-to-speech conversion. Do not include titles like "Script:" or notes.
    """
    try:
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        response = gemini_model.generate_content(prompt, safety_settings=safety_settings)
        script_text = ""
        try:
            script_text = response.text.strip()
        except ValueError:
            if response.parts:
                script_text = response.parts[0].text.strip()
            else:
                 raise ValueError("Gemini response blocked or empty for script generation.")

        if not script_text:
             print(f"Warning: Generated empty script for '{topic}'.")
             return f"Could not generate script for {topic}."

        print(f"Script generated for '{topic}'.")
        return script_text
    except Exception as e:
        print(f"Error during script generation for '{topic}': {e}")
        print("Full Response object (if available):", response if 'response' in locals() else 'N/A')
        return f"Error: Could not generate script for {topic}."



def generate_audio_elevenlabs(script_text, output_filepath):
    """
    Generates audio from text using ElevenLabs API and saves to a file.
    Returns True if successful, False otherwise.
    """
    print(f"Generating audio for: {os.path.basename(output_filepath)}...")
    if not script_text or len(script_text.strip()) < 10:
        print("Error: Script text is too short or empty. Skipping TTS.")
        return False
    data = {"text": script_text, "voice_settings": {"stability": 0.5, "similarity_boost": 0.5}}
    try:
        response = requests.post(ELEVENLABS_API_URL, headers=ELEVENLABS_HEADERS, json=data, timeout=180)
        if response.status_code == 200:
            with open(output_filepath, "wb") as f: f.write(response.content)
            print(f"Audio saved as {output_filepath}")
            return True
        else:
            print(f"ElevenLabs API Failed: {response.status_code} - {response.text}")
            try: print(f"   Error details: {response.json()}")
            except json.JSONDecodeError: pass
            return False
    except requests.exceptions.RequestException as e: print(f"Network error during TTS request: {e}"); return False
    except Exception as e: print(f"Unexpected error during TTS generation: {e}"); return False


def sanitize_filename(name):
    """Removes or replaces characters invalid for filenames."""
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    name = name.replace(" ", "_")
    return name[:50]


def process_single_request(user_prompt, session_history):
    """
    Processes a single user request: analyzes, determines topics,
    generates scripts & audio, saves files.
    Returns a dictionary with result info (folder path, topics, title) or None on failure.
    """
    print(f"\n>>> Processing request: '{user_prompt}' <<<")
    print(f"    Current history: {session_history}")

    analysis = analyze_user_prompt(user_prompt)

 
    if analysis.get("error"):
        print(f"Analysis error: {analysis['error']}. Cannot proceed.")
        return None # Indicate failure
    segments_needed = analysis.get("segments_needed", 0)
    requested_topics = analysis.get("requested_topics", [])
    requires_suggestion = analysis.get("requires_suggestion", False)
    segments_based_on_time = analysis.get("segments_based_on_time", False)

    if segments_needed == 0:
         print("Could not determine number of segments needed.")
         return None 
    
    final_topics = []
    if not requested_topics and requires_suggestion:
        print("Scenario: Suggesting a single topic...")
        single_suggestion = suggest_single_topic(user_prompt, session_history)
        if single_suggestion and "Error" not in single_suggestion:
            final_topics = [single_suggestion]
        else:
            print("Failed to get single topic suggestion.")
            return None 
    else:
        initial_topics_from_analysis = list(requested_topics)
        if len(initial_topics_from_analysis) < segments_needed:
            if segments_based_on_time or requires_suggestion:
                print(f"Scenario: Expanding/Suggesting topics for {segments_needed} segments...")
                final_topics = expand_or_suggest_topics(initial_topics_from_analysis, segments_needed)
            else:
                print(f"Scenario: Using only initial topics: {initial_topics_from_analysis}")
                final_topics = initial_topics_from_analysis
        elif len(initial_topics_from_analysis) > segments_needed:
            print(f"Warning: Using first {segments_needed} of {len(initial_topics_from_analysis)} topics.")
            final_topics = initial_topics_from_analysis[:segments_needed]
        else:
            print(f"Scenario: Using initial topics: {initial_topics_from_analysis}")
            final_topics = initial_topics_from_analysis

    if not final_topics:
        print("No topics determined to generate scripts for.")
        return None 


    playlist_timestamp = time.strftime("%Y%m%d_%H%M%S")
    folder_name_base = sanitize_filename(final_topics[0]) if final_topics else "general"

    base_output_dir = "generated_playlists"
    output_folder_name = f"playlist_{playlist_timestamp}_{folder_name_base}"
    output_folder_path = os.path.join(base_output_dir, output_folder_name)
    try:
        os.makedirs(output_folder_path, exist_ok=True)
        print(f"\n Saving audio files to folder: {output_folder_path}")
    except OSError as e:
        print(f" Error creating output folder '{output_folder_path}': {e}")
        return None # Indicate failure


    print(f"\nGenerating playlist for topics: {final_topics}")
    playlist_segments_data = []
    successfully_generated_topics_this_run = [] 

    for i, topic in enumerate(final_topics):
        if not topic or "Error" in topic:
             print(f"Skipping segment {i+1} due to invalid topic: '{topic}'")
             continue

        print(f"\n--- Processing Segment {i+1}/{len(final_topics)}: {topic} ---")
        context = search_web_for_topic(topic)
        time.sleep(1)
        script = generate_learning_script(topic, context)

        audio_success = False
        audio_filepath_relative = None 
        if script and "Error:" not in script:
            safe_topic_name = sanitize_filename(topic)
            audio_filename = f"segment_{i+1}_{safe_topic_name}.mp3"
            audio_filepath_absolute = os.path.join(output_folder_path, audio_filename)
            audio_filepath_relative = os.path.join(output_folder_name, audio_filename) 

            time.sleep(1)
            audio_success = generate_audio_elevenlabs(script, audio_filepath_absolute)
        else:
            print(f"Skipping audio generation for '{topic}' due to script error.")

        if audio_success and audio_filepath_relative:
             playlist_segments_data.append({
                 "segment_number": i + 1,
                 "topic": topic,
                 "script_preview": script[:100] + "...",
                 "audio_file": audio_filepath_relative
             })
             successfully_generated_topics_this_run.append(topic)
        else:
             print(f"Segment for '{topic}' failed.")

        time.sleep(1.5)

    if not playlist_segments_data:
         print(f"\n Failed to generate any valid audio segments. Check folder '{output_folder_path}'.")
        
         return None
    
    # --- Determine Playlist Title ---
    playlist_title = ""
    display_time = analysis.get('total_time_minutes', len(playlist_segments_data) * 5)
    if not requested_topics and requires_suggestion and len(playlist_segments_data) == 1:
         playlist_title = f"Suggested 5-Min Bite (History Considered)"
    elif len(successfully_generated_topics_this_run) == 1 and len(final_topics) == 1:
         playlist_title = f"5-Min Bite: {successfully_generated_topics_this_run[0]}"
    else:
         playlist_title = f"{display_time}-Min Playlist ({len(playlist_segments_data)} Segments)"

   
    output_summary_data = {
        "playlist_title": playlist_title,
        "output_folder_name": output_folder_name, 
        "output_folder_path": output_folder_path, 
        "total_segments": len(playlist_segments_data),
        "segments": playlist_segments_data
    }

   
    try:
        summary_filepath = os.path.join(output_folder_path, "playlist_summary.json")
        with open(summary_filepath, "w", encoding='utf-8') as f:
             json.dump(output_summary_data, f, indent=2, ensure_ascii=False)
        print(f"\nPlaylist summary saved to {summary_filepath}")
    except Exception as e:
        print(f"\nError saving playlist summary file: {e}")
       

    print(f">>> Request processing finished for '{user_prompt}' <<<")

  
    return {
        "folder_path": output_folder_path,
        "folder_name": output_folder_name,
        "generated_topics": successfully_generated_topics_this_run
    }


if __name__ == '__main__':
    print("Running model_wt_audio.py standalone for testing...")
    test_history = []
    while True:
         prompt = input("Enter test prompt (or 'quit'): ")
         if prompt.lower() == 'quit':
             break
         result = process_single_request(prompt, test_history)
         if result:
             print("\n--- Test Result ---")
             print(f"Folder Path: {result['folder_path']}")
             print(f"Generated Topics: {result['generated_topics']}")
             test_history.extend(result['generated_topics']) 
             print("-------------------\n")
         else:
             print("\n--- Test Failed ---")
