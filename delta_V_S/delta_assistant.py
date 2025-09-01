# give information regarding gestruture in this assistant--- model-5
import speech_recognition as sr
import pyttsx3
import joblib
import string
import re
import webbrowser
import pywhatkit
import wikipedia
import requests
import datetime
import pyjokes
import os
import platform
import random
import smtplib
from email.message import EmailMessage
import time
import threading
import subprocess
# import signal
import sys



# ==================================================================================
# IMPORTANT: Before you run, you MUST get these keys and update the placeholders.
# ==================================================================================
# 1. OpenWeatherMap API Key for weather forecasts (free from OpenWeatherMap)
OPENWEATHERMAP_API_KEY = "b49192f24c099d5f2f9cf4191d0c89fe" # api keys open weather

# 2. NewsAPI Key for news headlines (free from NewsAPI.org)
NEWSAPI_KEY = "AIzaSyAlY0wnK5GFfRQsnM9qk3yjX9ZufMoXj1Y"# api keys are gemini api

# 3. Gemini API Key for advanced Q&A (from Google AI Studio)
GEMINI_API_KEY = "AIzaSyAlY0wnK5GFfRQsnM9qk3yjX9ZufMoXj1Y"# api keys are gemini api

# ==================================================================================
# CONFIGURATION & USER DATA
# ==================================================================================
# MODEL_PATH = "delta_V_S/model5.joblib" # model 5.  Make sure your trained model is in the same folder
# MODEL_PATH = "delta_V_S/model_5_1.joblib" # new model
MODEL_PATH = "delta_V_S/model_5_2.joblib" # new model6
NOTES_FILE = "delta_V_S/notes.txt" # Notes will be saved in the gemini folder

# User-defined contacts for WhatsApp
contacts = {
    "akshay": "+918971573838",
    "yuvaraj": "+916362426223",
    "dad": "+911234567890", # Example number
    "mom": "+919112233445"  # Example number
}

# ==================================================================================
# INITIALIZATION
# ==================================================================================
try:
    engine = pyttsx3.init()
except Exception as e:
    print(f"Error initializing TTS engine: {e}")
    engine = None

# ==================================================================================
# HELPER FUNCTIONS
# ==================================================================================

def speak(text):
    """Converts text to speech."""
    if engine:
        print(f"🗣️ Delta: {text}")
        engine.say(text)
        engine.runAndWait()
    else:
        print(f"🗣️ Delta (TTS not available): {text}")

def listen_command(prompt="Listening for command..."):
    """Listens for a user command and converts it to text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print(f"🎤 {prompt}")
        recognizer.pause_threshold = 1
        recognizer.adjust_for_ambient_noise(source, duration=1)
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
        except sr.WaitTimeoutError:
            print("Listening timed out while waiting for phrase to start")
            return None

    try:
        command = recognizer.recognize_google(audio).lower()
        print(f"👂 You said: {command}")
        return command
    except sr.UnknownValueError:
        print("Could not understand audio")
        return None
    except sr.RequestError:
        speak("Sorry, my speech service is down.")
        return None
    except Exception as e:
        print(f"An error occurred during recognition: {e}")
        return None

def preprocess_text(text):
    """Cleans text for the model."""
    if not isinstance(text, str): return ""
    return text.lower().translate(str.maketrans('', '', string.punctuation))

def load_model(path):
    """Loads the pre-trained intent classification model."""
    try:
        model = joblib.load(path)
        print("✅ Model loaded successfully.")
        return model
    except FileNotFoundError:
        print(f"❌ Error: Model file not found at '{path}'.")
        return None
    except Exception as e:
        print(f"❌ An error occurred while loading the model: {e}")
        return None

def extract_entity(command, keywords):
    """Extracts the main subject/entity from a command."""
    for keyword in keywords:
        command = re.sub(r'\b' + re.escape(keyword) + r'\b', '', command, flags=re.IGNORECASE).strip()
    return command

def play_alert_sound():
    """Plays a simple alert sound (cross-platform)."""
    print("\a") # ASCII Bell character
    speak("Time is up!")

# ==================================================================================
# ACTION FUNCTIONS (NEW & IMPROVED)
# ==================================================================================

def set_alarm(command):
    """Sets an alarm. Tries to parse time from command first."""
    time_text = extract_entity(command, ["set alarm for", "set alarm at", "alarm"])
    match = re.search(r'(\d{1,2})[: ]?(\d{1,2})?\s*(am|pm)?', time_text)

    if not match:
        speak("What time should I set the alarm for? For example, say 6:30 am.")
        time_command = listen_command()
        if not time_command:
            speak("I didn't hear a time."); return
        match = re.search(r'(\d{1,2})[: ]?(\d{1,2})?\s*(am|pm)?', time_command)

    if not match:
        speak("Sorry, I couldn't understand the alarm time."); return

    hour, minute, period = int(match.group(1)), int(match.group(2) or 0), match.group(3)
    if period == "pm" and hour != 12: hour += 12
    elif period == "am" and hour == 12: hour = 0

    speak(f"Alarm set for {hour:02d}:{minute:02d}. I will notify you.")
    def alarm_thread():
        now = datetime.datetime.now()
        alarm_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if alarm_time < now: alarm_time += datetime.timedelta(days=1)
        time.sleep((alarm_time - now).total_seconds())
        play_alert_sound()
    threading.Thread(target=alarm_thread, daemon=True).start()

def set_reminder(command):
    """Sets a reminder. Tries to parse from command first."""
    speak("What should I remind you about?")
    reminder_text = listen_command()
    if not reminder_text:
        speak("I didn't catch the reminder message."); return

    speak("At what time should I remind you?")
    time_command = listen_command()
    if not time_command:
        speak("I didn't hear a time."); return

    match = re.search(r'(\d{1,2})[: ]?(\d{1,2})?\s*(am|pm)?', time_command)
    if not match:
        speak("Sorry, I couldn't understand the reminder time."); return

    hour, minute, period = int(match.group(1)), int(match.group(2) or 0), match.group(3)
    if period == "pm" and hour != 12: hour += 12
    elif period == "am" and hour == 12: hour = 0

    speak(f"Reminder set for {hour:02d}:{minute:02d}. I will remind you to {reminder_text}.")
    def reminder_thread():
        now = datetime.datetime.now()
        reminder_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if reminder_time < now: reminder_time += datetime.timedelta(days=1)
        time.sleep((reminder_time - now).total_seconds())
        speak(f"Reminder: {reminder_text}")
        play_alert_sound()
    threading.Thread(target=reminder_thread, daemon=True).start()

def set_timer(command):
    """Sets a countdown timer. Tries to parse from command first."""
    time_text = extract_entity(command, ["set timer for", "start timer for", "timer for"])
    min_match = re.search(r'(\d+)\s*minute', time_text)
    sec_match = re.search(r'(\d+)\s*second', time_text)
    
    if not min_match and not sec_match:
        speak("How long should I set the timer for?")
        time_command = listen_command()
        if not time_command: speak("I didn't hear a duration."); return
        min_match = re.search(r'(\d+)\s*minute', time_command)
        sec_match = re.search(r'(\d+)\s*second', time_command)

    minutes = int(min_match.group(1)) if min_match else 0
    seconds = int(sec_match.group(1)) if sec_match else 0
    total_seconds = minutes * 60 + seconds

    if total_seconds == 0:
        speak("Sorry, I couldn't recognize the timer duration."); return

    speak(f"Timer set for {minutes} minutes and {seconds} seconds.")
    def timer_thread():
        time.sleep(total_seconds)
        play_alert_sound()
    threading.Thread(target=timer_thread, daemon=True).start()

# def handle_notes(command):
#     """Allows user to add a multi-line note or have their notes read back."""
#     speak("Do you want to add a new note or read your existing notes?")
#     choice = listen_command()
#     if choice and "add" in choice:
#         speak("Okay, I'm ready to take your note. Say 'finish' or 'that's it' when you are done.")
#         note_lines = []
#         stop_keywords = ["that's it", "finish", "done", "end note", "stop note"]
#         while True:
#             line = listen_command(prompt="Listening to note...")
#             if not line or any(kw in line for kw in stop_keywords):
#                 break
#             note_lines.append(line.capitalize())
        
#         if note_lines:
#             timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             note_entry = f"\n--- Note on {timestamp} ---\n" + "\n".join(note_lines) + "\n"
            
#             # Ensure directory exists
#             os.makedirs(os.path.dirname(NOTES_FILE), exist_ok=True)
#             with open(NOTES_FILE, "a") as file:
#                 file.write(note_entry)
#             speak("Your note has been saved.")
#         else:
#             speak("No content was recorded for the note.")

#     elif choice and "read" in choice:
#         try:
#             with open(NOTES_FILE, 'r') as f:
#                 notes = f.read()
#                 if notes: speak("Here are your notes:"); speak(notes)
#                 else: speak("You don't have any notes yet.")
#         except FileNotFoundError:
#             speak("You don't have a notes file yet. Try adding a note first.")
#     else:
#         speak("I didn't catch that. Please say 'add' or 'read'.")

def handle_notes(command):
    """Allows user to add a multi-line note or have their notes read back."""
    speak("Do you want to add a new note or read your existing notes?")
    choice = listen_command()

    if choice and "new note" in choice.lower():
        speak("Okay, I'm ready to take your note. Say 'finish' or 'that's it' when you are done.")
        note_lines = []
        stop_keywords = ["that's it", "finish", "done", "end note", "stop note"]

        while True:
            line = listen_command()  # Removed prompt argument
            if not line or any(kw in line.lower() for kw in stop_keywords):
                break
            note_lines.append(line.capitalize())

        if note_lines:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            note_entry = f"\n--- Note on {timestamp} ---\n" + "\n".join(note_lines) + "\n"

            # Save notes directly (no os.makedirs needed if same dir)
            with open(NOTES_FILE, "a") as file:
                file.write(note_entry)

            speak("Your note has been saved.")
        else:
            speak("No content was recorded for the note.")

    elif choice and "read note" in choice.lower():
        try:
            with open(NOTES_FILE, 'r') as f:
                notes = f.read()
                if notes:
                    speak("Here are your notes:")
                    speak(notes)
                else:
                    speak("You don't have any notes yet.")
        except FileNotFoundError:
            speak("You don't have a notes file yet. Try adding a note first.")
    else:
        speak("I didn't catch that. Please say 'add note' or 'read note'.")

# ... [All other action functions remain the same as the previous version] ...
def get_fact(command):
    try:
        response = requests.get("https://uselessfacts.jsph.pl/random.json?language=en")
        fact_data = response.json()
        speak(fact_data['text'])
    except Exception as e:
        print(f"Fact API Error: {e}")
        speak("I couldn't fetch a fact right now, but here's one: The unicorn is the national animal of Scotland.")

def get_info_subject(command):
    entity = extract_entity(command, ["what is", "who is", "tell me about", "search for", "in wikipedia"])
    if not entity:
        speak("What subject are you interested in?")
        entity = listen_command()
        if not entity: return

    try:
        speak(f"Searching Wikipedia for {entity}...")
        result = wikipedia.summary(entity, sentences=2, auto_suggest=True, redirect=True)
        speak("According to Wikipedia:")
        speak(result)
    except (wikipedia.exceptions.PageError, wikipedia.exceptions.DisambiguationError):
        speak(f"I couldn't find a clear result for {entity} on Wikipedia. Would you like me to search Google instead?")
        if "yes" in (listen_command() or ""):
            search_google(f"search google for {entity}")
    except Exception as e:
        speak("Sorry, I ran into an error while searching.")
        print(f"Wikipedia/Google Search Error: {e}")

# def handle_messaging(command):
#     try:
#         match = re.search(r"send message to (.+?) and the message is (.+)", command)
#         if match:
#             name = match.group(1).strip().lower()
#             message = match.group(2).strip()
#             if name in contacts:
#                 phone_number = contacts[name]
#                 speak(f"Sending your message to {name.title()}.")
#                 pywhatkit.sendwhatmsg_instantly(phone_number, message, wait_time=15)
#                 speak("Message sent successfully.")
#             else:
#                 speak(f"Sorry, I don't have a contact named {name}.")
#         else:
#             speak("Please say it like this: 'send message to [name] and the message is [your message]'.")
#     except Exception as e:
#         print(f"WhatsApp Error: {e}")
#         speak("Something went wrong while sending the message.")
def handle_messaging(command):
    try:
        # Patterns to match variations in how the user may phrase the command
        patterns = [
            r"send message to (.+?) and the message is (.+)",
            r"send a message to (.+?) and saying (.+)",
            r"message (.+?) that (.+)",
            r"send (.+?) a message (.+)"
        ]

        name = message = None
        for pattern in patterns:
            match = re.search(pattern, command, re.IGNORECASE)
            if match:
                name = match.group(1).strip().lower()
                message = match.group(2).strip()
                break

        if name and message:
            if name in contacts:
                phone_number = contacts[name]
                speak(f"Sending your message to {name.title()}.")
                pywhatkit.sendwhatmsg_instantly(phone_number, message, wait_time=15)
                speak("Message sent successfully.")
            else:
                speak(f"Sorry, I don't have a contact named {name}.")
        else:
            speak("Please say something like 'send message to [name] and the message is [your message]' or 'message [name] that [message]'.")
    
    except Exception as e:
        print(f"WhatsApp Error: {e}")
        speak("Something went wrong while sending the message.")


def handle_email(command):
    speak("Who is the recipient?")
    recipient = listen_command()
    if not recipient: return
    recipient = recipient.replace(" at ", "@").replace(" dot ", ".").replace(" ", "")
    speak("What is the subject?")
    subject = listen_command()
    if not subject: return
    speak("What should the message say?")
    body = listen_command()
    if not body: return
    speak(f"Opening your email client to send an email to {recipient}.")
    webbrowser.open(f'mailto:{recipient}?subject={subject}&body={body}')

# def open_notepad(command):
#     speak("Opening a text editor.")
#     system = platform.system()
#     try:
#         if system == "Darwin": os.system("open -a TextEdit")
#         elif system == "Windows": os.system("start notepad")
#         else: os.system("gedit &")
#     except Exception as e:
#         speak("I couldn't open a text editor on your system.")
#         print(f"OS Command Error: {e}")
def notepad(command=None):
    import subprocess, shutil

    speak("Opening a text editor...")
    system = platform.system()

    try:
        editors = {
            "Darwin": ["TextEdit"],
            "Windows": ["notepad"],
            "Linux": ["gedit", "nano", "kate"]
        }

        available_editors = editors.get(system, [])

        # Try to open the first available editor
        for editor in available_editors:
            if shutil.which(editor.lower()):
                subprocess.Popen([editor])
                return

        speak("No default text editor found.")
    except Exception as e:
        speak("I couldn't open a text editor on your system.")
        print(f"OS Command Error: {e}")


def handle_exit(command):
    goodbyes = ["Goodbye!", "Shutting down. Have a great day!", "See you later!"]
    speak(random.choice(goodbyes))
    return "exit"

def handle_sleep(command):
    speak("Going to sleep. Say my name to wake me up.")
    return "sleep"

# def query_gemini(command):
#     if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
#         handle_fallback(command)
#         return
#     speak("That's an interesting question. Let me check...")
#     try:
#         url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"
#         payload = {"contents": [{"parts": [{"text": command}]}]}
#         headers = {'Content-Type': 'application/json'}
#         response = requests.post(url, json=payload, headers=headers)
#         response.raise_for_status()
#         result = response.json()
#         answer = result['candidates'][0]['content']['parts'][0]['text']
#         speak(answer)
#     except Exception as e:
#         print(f"Gemini API Error: {e}")
#         handle_fallback(command)
def query_gemini(command):
    """
    Sends a command to the Google Gemini API and speaks the response.
    """
    # 1. More robust API Key check
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY environment variable not set.")
        handle_fallback(command)
        return

    speak("That's an interesting question. Let me check...")

    # Set up the API call details
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"parts": [{"text": command}]}]}
    headers = {'Content-Type': 'application/json'}

    try:
        # 2. Added a timeout for robustness
        response = requests.post(url, json=payload, headers=headers, timeout=15)
        response.raise_for_status()  # This will raise an exception for HTTP errors (like 401, 404, 500)

        result = response.json()

        # 3. Safer response parsing to avoid crashes
        candidates = result.get('candidates', [])
        if candidates and candidates[0].get('content', {}).get('parts', []):
            answer = candidates[0]['content']['parts'][0].get('text')
        else:
            answer = None # Set answer to None if the structure is unexpected

        if answer:
            speak(answer)
        else:
            # Handle cases where the API replied, but without a valid answer text
            print("Gemini API returned an unexpected response format.")
            handle_fallback(command)

    except requests.exceptions.RequestException as e:
        # Catches network-related errors specifically (timeout, connection error, etc.)
        print(f"Network Error calling Gemini API: {e}")
        handle_fallback(command)
    except Exception as e:
        # Catches other potential errors (e.g., JSON parsing error)
        print(f"An unexpected error occurred: {e}")
        handle_fallback(command)

def get_time(command):
    now = datetime.datetime.now().strftime("%I:%M %p")
    speak(f"The current time is {now}")

def get_date(command):
    today = datetime.datetime.now().strftime("%B %d, %Y")
    speak(f"Today's date is {today}")

def get_news(command):
    if NEWSAPI_KEY == "YOUR_NEWSAPI_KEY":
        speak("News service is not configured. Please add your NewsAPI key.")
        return
    speak("Fetching the latest news headlines.")
    url = f"https://newsapi.org/v2/top-headlines?country=in&apiKey={NEWSAPI_KEY}"
    try:
        data = requests.get(url).json()
        articles = data.get("articles", [])[:4]
        if not articles: speak("I couldn't find any news articles.")
        else:
            for i, article in enumerate(articles): speak(f"Headline {i+1}: {article['title']}")
    except Exception as e:
        speak("Sorry, I'm having trouble fetching the news.")
        print(f"News API Error: {e}")

def get_joke(command): speak(pyjokes.get_joke())
def search_wikipedia(command): get_info_subject(command)
def open_youtube(command): speak("Opening YouTube."); webbrowser.open("https://www.youtube.com")
def open_whatsapp(command): speak("Opening WhatsApp."); webbrowser.open("https://web.whatsapp.com")
def open_chrome(command): speak("Opening Google Chrome."); webbrowser.open("https://www.google.com")
def open_website(command):
    speak("Which website should I open?")
    website = listen_command()
    if website:
        speak(f"Opening {website}.")
        if not website.startswith("http"): website = f"https://www.{website}.com"
        webbrowser.open(website)

def search_google(command):
    entity = extract_entity(command, ["search google for", "google", "search for"])
    if not entity:
        speak("What would you like to search for on Google?")
        entity = listen_command()
    if entity:
        speak(f"Searching Google for {entity}."); pywhatkit.search(entity)

def play_song_youtube(command):
    song = extract_entity(command, ["play", "on youtube", "song"])
    if not song:
        speak("What song would you like to play?"); song = listen_command()
    if song:
        speak(f"Playing {song} on YouTube."); pywhatkit.playonyt(song)
        
def get_weather(command):
    if OPENWEATHERMAP_API_KEY == "YOUR_OPENWEATHERMAP_API_KEY":
        speak("Weather service is not configured. Please add your OpenWeatherMap API key.")
        return

    # Extract city from command
    command = command.lower()
    city = None
    trigger_phrases = ["weather in", "weather at", "temperature in"]

    for phrase in trigger_phrases:
        if phrase in command:
            city = command.split(phrase)[-1].strip()
            break

    # If city not found, ask for it
    if not city:
        speak("For which city would you like the weather?")
        city = listen_command()

    if not city:
        speak("I didn't catch the city name. Please try again.")
        return

    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHERMAP_API_KEY}&units=metric"

    try:
        response = requests.get(url).json()
        if response.get("cod") == 200:
            temp = response["main"]["temp"]
            desc = response["weather"][0]["description"]
            speak(f"The weather in {city} is currently {desc} with a temperature of {temp} degrees Celsius.")
        else:
            speak(f"Sorry, I couldn't find the weather for {city}.")
    except Exception as e:
        speak("Sorry, I'm having trouble fetching the weather right now.")
        print(f"Weather API Error: {e}")


def show_calendar(command): speak("Showing your calendar.")

# def show_shopping_list(command): speak("Here is your shopping list.")
def show_shopping_list(command):
    notes_file = "delta_V_S/notes.txt"
    try:
        with open(notes_file, "r") as file:
            items = file.readlines()
        
        if not items:
            speak("Your shopping list is empty.")
            print("Shopping List: [Empty]")
        else:
            speak("Here is your shopping list.")
            shopping_list = ", ".join([item.strip() for item in items])
            print(f"Shopping List: {shopping_list}")
            speak(shopping_list)
    except FileNotFoundError:
        speak("I couldn't find your shopping list.")
        print("Error: notes.txt file not found.")


def handle_greeting(command):
    responses = ["Hello! How can I assist you today?", "Hi there! What can I do for you?", "Hey! I'm here to help."]
    speak(random.choice(responses))
def handle_gratitude(command):
    responses = ["You're welcome!", "No problem!", "Happy to help!"]
    speak(random.choice(responses))
def handle_chitchat(command):
    responses = ["I'm doing great, thanks for asking!", "I'm an AI, so I'm always running at peak performance!"]
    speak(random.choice(responses))
def handle_user_feedback(command): speak("Thank you for the feedback. I'm always learning.")
def handle_clarification(command): speak("My apologies. Could you please say that again?")
def handle_fallback(command): speak("I'm not quite sure how to handle that. Can you try asking in a different way?")

# gesture activation funtion
# ---------- Gesture control (start/stop NEWGEST.py) ----------
newgest_process = None  # holds subprocess.Popen object

def control_newgest(command):
    """
    Start or stop NEWGEST.py via voice/text commands.
    Recognised start phrases: "start gesture", "activate gesture", "start gesture mode", ...
    Recognised stop phrases: "stop gesture", "deactivate gesture", "stop gesture mode", ...
    """
    global newgest_process
    cmd = (command or "").lower()

    start_phrases = [
        "start gesture", "start gesture mode", "activate gesture",
        "activate gesture mode", "enable gesture", "run gesture",
        "start gesture recognition", "enable gesture mode"
    ]
    stop_phrases = [
        "stop gesture", "stop gesture mode", "deactivate gesture",
        "deactivate gesture mode", "disable gesture", "end gesture",
        "stop gesture recognition", "end gesture mode"
    ]

    # START
    if any(p in cmd for p in start_phrases):
        # already running?
        if newgest_process is not None and newgest_process.poll() is None:
            speak("Gesture mode is already running.")
            return

        # try to locate NEWGEST.py: script folder, current working dir, or raw name
        possible = [
            os.path.join(os.path.dirname(__file__), "NEWGEST.py"),
            os.path.join(os.getcwd(), "NEWGEST.py"),
            "NEWGEST.py"
        ]
        script_path = next((p for p in possible if p and os.path.exists(p)), None)
        if not script_path:
            speak("I couldn't find NEWGEST.py. Make sure NEWGEST.py is in the same folder as this script or current directory.")
            return

        try:
            # use same python interpreter that runs main.py
            newgest_process = subprocess.Popen([sys.executable, script_path],
                                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            speak("Gesture mode activated. NEWGEST.py started.")
        except Exception as e:
            print(f"Error starting NEWGEST.py: {e}")
            speak("Sorry, I couldn't start NEWGEST.py.")

    # STOP
    elif any(p in cmd for p in stop_phrases):
        if newgest_process is not None and newgest_process.poll() is None:
            try:
                newgest_process.terminate()   # polite shutdown
                try:
                    newgest_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    newgest_process.kill()     # force kill if not exiting
                speak("Gesture mode deactivated. NEWGEST.py stopped.")
            except Exception as e:
                print(f"Error stopping NEWGEST.py: {e}")
                speak("There was a problem stopping NEWGEST.py. I tried to force stop it.")
                try:
                    newgest_process.kill()
                except Exception:
                    pass
            finally:
                newgest_process = None
        else:
            speak("Gesture mode is not running right now.")
    else:
        # If function called without clear start/stop words
        speak("Please say 'start gesture' to run NEWGEST or 'stop gesture' to stop it.")




# ==================================================================================
# INTENT MAPPING
# ==================================================================================
INTENT_FUNCTIONS = {
    "gesture_activation": control_newgest,
    "time": get_time, "date": get_date, "weather": get_weather, "news": get_news, "fact": get_fact,
    "joke": get_joke, "get_info_subject": get_info_subject, "search_wikipedia": search_wikipedia,
    "open_chrome": open_chrome, "open_website": open_website, "open_youtube": open_youtube,
    "open_whatsapp": open_whatsapp, "open_wikipedia": search_wikipedia, "notepad": notepad,
    "search_google": search_google, "search_website": search_google, "play_song_youtube": play_song_youtube,
    "alarm": set_alarm, "reminder": set_reminder, "timer": set_timer, "calendar": show_calendar,
    "notes": handle_notes, "shopping_list": show_shopping_list, "email": handle_email,
    "messaging": handle_messaging, "greeting": handle_greeting, "gratitude": handle_gratitude,
    "chitchat": handle_chitchat, "user_feedback": handle_user_feedback,
    "clarification": handle_clarification, "fallback": query_gemini,
    "recommendations": lambda cmd: speak("I can look that up for you. What are you looking for?"),
    "audiobook": lambda cmd: speak("I can't handle audiobooks yet."),
    "finance": lambda cmd: speak("I am not equipped to provide financial information."),
    "games": lambda cmd: speak("Let's play a game! How about rock, paper, scissors?"),
    "podcast": lambda cmd: speak("Which podcast would you like to listen to?"),
    "recipe": lambda cmd: speak("I can find that recipe for you."),
    "search": search_google, "translation": lambda cmd: speak("What would you like to translate?"),
    "exit": handle_exit, "sleep": handle_sleep
}

# ==================================================================================
# MAIN ASSISTANT LOOP (IMPROVED LOGIC)
# ==================================================================================
def run_delta():
    """Main function to run the voice assistant with state-based wake word detection."""
    model = load_model(MODEL_PATH)
    if not model: return

    assistant_name = "delta"
    is_active = False
    
    speak("Delta assistant is now online. Say my name to activate.")

    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1.5)
        print(f"✅ Delta is running. Listening for '{assistant_name}'...")

        while True:
            if is_active:
                command = listen_command()
            else:
                command = listen_command(prompt=f" Muted. Listening for '{assistant_name}'...")
            
            if not command:
                continue

            if not is_active:
                if assistant_name in command:
                    is_active = True
                    speak("Hello sir, how can I help you?")
                continue
            
            # --- Active State Logic ---
            # 1. Check for critical state-changing commands first
            if any(word in command for word in ["go to sleep", "sleep mode", "sleep"]):
                if handle_sleep(command) == "sleep": is_active = False
                continue
            
            if any(word in command for word in ["exit", "stop", "shut down"]):
                if handle_exit(command) == "exit": break

            # 2. Check for keyword overrides for reliability
            intent = None
            if "send message to" in command and "and the message is" in command: intent = "messaging"
            elif "notepad" in command or "text editor" in command: intent = "open_notepad"
            elif "date" in command: intent = "date"
            elif "make a note" in command or "take a note" in command: intent = "notes"
            elif "gesture" in command:
                control_newgest(command)
                continue
            # 3. If no override, use the AI model
            if not intent:
                processed_command = preprocess_text(command)
                intent = model.predict([processed_command])[0]
            
            print(f"📌 Intent classified as: '{intent}'")
            action_function = INTENT_FUNCTIONS.get(intent, query_gemini)
            action_function(command)

# ==================================================================================
# SCRIPT ENTRY POINT
# ==================================================================================
if __name__ == "__main__":
    run_delta()

