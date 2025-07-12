import speech_recognition as sr
import pyttsx3
import os
import webbrowser
import platform
import time
import pyjokes
import pywhatkit
import pickle
from datetime import datetime
import re
import wikipedia

# ========== Load Trained ML Model and Vectorizer ==========
try:
    with open("jarvis_training/jarvis_command_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("jarvis_training/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
except Exception as e:
    print(f"Model loading error: {e}")
    exit()

def predict_action(command):
    try:
        vec = vectorizer.transform([command.lower()])
        return model.predict(vec)[0]
    except Exception as e:
        print(f"Prediction error: {e}")
        return "unknown"

# ========== Assistant Setup ==========
engine = pyttsx3.init()
engine.setProperty('rate', 150)
assistant_name = "alpha"
is_active = False

def speak(text):
    print("alpha:", text)
    engine.say(text)
    engine.runAndWait()

contacts = {
    "akshay": "+918971573838",
    "yuvaraj": "+916362426223",
    "dad": "+911234567890",
    "mom": "+919112233445"
}

def listen_command():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=0.5)
        print("Listening...")
        try:
            audio = r.listen(source, timeout=3, phrase_time_limit=4)
            command = r.recognize_google(audio).lower()
            print("You said:", command)
            return command
        except:
            return ""

def open_chrome():
    if platform.system() == "Darwin":
        os.system("open -a 'Google Chrome'")
    elif platform.system() == "Windows":
        os.system("start chrome")
    else:
        os.system("google-chrome &")

def open_notepad():
    if platform.system() == "Darwin":
        os.system("open -a 'TextEdit'")
    elif platform.system() == "Windows":
        os.system("start notepad")
    else:
        os.system("gedit &")

def tell_joke():
    joke = pyjokes.get_joke()
    speak(joke)

# ========== Main Loop ==========
while True:
    command = listen_command()

    if not command:
        continue

    if not is_active:
        if assistant_name in command:
            is_active = True
            speak("Hello sir, how can I help you?")
        continue

    action = predict_action(command)

    if action == "get_time":
        now = datetime.now().strftime("%I:%M %p")
        speak(f"The time is {now}")

    elif action == "send_whatsapp":
        try:
            match = re.search(r"send message to (.+?) and the message is (.+)", command)
            if match:
                name_or_number = match.group(1).strip()
                message = match.group(2).strip()
                phone = contacts.get(name_or_number.lower(), name_or_number)
                speak(f"Sending your message to {name_or_number}")
                pywhatkit.sendwhatmsg_instantly(phone, message)
            else:
                speak("Please say it like 'send message to [name] and the message is [your message]'")
        except Exception as e:
            print("Error:", e)
            speak("Something went wrong while sending the message.")

    elif action == "open_chrome":
        speak("Opening Chrome.")
        open_chrome()

    elif action == "play_song" or action == "play_music":
        song = command.replace("play song", "").replace("play", "").strip()
        if song:
            speak(f"Playing {song} on YouTube")
            pywhatkit.playonyt(song)
        else:
            speak("Please tell me the name of the song.")

    elif action == "tell_joke":
        tell_joke()

    elif action == "sleep_mode":
        speak("Going to sleep.")
        is_active = False

    elif action == "say_name":
        speak("I am alpha, your assistant.")

    elif action == "search_wikipedia":
        query = command.replace("search wikipedia for", "").strip()
        speak(f"Searching Wikipedia for {query}")
        try:
            result = wikipedia.summary(query, sentences=2)
            speak(result)
        except:
            speak("Sorry, I couldn’t find that.")

    elif action == "open_youtube":
        speak("Opening YouTube.")
        webbrowser.open("https://www.youtube.com")

    elif action == "open_google":
        speak("Opening Google.")
        webbrowser.open("https://www.google.com")

    elif action == "google_search":
        query = command.replace("search for", "").strip()
        speak(f"Searching Google for {query}")
        webbrowser.open(f"https://www.google.com/search?q={query}")

    elif action == "open_notepad":
        speak("Opening Notepad.")
        open_notepad()

    elif action == "system_info":
        info = f"You are using {platform.system()} with {platform.processor()}."
        speak(info)

    elif action == "purpose":
        speak("I exist to help you, make life easier, and occasionally tell bad jokes.")

    elif action == "shutdown":
        speak("Are you sure you want to shut down?")
        confirm = listen_command()
        if "yes" in confirm:
            speak("Shutting down.")
            if platform.system() == "Windows":
                os.system("shutdown /s /t 1")
            elif platform.system() == "Darwin":
                os.system("sudo shutdown -h now")

    elif action == "restart":
        speak("Are you sure you want to restart?")
        confirm = listen_command()
        if "yes" in confirm:
            speak("Restarting.")
            if platform.system() == "Windows":
                os.system("shutdown /r /t 1")
            elif platform.system() == "Darwin":
                os.system("sudo shutdown -r now")

    elif action == "exit" or action == "stop":
        speak("Goodbye sir!")
        break

    elif action == "unknown":
        speak("Sorry, I didn’t understand that.")

    time.sleep(0.5)
