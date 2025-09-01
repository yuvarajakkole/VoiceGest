
# delta_text.py model 5
import sys
import os

# Import your assistant module
sys.path.append(os.path.abspath("delta_V_S"))
import delta_assistant as da

def run_text_assistant():
    # Load the model once
    model = da.load_model(da.MODEL_PATH)
    if not model:
        print("❌ Could not load model. Exiting.")
        return

    da.speak("Delta text assistant is ready. Type 'exit' to quit.")

    while True:
        # Take text input instead of speech
        command = input("⌨️ Enter your command: ").strip().lower()
        if not command:
            continue

        if command in ["exit", "quit", "stop"]:
            da.handle_exit(command)
            break

        # --- Detect intent ---
        processed_command = da.preprocess_text(command)
        try:
            intent = model.predict([processed_command])[0]
        except Exception:
            intent = "fallback"

        print(f"📌 Intent classified as: {intent}")

        # --- Execute mapped action ---
        action_function = da.INTENT_FUNCTIONS.get(intent, da.query_gemini)
        action_function(command)

if __name__ == "__main__":
    run_text_assistant()
