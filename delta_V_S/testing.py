import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import re
import os

# --- Important: Make sure these match your training script ---
MODEL_FILENAME = 'delta_V_S/model_5_2.joblib'
DATASET_FILENAME = 'delta_V_S/datasets/dataset6.csv'
TEST_SPLIT_SIZE = 0.3
RANDOM_STATE = 55

# --- 1. Load Model and Data ---

def preprocess_text(text):
    """Cleans text. Must be identical to the function used in training."""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

print(f"Loading model from '{MODEL_FILENAME}'...")
try:
    model_dir = os.path.dirname(MODEL_FILENAME)
    if model_dir and not os.path.exists(model_dir):
        print(f"Error: Directory '{model_dir}' does not exist.")
        exit()
    model = joblib.load(MODEL_FILENAME)
except FileNotFoundError:
    print(f"Error: Model file '{MODEL_FILENAME}' not found.")
    print("Please run the training script first.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    exit()

print(f"Loading data from '{DATASET_FILENAME}' to create the test set...")
try:
    df = pd.read_csv(DATASET_FILENAME)
except FileNotFoundError:
    print(f"Error: Dataset file '{DATASET_FILENAME}' not found.")
    exit()

# --- 2. Prepare the Test Data ---

# Rename column to match training script
df.rename(columns={"command": "user_utterance"}, inplace=True)

# Drop missing values and preprocess
df.dropna(subset=['user_utterance', 'intent'], inplace=True)
df['cleaned_utterance'] = df['user_utterance'].apply(preprocess_text)

X = df['cleaned_utterance']
y = df['intent']

# Split test set same way as training
_, X_test, _, y_test = train_test_split(
    X, y, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"✅ Test set created with {len(X_test)} samples.")

# --- 3. Evaluate the Model ---

print("\n🔎 Making predictions on the test set...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Model Accuracy on Test Set: {accuracy:.4f}\n")
print("📊 Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# --- 4. Confusion Matrix Visualization ---

print("📉 Generating confusion matrix visualization...")
class_names = model.classes_
cm = confusion_matrix(y_test, y_pred, labels=class_names)

plt.figure(figsize=(18, 15))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix', fontsize=20)
plt.ylabel('Actual Intent', fontsize=16)
plt.xlabel('Predicted Intent', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# --- 5. Interactive Testing ---
print("\n--- 🎤 Interactive Model Test ---")
print("Type your command to see the model's prediction.")
print("Type 'exit' or 'quit' to stop.\n")

while True:
    try:
        user_input = input("Enter a command: ")
        if user_input.lower() in ['exit', 'quit']:
            print("👋 Exiting interactive test.")
            break

        cleaned_input = preprocess_text(user_input)
        predicted_intent = model.predict([cleaned_input])[0]

        probabilities = model.predict_proba([cleaned_input])[0]
        prob_per_class = dict(zip(model.classes_, probabilities))
        confidence_score = prob_per_class[predicted_intent]

        print(f"👉 Predicted Intent: '{predicted_intent}' (Confidence: {confidence_score:.2%})\n")

    except Exception as e:
        print(f"⚠️ An error occurred: {e}")

