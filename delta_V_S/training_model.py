# this is dataset_5
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
import re

# --- 1. Data Loading and Preprocessing ---

def preprocess_text(text):
    """
    Cleans the input text by converting to lowercase and removing punctuation.
    """
    # Ensure text is a string
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    return text

# Define the correct filename for your new dataset
# dataset_filename = 'delta_V_S/datasets/updated_dataset.csv' # this for this dataset delta_V_S/datasets/updated_dataset.csv
dataset_filename = 'delta_V_S/datasets/dataset6.csv'
# Load the dataset from the provided CSV file
try:
    df = pd.read_csv(dataset_filename)
    print("Dataset loaded successfully.")
    print("Dataset shape:", df.shape)
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
except FileNotFoundError:
    print(f"Error: '{dataset_filename}' not found.")
    print("Please make sure the CSV file is in the same directory as this script.")
    exit()

# Drop rows with missing values in key columns
df.dropna(subset=['user_utterance', 'intent'], inplace=True)

# Apply the preprocessing function to the user utterances
df['cleaned_utterance'] = df['user_utterance'].apply(preprocess_text)

# Define our features (X) and target (y)
X = df['cleaned_utterance']
y = df['intent']

# --- 2. Train-Test Split ---

# Split the data into training and testing sets (80% train, 20% test)
# random_state ensures that the splits are the same every time we run the code
# stratify=y is important for imbalanced datasets to keep class distribution similar in train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")


# --- 3. Model Building and Training ---

# We will use a Pipeline to chain together the vectorizer and the classifier.
# This is best practice as it prevents data leakage from the test set.

# 1. TfidfVectorizer: Converts text data into numerical vectors.
#    TF-IDF (Term Frequency-Inverse Document Frequency) reflects how important
#    a word is to a document in a collection or corpus.
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2)) # Use 1-grams and 2-grams

# 2. LogisticRegression: A robust and efficient classification model.
#    'saga' solver is good for large datasets, and class_weight='balanced'
#    helps with imbalanced classes.
classifier = LogisticRegression(solver='saga', max_iter=1000, random_state=42, class_weight='balanced')

# Create the scikit-learn pipeline
model_pipeline = Pipeline([
    ('tfidf', tfidf_vectorizer),
    ('clf', classifier)
])

print("\nTraining the model on the new dataset...")
# Train the model on the training data
model_pipeline.fit(X_train, y_train)
print("Model training complete.")


# --- 4. Model Evaluation ---

print("\nEvaluating the model...")
# Make predictions on the test set
y_pred = model_pipeline.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

# Display a detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))


# --- 5. Saving the Trained Model ---

# Save the entire pipeline (vectorizer + classifier) to a single file--------------------model-------

model_filename = 'delta_V_S/model_5_2.joblib' 
joblib.dump(model_pipeline, model_filename)
print(f"\nModel saved successfully as '{model_filename}'")


# --- 6. Example of How to Load and Use the Model ---

print("\n--- Example Usage ---")
# Load the saved model from the file
loaded_model = joblib.load(model_filename)
print("Model loaded from file.")

# Let's test it with some new commands relevant to the updated dataset
test_commands = [
    "what is the weather like in london",
    "remind me to buy milk tomorrow at 8am",
    "play the latest episode of my favorite podcast",
    "turn on the living room lights",
    "how do you say hello in japanese",
    "send an email to my manager",
    "what's the price of google stock"
]

print("\nMaking predictions on new commands:")
for command in test_commands:
    # The model expects a list or iterable of texts
    predicted_intent = loaded_model.predict([command])[0]
    print(f"  Command: '{command}'")
    print(f"  Predicted Intent: -> {predicted_intent}\n")




