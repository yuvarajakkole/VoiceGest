import pandas as pd

# Load dataset
df = pd.read_csv("delta_V_S/datasets/updated_dataset_with_gesture.csv")

# Normalize intent names
df['intent'] = df['intent'].str.strip().str.lower()

# Count intents
intent_counts = df['intent'].value_counts()

# Convert to DataFrame
intent_summary = intent_counts.reset_index()
intent_summary.columns = ['intent', 'count']

# Print to terminal
print(intent_summary)
