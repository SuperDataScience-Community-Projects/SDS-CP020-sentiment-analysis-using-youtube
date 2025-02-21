import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from collections import Counter
from wordcloud import WordCloud
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset 
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import Dataset, DataLoader

# Streamlit App Title
st.title("Sentiment Analysis with Streamlit")

# Download NLTK data
nltk.download('punkt')        # For tokenization
nltk.download("stopwords")    # Stopwords list
nltk.download("wordnet")      # Lemmatization

# Load data
with open("comments.txt", "r", encoding="utf-8") as file:
    lines = [line.strip() for line in file]  # Remove extra spaces/newlines

df = pd.DataFrame(lines, columns=["Comments"])
st.write("### Raw Data")
st.dataframe(df.head(n=20))

# Initialize Tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'&#\d+;', '', text)  # Remove HTML entities
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation and special characters
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\b#\w+\b', '', text)  # Remove words starting with '#'
    words = word_tokenize(text)  # Tokenize words
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Lemmatization & remove stopwords
    return " ".join(words)

# Apply the function to all comments
df["Cleaned_Comments"] = df["Comments"].apply(clean_text)
st.write("### Cleaned Data")
st.dataframe(df.head(n=50))

# Load pre-trained tokenizer and model for sentiment analysis
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# Tokenize comments
inputs = tokenizer(df["Cleaned_Comments"].tolist(), padding=True, truncation=True, return_tensors="pt", max_length=512)

# Get model predictions
with torch.no_grad():
    outputs = model(**inputs)

# Convert logits to probabilities
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

# Get sentiment labels (0 = Negative, 1 = Positive)
labels = torch.argmax(predictions, axis=1).tolist()

# Convert Sentiment to 0, 1, 2
sentiment_mapping = {0: 1, 1: 0}  # 0 → Negative (1), 1 → Positive (0)
df["Sentiment_Label"] = [sentiment_mapping[label] for label in labels]

# Assign Neutral (2) to Some Cases
df.loc[predictions.max(dim=1).values.cpu().numpy() < 0.6, "Sentiment_Label"] = 2  # If model is unsure, mark as Neutral

# Display the labeled data
st.write("### Labeled Data")
st.dataframe(df.head(n=20))

# Visualize Sentiment Distribution in a Pie Chart
st.write("### Sentiment Distribution (Pie Chart)")
sentiment_counts = df["Sentiment_Label"].value_counts()
sentiment_labels = ["Negative", "Positive", "Neutral"]
available_labels = [label for label in sentiment_labels if sentiment_counts.get(sentiment_labels.index(label), 0) > 0]
available_counts = [sentiment_counts.get(sentiment_labels.index(label), 0) for label in available_labels]

fig, ax = plt.subplots()
ax.pie(available_counts, labels=available_labels, autopct="%1.1f%%", startangle=90, colors=["red", "green", "blue"])
ax.axis("equal")
st.pyplot(fig)

# Tokenize the comments
tokens = tokenizer(df["Cleaned_Comments"].to_list(), padding=True, truncation=True, return_tensors="pt", max_length=512)

# Convert to Hugging Face Dataset
dataset = HFDataset.from_dict({
    "input_ids": tokens["input_ids"],
    "attention_mask": tokens["attention_mask"],
    "label": df["Sentiment_Label"].to_list()
})

# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",          # output directory
    eval_strategy="epoch",          # evaluation frequency
    learning_rate=2e-5,             # learning rate
    per_device_train_batch_size=8,  # batch size for training
    per_device_eval_batch_size=8,   # batch size for evaluation
    num_train_epochs=3,             # number of training epochs
    weight_decay=0.01,              # strength of weight decay
)

# Trainer
trainer = Trainer(
    model=model,                    # the pre-trained model
    args=training_args,             # training arguments
    train_dataset=dataset,          # training dataset
    eval_dataset=dataset,           # validation dataset (can be a separate dataset)
)

# Fine-tune the model
st.write("### Fine-tuning the Model")
trainer.train()

# Evaluate the model
st.write("### Evaluating the Model")
eval_results = trainer.evaluate()
st.write("Evaluation Results:", eval_results)

# Predict sentiment on new comments
st.write("### Predict Sentiment on New Comments")

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
# Define label mapping
label_mapping = {0: "Negative", 1: "Positive", 2: "Neutral"}
# Predict sentiment on new comments
def predict_sentiment(comment):
    inputs = tokenizer(comment, return_tensors="pt", padding=True, truncation=True).to(device)
    output = model(**inputs)
    logits = output.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    print("Probabilities:", probs)
    sentiment_label = torch.argmax(logits, dim=1).item()  # Get predicted sentiment label
    sentiment_name = label_mapping.get(sentiment_label, "Unknown")  # Map label to name
    return sentiment_name

comment = st.text_input("Enter a comment to analyze its sentiment:")
if comment and comment.strip():  # Ensure the comment is not empty
    sentiment = predict_sentiment(comment)
    st.write(f"Predicted sentiment: {sentiment}")

# Classification Report
st.write("### Classification Report")
predictions = trainer.predict(dataset)
preds = np.argmax(predictions.predictions, axis=-1)
true_labels = dataset["label"]
report = classification_report(true_labels, preds, target_names=["Negative", "Positive", "Neutral"], output_dict=True)
st.write(pd.DataFrame(report).transpose())

# Accuracy
accuracy = accuracy_score(true_labels, preds)  # Calculate accuracy
st.write(f"Accuracy: {accuracy:.2f}")