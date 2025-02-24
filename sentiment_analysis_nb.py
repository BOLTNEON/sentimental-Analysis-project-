import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Sample dataset (replace with actual dataset)
data = {
    "text": [
        "I love this product! It's amazing.",
        "This is the worst experience I've ever had.",
        "Absolutely fantastic! Highly recommend.",
        "Terrible, never buying again.",
        "It's okay, not the best but not the worst.",
    ],
    "sentiment": [1, 0, 1, 0, 1]  # 1 = Positive, 0 = Negative
}
df = pd.DataFrame(data)

# Text Preprocessing
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text.lower())  # Tokenization & lowercasing
    tokens = [word for word in tokens if word.isalnum()]  # Remove punctuation
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
    return " ".join(tokens)

# Apply preprocessing
df["cleaned_text"] = df["text"].apply(preprocess_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["cleaned_text"])
y = df["sentiment"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naïve Bayes Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Evaluate Model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Test with a new sample
new_text = ["The product quality is very bad and I hate it."]
new_text_cleaned = [preprocess_text(text) for text in new_text]
new_text_vectorized = vectorizer.transform(new_text_cleaned)
print("Sentiment Prediction:", model.predict(new_text_vectorized))
