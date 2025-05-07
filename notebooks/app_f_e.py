import pandas as pd
import spacy
from feature_extraction import convert_to_sentiment
from sklearn.naive_bayes import MultinomialNB # ComplementNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

nlp = spacy.load('en_core_web_sm')
df = pd.read_csv("../data/yelp_preprocessed.csv")

df['processed_text'] = df['processed_text'].fillna('')
# Feature Extraction steps
df['sentiment'] = df['stars'].apply(convert_to_sentiment)

sentiment_counts = df['sentiment'].value_counts()
print("Sentiment distribution:")
print(sentiment_counts)

X_train, X_test, y_train, y_test = train_test_split(
    df['processed_text'],
    df['sentiment'],
    test_size=0.2,
    random_state=42,
    stratify=df['sentiment']
)

# Feature extraction and model training with Bag of Words
count_vectorizer = CountVectorizer(max_features=5000)
X_train_counts = count_vectorizer.fit_transform(X_train)
X_test_counts = count_vectorizer.transform(X_test)

nb_counts = MultinomialNB()
nb_counts.fit(X_train_counts, y_train)
y_pred_counts = nb_counts.predict(X_test_counts)

# Feature extraction and model training with TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

nb_tfidf = MultinomialNB()
nb_tfidf.fit(X_train_tfidf, y_train)
y_pred_tfidf = nb_tfidf.predict(X_test_tfidf)

# Evaluate both models
print("Naive Bayes with Count Vectorizer:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_counts):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_counts))

print("\nNaive Bayes with TF-IDF Vectorizer:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_tfidf):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_tfidf))

# Save the best model and its vectorizer
import pickle

if accuracy_score(y_test, y_pred_counts) > accuracy_score(y_test, y_pred_tfidf):
    best_model = nb_counts
    best_vectorizer = count_vectorizer
    best_approach = "Count Vectorizer"
else:
    best_model = nb_tfidf
    best_vectorizer = tfidf_vectorizer
    best_approach = "TF-IDF Vectorizer"

# Create folder if it doesn't exist
import os
os.makedirs('../models', exist_ok=True)

with open('../models/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('../models/best_vectorizer.pkl', 'wb') as f:
    pickle.dump(best_vectorizer, f)

print(f"Best model saved using {best_approach}")
