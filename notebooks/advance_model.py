import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import joblib
import warnings
warnings.filterwarnings('ignore')

# NLP libraries
import spacy
from feature_extraction import convert_to_sentiment

# Machine learning libraries
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Create output directories
os.makedirs('../models/advanced', exist_ok=True)
os.makedirs('../results/advanced', exist_ok=True)

print("Loading data...")
df = pd.read_csv("../data/yelp_preprocessed.csv")

# Data cleaning and preparation
print("Preparing data...")
df['processed_text'] = df['processed_text'].fillna('')
df['sentiment'] = df['stars'].apply(convert_to_sentiment)

print(f"Dataset shape after cleaning: {df.shape}")
print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")

# Define feature engineering functions
def add_text_features(df, text_column):
    """Add engineered features based on text data"""
    # Create a copy of the dataframe
    df_features = df.copy()

    # Add text length feature
    df_features['text_length'] = df_features[text_column].apply(len)

    # Add word count feature
    df_features['word_count'] = df_features[text_column].apply(lambda x: len(x.split()))

    # Add average word length feature
    df_features['avg_word_length'] = df_features[text_column].apply(
        lambda x: np.mean([len(word) for word in x.split()]) if len(x.split()) > 0 else 0
    )

    return df_features

# Create feature-enhanced dataset
df_features = add_text_features(df, 'processed_text')

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    df_features['processed_text'],
    df_features['sentiment'],
    test_size=0.2,
    random_state=42,
    stratify=df_features['sentiment']
)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Define model pipelines
print("Defining model pipelines...")
pipelines = {
    'MultinomialNB_CountVec': Pipeline([
        ('vectorizer', CountVectorizer(max_features=10000, ngram_range=(1, 2))),
        ('classifier', MultinomialNB())
    ]),
    'MultinomialNB_TfidfVec': Pipeline([
        ('vectorizer', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ('classifier', MultinomialNB())
    ]),
    'LinearSVC_TfidfVec': Pipeline([
        ('vectorizer', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ('classifier', LinearSVC(random_state=42))
    ]),
    'LogisticRegression_TfidfVec': Pipeline([
        ('vectorizer', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
}

# Cross-validation scores for each pipeline
print("Performing cross-validation...")
cv_results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, pipeline in pipelines.items():
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')
    cv_results[name] = {
        'scores': cv_scores,
        'mean': cv_scores.mean(),
        'std': cv_scores.std()
    }
    print(f"{name} CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Find the best model from cross-validation
best_model_name = max(cv_results, key=lambda k: cv_results[k]['mean'])
print(f"\nBest model from cross-validation: {best_model_name}")

# Hyperparameter tuning for the best model
print("\nPerforming hyperparameter tuning...")

# Define parameter grids for different models
param_grids = {
    'MultinomialNB_CountVec': {
        'vectorizer__max_features': [5000, 10000, 15000],
        'vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'classifier__alpha': [0.01, 0.1, 1.0]
    },
    'MultinomialNB_TfidfVec': {
        'vectorizer__max_features': [5000, 10000, 15000],
        'vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'classifier__alpha': [0.01, 0.1, 1.0]
    },
    'LinearSVC_TfidfVec': {
        'vectorizer__max_features': [5000, 10000, 15000],
        'vectorizer__ngram_range': [(1, 1), (1, 2)],
        'classifier__C': [0.1, 1.0, 10.0]
    },
    'LogisticRegression_TfidfVec': {
        'vectorizer__max_features': [5000, 10000, 15000],
        'vectorizer__ngram_range': [(1, 1), (1, 2)],
        'classifier__C': [0.1, 1.0, 10.0]
    }
}

# Run grid search on the best model
grid_search = GridSearchCV(
    pipelines[best_model_name],
    param_grids[best_model_name],
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Get the best model from grid search
best_pipeline = grid_search.best_estimator_

# Evaluate the best model on test data
print("\nEvaluating best model on test data...")
y_pred = best_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Best model test accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Best Model')
plt.savefig('../results/advanced/confusion_matrix_best_model.png')
plt.close()

# Feature importance analysis for the best model
print("\nAnalyzing feature importance...")

# For vectorizer-based models
feature_names = best_pipeline.named_steps['vectorizer'].get_feature_names_out()

# For different classifier types
if best_model_name.startswith('MultinomialNB'):
    classifier = best_pipeline.named_steps['classifier']
    feature_importance = classifier.feature_log_prob_[1] - classifier.feature_log_prob_[0]

elif best_model_name.startswith('LinearSVC') or best_model_name.startswith('LogisticRegression'):
    classifier = best_pipeline.named_steps['classifier']
    feature_importance = classifier.coef_[0]

else:  # Default for other models
    feature_importance = np.zeros(len(feature_names))

# Get top features
top_n = 20
top_indices = np.argsort(feature_importance)[-top_n:]
top_features = [(feature_names[i], feature_importance[i]) for i in top_indices]
bottom_indices = np.argsort(feature_importance)[:top_n]
bottom_features = [(feature_names[i], feature_importance[i]) for i in bottom_indices]

print("\nTop positive features:")
for feature, importance in reversed(top_features):
    print(f"{feature}: {importance:.4f}")

print("\nTop negative features:")
for feature, importance in bottom_features:
    print(f"{feature}: {importance:.4f}")

# Visualize feature importance
plt.figure(figsize=(12, 8))
plt.barh([f[0] for f in reversed(top_features)], [f[1] for f in reversed(top_features)])
plt.xlabel('Importance')
plt.title('Top Positive Features')
plt.tight_layout()
plt.savefig('../results/advanced/top_positive_features.png')
plt.close()

plt.figure(figsize=(12, 8))
plt.barh([f[0] for f in bottom_features], [f[1] for f in bottom_features])
plt.xlabel('Importance')
plt.title('Top Negative Features')
plt.tight_layout()
plt.savefig('../results/advanced/top_negative_features.png')
plt.close()

# Save the best model
print("\nSaving the best model...")
with open('../models/advanced/best_pipeline.pkl', 'wb') as f:
    pickle.dump(best_pipeline, f)

def predict_sentiment(text, pipeline=best_pipeline):
    # Transform text to features
    if hasattr(pipeline.named_steps['classifier'], 'predict_proba'):
        prediction = pipeline.predict([text])[0]
        probabilities = pipeline.predict_proba([text])[0]
        confidence = probabilities.max()
    else:
        # For models without predict_proba (like LinearSVC)
        prediction = pipeline.predict([text])[0]
        # Use decision function as a proxy for confidence
        decision_values = pipeline.decision_function([text])[0]
        if isinstance(decision_values, np.ndarray):
            confidence = abs(decision_values).max()
        else:
            confidence = abs(decision_values)

    return {
        'sentiment': prediction,
        'confidence': confidence
    }

# Save the prediction function
with open('../models/advanced/predict_function.pkl', 'wb') as f:
    pickle.dump(predict_sentiment, f)

# Test the prediction function
print("\nTesting prediction function...")
example_texts = [
    "I absolutely love this restaurant! The food is amazing and the service is excellent.",
    "This place is terrible. I regret eating here. Complete waste of money.",
    "It's okay, not great but not terrible either.",
    "The staff was friendly and helpful, but the food was disappointing.",
    "The quality is poor and the prices are too high for what you get."
]

for text in example_texts:
    result = predict_sentiment(text)
    print(f"Text: {text}")
    print(f"Predicted sentiment: {result['sentiment']}")
    print(f"Confidence: {result['confidence']:.4f}\n")

print("Advanced model training and evaluation complete!")

# Additional: Create a simple demo script
demo_script = """
import pickle
import sys

# Load the prediction function
with open('../models/advanced/predict_function.pkl', 'rb') as f:
    predict_sentiment = pickle.load(f)

def main():
    if len(sys.argv) > 1:
        text = ' '.join(sys.argv[1:])
    else:
        text = input("Enter a Yelp review to analyze sentiment: ")
    
    result = predict_sentiment(text)
    print(f"\\nText: {text}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Confidence: {result['confidence']:.4f}")

if __name__ == "__main__":
    main()
"""

with open('../models/advanced/yelp_sentiment_demo.py', 'w') as f:
    f.write(demo_script)

print("\nDemo script created at ../models/advanced/yelp_sentiment_demo.py")
print("You can run the demo with: python ../models/advanced/yelp_sentiment_demo.py \"Your review text here\"")