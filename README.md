# Yelp Reviews Sentiment Analysis Using NLP
## Project Overview
This repository explores sentiment analysis on **Yelp reviews** using **Natural Language Processing (NLP)** techniques. We analyze customer reviews to automatically classify sentiment as **positive**, **negative**, or **neutral** based on the text content. By leveraging spaCy for text preprocessing and various feature extraction methods, we build and evaluate Naive Bayes classification models. The project demonstrates the effectiveness of different text representation approaches (Bag-of-Words, TF-IDF, and word embeddings) and illustrates the power of NLP for understanding customer sentiment.

## File/Folder Organization
- **data/**  
  Contains the dataset used for training and evaluation.  
  - **yelp.csv**: The original dataset containing Yelp reviews with star ratings.
  - **yelp_preprocessed.csv**: The cleaned and preprocessed dataset ready for model training.

- **notebooks/**  
  Contains Python scripts that detail the project workflow:
  - **app.py** – Text preprocessing script using spaCy to clean and prepare the review text.
  - **preprocess_functions.py** – Helper functions for text cleaning and preprocessing.
  - **app_f_e.py** – Feature extraction and model training script.
  - **feature_extraction.py** – Helper functions for feature extraction and sentiment conversion.
  - **model_analysis.py** - Analyzing feature importance and model performance.

- **articles/**  
  Contains write-ups and documentation summarizing the main findings, methodologies, and insights from the analysis.

- **presentations/**  
  Contains presentation materials (e.g., slides, PDFs) that provide a high-level summary of the project, including key metrics, methodologies, and future directions.

- **models/**  
  Contains saved models and vectorizers:
  - **best_model.pkl** – The trained Naive Bayes classifier with the highest performance.
  - **best_vectorizer.pkl** – The corresponding vectorizer (either Count or TF-IDF) used with the best model.

- **figures/**  
  Contains visualizations of model performance, feature importance, and sentiment distribution.
  - **confusion_matrices.png** – Visualization of model prediction accuracy.
  - **top_features_positive.png** – Words most strongly associated with positive sentiment.
  - **top_features_negative.png** – Words most strongly associated with negative sentiment.
  - **top_features_neutral.png** – Words most strongly associated with neutral sentiment.

- **results/advance/**  
  Contains visualizations of model performance, feature importance, and sentiment distribution.
  - **confusion_matrices_best_model.png** – Visualization of the best model prediction accuracy.
  - **top_negative_features.png** – Words most strongly associated with negative sentiment.
  - **top_positive_features.png** – Words most strongly associated with positive sentiment.

## Key Findings
- **Preprocessing Impact:**  
  spaCy's lemmatization and stopword removal significantly improve model performance by reducing vocabulary size and focusing on meaningful content.
  
- **Feature Representation:**  
  Comparative analysis of different text representation methods (Bag-of-Words vs. TF-IDF) reveals their strengths and weaknesses for sentiment classification.
  
- **Sentiment Predictors:**  
  Analysis of feature importance identifies key words and phrases that strongly indicate positive or negative sentiment in restaurant reviews.
  
- **Model Performance:**  
  Naive Bayes classifiers achieve strong performance, with detailed analysis of where they succeed and areas for improvement.

## Data Source
The dataset consists of 10,000 Yelp reviews, each with a star rating (1-5) and the corresponding review text. The sentiment classes were derived from star ratings:
- 1-2 stars: Negative sentiment
- 3 stars: Neutral sentiment
- 4-5 stars: Positive sentiment

## How to Use
1. **Clone or Download** this repository.

2. **Install Dependencies:**
   ```
   pip install pandas numpy scikit-learn spacy tqdm matplotlib seaborn
   python -m spacy download en_core_web_sm
   ```

3. **Explore the project** in the following order:
   - Run the preprocessing script first:
     ```
     python notebooks/app.py
     ```
   - Then train and evaluate the models:
     ```
     python notebooks/app_f_e.py
     ```
   - Analyze feature importance and model insights:
     ```
     python notebooks/model_analysis.py
     ```

## Future Enhancements
- Implement more advanced NLP techniques such as word embeddings (Word2Vec, GloVe) for improved semantic understanding
- Experiment with deep learning models like LSTM or BERT for potentially higher accuracy
- Create a simple web application for real-time sentiment analysis of user-input reviews
- Expand the analysis to include aspect-based sentiment analysis (identifying sentiment towards specific aspects of the restaurant)

## Acknowledgments
- Data sourced from the Yelp Open Dataset
- Project completed as part of NLP and machine learning coursework
