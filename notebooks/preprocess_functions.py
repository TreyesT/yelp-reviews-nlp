import spacy
import re

def clean_text(text):
    text = text.lower()

    # Remove punctuation, numbers, whitespaces, and special characters
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def preprocessing(text, nlp, remove_stopwords=True, lemmatize=True):
    cleaned_text = clean_text(text)
    doc = nlp(cleaned_text)

    processed_tokens = []

    for token in doc:
        # Skip stopwords if requested
        if remove_stopwords and token.is_stop:
            continue
        # Skip punctuation and whitespace
        if token.is_punct or token.is_space:
            continue
        # Get the base form (lemma) if requested, otherwise use the original form
        processed_token = token.lemma_ if lemmatize else token.text

        processed_tokens.append(processed_token)

    # Join tokens back into string
    processed_text = ' '.join(processed_tokens)

    return processed_text