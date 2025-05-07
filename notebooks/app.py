import pandas as pd
import spacy
from tqdm import tqdm
from preprocess_functions import clean_text, preprocessing

nlp = spacy.load('en_core_web_sm')
df = pd.read_csv("../data/yelp.csv")

df['cleaned_text'] = df['text'].apply(clean_text)
df['processed_text'] = [preprocessing(text, nlp) for text in tqdm(df['cleaned_text'])]

df.to_csv('../data/yelp_preprocessed.csv', index=False)