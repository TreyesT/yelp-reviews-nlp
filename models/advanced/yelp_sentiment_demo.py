
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
    print(f"\nText: {text}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Confidence: {result['confidence']:.4f}")

if __name__ == "__main__":
    main()
