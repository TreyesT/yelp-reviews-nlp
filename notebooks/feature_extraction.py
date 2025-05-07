def convert_to_sentiment(stars):
    if stars <= 2:
        return 'negative'
    elif stars == 3:
        return 'neutral'
    else:  # 4 or 5 stars
        return 'positive'