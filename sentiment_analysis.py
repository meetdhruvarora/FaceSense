def classify_sentiment(polarity):
    if polarity < -0.3:
        return "Negative"
    elif polarity > 0.3:
        return "Positive"
    else:
        return "Neutral"


def classify_tone(subjectivity):
    if subjectivity > 0.7:
        return "Highly Subjective"
    elif subjectivity < 0.3:
        return "Objective"
    else:
        return "Somewhat Subjective"
    

def analyze_sentiment_and_tone(text):
    from textblob import TextBlob

    blob = TextBlob(text)
    polarity, subjectivity = blob.sentiment.polarity, blob.sentiment.subjectivity

    sentiment = classify_sentiment(polarity)
    tone = classify_tone(subjectivity)
    print(f"Sentiment: {sentiment}, Tone: {tone}")
    return sentiment, tone

# text = "I feel really confident about this position, and I think I have the skills required."
# sentiment, tone = analyze_sentiment_and_tone(text)
# print(f"Sentiment: {sentiment}, Tone: {tone}")
