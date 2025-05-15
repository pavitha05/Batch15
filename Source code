import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import re

# Download required NLTK data (run this once)
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')  # Add this line
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        text = re.sub(r'http\S+', '', text) # Remove URLs
        text = re.sub(r'@\S+', '', text) # Remove mentions
        text = re.sub(r'#\S+', '', text) # Remove hashtags
        text = re.sub(r'[^a-zA-Z\s]', '', text, re.IGNORECASE) # Remove special characters and numbers
        tokens = word_tokenize(text.lower())
        tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in self.stop_words and len(w) > 2]
        return " ".join(tokens)

    def analyze_sentiment(self, text):
        preprocessed_text = self.preprocess_text(text)
        if not preprocessed_text:
            return {'compound': 0.0, 'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'sentiment': 'neutral'}
        vs = self.analyzer.polarity_scores(preprocessed_text)
        if vs['compound'] >= 0.05:
            sentiment = 'positive'
        elif vs['compound'] <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        return {**vs, 'sentiment': sentiment}

# Example Usage with a list of social media posts
if __name__ == "__main__":
    social_media_posts = [
        "I am so happy about the new update! It's amazing.",
        "This product is absolutely terrible. What a waste of money.",
        "Just watched a movie. It was okay, nothing special.",
        "Feeling really down today :(",
        "The weather is beautiful and sunny! #goodvibes",
        "@user1 Thanks for the support!",
        "Check out my new blog post: https://example.com/blog",
        "This is an interesting discussion.",
        "I'm so excited for the concert tonight!",
        "The customer service was incredibly unhelpful."
    ]

    sentiment_analyzer = SentimentAnalyzer()
    results = []

    for post in social_media_posts:
        sentiment = sentiment_analyzer.analyze_sentiment(post)
        results.append({'post': post, **sentiment})

    df = pd.DataFrame(results)
    print(df)

    # You can further analyze the results, e.g., calculate the distribution of sentiments
    print("\nSentiment Distribution:")
    print(df['sentiment'].value_counts())
