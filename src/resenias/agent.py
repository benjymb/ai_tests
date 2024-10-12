import pandas as pd
import re
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


# Pregunta 1 : Análisis de sentimientos


class SentimentAnalyzer:
    
    def __init__(self, data_path):
        # Initialize with the path to the dataset
        self.data = pd.read_csv(data_path)
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model = LogisticRegression()
    
    def preprocess_text(self, text):
        """ Clean and preprocess the text. """
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        words = text.split()
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        return ' '.join(words)
    
    def preprocess_reviews(self):
        """ Apply preprocessing to all reviews in the dataset. """
        self.data['cleaned_reviews'] = self.data['review'].apply(self.preprocess_text)
    
    def vectorize_reviews(self):
        """ Convert the cleaned reviews to a TF-IDF matrix. """
        X = self.vectorizer.fit_transform(self.data['cleaned_reviews'])
        return X
    
    def split_data(self):
        """ Split the dataset into training and testing sets. """
        X = self.vectorize_reviews()
        def to_number(sentiment):
            if sentiment == 'positive':
                return 1
            else:
                return 0
        y = self.data['sentiment'].apply(to_number)
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def train_model(self):
        """ Train a Logistic Regression model on the data. """
        X_train, X_test, y_train, y_test = self.split_data()
        self.model.fit(X_train, y_train)
        return X_test, y_test
    
    def evaluate_model(self):
        """ Evaluate the trained model on the test set. """
        X_test, y_test = self.train_model()
        y_pred = self.model.predict(X_test)
        
        # Print accuracy and classification report
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        print(classification_report(y_test, y_pred))
    
    def analyze_word_frequency(self):
        """ Analyze the most common words in positive and negative reviews. """
        positive_reviews = self.data[self.data['sentiment'] == 'positive']['cleaned_reviews']
        negative_reviews = self.data[self.data['sentiment'] == 'negative']['cleaned_reviews']
        
        # Tokenize and count words
        positive_words = ' '.join(positive_reviews).split()
        negative_words = ' '.join(negative_reviews).split()
        
        positive_word_count = Counter(positive_words)
        negative_word_count = Counter(negative_words)
        
        print("Most common words in positive reviews:", positive_word_count.most_common(10))
        print("Most common words in negative reviews:", negative_word_count.most_common(10))

    def predict_new_review(self, review_text):
        """ Predict the sentiment of a new review. """
        cleaned_review = self.preprocess_text(review_text)
        review_tfidf = self.vectorizer.transform([cleaned_review])
        prediction = self.model.predict(review_tfidf)
        return 'Positive' if prediction == 1 else 'Negative'


