from resenias.agent import SentimentAnalyzer
from helpers.file_helper import get_filepath

# Example usage
if __name__ == '__main__':
    # Initialize the SentimentAnalyzer with a dataset
    analyzer = SentimentAnalyzer(data_path=get_filepath('resenias_peliculas.csv'))

    # Preprocess the reviews
    analyzer.preprocess_reviews()

    # Train and evaluate the model
    analyzer.evaluate_model()

    # Analyze word frequency in positive and negative reviews
    analyzer.analyze_word_frequency()

    # Analyze sentiment over time (requires a 'date' column in the dataset)
    #analyzer.sentiment_over_time()

    # Predict the sentiment of a new review
    new_review = "I love this product, it's amazing!"
    prediction = analyzer.predict_new_review(new_review)
    print(f"Sentiment of new review: {prediction}")