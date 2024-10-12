import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

def to_number(data):
    if data == 'True':
        return 1
    else:
        return 0

class ChurnPredictionModel:
    def __init__(self, data_path):
        """Initializes the model with data."""
        self.data = pd.read_csv(data_path)
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def feature_engineering(self):
        """Performs feature engineering on the dataset."""
        # Example: Calculate user-specific features like watch frequency and abandonment rate
        # Group by User_ID to get the user's behavior
        user_behavior = self.data.groupby('User_ID').agg(
            user_watch_freq=('Watched_to_End', 'count'),  # Total series watched
            user_abandon_rate=('Watched_to_End', lambda x: 1 - x.mean())  # Rate of not watching to the end
        ).reset_index()
        
        # Example: Calculate series-specific features like popularity (completion rate)
        series_behavior = self.data.groupby('Series_Title').agg(
            series_popularity=('Watched_to_End', 'mean')  # How often series are finished
        ).reset_index()
        
        # Merge user and series features back into the original data
        self.data = pd.merge(self.data, user_behavior, on='User_ID', how='left')
        self.data = pd.merge(self.data, series_behavior, on='Series_Title', how='left')
        
        # Encoding Age_Group as categorical if not already done
        self.data = pd.get_dummies(self.data, columns=['Age_Group'], drop_first=True)
        
    def prepare_data(self):
        """Prepares the data for model training."""
        # Feature selection (excluding 'User_ID', 'Series_Title', and 'Watched_to_End' as features)
        X = self.data.drop(columns=['User_ID', 'Series_Title', 'Watched_to_End'])
        y = self.data['Watched_to_End']
        
        # Split into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self):
        """Trains a Random Forest model."""
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        """Evaluates the trained model."""
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f'Accuracy: {accuracy}')
        print('Classification Report:')
        print(classification_report(self.y_test, y_pred))

    def feature_importance(self):
        """Shows the feature importance."""
        feature_importances = pd.Series(self.model.feature_importances_, index=self.X_train.columns).sort_values(ascending=False)
        print('Feature Importances:')
        print(feature_importances)