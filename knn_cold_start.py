import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# Helper functions


def get_interaction_counts(df):
    counts = df.groupby('user_id')['user_id'].count()
    return counts


def filter_users(counts, max_interactions):
    filtered_users = counts[counts['user_id'] < max_interactions].index
    return filtered_users


def filter_data(df, filtered_users):
    df_filtered = df[df['user_id'].isin(filtered_users)]
    return df_filtered


def encode_categorical_columns(df, cols):
    le = LabelEncoder()
    for col in cols:
        df[col] = le.fit_transform(df[col])
    return df


def compute_avg_rating(df):
    return df[['item_score1', 'item_score2', 'item_score3']].mean(axis=1)


def create_nn_model(input_shape):
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=input_shape))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer='adam', loss='mse')

    return model


def train_model(X, y, model):
    early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min')

    model.fit(X, y,
              epochs=20,
              batch_size=64,
              validation_split=0.2,
              callbacks=[early_stop])

    return model


def find_similar_items(item_id, df, num_neighbors=10):

    knn = NearestNeighbors(metric='cosine', n_neighbors=num_neighbors)

    knn.fit(df[['category_second', 'category_first',
                'item_score1', 'item_score2', 'item_score3']])

    distances, indices = knn.kneighbors(
        df.iloc[item_id, :].values.reshape(1, -1))

    similar_items = df.iloc[indices.squeeze()]['item_id'].tolist()

    return similar_items


def find_similar_users(user_id, df):

    user_ratings = df[df['user_id'] == user_id].set_index('item_id')[
        'avg_rating']

    similar_users = df[df['item_id'].isin(user_ratings.index) &
                       (df['user_id'] != user_id)].groupby('user_id')['item_id'].apply(set).apply(list)

    similar_users = similar_users.reset_index()

    # Compute similarity score
    similar_users['similarity'] = similar_users['item_id'].apply(lambda x: len(
        set(x) & set(user_ratings.index)) / len(set(x) | set(user_ratings.index)))

    # Filter and sort
    similar_users = similar_users[similar_users['similarity'] > 0].sort_values(
        'similarity', ascending=False)['user_id'].tolist()

    return similar_users


def get_recommendations(user_id, X, y, model, num_recs):

    user_ratings = X[X['user_id'] == user_id]['avg_rating']
    items_rated = X[X['user_id'] == user_id]['item_id'].unique()

    if len(user_ratings) < 5:

        # Use content based
        similar_items = find_similar_items(
            items_rated[0])  # Can pass any rated item id
        recs = similar_items[:num_recs]

    else:

        # Use collaborative filtering
        similar_users = find_similar_users(user_id)

        items_rated_by_similar_users = X[X['user_id'].isin(
            similar_users)]['item_id'].unique()

        train_data = X[(X['user_id'].isin(similar_users)) &
                       (X['item_id'].isin(items_rated_by_similar_users))]

        train_labels = y[train_data.index]

        # Retrain model on this subset
        model.fit(train_data, train_labels)

        # Get predictions
        unrated_items = np.setdiff1d(X['item_id'].unique(), items_rated)

        preds = model.predict(X[X['item_id'].isin(unrated_items)])

        items_to_recommend = X[X['item_id'].isin(
            unrated_items)]['item_id'].unique()

        preds_filtered = preds[X[X['item_id'].isin(unrated_items)].index]

        # Rank predictions
        recommended_items = pd.DataFrame(
            {'item_id': items_to_recommend, 'pred': preds_filtered})

        recommended_items = recommended_items.sort_values(
            'pred', ascending=False)

        recs = recommended_items.head(num_recs)['item_id'].tolist()

    return recs


def evaluate_recommendations(recommended_items, X_train, X_test):

    precision = []
    recall = []
    f1 = []

    for user_id in X_test['user_id'].unique():

        actual_items = X_test[X_test['user_id'] == user_id]['item_id'].values
        recommended = recommended_items[user_id]

        num_relevant = len(np.intersect1d(actual_items, recommended))

        if len(actual_items) > 0:
            precision.append(num_relevant / len(recommended))
            recall.append(num_relevant / len(actual_items))
            f1.append(2 * precision[-1] * recall[-1] /
                      (precision[-1] + recall[-1]))
        else:
            precision.append(0)
            recall.append(0)
            f1.append(0)

    precision = np.mean(precision)
    recall = np.mean(recall)
    f1 = np.mean(f1)

    return precision, recall, f1

# Recommender class


class Recommender:

    def __init__(self, data):
        self.data = data
        self.data_encoded = None
        self.model = None

    def preprocess(self):

        # Filter users
        interaction_counts = get_interaction_counts(self.data)
        filtered_users = filter_users(interaction_counts, 500)
        self.data = filter_data(self.data, filtered_users)

        # Encode categorical columns
        cat_cols = ['read', 'share', 'like', 'follow', 'favorite']
        self.data_encoded = encode_categorical_columns(self.data, cat_cols)

        # Compute avg rating
        self.data_encoded['avg_rating'] = compute_avg_rating(self.data_encoded)

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data_encoded,
                                                                                self.data_encoded['avg_rating'],
                                                                                test_size=0.2,
                                                                                random_state=42)

    def build_model(self):
        input_shape = (self.X_train.shape[1],)
        self.model = create_nn_model(input_shape)

    def train(self):
        self.model = train_model(self.X_train, self.y_train, self.model)

    def predict(self):
        self.y_pred = self.model.predict(self.X_test)

    def evaluate(self):
        self.mse = mean_squared_error(self.y_test, self.y_pred)
        self.precision = precision_score(
            self.y_test, self.y_pred, average='weighted')
        self.recall = recall_score(
            self.y_test, self.y_pred, average='weighted')

    def get_recommendations(self, user_id, num_recs):
        return get_recommendations(user_id, self.X_train, self.y_train, self.model, num_recs)

    def recommend_items(self):
        user_id = 12333
        print(self.get_recommendations(user_id, 10))
        # Recommend items
        recommended_items = {user_id: get_recommendations(
            user_id) for user_id in self.X_test['user_id'].unique()}

        # Evaluate recommendations
        precision, recall, f1 = evaluate_recommendations(
            recommended_items, self.X_train, self.X_test)

        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1: {f1:.4f}")

# Main workflow


def run_experiment_knn():

    data = pd.read_csv('Data/cold_data_0.3.csv')

    r = Recommender(data)
    r.preprocess()
    r.split_data()
    r.build_model()
    r.train()
    r.predict()
    r.evaluate()
    r.recommend_items()
