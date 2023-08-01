# Install the required libraries
# from google.colab import drive
from keras.models import load_model
from scipy.sparse import csr_matrix
from annoy import AnnoyIndex
from surprise.model_selection import train_test_split as surprise_train_test_split
from surprise import Dataset, Reader, SVD
from keras.optimizers import Adam
from keras.layers import LSTM, Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# !pip install scikit-surprise
# !pip install annoy
# Mount Google Drive to access the dataset
# drive.mount('/content/drive')


class RecommenderSystem_RNN:
    def __init__(self, dataset_path):
        self.data = pd.read_csv(dataset_path)

    def preprocess_data(self):
        # Load the dataset from Google Drive
        data = pd.read_csv(dataset_path)

        # Perform exploratory data analysis, statistical tests, data cleaning, normalization, etc.
        data.info()
        data.head()
        data.describe()
        data.isna().sum()
        data.duplicated().sum()
        data = data.drop_duplicates()
        plt.figure(figsize=(20, 10))
        data.hist()
        plt.tight_layout()
        plt.show()

        return data

    def train_svd_model(self):
        # Load the dataset for Surprise and split into train and test sets
        reader = Reader(rating_scale=(0, 1))
        surprise_data = Dataset.load_from_df(
            self.data[['user_id', 'item_id', 'click']], reader)
        trainset, testset = surprise_train_test_split(
            surprise_data, test_size=0.2, random_state=42)

        # Initialize and train the SVD model
        svd_model = SVD(n_factors=64, random_state=42)
        svd_model.fit(trainset)

        # Obtain user embeddings for the training data using the Latent Factor Model (SVD)
        trainset = svd_model.trainset
        user_ids_train = [trainset.to_raw_uid(
            user_inner_id) for user_inner_id in range(trainset.n_users)]
        user_embeddings_train = np.array(
            [svd_model.pu[trainset.to_inner_uid(user_id)] for user_id in user_ids_train])

        # Comment this part out if you need the values to be recalculated. To be noted, it takes a very long time.
        # # Filter users based on their availability in user_embeddings_train
        # filtered_user_ids_train = [user_id for user_id in self.data['user_id'].values if user_id in user_ids_train]
        # filtered_indices = [np.where(user_ids_train == user_id)[0][0] for user_id in filtered_user_ids_train]

        # # Save the filtered_user_ids_train and filtered_indices as numpy arrays
        # np.save('filtered_user_ids_train.npy',
        #         np.array(filtered_user_ids_train))
        # np.save('filtered_indices.npy', np.array(filtered_indices))

        # Load the saved numpy arrays
        filtered_user_ids_train = np.load(
            'Models/RNN/filtered_user_ids_train.npy')
        filtered_indices = np.load(
            'Models/RNN/filtered_indices.npy')

        return filtered_user_ids_train, filtered_indices, svd_model, testset

    def train_lstm_model(self):
        # Assuming train_data contains user features (gender, age, etc.)
        X_train_context = self.data[['gender', 'age', 'click_count', 'like_count', 'comment_count',
                                     'read_percentage', 'item_score1', 'item_score2', 'category_second', 'category_first', 'item_score3']].values

        # Get filtered X_train_context and user_embeddings_train
        X_train_context_filtered = X_train_context[filtered_indices]
        user_embeddings_train_filtered = user_embeddings_train[filtered_indices]

        # Define the Context Factor Model as an LSTM-based RNN
        context_factor_model = Sequential()
        context_factor_model.add(LSTM(64, activation='relu', input_shape=(
            X_train_context_filtered.shape[1], 1)))  # Input shape for LSTM
        # Output layer with the same size as user embeddings
        context_factor_model.add(Dense(64))

        # Compile the model
        context_factor_model.compile(
            loss='mse', optimizer=Adam(learning_rate=0.001))

        # Reshape the input data for LSTM (add a time step of 1)
        X_train_context_filtered = X_train_context_filtered.reshape(
            X_train_context_filtered.shape[0], X_train_context_filtered.shape[1], 1)

        # Train the model
        context_factor_model.fit(
            X_train_context_filtered, user_embeddings_train_filtered, epochs=10, batch_size=128)

        # Save the trained LSTM-based Context Factor Model using Keras
        context_factor_model.save('context_factor_model.h5')

        return X_train_context_filtered, user_embeddings_train_filtered

    def build_annoy_index(self, X_train_context_filtered, user_embeddings_train_filtered, filtered_user_ids_train):
        # Reshape X_train_context_filtered to remove the singleton dimension
        X_train_context_filtered = np.squeeze(X_train_context_filtered)

        # Concatenate them along axis=1
        latent_cross_embeddings_train = np.concatenate(
            (user_embeddings_train_filtered, X_train_context_filtered), axis=1)

        # Convert latent_cross_embeddings_train to a sparse matrix to reduce memory consumption
        latent_cross_embeddings_sparse = csr_matrix(
            latent_cross_embeddings_train)

        # Initialize Annoy index for approximate nearest neighbor search
        # Number of trees in the Annoy index (you can tune this parameter)
        num_trees = 100
        annoy_index = AnnoyIndex(
            latent_cross_embeddings_sparse.shape[1], 'angular')

        # Add items to the Annoy index
        for item_idx in range(latent_cross_embeddings_sparse.shape[0]):
            item_embedding = latent_cross_embeddings_sparse.getrow(
                item_idx).toarray().flatten()
            annoy_index.add_item(item_idx, item_embedding)

        # Build the index
        annoy_index.build(num_trees)

        # Number of similar items to retrieve
        N = 10

        # Find the N most similar items for a given item index (replace 'item_index' with the actual item index)
        item_index = 0  # Replace 'item_index' with the actual item index for which you want to find similar items
        similar_item_indices = annoy_index.get_nns_by_item(item_index, N)

        # Now, 'similar_item_indices' contains the indices of the N most similar items to the given item

        # Retrieve the actual item IDs from the filtered_user_ids_train list
        similar_item_ids = [filtered_user_ids_train[idx]
                            for idx in similar_item_indices]
        return similar_item_ids

    def evaluate_cold_start_users(self, testset, svd_model):
        # Assuming you have a DataFrame named 'testset_df' containing the testset interactions
        # with columns 'user_id', 'item_id', and 'rating'

        # Convert the testset to a DataFrame
        testset_df = pd.DataFrame(
            testset, columns=['user_id', 'item_id', 'rating'])

        # Group the testset interactions by 'user_id' and aggregate the 'item_id' into lists
        ground_truth_data_df = testset_df.groupby(
            'user_id')['item_id'].agg(list).reset_index()

        # Load the original testset as a DataFrame
        testset_df = pd.DataFrame(
            testset, columns=['user_id', 'item_id', 'true_rating'])

        # Get the list of unique user IDs in the testset
        testset_users = testset_df['user_id'].unique()

        # Get the list of unique user IDs in the training set (user_ids_train)
        trainset_users = np.array([trainset.to_raw_uid(
            user_inner_id) for user_inner_id in range(trainset.n_users)])

        # Find the user IDs that are only present in the testset (cold start users)
        cold_start_users = np.setdiff1d(testset_users, trainset_users)

        # Filter the testset to include only the cold start users
        cold_start_testset_df = testset_df[testset_df['user_id'].isin(
            cold_start_users)]

        # Convert the cold start testset to the Surprise Dataset
        cold_start_surprise_data = Dataset.load_from_df(
            cold_start_testset_df[['user_id', 'item_id', 'true_rating']], reader)

        # Get the predictions for the cold start users
        cold_start_predictions = svd_model.test(
            cold_start_surprise_data.build_full_trainset().build_testset())

        # Step 3: Sort the predictions by predicted ratings (est)
        cold_start_predictions.sort(key=lambda x: x.est, reverse=True)

        # Step 4: Recommend the top-N items with the highest predicted ratings
        N = 10  # Replace 'N' with the number of items you want to recommend
        top_N_item_ids = [int(prediction.iid)
                          for prediction in cold_start_predictions[:N]]

        # Create a dictionary to store the ground truth data for cold start users
        ground_truth_data_cold_start = dict(
            zip(ground_truth_data_df['user_id'], ground_truth_data_df['item_id']))
        return top_N_item_ids, ground_truth_data_cold_start

    def evaluate_recommendations(self, recommended_items, ground_truth_data):
        precision = []
        recall = []
        f1 = []
        ndcg = []
        num_users = len(ground_truth_data)

        for user_id in range(num_users):
            # Check if the user id exists in the ground truth data
            if user_id not in ground_truth_data:
                continue
            # Get the actual items rated by the user in the test set
            actual_items = ground_truth_data[user_id]
            # Get the recommended items for the user
            recommended = recommended_items
            # Compute the number of recommended items that are relevant (i.e., in the ground truth data)
            num_relevant = len(np.intersect1d(actual_items, recommended))
            # Compute precision, recall, and F1 score
            if len(recommended) > 0:
                precision.append(num_relevant / len(recommended))
            else:
                precision.append(0)

            if len(actual_items) > 0:
                recall.append(num_relevant / len(actual_items))
            else:
                recall.append(0)

            if precision[-1] + recall[-1] > 0:
                f1.append(2 * precision[-1] * recall[-1] /
                          (precision[-1] + recall[-1]))
            else:
                f1.append(0)

            # Compute NDCG
            idcg = np.sum(1 / np.log2(np.arange(2, len(actual_items) + 2)))
            dcg = np.sum(
                [(1 / np.log2(i + 2)) if item in actual_items else 0 for i, item in enumerate(recommended)])
            if idcg > 0:
                ndcg.append(dcg / idcg)
            else:
                ndcg.append(0)

        # Compute the average precision, recall, F1 score, and NDCG across all users
        precision_avg = np.mean(precision)
        recall_avg = np.mean(recall)
        f1_avg = np.mean(f1)
        ndcg_avg = np.mean(ndcg)

        return precision_avg, recall_avg, f1_avg, ndcg_avg

    def calculate_bias_metrics(self, typeData, data):
        bias_metrics = {}
        gender_groups = data.groupby(typeData)
        for gender in data[typeData].unique():
            data = gender_groups.get_group(gender)
            # Group the testset interactions by 'user_id' and aggregate the 'item_id' into lists
            ground_truth_data_df = data.groupby(
                'user_id')['item_id'].agg(list).reset_index()

            # Convert the cold start testset to the Surprise Dataset
            cold_start_surprise_data = Dataset.load_from_df(
                data[['user_id', 'item_id', 'click']], reader)

            # Get the predictions for the cold start users
            cold_start_predictions = svd_model.test(
                cold_start_surprise_data.build_full_trainset().build_testset())

            # Step 3: Sort the predictions by predicted ratings (est)
            cold_start_predictions.sort(key=lambda x: x.est, reverse=True)

            # Step 4: Recommend the top-N items with the highest predicted ratings
            N = 10  # Replace 'N' with the number of items you want to recommend
            top_N_item_ids = [int(prediction.iid)
                              for prediction in cold_start_predictions[:N]]

            # Create a dictionary to store the ground truth data for cold start users
            ground_truth_data_cold_start = dict(
                zip(ground_truth_data_df['user_id'], ground_truth_data_df['item_id']))

            precision_avg, recall_avg, f1_avg, ndcg_avg = self.evaluate_recommendations(
                top_N_item_ids, ground_truth_data_cold_start)

            bias_metrics[('all', gender)] = {
                'Precision': precision_avg,
                'Recall': recall_avg,
                'ndcg': ndcg_avg,
                'F1 Score': f1_avg
            }

            print(typeData, gender,
                  '\n Precision: ', precision_avg,
                  'Recall: ', recall_avg,
                  'ndcg:', ndcg_avg,
                  'F1 Score: ', f1_avg)

        return bias_metrics


def run_experiment_rnn():
    dataset_path = 'Data/cold_data_0.7.csv'
    recommender = RecommenderSystem_RNN(dataset_path)

    data = recommender.preprocess_data()
    filtered_user_ids_train, filtered_indices, svd_model, testset = recommender.train_svd_model()
    X_train_context_filtered, user_embeddings_train_filtered = recommender.train_lstm_model()
    similar_item_ids = recommender.build_annoy_index(
        X_train_context_filtered, user_embeddings_train_filtered, filtered_user_ids_train)
    top_N_item_ids, ground_truth_data_cold_start = recommender.evaluate_cold_start_users(
        testset, svd_model)
    precision_avg, recall_avg, f1_avg, ndcg_avg = recommender.evaluate_recommendations(
        top_N_item_ids, ground_truth_data_cold_start)
    bias_metrics_model_gender = recommender.calculate_bias_metrics(
        'gender', testset)
    bias_metrics_model_age = recommender.calculate_bias_metrics('age', testset)
