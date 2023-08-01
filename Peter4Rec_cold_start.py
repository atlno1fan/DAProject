# import libiraries
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch.utils.data as data_utils
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
import time
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_score, recall_score
import torch
from torch.nn.init import uniform_, xavier_normal_, constant_, normal_
from torch.utils.data import DistributedSampler, DataLoader


def analyze_data(all_data, all_data_70):
    all_data.head(7)  # get the first 7 rows
    all_data_70.head(7)  # get the first 7 rows
    all_data_70.shape  # shape
    all_data.shape  # shape
    all_data_70.duplicated().sum()  # check for the dublicates
    all_data.duplicated().sum()  # check for the dublicates
    all_data_70 = all_data_70.drop_duplicates()  # drop the duplicates
    all_data_70.duplicated().sum()
    all_data = all_data.drop_duplicates()  # drop the duplicates
    all_data.duplicated().sum()
    all_data_70.isna().sum()  # check for the nulls
    all_data.isna().sum()  # check for the nulls
    all_data_70.info()  # information
    all_data.info()  # information
    all_data_70.describe(include="all")  # describtion of the data
    all_data.describe(include="all")  # describtion of the data

    # plotting the heatmap for correlation
    ax = pyplot.subplots(figsize=(20, 10))
    ax = sns.heatmap(all_data_70.corr(), annot=True)

    # plotting the heatmap for correlation
    ax = pyplot.subplots(figsize=(20, 10))
    ax = sns.heatmap(all_data.select_dtypes(
        include=np.number).corr(), annot=True)
    # Create a list of consecutive integers
    s = all_data_70['gender']
    i = range(len(s))

    plt.figure(figsize=(16, 8))

    # Plot a scatterplot
    plt.scatter(i, s, c='red', alpha=0.5)
    plt.show()

    # removing the outliers from the target column

    all_data_70 = all_data_70[all_data_70['gender'] >= 1]

    # plotting the outlier against the constinous distn. in a scatter plot again to check
    # Create a list of consecutive integers
    s = all_data_70['gender']
    i = range(len(s))

    plt.figure(figsize=(16, 8))

    # Plot a scatterplot
    plt.scatter(i, s, c='red', alpha=0.5)
    plt.show()

    # Create a list of consecutive integers
    s = all_data['gender']
    i = range(len(s))

    plt.figure(figsize=(16, 8))

    # Plot a scatterplot
    plt.scatter(i, s, c='red', alpha=0.5)
    plt.show()

    # removing the outliers from the target column

    all_data = all_data[all_data['gender'] >= 1]

    # plotting the outlier against the constinous distn. in a scatter plot again to check
    # Create a list of consecutive integers
    s = all_data['gender']
    i = range(len(s))

    plt.figure(figsize=(16, 8))

    # Plot a scatterplot
    plt.scatter(i, s, c='red', alpha=0.5)
    plt.show()

    # Here, we just want to make sure that ther are 2 categories only (male, female from our assumption)
    all_data_70['gender'].unique()

    # Here, we just want to make sure that ther are 2 categories only (male, female from our assumption)
    all_data['gender'].unique()

    # Create a list of consecutive integers
    s = all_data_70['age']
    i = range(len(s))

    plt.figure(figsize=(16, 8))

    # Plot a scatterplot
    plt.scatter(i, s, c='red', alpha=0.5)
    plt.show()

    # removing the outliers from the target column

    all_data_70 = all_data_70[all_data_70['age'] >= 1]

    # plotting the outlier against the constinous distn. in a scatter plot again to check
    # Create a list of consecutive integers
    s = all_data_70['age']
    i = range(len(s))

    plt.figure(figsize=(16, 8))

    # Plot a scatterplot
    plt.scatter(i, s, c='red', alpha=0.5)
    plt.show()
    # Here, we just want to make sure that ther are 7 categories only (1->7 from our assumption)
    all_data_70['age'].unique()
    # Create a list of consecutive integers
    s = all_data['age']
    i = range(len(s))

    plt.figure(figsize=(16, 8))

    # Plot a scatterplot
    plt.scatter(i, s, c='red', alpha=0.5)
    plt.show()

    # removing the outliers from the target column

    all_data = all_data[all_data['age'] >= 1]

    # plotting the outlier against the constinous distn. in a scatter plot again to check
    # Create a list of consecutive integers
    s = all_data['age']
    i = range(len(s))

    plt.figure(figsize=(16, 8))

    # Plot a scatterplot
    plt.scatter(i, s, c='red', alpha=0.5)
    plt.show()

    # Here, we just want to make sure that ther are 7 categories only (1->7 from our assumption)
    all_data['age'].unique()
    aggregate_data(all_data, all_data_70)
    # all_data, all_data_70 = standerdize_data(all_data, all_data_70)

    # return all_data, all_data_70


def aggregate_data(all_data, all_data_70):
    all_data_70_agg = all_data_70.groupby('user_id').agg(
        {'click_count': 'sum', 'read_percentage': 'mean', 'read_time': 'sum'}).reset_index()
    all_data_70_agg
    all_data_70_useritem = all_data_70.groupby(
        'user_id').agg({'item_id': 'count'}).reset_index()
    all_data_70_useritem
    all_data_agg = all_data.groupby('user_id').agg(
        {'watching_times': 'sum'}).reset_index()
    all_data_agg
    all_data_useritem = all_data.groupby('user_id').agg(
        {'item_id': 'count'}).reset_index()
    all_data_useritem


def standerdize_data(all_data, all_data_70):
    all_data_70.info()
    all_data.info()
    all_data_70_drop_category = all_data_70.drop(
        ['gender', 'age', 'click', 'user_id', 'item_id', 'read', 'share', 'like', 'follow', 'favorite'], axis=1)  # 70%
    all_data_drop_category = all_data.drop(
        ['gender', 'age', 'follow', 'like', 'share', 'click', 'user_id', 'item_id', 'video_category'], axis=1)  # sbr
    all_data_70_category = all_data_70.drop(['click_count', 'like_count', 'comment_count', 'read_percentage',
                                            'item_score1', 'item_score2', 'category_second', 'category_first', 'item_score3', 'read_time'], axis=1)
    all_data_category = all_data.drop(['watching_times'], axis=1)
    # For the 705 cold start
    scaler = StandardScaler()
    standardized_data_07 = scaler.fit_transform(all_data_70_drop_category)
    print(standardized_data_07)
    # For the sbr dataset
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(all_data_drop_category)
    print(standardized_data)
    all_data_drop_category.select_dtypes
    standardized_data_07 = pd.DataFrame(
        standardized_data_07, columns=all_data_70_drop_category.columns)
    standardized_data_07
    standardized_data = pd.DataFrame(
        standardized_data, columns=all_data_drop_category.columns)
    standardized_data
    all_data_70_final_data = pd.concat(
        [standardized_data_07, all_data_70_category], axis=1)
    all_data_final_data = pd.concat(
        [standardized_data, all_data_category], axis=1)
    return all_data_final_data, all_data_70_final_data

# utils.py


def get_train_loader(dataset, is_parallel=False):
    """
    This cell defines the functions to create data loaders for your training, validation, and testing datasets.
    Data loaders are PyTorch's way to handle large datasets that can't fit into memory.
    They allow you to load data in small batches, rather than all at once.
    """
    if is_parallel:
        # Create a distributed data loader using PyTorch's DistributedSampler class
        dataloader = data_utils.DataLoader(
            dataset, batch_size=1024, sampler=DistributedSampler(dataset))
    else:
        # Create a regular data loader with a batch size of 1024 and shuffle the data
        dataloader = data_utils.DataLoader(
            dataset, batch_size=1024, shuffle=True, pin_memory=True)
    return dataloader


def get_val_loader(dataset, is_parallel=False):
    """
    This function creates a data loader for the validation dataset.

    Args:
        dataset: PyTorch dataset object containing the validation data.
        is_parallel: Boolean flag indicating whether to use distributed data loading.
                     Defaults to False.

    Returns:
        A PyTorch DataLoader object for the validation dataset.
    """
    if is_parallel:
        # Create a distributed data loader using PyTorch's DistributedSampler class
        dataloader = data_utils.DataLoader(
            dataset, batch_size=1024, sampler=DistributedSampler(dataset))
    else:
        # Create a regular data loader with a batch size of 1024 and do not shuffle the data
        dataloader = data_utils.DataLoader(
            dataset, batch_size=1024, shuffle=False, pin_memory=True)
    return dataloader


def get_test_loader(dataset, is_parallel=False):
    """
    This function creates a data loader for the test dataset.

    Args:
        dataset: PyTorch dataset object containing the test data.
        is_parallel: Boolean flag indicating whether to use distributed data loading.
                     Defaults to False.

    Returns:
        A PyTorch DataLoader object for the test dataset.
    """
    if is_parallel:
        # Create a distributed data loader using PyTorch's DistributedSampler class
        dataloader = data_utils.DataLoader(
            dataset, batch_size=1024, sampler=DistributedSampler(dataset))
    else:
        # Create a regular data loader with a batch size of 1024 and do not shuffle the data
        dataloader = data_utils.DataLoader(
            dataset, batch_size=1024, shuffle=False, pin_memory=True)
    return dataloader


class ColdResetDF:
    def __init__(self):
        self.item_enc1 = LabelEncoder()
        self.item_enc2 = LabelEncoder()
        self.user_enc = LabelEncoder()

    def fit_transform(self, df1, df2):
        df = pd.concat([df1['user_id'], df2['user_id']], ignore_index=True)
        df = self.user_enc.fit_transform(df) + 1
        df1['item_id'] = self.item_enc1.fit_transform(df1['item_id']) + 1
        df1['user_id'] = df[:len(df1)]
        df2['item_id'] = self.item_enc2.fit_transform(df2['item_id']) + 1
        df2['user_id'] = df[len(df1):]
        return df1, df2

    def inverse_transform(self, df):
        df['item_id'] = self.item_enc1.inverse_transform(df['item_id'] - 1)
        df['user_id'] = self.user_enc.inverse_transform(df['user_id'] - 1)
        return df


class ItemResetDF:
    def __init__(self):
        self.item_enc = LabelEncoder()

    def fit_transform(self, df):
        df['item_id'] = self.item_enc.fit_transform(df['item_id']) + 1
        return df

    def inverse_transform(self, df):
        df['item_id'] = self.item_enc.inverse_transform(df['item_id'] - 1)
        return df


def construct_data(task, item_min, all_data_70_final_data, all_data_final_data):

    if task != 2:
        df1 = all_data_70_final_data[['user_id',
                                      'item_id', 'age', 'gender', 'click']]
#         df1 = pd.read_csv(path1, usecols=['user_id', 'item_id', 'click'])
        df1 = df1[df1.click == 1]
    else:
        df1 = all_data_70_final_data[['user_id',
                                      'item_id', 'age', 'gender', 'like']]
        df1 = df1[df1.like == 1]

    df2 = all_data_final_data[['user_id', 'item_id', 'age', 'gender', 'click']]
    df2 = df2[df2.click == 1]

    user_counts = df2.groupby('user_id').size()
    user_subset = np.in1d(
        df2.user_id, user_counts[user_counts >= item_min].index)
    df2 = df2[user_subset].reset_index(drop=True)

    assert (df2.groupby('user_id').size() < item_min).sum() == 0
    s_item_count = len(set(df2['item_id']))

    reset_ob = ColdResetDF()
    df2, df1 = reset_ob.fit_transform(df2, df1)

    user1 = set(df1.user_id.values.tolist())
    user2 = set(df2.user_id.values.tolist())
    user = user1 & user2
    df1 = df1[df1.user_id.isin(list(user))]
    df2 = df2[df2.user_id.isin(list(user))]

    new_data1 = []
    new_data2 = []
    for u in tqdm(user):
        tmp_data2 = df2[df2.user_id == u][:-3].values.tolist()
        tmp_data1 = df1[df1.user_id == u].values.tolist()
        new_data1.extend(tmp_data1)
        new_data2.extend(tmp_data2)
    new_data1 = pd.DataFrame(new_data1, columns=df1.columns)
    new_data2 = pd.DataFrame(new_data2, columns=df2.columns)
    user_count = len(set(new_data1.user_id.values.tolist()))

    reset_item = ItemResetDF()
    new_data1 = reset_item.fit_transform(new_data1)

    t_item_count = len(set(new_data1['item_id']))
    print(new_data1.columns)
    return new_data1, new_data2, user_count, t_item_count, s_item_count

# utils.py


def colddataset(item_min, task, all_data_70_final_data, all_data_final_data):
    """This function constructs a cold start recommendation dataset from the data returned by the `construct_data` function.
    It creates a dictionary of user histories from the source dataset and a dictionary of target items from the target dataset.
    It then iterates over the target items and appends each one to its corresponding user's history,
    along with a special padding value of 0.
    It creates a new Pandas dataframe with the resulting source and target sequences and returns
    it along with some summary statistics.

    Args:
        item_min (int): The minimum number of items a user must have clicked on in the second dataset.
        task (int): The task to perform. If task is not 2, the function selects the 'click' column from the 'cold_data_0.3.csv' dataset. Otherwise, it selects the 'like' column from the same dataset.

    Returns:
        A tuple containing four elements:
        - A Pandas dataframe containing the source and target sequences for each example in the cold start recommendation dataset.
        - The number of selected users in the dataset.
        - The number of items in the original 'sbr_data_1M.csv' dataset.
        - The number of items in the resulting dataset for cold start recommendation.
    """
    target_data, source_data, user_count, t_item_count, s_item_count = construct_data(
        task, item_min, all_data_70_final_data, all_data_final_data)

    print("+++user_history+++")
    user_history = source_data.groupby('user_id').item_id.apply(list).to_dict()
    target = target_data.groupby('user_id').item_id.apply(list).to_dict()

    # Modified part to include age and gender
    age_gender_mapping = target_data.groupby('user_id').agg(
        {'age': 'first', 'gender': 'first'}).to_dict()

    examples = []
    for u, t_list in tqdm(target.items()):
        for t in t_list:
            # Modified part to include age and gender in examples
            age_val = age_gender_mapping['age'][u]
            gender_val = age_gender_mapping['gender'][u]
            # Add age and gender to the example list
            e_list = [user_history[u] + [0], t, age_val, gender_val]
            examples.append(e_list)
    examples = pd.DataFrame(
        examples, columns=['source', 'target', 'age', 'gender'])
    return examples, user_count, s_item_count, t_item_count

# utils.py


class ColdDataset(data_utils.Dataset):
    """This class defines a PyTorch dataset for the cold start recommendation task. It takes as input two sequences of item IDs, `x` and `y`,
    a maximum sequence length `max_len`, and a mask token `mask_token`.
    It implements the `__len__` and `__getitem__` methods required by the PyTorch dataset API to enable iteration over the dataset.


    Args:
        x (list): A list of sequences of item IDs.
        y (list): A list of target item IDs.
        max_len (int): The maximum sequence length.
        mask_token (int): The mask token ID.

    Returns:
        A tuple containing two PyTorch tensors:
        - A tensor of shape (max_len,) containing the input sequence with padding if necessary.
        - A tensor of shape (1,) containing the target item ID.
    """

    def __init__(self, x, y, max_len, mask_token):
        self.seqs = x
        self.targets = y
        self.max_len = max_len
        self.mask_token = mask_token

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        seq = self.seqs[index]
        target = self.targets[index]
        seq = seq[-self.max_len:]
        seq_len = len(seq)
        seq_mask_len = self.max_len - seq_len
        seq = [0] * seq_mask_len + seq
        return torch.LongTensor(seq), torch.LongTensor([target])

 # utils.py


class ColdEvalDataset(data_utils.Dataset):
    """This class defines a PyTorch dataset for evaluating the cold start recommendation task. It takes as input two sequences of item IDs, `x` and `y`, a maximum sequence length `max_len`, a mask token `mask_token`, and the number of items in the dataset `num_item`. It implements the `__len__` and `__getitem__` methods required by the PyTorch dataset API to enable iteration over the dataset.

    Args:
        x (list): A list of sequences of item IDs.
        y (list): A list of target item IDs.
        max_len (int): The maximum sequence length.
        mask_token (int): The mask token ID.
        num_item (int): The number of items in the dataset.

    Returns:
        A tuple containing two PyTorch tensors:
        - A tensor of shape (max_len,) containing the input sequence with padding if necessary.
        - A tensor of shape (num_item+1,) containing the one-hot encoded labels for the target item.
    """

    def __init__(self, x, y, max_len, mask_token, num_item):
        self.seqs = x
        self.targets = y
        self.max_len = max_len
        self.mask_token = mask_token
        self.num_item = num_item + 1

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        seq = self.seqs[index]
        target = self.targets[index]
        labels = [0] * self.num_item
        labels[target] = 1
        seq = seq[-self.max_len:]
        seq_len = len(seq)
        seq_mask_len = self.max_len - seq_len
        seq = [self.mask_token] * seq_mask_len + seq
        return torch.LongTensor(seq), torch.LongTensor(labels)


# main.py
def get_data(all_data_70_final_data, all_data_final_data, x_test_for_bias_later, y_test_for_bias_later, max_len=20, item_min=10, task=2, pad_token=0):
    """
    This function retrieves the data for the COLD model.

    Args:
        max_len: Maximum length of input sequences. Defaults to 20.
        item_min: Minimum number of times an item must appear in the dataset. Defaults to 10.
        task: Task number for the COLD dataset. Defaults to 2.
        pad_token: Token used for padding shorter sequences. Defaults to 0.

    Returns:
        train_dataloader: PyTorch DataLoader object for the training dataset.
        valid_dataloader: PyTorch DataLoader object for the validation dataset.
        test_dataloader: PyTorch DataLoader object for the test dataset.
        num_users: Number of users in the dataset.
        num_items: Number of items in the dataset.
        num_embeddings: Size of the vocabulary.
    """

    data, user_count, vocab_size, item_count = colddataset(
        item_min, task, all_data_70_final_data, all_data_final_data)

    # Split data into train, validation, and test sets
    x_train, x_test, y_train, y_test = train_test_split(data[['source', 'age', 'gender']],
                                                        data['target'],
                                                        test_size=0.2,
                                                        random_state=512)
    x_val, x_test, y_val, y_test = train_test_split(x_test,
                                                    y_test,
                                                    test_size=0.5,
                                                    random_state=512)

    x_test_for_bias_later.append(x_test)
    y_test_for_bias_later.append(y_test)

    # Convert the dataframes to arrays and then lists
    x_train = {'seq_column': x_train['source'].values.tolist(),
               'age_column': x_train['age'].values.tolist(),
               'gender_column': x_train['gender'].values.tolist()}
    y_train = y_train.values.tolist()

    x_val = {'seq_column': x_val['source'].values.tolist(),
             'age_column': x_val['age'].values.tolist(),
             'gender_column': x_val['gender'].values.tolist()}
    y_val = y_val.values.tolist()

    x_test = {'seq_column': x_test['source'].values.tolist(),
              'age_column': x_test['age'].values.tolist(),
              'gender_column': x_test['gender'].values.tolist()}
    y_test = y_test.values.tolist()

    train_dataset = ColdDataset(
        x_train['seq_column'], y_train, max_len, pad_token)
    valid_dataset = ColdEvalDataset(
        x_val['seq_column'], y_val, max_len, pad_token, item_count)
    test_dataset = ColdEvalDataset(
        x_test['seq_column'], y_test, max_len, pad_token, item_count)

    num_users = user_count
    num_items = item_count
    num_embeddings = vocab_size

    train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=1024, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    return train_dataloader, valid_dataloader, test_dataloader, num_users, num_items, num_embeddings

# trainer.py


def recalls_and_ndcgs_for_ks(scores, labels, ks):
    """
    This function computes recall and NDCG scores for a given set of predictions and labels.

    Args:
        scores: Tensor containing the predicted scores for each item.
        labels: Tensor containing the true labels for each item.
        ks: List of integers representing the cutoffs for recall and NDCG.

    Returns:
        metrics: Dictionary containing the recall and NDCG scores for each cutoff k.
    """
    metrics = {}
    # Compute the number of correct answers for each user and the total number of answers
    answer_count = labels.sum(1)
    answer_count_float = answer_count.float()
    labels_float = labels.float()
    # Sort the predicted scores and get the indices of the top k items for each user
    rank = (-scores).argsort(dim=1)
    cut = rank
    # Compute recall and NDCG scores for each cutoff k
    for k in sorted(ks, reverse=True):
        cut = cut[:, :k]
        hits = labels_float.gather(1, cut)
        metrics['Recall@%d' %
                k] = (hits.sum(1) / answer_count_float).mean().item()

        position = torch.arange(2, 2+k)
        weights = 1 / torch.log2(position.float()).to('cpu')
        dcg = (hits * weights).sum(1)
        idcg = torch.Tensor([weights[:min(n, k)].sum()
                            for n in answer_count]).to('cpu')
        ndcg = (dcg / idcg).mean()
        metrics['NDCG@%d' % k] = ndcg

    return metrics

# trainer.py


def Sequence_full_Validate(epoch, model, dataloader, writer, test=False):
    """
    This function performs a full validation pass on the given data using the given model.

    Args:
        epoch: Current epoch number.
        model: PyTorch model to use for validation.
        dataloader: PyTorch DataLoader object for the validation data.
        writer: SummaryWriter object for logging.
        test: Whether to perform testing instead of validation. Defaults to False.

    Returns:
        avg_metrics: Dictionary containing the average recall and NDCG scores for the validation data.
    """
    print("+" * 20, "Valid Epoch {}".format(epoch + 1), "+" * 20)
    # Set the model to evaluation mode
    model.eval()
    # Initialize a dictionary for storing the average recall and NDCG scores
    avg_metrics = {}
    # Initialize a counter for the number of batches processed
    i = 0
    # Iterate over the validation data
    with torch.no_grad():
        tqdm_dataloader = dataloader
        # Move the data to the CPU
        for data in tqdm_dataloader:
            data = [x.to('cpu') for x in data]
            seqs, labels = data
            # Forward pass through the model to get predicted scores
            if test:
                scores = model.predict(seqs)
            else:
                scores = model(seqs)
            scores = scores.mean(1)
            # Compute recall and NDCG scores for the predicted scores and true labels
            metrics = recalls_and_ndcgs_for_ks(scores, labels, [5, 20])
            # Update the average metrics dictionary and the batch counter
            i += 1
            for key, value in metrics.items():
                if key not in avg_metrics:
                    avg_metrics[key] = value
                else:
                    avg_metrics[key] += value
     # Compute the average recall and NDCG score
    for key, value in avg_metrics.items():
        avg_metrics[key] = value / i
        # Log the NDCG scores to TensorBoard
    print(avg_metrics)
    for k in sorted([5, 20], reverse=True):
        writer.add_scalar('Train/NDCG@{}'.format(k),
                          avg_metrics['NDCG@%d' % k], epoch)
#          Return the average recall and NDCG scores
    return avg_metrics

# trainer.py


def SequenceTrainer(epoch, model, dataloader, optimizer, writer):  # schedular,
    """
    This function trains the given model on the given data using the given optimizer.

    Args:
        epoch: Current epoch number.
        model: PyTorch model to train.
        dataloader: PyTorch DataLoader object for the training data.
        optimizer: PyTorch optimizer to use for training.
        writer: SummaryWriter object for logging.

    Returns:
        optimizer: PyTorch optimizer after training.
    """
    print("+" * 20, "Train Epoch {}".format(epoch + 1), "+" * 20)
    # Set the model to training mode
    model.train()
    # Initialize a variable for storing the running loss
    running_loss = 0
    # Use cross-entropy loss as the loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    # Iterate over the training data
    for data in dataloader:
        # Zero out the gradients
        optimizer.zero_grad()
        data = [x.to('cpu') for x in data]
        seqs, labels = data
        # Forward pass through the model to get logits and compute the loss
        logits = model(seqs)  # B x T x V
        logits = logits.mean(1)
        labels = labels.view(-1)
        # Backward pass and optimization step
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.detach().cpu().item()
        # Log the average loss to TensorBoard and print it to the console
    writer.add_scalar('Train/loss', running_loss / len(dataloader), epoch)
    print("Training CE Loss: {:.5f}".format(running_loss / len(dataloader)))
    # Return the optimizer after training
    return optimizer

# trainer.py


def SeqTrain(epochs, model, train_loader, val_loader, writer, is_parallel, is_pretrain, lr, weight_decay, local_rank):
    """
    This function trains the given model on the given training data and validates it on the given validation data.

    Args:
        epochs: Number of training epochs.
        model: PyTorch model to train.
        train_loader: PyTorch DataLoader object for the training data.
        val_loader: PyTorch DataLoader object for the validation data.
        writer: SummaryWriter object for logging.
        is_parallel: Whether to use data parallelism for training. Defaults to False.
        is_pretrain: Whether to use pre-trained model. Defaults to 0.
        lr: Learning rate for optimizer.
        weight_decay: Weight decay for optimizer.
        local_rank: Local rank for distributed training.

    Returns:
        best_model: PyTorch model with the best performance on the validation data.
    """
    # Initialize the optimizer
    if is_pretrain == 0:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay)

    model = model.to('cpu')
    if is_parallel:
        model = torch.nn.parallel.DistributedDataParallel(
            model,  find_unused_parameters=True, device_ids=[local_rank], output_device=local_rank)
    best_metric = 0
    all_time = 0
    val_all_time = 0
    for epoch in range(epochs):
        since = time.time()
        optimizer = SequenceTrainer(
            epoch, model, train_loader, optimizer, writer)
        tmp = time.time() - since
        print('one epoch train:', tmp)
        all_time += tmp
        val_since = time.time()
        metrics = Sequence_full_Validate(epoch, model, val_loader, writer)
        # Validate the model and measure the time
        val_tmp = time.time() - val_since
        print('one epoch val:', val_tmp)
        val_all_time += val_tmp
        # Update the best model if necessary
        i = 1
        current_metric = metrics['NDCG@5']
        if best_metric <= current_metric:
            best_metric = current_metric
            best_model = deepcopy(model)
            state_dict = model.state_dict()
        else:
            i += 1
            if i == 10:
                print('early stop!')
                break
    # Print the total training and validation times and return the best model
    print('train_time:', all_time)
    print('val_time:', val_all_time)
    return best_model


# -*- coding: utf-8 -*-
'''
Reference:
    [1]Fajie Yuan et al. Parameter-efficient transfer from sequential behaviors for user modeling and recommendation. In SIGIR, pages 1469–1478, 2020.
'''


class Peter4Coldstart(nn.Module):

    def __init__(self, embedding_size, block_num, dilations, kernel_size, num_items, num_embedding, is_mp, pad_token):
        super(Peter4Coldstart, self).__init__()

        # load parameters info
        self.embedding_size = embedding_size
        self.residual_channels = embedding_size
        self.block_num = block_num
        self.dilations = dilations * self.block_num
        self.kernel_size = kernel_size
        self.output_dim = num_items
        self.vocab_size = num_embedding
        self.is_mp = is_mp

        self.pad_token = pad_token

        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.vocab_size+1, self.embedding_size, padding_idx=self.pad_token)

        # residual blocks
        rb = [
            ResidualBlock_b_2mp_parallel(
                self.residual_channels, self.residual_channels, kernel_size=self.kernel_size, dilation=dilation, is_mp=self.is_mp
            ) for dilation in self.dilations
        ]
        self.residual_blocks = nn.Sequential(*rb)

        # fully-connected layer
        self.final_layer = nn.Linear(self.residual_channels, self.output_dim+1)

        # parameters initialization
        # self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            stdv = np.sqrt(1. / (self.output_dim+1))
            uniform_(module.weight.data, -stdv, stdv)
        elif isinstance(module, nn.Linear):
            # xavier_normal_(module.weight.data)
            normal_(module.weight.data, 0.0, 0.1)
            if module.bias is not None:
                constant_(module.bias.data, 0.1)

    def forward(self, item_seq):  # , pos, neg
        # [batch_size, seq_len, embed_size]
        item_seq_emb = self.item_embedding(item_seq)
        # Residual locks
        dilate_outputs = self.residual_blocks(item_seq_emb)
        # [batch_size, embedding_size]hidden
        seq_output = self.final_layer(dilate_outputs)

        return seq_output

    def predict(self, item_seq, item):
        # [batch_size, seq_len, embed_size]
        item_seq_emb = self.item_embedding(item_seq)
        dilate_outputs = self.residual_blocks(item_seq_emb)
        item_embs = self.item_embedding(item)
        logits = dilate_outputs.matmul(item_embs.transpose(1, 2))
        logits = logits.mean(1)
        return logits


class mp(nn.Module):
    #      it implements a simple max pooling operation on the output of a 1D CNN.
    def __init__(self, channel):
        super(mp, self).__init__()
        self.hidden_size = int(channel / 4)
        self.conv1 = nn.Conv1d(channel, self.hidden_size, 1)
        self.conv2 = nn.Conv1d(self.hidden_size, channel, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        return x


class ResidualBlock_a(nn.Module):
    r"""
    Residual block (a) in the paper
    """

    def __init__(self, in_channel, out_channel, kernel_size=3, dilation=None):
        super(ResidualBlock_a, self).__init__()

        half_channel = out_channel // 2
        self.ln1 = nn.LayerNorm(out_channel, eps=1e-8)
        self.conv1 = nn.Conv2d(in_channel, half_channel,
                               kernel_size=(1, 1), padding=0)

        self.ln2 = nn.LayerNorm(half_channel, eps=1e-8)
        self.conv2 = nn.Conv2d(half_channel, half_channel, kernel_size=(
            1, kernel_size), padding=0, dilation=dilation)

        self.ln3 = nn.LayerNorm(half_channel, eps=1e-8)
        self.conv3 = nn.Conv2d(half_channel, out_channel,
                               kernel_size=(1, 1), padding=0)

        self.dilation = dilation
        self.kernel_size = kernel_size

    def forward(self, x):  # x: [batch_size, seq_len, embed_size], move forward

        out = F.relu(self.ln1(x))
        out = out.permute(0, 2, 1).unsqueeze(2)
        out = self.conv1(out).squeeze(2).permute(0, 2, 1)

        out2 = F.relu(self.ln2(out))
        out2 = self.conv_pad(out2, self.dilation)
        out2 = self.conv2(out2).squeeze(2).permute(0, 2, 1)

        out3 = F.relu(self.ln3(out2))
        out3 = out3.permute(0, 2, 1).unsqueeze(2)
        out3 = self.conv3(out3).squeeze(2).permute(0, 2, 1)
        return out3 + x

    def conv_pad(self, x, dilation):  # x: [batch_size, seq_len, embed_size]
        r""" Dropout-mask: To avoid the future information leakage problem, this paper proposed a masking-based dropout
        trick for the 1D dilated convolution to prevent the network from seeing the future items.
        Also the One-dimensional transformation is completed in this function.
        """
        inputs_pad = x.permute(0, 2, 1)  # [batch_size, embed_size, seq_len]
        # [batch_size, embed_size, 1, seq_len]
        inputs_pad = inputs_pad.unsqueeze(2)
        pad = nn.ZeroPad2d(((self.kernel_size - 1) * dilation, 0, 0, 0))
        # padding operation  args：(left,right,top,bottom)
        # [batch_size, embed_size, 1, seq_len+(self.kernel_size-1)*dilations]
        inputs_pad = pad(inputs_pad)
        return inputs_pad


class ResidualBlock_b_2mp_parallel(nn.Module):
    r"""
    Residual block (b) in the paper
    """

    def __init__(self, in_channel, out_channel, kernel_size=3, dilation=None, is_mp=False):
        super(ResidualBlock_b_2mp_parallel, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(
            1, kernel_size), padding=0, dilation=dilation)
        self.ln1 = nn.LayerNorm(out_channel, eps=1e-8)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(
            1, kernel_size), padding=0, dilation=dilation * 2)
        self.ln2 = nn.LayerNorm(out_channel, eps=1e-8)

        self.dilation = dilation
        self.kernel_size = kernel_size
        self.is_mp = is_mp
        self.rez = nn.Parameter(torch.FloatTensor([1]))
        if self.is_mp:
            self.mp1 = mp(in_channel)
            self.mp2 = mp(in_channel)

    def forward(self, x):  # x: [batch_size, seq_len, embed_size]
        # [batch_size, embed_size, 1, seq_len+(self.kernel_size-1)*dilations]
        x_pad = self.conv_pad(x, self.dilation)
        out = self.conv1(x_pad).squeeze(2).permute(0, 2, 1)
        # [batch_size, seq_len+(self.kernel_size-1)*dilations-kernel_size+1, embed_size]
        if self.is_mp:
            mp_out = self.mp1(x)
            out = mp_out + out
        out = F.relu(self.ln1(out))
        out_pad = self.conv_pad(out, self.dilation * 2)
        out2 = self.conv2(out_pad).squeeze(2).permute(0, 2, 1)
        if self.is_mp:
            mp_out2 = self.mp2(out)
            out2 = mp_out2 + out2
        out2 = F.relu(self.ln2(out2))
        return out2 * self.rez + x

    def conv_pad(self, x, dilation):
        r""" Dropout-mask: To avoid the future information leakage problem, this paper proposed a masking-based dropout
        trick for the 1D dilated convolution to prevent the network from seeing the future items.
        Also the One-dimensional transformation is completed in this function.
        """
        inputs_pad = x.permute(0, 2, 1)
        inputs_pad = inputs_pad.unsqueeze(2)
        pad = nn.ZeroPad2d(((self.kernel_size - 1) * dilation, 0, 0, 0))
        inputs_pad = pad(inputs_pad)
        return inputs_pad


class ResidualBlock_b_2mp_serial(nn.Module):
    r"""
    Residual block (b) in the paper

    in the residual block, the convolutional layer applies a set of filters to the input sequence to capture local dependencies
    between adjacent items.
    The dilations parameter controls the spacing between the elements of the filter,
    so that the filter can capture dependencies over longer distances in the sequence.
    By setting different dilation values, the model can capture dependencies at different scales.
    is_mp parameter controls whether or not max pooling is applied after the convolutional layer
    """

    def __init__(self, in_channel, out_channel, kernel_size=3, dilation=None, is_mp=False):
        super(ResidualBlock_b_2mp_serial, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(
            1, kernel_size), padding=0, dilation=dilation)
        self.ln1 = nn.LayerNorm(out_channel, eps=1e-8)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(
            1, kernel_size), padding=0, dilation=dilation * 2)
        self.ln2 = nn.LayerNorm(out_channel, eps=1e-8)
        # self.mp = mp(in_channel)
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.is_mp = is_mp
        if self.is_mp:
            self.mp1 = mp(in_channel)
            self.mp2 = mp(in_channel)

    def forward(self, x):  # x: [batch_size, seq_len, embed_size]
        # [batch_size, embed_size, 1, seq_len+(self.kernel_size-1)*dilations]
        x_pad = self.conv_pad(x, self.dilation)
        out = self.conv1(x_pad).squeeze(2).permute(0, 2, 1)
        # [batch_size, seq_len+(self.kernel_size-1)*dilations-kernel_size+1, embed_size]
        if self.is_mp:
            mp_out = self.mp1(x)
            out = mp_out
        out = F.relu(self.ln1(out))
        out_pad = self.conv_pad(out, self.dilation * 2)
        out2 = self.conv2(out_pad).squeeze(2).permute(0, 2, 1)
        if self.is_mp:
            mp_out2 = self.mp2(out)
            out2 = mp_out2
        out2 = F.relu(self.ln2(out2))
        return out2 + x

    def conv_pad(self, x, dilation):
        r""" Dropout-mask: To avoid the future information leakage problem, this paper proposed a masking-based dropout
        trick for the 1D dilated convolution to prevent the network from seeing the future items.
        Also the One-dimensional transformation is completed in this function.
        """
        inputs_pad = x.permute(0, 2, 1)
        inputs_pad = inputs_pad.unsqueeze(2)
        pad = nn.ZeroPad2d(((self.kernel_size - 1) * dilation, 0, 0, 0))
        inputs_pad = pad(inputs_pad)
        return inputs_pad


class ResidualBlock_b_mp_serial(nn.Module):
    r"""
    Residual block (b) in the paper
    """

    def __init__(self, in_channel, out_channel, kernel_size=3, dilation=None, is_mp=False):
        super(ResidualBlock_b_mp_serial, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(
            1, kernel_size), padding=0, dilation=dilation)
        self.ln1 = nn.LayerNorm(out_channel, eps=1e-8)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(
            1, kernel_size), padding=0, dilation=dilation * 2)
        self.ln2 = nn.LayerNorm(out_channel, eps=1e-8)
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.is_mp = is_mp
        if self.is_mp:
            self.mp = mp(in_channel)

    def forward(self, x):  # x: [batch_size, seq_len, embed_size]
        # [batch_size, embed_size, 1, seq_len+(self.kernel_size-1)*dilations]
        x_pad = self.conv_pad(x, self.dilation)
        out = self.conv1(x_pad).squeeze(2).permute(0, 2, 1)
        # [batch_size, seq_len+(self.kernel_size-1)*dilations-kernel_size+1, embed_size]
        out = F.relu(self.ln1(out))
        out_pad = self.conv_pad(out, self.dilation * 2)
        out2 = self.conv2(out_pad).squeeze(2).permute(0, 2, 1)
        out2 = F.relu(self.ln2(out2))
        if self.is_mp:
            mp_out2 = self.mp(out)
            out2 = mp_out2
        return out2 + x

    def conv_pad(self, x, dilation):
        r""" Dropout-mask: To avoid the future information leakage problem, this paper proposed a masking-based dropout
        trick for the 1D dilated convolution to prevent the network from seeing the future items.
        Also the One-dimensional transformation is completed in this function.
        """
        inputs_pad = x.permute(0, 2, 1)
        inputs_pad = inputs_pad.unsqueeze(2)
        pad = nn.ZeroPad2d(((self.kernel_size - 1) * dilation, 0, 0, 0))
        inputs_pad = pad(inputs_pad)
        return inputs_pad


def calculate_accuracy(predictions, targets):
    return accuracy_score(targets, predictions)


def calculate_confusion_matrix(predictions, targets):
    return confusion_matrix(targets, predictions)


def calculate_f1_score(predictions, targets):
    return f1_score(targets, predictions, average='macro')


def calculate_recall(predictions, targets):
    return recall_score(targets, predictions, average='macro')


def calculate_precision(predictions, targets):
    return precision_score(targets, predictions, average='macro')


def calculate_bias_metrics(predictions, targets, subgroup_attribute):
    metrics = {}

    unique_subgroups = np.unique(subgroup_attribute)
    for subgroup in unique_subgroups:
        mask = (subgroup_attribute == subgroup)
        subgroup_predictions = predictions[mask]
        subgroup_targets = targets[mask]

        # Calculate metrics for the subgroup
        subgroup_accuracy = calculate_accuracy(
            subgroup_predictions, subgroup_targets)
        subgroup_confusion_matrix = calculate_confusion_matrix(
            subgroup_predictions, subgroup_targets)
        subgroup_f1_score = calculate_f1_score(
            subgroup_predictions, subgroup_targets)
        subgroup_recall = calculate_recall(
            subgroup_predictions, subgroup_targets)
        subgroup_precision = calculate_precision(
            subgroup_predictions, subgroup_targets)

        # Store the metrics in the dictionary
        metrics[subgroup] = {
            'accuracy': subgroup_accuracy,
            'confusion_matrix': subgroup_confusion_matrix,
            'f1_score': subgroup_f1_score,
            'recall': subgroup_recall,
            'precision': subgroup_precision,
        }

    return metrics


def calculate_odds_ratio(positive_predictions, total_predictions):
    return (positive_predictions[0] * total_predictions[1]) / (positive_predictions[1] * total_predictions[0])


def calculate_log_odds_ratio(positive_predictions, total_predictions):
    odds_ratio = calculate_odds_ratio(positive_predictions, total_predictions)
    log_odds_ratio = np.log(odds_ratio)
    return log_odds_ratio


def measure_bias_with_odds_and_log_odds_ratios(bias_metrics):
    odds_and_log_odds_ratios = {}
    total_positives = sum(m['confusion_matrix'][1][1] for m in bias_metrics.values(
    ) if len(m['confusion_matrix']) > 1 and len(m['confusion_matrix'][0]) > 1)
    total_negatives = sum(m['confusion_matrix'][0][0] for m in bias_metrics.values(
    ) if len(m['confusion_matrix']) > 0 and len(m['confusion_matrix'][0]) > 0)
    total_total_positives = sum(sum(m['confusion_matrix'][1])
                                for m in bias_metrics.values() if len(m['confusion_matrix']) > 1)
    total_total_negatives = sum(sum(m['confusion_matrix'][0])
                                for m in bias_metrics.values() if len(m['confusion_matrix']) > 0)

    for subgroup, metrics in bias_metrics.items():
        confusion_matrix = metrics['confusion_matrix']
        if len(confusion_matrix) == 1 and len(confusion_matrix[0]) == 1:
            # Handle the case when the confusion matrix is [[1]]
            subgroup_positives = confusion_matrix[0][0]
            subgroup_negatives = 0
            subgroup_total_positives = subgroup_positives
            subgroup_total_negatives = 0
        else:
            subgroup_positives = confusion_matrix[1][1] if len(
                confusion_matrix) > 1 and len(confusion_matrix[0]) > 1 else 0
            subgroup_negatives = confusion_matrix[0][0] if len(
                confusion_matrix) > 0 and len(confusion_matrix[0]) > 0 else 0
            subgroup_total_positives = sum(confusion_matrix[1]) if len(
                confusion_matrix) > 1 else subgroup_positives
            subgroup_total_negatives = sum(confusion_matrix[0]) if len(
                confusion_matrix) > 0 else subgroup_negatives

        positive_odds = (subgroup_positives, total_positives)
        negative_odds = (subgroup_negatives, total_negatives)
        total_positive_odds = (subgroup_total_positives, total_total_positives)
        total_negative_odds = (subgroup_total_negatives, total_total_negatives)

        subgroup_odds_ratio = calculate_odds_ratio(
            positive_odds, total_positive_odds)
        overall_odds_ratio = calculate_odds_ratio(
            negative_odds, total_negative_odds)

        positive_odds = (subgroup_positives, total_positives)
        negative_odds = (subgroup_negatives, total_negatives)
        total_positive_odds = (subgroup_total_positives, total_total_positives)
        total_negative_odds = (subgroup_total_negatives, total_total_negatives)

        subgroup_log_odds_ratio = calculate_log_odds_ratio(
            positive_odds, total_positive_odds)
        overall_log_odds_ratio = calculate_log_odds_ratio(
            negative_odds, total_negative_odds)

        odds_ratio = calculate_odds_ratio(positive_odds, total_positive_odds)
        log_odds_ratio = calculate_log_odds_ratio(
            positive_odds, total_positive_odds)

        odds_and_log_odds_ratios[subgroup] = {
            'subgroup_odds_ratio': subgroup_odds_ratio,
            'overall_odds_ratio': overall_odds_ratio,
            'subgroup_log_odds_ratio': subgroup_log_odds_ratio,
            'overall_log_odds_ratio': overall_log_odds_ratio,
        }

    return odds_and_log_odds_ratios


def run_experiment_peter4rec():
    all_data = pd.read_csv('Data/sbr_data_1M.csv')  # read the data
    all_data_70 = pd.read_csv('Data/cold_data_0.7.csv')  # read the data

    all_data, all_data_70 = analyze_data(all_data, all_data_70)

    x_test_for_bias_later = []
    y_test_for_bias_later = []

    all_data_70_final_data, all_data_final_data = standerdize_data(
        all_data, all_data_70)

    train_dataloader, valid_dataloader, test_dataloader, num_users, num_items, num_embeddings = get_data(
        x_test_for_bias_later, y_test_for_bias_later, all_data_70_final_data, all_data_final_data)
    num_items, max_len, block_num, num_heads, num_embedding, hidden_size, is_mp, dropout, embedding_size, dilations, kernel_size, pad_token = num_items, 40, 2, 4, num_embeddings, 128, False, 0.3, 128, [
        1, 4], 4, 0
    model = Peter4Coldstart(embedding_size, block_num, dilations,
                            kernel_size, num_items, num_embedding, is_mp, pad_token)
    writer = SummaryWriter()
    Model_1 = SeqTrain(10, model, train_dataloader, valid_dataloader, writer,
                       is_pretrain=1, is_parallel=False, lr=0.001, weight_decay=0.001, local_rank=1)

    train_dataloader, valid_dataloader, test_dataloader, num_users, num_items, num_embeddings = get_data(
        x_test_for_bias_later, y_test_for_bias_later, all_data_70_final_data, all_data_final_data)
    num_items, max_len, block_num, num_heads, num_embedding, hidden_size, is_mp, dropout, embedding_size, dilations, kernel_size, pad_token = num_items, 40, 2, 4, num_embeddings, 128, False, 0.3, 128, [
        1, 4], 4, 0
    model = Peter4Coldstart(embedding_size, block_num, dilations,
                            kernel_size, num_items, num_embedding, is_mp, pad_token)
    writer = SummaryWriter()
    Model_2 = SeqTrain(10, model, train_dataloader, valid_dataloader, writer,
                       is_pretrain=1, is_parallel=False, lr=0.01, weight_decay=0.01, local_rank=1)

    train_dataloader, valid_dataloader, test_dataloader, num_users, num_items, num_embeddings = get_data(
        x_test_for_bias_later, y_test_for_bias_later, all_data_70_final_data, all_data_final_data)
    num_items, max_len, block_num, num_heads, num_embedding, hidden_size, is_mp, dropout, embedding_size, dilations, kernel_size, pad_token = num_items, 40, 3, 4, num_embeddings, 128, False, 0.3, 128, [
        1, 4], 4, 0
    model = Peter4Coldstart(embedding_size, block_num, dilations,
                            kernel_size, num_items, num_embedding, is_mp, pad_token)
    writer = SummaryWriter()
    Model_3 = SeqTrain(10, model, train_dataloader, valid_dataloader, writer,
                       is_pretrain=1, is_parallel=False, lr=0.001, weight_decay=0.01, local_rank=1)

    train_dataloader, valid_dataloader, test_dataloader, num_users, num_items, num_embeddings = get_data(
        x_test_for_bias_later, y_test_for_bias_later, all_data_70_final_data, all_data_final_data)
    model = Peter4Coldstart(embedding_size, block_num, dilations,
                            kernel_size, num_items, num_embedding, is_mp, pad_token)
    writer = SummaryWriter()
    Model_4 = SeqTrain(10, model, train_dataloader, valid_dataloader, writer,
                       is_pretrain=1, is_parallel=False, lr=0.001, weight_decay=0.01, local_rank=1)

    # Evaluate the models on the test dataset to get the predictions, the test data for each model will be similar to the one used
    # during the evaluation above, so it will be each models target data. The only reason is so that we have access to the predicted
    # labels.
    Model_1.eval()

    with torch.no_grad():
        all_predictions_model_1 = []  # Empty list to hold all predictions
        for data in tqdm(test_dataloader):
            data = [x.to('cpu') for x in data]
            seqs, labels = data
            test_predictions_model_1 = Model_1.predict(seqs, labels)
            all_predictions_model_1.append(test_predictions_model_1)

    # Concatenate all predictions into a single tensor
    all_predictions_model_1 = torch.cat(all_predictions_model_1, dim=0)

    # Convert the predictions to class labels (assuming multi-class classification)
    all_test_predictions_labels_model_1 = torch.argmax(
        all_predictions_model_1, dim=1).numpy()

    # Calculate bias metrics for age and gender subgroups
    age_bias_metrics_model_1 = calculate_bias_metrics(
        all_test_predictions_labels_model_1, y_test_for_bias_later[0], x_test_for_bias_later[0]['age'])
    gender_bias_metrics_model_1 = calculate_bias_metrics(
        all_test_predictions_labels_model_1, y_test_for_bias_later[0], x_test_for_bias_later[0]['gender'])

    print("Bias Metrics for Age Subgroups in Model :")
    print(age_bias_metrics_model_1)

    print("Bias Metrics for Gender Subgroups in Model:")
    print(gender_bias_metrics_model_1)

    # Measure bias using log odds ratios
    age_ratios_1 = measure_bias_with_odds_and_log_odds_ratios(
        age_bias_metrics_model_1)
    gender_ratios_1 = measure_bias_with_odds_and_log_odds_ratios(
        gender_bias_metrics_model_1)

    print("Ratios for Age Subgroups in Model :")
    print(age_ratios_1)

    print("Ratios for Gender Subgroups in Model :")
    print(gender_ratios_1)

    Model_2.eval()
    with torch.no_grad():
        all_predictions_model_2 = []  # Empty list to hold all predictions
        for data in tqdm(test_dataloader):
            data = [x.to('cpu') for x in data]
            seqs, labels = data
            test_predictions_model_2 = Model_2.predict(seqs, labels)
            all_predictions_model_2.append(test_predictions_model_2)

    # Concatenate all predictions into a single tensor
    all_predictions_model_2 = torch.cat(all_predictions_model_2, dim=0)

    # Convert the predictions to class labels (assuming multi-class classification)
    all_test_predictions_labels_model_2 = torch.argmax(
        all_predictions_model_2, dim=1).numpy()

    # Calculate bias metrics for age and gender subgroups
    age_bias_metrics_model_2 = calculate_bias_metrics(
        all_test_predictions_labels_model_2, y_test_for_bias_later[1], x_test_for_bias_later[1]['age'])
    gender_bias_metrics_model_2 = calculate_bias_metrics(
        all_test_predictions_labels_model_2, y_test_for_bias_later[1], x_test_for_bias_later[1]['gender'])

    print("Bias Metrics for Age Subgroups in Model :")
    print(age_bias_metrics_model_2)

    print("Bias Metrics for Gender Subgroups in Model :")
    print(gender_bias_metrics_model_2)

    # Measure bias using log odds ratios
    age_ratios_2 = measure_bias_with_odds_and_log_odds_ratios(
        age_bias_metrics_model_2)
    gender_ratios_2 = measure_bias_with_odds_and_log_odds_ratios(
        gender_bias_metrics_model_2)

    print("Ratios for Age Subgroups in Model :")
    print(age_ratios_2)

    print("Ratios for Gender Subgroups in Model :")
    print(gender_ratios_2)

    Model_3.eval()

    with torch.no_grad():
        all_predictions_model_3 = []  # Empty list to hold all predictions
        for data in tqdm(test_dataloader):
            data = [x.to('cpu') for x in data]
            seqs, labels = data
            test_predictions_model_3 = Model_3.predict(seqs, labels)
            all_predictions_model_3.append(test_predictions_model_3)

    # Concatenate all predictions into a single tensor
    all_predictions_model_3 = torch.cat(all_predictions_model_3, dim=0)

    # Convert the predictions to class labels (assuming multi-class classification)
    all_test_predictions_labels_model_3 = torch.argmax(
        all_predictions_model_3, dim=1).numpy()

    # Calculate bias metrics for age and gender subgroups
    age_bias_metrics_model_3 = calculate_bias_metrics(
        all_test_predictions_labels_model_3, y_test_for_bias_later[2], x_test_for_bias_later[2]['age'])
    gender_bias_metrics_model_3 = calculate_bias_metrics(
        all_test_predictions_labels_model_3, y_test_for_bias_later[2], x_test_for_bias_later[2]['gender'])

    print("Bias Metrics for Age Subgroups in Model On Seventy Percent Cold Data:")
    print(age_bias_metrics_model_3)

    print("Bias Metrics for Gender Subgroups in Model On Seventy Percent Cold Data:")
    print(gender_bias_metrics_model_3)

    # Measure bias using odds ratios
    age_ratios_3 = measure_bias_with_odds_and_log_odds_ratios(
        age_bias_metrics_model_3)
    gender_ratios_3 = measure_bias_with_odds_and_log_odds_ratios(
        gender_bias_metrics_model_3)

    print("Ratios for Age Subgroups in Model On :")
    print(age_ratios_3)

    print("Ratios for Gender Subgroups in Model :")
    print(gender_ratios_3)

    Model_4.eval()
    with torch.no_grad():
        all_predictions_model_4 = []  # Empty list to hold all predictions
        for data in tqdm(test_dataloader):
            data = [x.to('cpu') for x in data]
            seqs, labels = data
            test_predictions_model_4 = Model_4.predict(seqs, labels)
            all_predictions_model_4.append(test_predictions_model_4)

    # Concatenate all predictions into a single tensor
    all_predictions_model_4 = torch.cat(all_predictions_model_4, dim=0)

    # Convert the predictions to class labels (assuming multi-class classification)
    all_test_predictions_labels_model_4 = torch.argmax(
        all_predictions_model_4, dim=1).numpy()

    # Calculate bias metrics for age and gender subgroups
    age_bias_metrics_model_4 = calculate_bias_metrics(
        all_test_predictions_labels_model_4, y_test_for_bias_later[3], x_test_for_bias_later[3]['age'])
    gender_bias_metrics_model_4 = calculate_bias_metrics(
        all_test_predictions_labels_model_4, y_test_for_bias_later[3], x_test_for_bias_later[3]['gender'])

    print("Bias Metrics for Age Subgroups in Model :")
    print(age_bias_metrics_model_4)

    print("Bias Metrics for Gender Subgroups in Model :")
    print(gender_bias_metrics_model_4)

    # Measure bias using odds ratios
    age_ratios_4 = measure_bias_with_odds_and_log_odds_ratios(
        age_bias_metrics_model_4)
    gender_ratios_4 = measure_bias_with_odds_and_log_odds_ratios(
        gender_bias_metrics_model_4)

    print("Ratios for Age Subgroups in Model :")
    print(age_ratios_4)

    print("Ratios for Gender Subgroups in Model :")
    print(gender_ratios_4)
