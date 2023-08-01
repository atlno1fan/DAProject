# Import the required libraries
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score
from IPython.core.interactiveshell import InteractiveShell
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.utils.data import DistributedSampler, DataLoader
import torch.nn as nn
import time
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import math

from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


pd.set_option('display.max_columns', None)

InteractiveShell.ast_node_interactivity = "all"


def visualize_data_bias(video_data, article_data_0, article_data_03, article_data_07, article_data_1):
    # Check if any of the data contains duplicates or nulls
    print("Video data shape: ", video_data.shape)
    print("Video data total dublicates: ")
    video_data.duplicated().sum()
    print("Video data total nulls: ")
    video_data.isna().sum()

    print("article_data_0 shape: ", article_data_0.shape)
    print("article_data_0 total dublicates: ")
    article_data_0.duplicated().sum()
    print("article_data_0 total nulls: ")
    article_data_0.isna().sum()

    print("article_data_03 shape: ", article_data_03.shape)
    print("article_data_03 total dublicates: ")
    article_data_03.duplicated().sum()
    print("article_data_03 total nulls: ")
    article_data_03.isna().sum()

    print("article_data_07 shape: ", article_data_07.shape)
    print("article_data_07 total dublicates: ")
    article_data_07.duplicated().sum()
    print("article_data_07 total nulls: ")
    article_data_07.isna().sum()

    print("article_data_1 shape: ", article_data_1.shape)
    print("article_data_1 total dublicates: ")
    article_data_1.duplicated().sum()
    print("article_data_1 total nulls: ")
    article_data_1.isna().sum()


def visualize_data_info(video_data, article_data_0, article_data_03, article_data_07, article_data_1):
    # Display the dataframe information
    video_data.info()
    article_data_0.info()
    article_data_03.info()
    article_data_07.info()
    article_data_1.info()
    # For the source data
    video_data_for_info_only = video_data.copy()
    video_data_for_info_only['video_category'] = video_data_for_info_only['video_category'].astype(
        'category')

    video_data_for_info_only['like'] = video_data_for_info_only['like'].astype(
        'category')
    video_data_for_info_only['follow'] = video_data_for_info_only['follow'].astype(
        'category')

    video_data_for_info_only['gender'] = video_data_for_info_only['gender'].astype(
        'category')
    video_data_for_info_only['age'] = video_data_for_info_only['age'].astype(
        'category')
    video_data_for_info_only['click'] = video_data_for_info_only['click'].astype(
        'category')

    video_data_for_info_only['user_id'] = video_data_for_info_only['user_id'].astype(
        'category')
    video_data_for_info_only['item_id'] = video_data_for_info_only['item_id'].astype(
        'category')
    video_data_for_info_only['share'] = video_data_for_info_only['share'].astype(
        'category')

    video_data_for_info_only.describe(include='all')

    # For the target data with zero percent cold data
    article_data_0_for_info_only = article_data_0.copy()

    article_data_0_for_info_only['read'] = article_data_0_for_info_only['read'].astype(
        'category')
    article_data_0_for_info_only['share'] = article_data_0_for_info_only['share'].astype(
        'category')
    article_data_0_for_info_only['like'] = article_data_0_for_info_only['like'].astype(
        'category')
    article_data_0_for_info_only['follow'] = article_data_0_for_info_only['follow'].astype(
        'category')
    article_data_0_for_info_only['favorite'] = article_data_0_for_info_only['favorite'].astype(
        'category')

    article_data_0_for_info_only['gender'] = article_data_0_for_info_only['gender'].astype(
        'category')
    article_data_0_for_info_only['age'] = article_data_0_for_info_only['age'].astype(
        'category')
    article_data_0_for_info_only['click'] = article_data_0_for_info_only['click'].astype(
        'category')

    article_data_0_for_info_only['user_id'] = article_data_0_for_info_only['user_id'].astype(
        'category')
    article_data_0_for_info_only['item_id'] = article_data_0_for_info_only['item_id'].astype(
        'category')

    article_data_0_for_info_only.describe(include='all')

    # For the target data with thirty percent cold data
    article_data_03_for_info_only = article_data_03.copy()

    article_data_03_for_info_only['read'] = article_data_03_for_info_only['read'].astype(
        'category')
    article_data_03_for_info_only['share'] = article_data_03_for_info_only['share'].astype(
        'category')
    article_data_03_for_info_only['like'] = article_data_03_for_info_only['like'].astype(
        'category')
    article_data_03_for_info_only['follow'] = article_data_03_for_info_only['follow'].astype(
        'category')
    article_data_03_for_info_only['favorite'] = article_data_03_for_info_only['favorite'].astype(
        'category')

    article_data_03_for_info_only['gender'] = article_data_03_for_info_only['gender'].astype(
        'category')
    article_data_03_for_info_only['age'] = article_data_03_for_info_only['age'].astype(
        'category')
    article_data_03_for_info_only['click'] = article_data_03_for_info_only['click'].astype(
        'category')

    article_data_03_for_info_only['user_id'] = article_data_03_for_info_only['user_id'].astype(
        'category')
    article_data_03_for_info_only['item_id'] = article_data_03_for_info_only['item_id'].astype(
        'category')

    article_data_03_for_info_only.describe(include='all')

    # For the target data with seventy percent cold data
    article_data_07_for_info_only = article_data_07.copy()

    article_data_07_for_info_only['read'] = article_data_07_for_info_only['read'].astype(
        'category')
    article_data_07_for_info_only['share'] = article_data_07_for_info_only['share'].astype(
        'category')
    article_data_07_for_info_only['like'] = article_data_07_for_info_only['like'].astype(
        'category')
    article_data_07_for_info_only['follow'] = article_data_07_for_info_only['follow'].astype(
        'category')
    article_data_07_for_info_only['favorite'] = article_data_07_for_info_only['favorite'].astype(
        'category')

    article_data_07_for_info_only['gender'] = article_data_07_for_info_only['gender'].astype(
        'category')
    article_data_07_for_info_only['age'] = article_data_07_for_info_only['age'].astype(
        'category')
    article_data_07_for_info_only['click'] = article_data_07_for_info_only['click'].astype(
        'category')

    article_data_07_for_info_only['user_id'] = article_data_07_for_info_only['user_id'].astype(
        'category')
    article_data_07_for_info_only['item_id'] = article_data_07_for_info_only['item_id'].astype(
        'category')

    article_data_07_for_info_only.describe(include='all')
    # For the target data with hundred percent cold data
    article_data_1_for_info_only = article_data_1.copy()

    article_data_1_for_info_only['read'] = article_data_1_for_info_only['read'].astype(
        'category')
    article_data_1_for_info_only['share'] = article_data_1_for_info_only['share'].astype(
        'category')
    article_data_1_for_info_only['like'] = article_data_1_for_info_only['like'].astype(
        'category')
    article_data_1_for_info_only['follow'] = article_data_1_for_info_only['follow'].astype(
        'category')
    article_data_1_for_info_only['favorite'] = article_data_1_for_info_only['favorite'].astype(
        'category')

    article_data_1_for_info_only['gender'] = article_data_1_for_info_only['gender'].astype(
        'category')
    article_data_1_for_info_only['age'] = article_data_1_for_info_only['age'].astype(
        'category')
    article_data_1_for_info_only['click'] = article_data_1_for_info_only['click'].astype(
        'category')

    article_data_1_for_info_only['user_id'] = article_data_1_for_info_only['user_id'].astype(
        'category')
    article_data_1_for_info_only['item_id'] = article_data_1_for_info_only['item_id'].astype(
        'category')

    article_data_1_for_info_only.describe(include='all')

    # Calculate demographic distributions for each dataset
    age_distribution_video_data = video_data['age'].value_counts()
    age_distribution_article_data_0 = article_data_0['age'].value_counts()
    age_distribution_article_data_03 = article_data_03['age'].value_counts()
    age_distribution_article_data_07 = article_data_07['age'].value_counts()
    age_distribution_article_data_1 = article_data_1['age'].value_counts()

    gender_distribution_video_data = video_data['gender'].value_counts()
    gender_distribution_article_data_0 = article_data_0['gender'].value_counts(
    )
    gender_distribution_article_data_03 = article_data_03['gender'].value_counts(
    )
    gender_distribution_article_data_07 = article_data_07['gender'].value_counts(
    )
    gender_distribution_article_data_1 = article_data_1['gender'].value_counts(
    )

    plt.figure(figsize=(20, 8))

    plt.subplot(1, 2, 1)
    plt.bar(age_distribution_video_data.index,
            age_distribution_video_data.values)
    plt.xlabel('Age Group')
    plt.ylabel('Count')
    plt.title('Age Distribution video_data')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.pie(gender_distribution_video_data.values,
            labels=gender_distribution_video_data.index, autopct='%1.1f%%')
    plt.title('Gender Distribution video_data')
    plt.legend()

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(20, 8))

    plt.subplot(1, 2, 1)
    plt.bar(age_distribution_article_data_0.index,
            age_distribution_article_data_0.values)
    plt.xlabel('Age Group')
    plt.ylabel('Count')
    plt.title('Age Distribution article_data_0')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.pie(gender_distribution_article_data_0.values,
            labels=gender_distribution_article_data_0.index, autopct='%1.1f%%')
    plt.title('Gender Distribution article_data_0')
    plt.legend()

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(20, 8))

    plt.subplot(1, 2, 1)
    plt.bar(age_distribution_article_data_03.index,
            age_distribution_article_data_03.values)
    plt.xlabel('Age Group')
    plt.ylabel('Count')
    plt.title('Age Distribution article_data_03')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.pie(gender_distribution_article_data_03.values,
            labels=gender_distribution_article_data_03.index, autopct='%1.1f%%')
    plt.title('Gender Distribution article_data_03')
    plt.legend()

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(20, 8))

    plt.subplot(1, 2, 1)
    plt.bar(age_distribution_article_data_07.index,
            age_distribution_article_data_07.values)
    plt.xlabel('Age Group')
    plt.ylabel('Count')
    plt.title('Age Distribution article_data_07')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.pie(gender_distribution_article_data_07.values,
            labels=gender_distribution_article_data_07.index, autopct='%1.1f%%')
    plt.title('Gender Distribution article_data_07')
    plt.legend()

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(20, 8))

    plt.subplot(1, 2, 1)
    plt.bar(age_distribution_article_data_1.index,
            age_distribution_article_data_1.values)
    plt.xlabel('Age Group')
    plt.ylabel('Count')
    plt.title('Age Distribution article_data_1')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.pie(gender_distribution_article_data_1.values,
            labels=gender_distribution_article_data_1.index, autopct='%1.1f%%')
    plt.title('Gender Distribution article_data_1')
    plt.legend()

    plt.tight_layout()
    plt.show()


def preprocess_data(video_data, article_data_0, article_data_03, article_data_07, article_data_1):
    # As there are duplicates found in all dataframes they will all be dropped
    video_data.drop_duplicates(inplace=True)
    article_data_0.drop_duplicates(inplace=True)
    article_data_03.drop_duplicates(inplace=True)
    article_data_07.drop_duplicates(inplace=True)
    article_data_1.drop_duplicates(inplace=True)


def get_data_loader(dataset, is_parallel=False, batch_size=512, shuffle=False):
    """
    This function creates a data loader for a given dataset. Data loaders are PyTorch's way to handle large datasets
    that can't fit into memory. They allow you to load data in small batches, rather than all at once.
    """
    if is_parallel:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        sampler=sampler
    )
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


def construct_data(task, item_min, path1, path2):
    if task != 2:
        df1 = path1[['user_id', 'item_id', 'age', 'gender', 'click']]
#         df1 = pd.read_csv(path1, usecols=['user_id', 'item_id', 'click'])
        df1 = df1[df1.click == 1]
    else:
        df1 = path1[['user_id', 'item_id', 'age', 'gender', 'like']]
        df1 = df1[df1.like == 1]

    df2 = path2[['user_id', 'item_id', 'age', 'gender', 'click']]
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


def colddataset(item_min, task, path1, path2):
    target_data, source_data, user_count, t_item_count, s_item_count = construct_data(
        task, item_min, path1, path2)

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


class ColdDataset(data_utils.Dataset):
    def __init__(self, x, y, max_len, pad_token):
        self.seqs = x
        self.targets = y
        self.max_len = max_len
        self.pad_token = pad_token

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        seq = self.seqs[index][-self.max_len:]
        seq_len = len(seq)
        seq_mask_len = self.max_len - seq_len
        seq = [self.pad_token] * seq_mask_len + seq
        target = self.targets[index]
        return torch.LongTensor(seq), torch.LongTensor([target])


class ColdEvalDataset(data_utils.Dataset):
    def __init__(self, x, y, max_len, pad_token, num_item):
        self.seqs = x
        self.targets = y
        self.max_len = max_len
        self.pad_token = pad_token
        self.num_item = num_item + 1

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        seq = self.seqs[index][-self.max_len:]
        seq_len = len(seq)
        seq_mask_len = self.max_len - seq_len
        seq = [self.pad_token] * seq_mask_len + seq
        target = self.targets[index]
        labels = [0] * self.num_item
        labels[target] = 1
        return torch.LongTensor(seq), torch.LongTensor(labels)


def get_data(max_len=20, item_min=10, task=2, pad_token=0, path1=article_data_03, path2=video_data):
    path1 = path1
    path2 = path2

    data, user_count, vocab_size, item_count = colddataset(
        item_min, task, path1, path2)

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


def recalls_and_ndcgs_for_ks(scores, labels, ks):
    metrics = {}

    answer_count = labels.sum(1)
    answer_count_float = answer_count.float()
    labels_float = labels.float()

    rank = (-scores).argsort(dim=1)
    cut = rank

    for k in sorted(ks, reverse=True):
        cut = cut[:, :k]
        hits = labels_float.gather(1, cut)
        metrics[f'Recall@{k}'] = (hits.sum(1) /
                                  answer_count_float).mean().item()

        position = torch.arange(2, 2 + k)
        weights = 1 / torch.log2(position.float()).to('cpu')
        dcg = (hits * weights).sum(1)
        idcg = torch.Tensor([weights[:min(n, k)].sum()
                            for n in answer_count]).to('cpu')
        ndcg = (dcg / idcg).mean()
        metrics[f'NDCG@{k}'] = ndcg.item()

        # Compute mAP
        ap_sum = 0
        for i in range(len(hits)):
            num_relevant_items = hits[i].sum().item()
            if num_relevant_items > 0:
                precision_at_k = (
                    hits[i] / torch.arange(1, k + 1).float().to('cpu')).mean().item()
                ap_sum += precision_at_k
        mAP = ap_sum / len(hits)
        metrics[f'mAP@{k}'] = mAP

    return metrics


def Sequence_full_Validate(epoch, model, dataloader, writer, test=False):
    print(
        "+" * 20, f"{'Test' if test else 'Valid'} Epoch {epoch + 1}", "+" * 20)
    model.eval()
    avg_metrics = {}
    i = 0

    with torch.no_grad():
        for data in tqdm(dataloader):
            data = [x.to('cpu') for x in data]
            seqs, labels = data

            if test:
                scores = model.predict(seqs)
            else:
                scores = model(seqs)

            scores = scores.mean(1)
            metrics = recalls_and_ndcgs_for_ks(scores, labels, [5, 20])
            i += 1

            for key, value in metrics.items():
                avg_metrics[key] = avg_metrics.get(key, 0) + value

    for key, value in avg_metrics.items():
        avg_metrics[key] /= i

    print(avg_metrics)

    for k in sorted([5, 20], reverse=True):
        writer.add_scalar(
            f"{'Test' if test else 'Train'}/NDCG@{k}", avg_metrics[f"NDCG@{k}"], epoch)

    return avg_metrics


def SequenceTrainer(epoch, model, dataloader, optimizer, writer):
    print("+" * 20, "Train Epoch {}".format(epoch + 1), "+" * 20)
    model.train()
    running_loss = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for data in tqdm(dataloader):
        optimizer.zero_grad()
        data = [x.to('cpu') for x in data]
        seqs, labels = data

        logits = model(seqs)  # B x T x V
        logits = logits.mean(1)
        labels = labels.view(-1)

        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.detach().cpu().item()

    writer.add_scalar('Train/loss', running_loss / len(dataloader), epoch)
    print("Training CE Loss: {:.5f}".format(running_loss / len(dataloader)))

    return optimizer


def SeqTrain(epochs, model, train_loader, val_loader, writer, is_parallel, is_pretrain, lr, weight_decay, local_rank):
    device = torch.device('cpu')  # Use CPU as the default device
    if torch.cuda.is_available():
        # If GPU is available, use it as the default device
        device = torch.device('cuda')

    if is_pretrain == 0:
        optimizer = optim.Adam(filter(
            lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr,
                               weight_decay=weight_decay)

    model = model.to(device)  # Move the model to the chosen device
    if is_parallel:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank)

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
        val_tmp = time.time() - val_since
        print('one epoch val:', val_tmp)
        val_all_time += val_tmp

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

    print('train_time:', all_time)
    print('val_time:', val_all_time)
    return best_model


class Attention(nn.Module):
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)
                              ) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linear_layers = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        x, attn = self.attention(
            query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(
            batch_size, -1, self.h * self.d_k)
        return self.output_linear(x)


class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super(PositionalEmbedding, self).__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super(TokenEmbedding, self).__init__(
            vocab_size, embed_size, padding_idx=0)


class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, max_len, dropout=0.1):
        super(BERTEmbedding, self).__init__()
        self.token = TokenEmbedding(
            vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(
            max_len=max_len, d_model=embed_size)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence):
        x = self.token(sequence) + self.position(sequence)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadedAttention(
            h=attn_heads, d_model=hidden, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(
            d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(
            x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


class bottle_net(nn.Module):
    def __init__(self, hidden):
        super(bottle_net, self).__init__()
        self.hidden = hidden
        self.hidden_size = int(hidden / 4)
        self.linear1 = nn.Linear(self.hidden, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden)
        self.ln = nn.LayerNorm(self.hidden, eps=1e-8)

    def forward(self, x):
        out = self.linear1(x)
        out = F.leaky_relu(out)
        out = self.linear2(out)
        out = self.ln(out)
        return
        return out


class BERT(nn.Module):
    def __init__(self, max_len, block_num, num_embedding, hidden_size, is_mp, dropout):
        super(BERT, self).__init__()
        max_len = max_len
        n_layers = block_num
        heads = block_num
        vocab_size = num_embedding + 1
        hidden = hidden_size
        self.hidden = hidden
        self.is_mp = is_mp
        dropout = dropout

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(
            vocab_size=vocab_size, embed_size=self.hidden, max_len=max_len, dropout=dropout)

        transformer_blocks = []
        for _ in range(n_layers):
            transformer_blocks.append(TransformerBlock(
                hidden, heads, hidden * 4, dropout))
            if self.is_mp:
                transformer_blocks.append(bottle_net(self.hidden))
        self.transformer_blocks = nn.ModuleList(transformer_blocks)

    def forward(self, x):
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to a sequence of vectors
        x = self.embedding(x)

        # running over multiple transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            mp_input = x
            if i % 2 == 0:
                x = block.forward(x, mask)
            else:
                if self.is_mp:
                    mp_out = block(mp_input)
                    x = mp_out + x
        return x

    def init_weights(self):
        pass


class BERT_ColdstartModel(nn.Module):
    def __init__(self, num_items, max_len, block_num, num_heads, num_embedding, hidden_size, is_mp, dropout):
        super(BERT_ColdstartModel, self).__init__()
        self.bert = BERT(max_len, block_num, num_embedding,
                         hidden_size, is_mp, dropout)
        self.num_items = num_items
        self.out = nn.Linear(self.bert.hidden, num_items + 1)

    def forward(self, x):
        x = self.bert(x)
        return self.out(x)

    def predict(self, x, item):
        x = self.bert(x)
        item_emb = self.bert.embedding.token(item)
        logits = x.matmul(item_emb.transpose(1, 2))
        logits = logits.mean(1)

        return logits


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


def run_experiment_bert4rec():
    # -------------------------Load Data---------------------------------------------------------------------
    # Data of users from a video platform to be used as the source/training data
    video_data_path = 'Data/sbr_data_1M.csv'
    # Similar data but collected from articels to be used as the target/testing data
    # Zero percent cold data
    article_data_0_path = 'Data/cold_data.csv'
    # Thirty percent cold data
    article_data_03_path = 'Data/cold_data_0.3.csv'
    # Seventy percent cold data
    article_data_07_path = 'Data/cold_data_0.7.csv'
    # Hundered percent cold data
    article_data_1_path = 'Data/cold_data_1.csv'

    # Read the data from the csv files
    video_data = pd.read_csv(video_data_path)
    article_data_0 = pd.read_csv(article_data_0_path)
    article_data_03 = pd.read_csv(article_data_03_path)
    article_data_07 = pd.read_csv(article_data_07_path)
    article_data_1 = pd.read_csv(article_data_1_path)
 # -------------------------Visualize the Data-----------------------------------------------------------------
    visualize_data_bias(video_data, article_data_0,
                        article_data_03, article_data_07, article_data_1)

 # -------------------------Train the Models-------------------------------------------------------------------
    x_test_for_bias_later = []
    y_test_for_bias_later = []

 # -------------------------Model_0----------------------------------------------------------------------------
    print('Fetching the data and parameters for model_30')
    train_dataloader, valid_dataloader, test_dataloader_0, num_users, num_items, num_embeddings = get_data(
        path1=article_data_0, path2=video_data)
    num_items, max_len, block_num, num_heads, num_embedding, hidden_size, is_mp, dropout = num_items, 20, 2, 4, num_embeddings, 128, False, 0.3
    model_0 = BERT_ColdstartModel(
        num_items, max_len, block_num, num_heads, num_embedding, hidden_size, is_mp, dropout)
    writer = SummaryWriter()
    print('Training the Model')
    model_0_best = SeqTrain(10, model_0, train_dataloader, valid_dataloader, writer,
                            is_pretrain=1, is_parallel=False, lr=0.001, weight_decay=0.0001, local_rank=1)

 # -------------------------Model_30---------------------------------------------------------------------------
    print('Fetching the data and parameters')
    train_dataloader, valid_dataloader, test_dataloader_30, num_users, num_items, num_embeddings = get_data(
        path1=article_data_03, path2=video_data)
    num_items, max_len, block_num, num_heads, num_embedding, hidden_size, is_mp, dropout = num_items, 20, 2, 4, num_embeddings, 128, False, 0.3
    model_30 = BERT_ColdstartModel(
        num_items, max_len, block_num, num_heads, num_embedding, hidden_size, is_mp, dropout)
    writer = SummaryWriter()
    print('Training the Model')
    model_30_best = SeqTrain(10, model_30, train_dataloader, valid_dataloader, writer,
                             is_pretrain=1, is_parallel=False, lr=0.001, weight_decay=0.0001, local_rank=1)

 # -------------------------Model_70----------------------------------------------------------------------------
    print('Fetching the data and parameters')
    train_dataloader, valid_dataloader, test_dataloader_70, num_users, num_items, num_embeddings = get_data(
        path1=article_data_07, path2=video_data)
    num_items, max_len, block_num, num_heads, num_embedding, hidden_size, is_mp, dropout = num_items, 20, 2, 4, num_embeddings, 128, False, 0.3
    model_70 = BERT_ColdstartModel(
        num_items, max_len, block_num, num_heads, num_embedding, hidden_size, is_mp, dropout)
    writer = SummaryWriter()
    print('Training the Model')
    model_70_best = SeqTrain(10, model_70, train_dataloader, valid_dataloader, writer,
                             is_pretrain=1, is_parallel=False, lr=0.001, weight_decay=0.0001, local_rank=1)

 # -------------------------Model_1------------------------------------------------------------------------------
    print('Fetching the data and parameters')
    train_dataloader, valid_dataloader, test_dataloader_1, num_users, num_items, num_embeddings = get_data(
        path1=article_data_1, path2=video_data)
    num_items, max_len, block_num, num_heads, num_embedding, hidden_size, is_mp, dropout = num_items, 20, 2, 4, num_embeddings, 128, False, 0.3
    model_1 = BERT_ColdstartModel(
        num_items, max_len, block_num, num_heads, num_embedding, hidden_size, is_mp, dropout)
    writer = SummaryWriter()
    print('Training the Model')
    model_1_best = SeqTrain(10, model_1, train_dataloader, valid_dataloader, writer,
                            is_pretrain=1, is_parallel=False, lr=0.001, weight_decay=0.0001, local_rank=1)

 # -------------------------Evaluation----------------------------------------------------------------------------
    # Evaluate the models on the test dataset to get the predictions, the test data for each model will be similar to the one used
    # during the evaluation above, so it will be each models target data. The only reason is so that we have access to the predicted
    # labels.

    # --------------------------Evaluation for Model_0-------------------------------------------------------------
    model_0_best.eval()

    with torch.no_grad():
        all_predictions_model_0 = []  # Empty list to hold all predictions
        for data in tqdm(test_dataloader_0):
            data = [x.to('cpu') for x in data]
            seqs, labels = data
            test_predictions_model_0 = model_0_best.predict(seqs, labels)
            all_predictions_model_0.append(test_predictions_model_0)

    # Concatenate all predictions into a single tensor
    all_predictions_model_0 = torch.cat(all_predictions_model_0, dim=0)

    # Convert the predictions to class labels (assuming multi-class classification)
    all_test_predictions_labels_model_0 = torch.argmax(
        all_predictions_model_0, dim=1).numpy()

    # Calculate bias metrics for age and gender subgroups
    age_bias_metrics_model_0 = calculate_bias_metrics(
        all_test_predictions_labels_model_0, y_test_for_bias_later[0], x_test_for_bias_later[0]['age'])
    gender_bias_metrics_model_0 = calculate_bias_metrics(
        all_test_predictions_labels_model_0, y_test_for_bias_later[0], x_test_for_bias_later[0]['gender'])

    print("Bias Metrics for Age Subgroups in Model On Zero Percent Cold Data:")
    print(age_bias_metrics_model_0)

    print("Bias Metrics for Gender Subgroups in Model On Zero Percent Cold Data:")
    print(gender_bias_metrics_model_0)

    # Measure bias using log odds ratios
    age_ratios_0 = measure_bias_with_odds_and_log_odds_ratios(
        age_bias_metrics_model_0)
    gender_ratios_0 = measure_bias_with_odds_and_log_odds_ratios(
        gender_bias_metrics_model_0)

    print("Ratios for Age Subgroups in Model On Zero Percent Cold Data:")
    print(age_ratios_0)

    print("Ratios for Gender Subgroups in Model On Zero Percent Cold Data:")
    print(gender_ratios_0)

    # -------------------------Evaluation for Model_30-------------------------------------------------------------
    model_30_best.eval()
    with torch.no_grad():
        all_predictions_model_30 = []  # Empty list to hold all predictions
        for data in tqdm(test_dataloader_30):
            data = [x.to('cpu') for x in data]
            seqs, labels = data
            test_predictions_model_30 = model_30_best.predict(seqs, labels)
            all_predictions_model_30.append(test_predictions_model_30)

    # Concatenate all predictions into a single tensor
    all_predictions_model_30 = torch.cat(all_predictions_model_30, dim=0)

    # Convert the predictions to class labels (assuming multi-class classification)
    all_test_predictions_labels_model_30 = torch.argmax(
        all_predictions_model_30, dim=1).numpy()

    # Calculate bias metrics for age and gender subgroups
    age_bias_metrics_model_30 = calculate_bias_metrics(
        all_test_predictions_labels_model_30, y_test_for_bias_later[1], x_test_for_bias_later[1]['age'])
    gender_bias_metrics_model_30 = calculate_bias_metrics(
        all_test_predictions_labels_model_30, y_test_for_bias_later[1], x_test_for_bias_later[1]['gender'])

    print("Bias Metrics for Age Subgroups in Model On Thirty Percent Cold Data:")
    print(age_bias_metrics_model_30)

    print("Bias Metrics for Gender Subgroups in Model On Thirty Percent Cold Data:")
    print(gender_bias_metrics_model_30)

    # Measure bias using odds ratios
    age_ratios_30 = measure_bias_with_odds_and_log_odds_ratios(
        age_bias_metrics_model_30)
    gender_ratios_30 = measure_bias_with_odds_and_log_odds_ratios(
        gender_bias_metrics_model_30)

    print("Ratios for Age Subgroups in Model On Thirty Percent Cold Data:")
    print(age_ratios_30)

    print("Ratios for Gender Subgroups in Model On Thirty Percent Cold Data:")
    print(gender_ratios_30)

    # -------------------------Evaluation for Model_70-------------------------------------------------------------
    model_70_best.eval()

    with torch.no_grad():
        all_predictions_model_70 = []  # Empty list to hold all predictions
        for data in tqdm(test_dataloader_70):
            data = [x.to('cpu') for x in data]
            seqs, labels = data
            test_predictions_model_70 = model_70_best.predict(seqs, labels)
            all_predictions_model_70.append(test_predictions_model_70)

    # Concatenate all predictions into a single tensor
    all_predictions_model_70 = torch.cat(all_predictions_model_70, dim=0)

    # Convert the predictions to class labels (assuming multi-class classification)
    all_test_predictions_labels_model_70 = torch.argmax(
        all_predictions_model_70, dim=1).numpy()

    # Calculate bias metrics for age and gender subgroups
    age_bias_metrics_model_70 = calculate_bias_metrics(
        all_test_predictions_labels_model_70, y_test_for_bias_later[2], x_test_for_bias_later[2]['age'])
    gender_bias_metrics_model_70 = calculate_bias_metrics(
        all_test_predictions_labels_model_70, y_test_for_bias_later[2], x_test_for_bias_later[2]['gender'])

    print("Bias Metrics for Age Subgroups in Model On Seventy Percent Cold Data:")
    print(age_bias_metrics_model_70)

    print("Bias Metrics for Gender Subgroups in Model On Seventy Percent Cold Data:")
    print(gender_bias_metrics_model_70)

    # Measure bias using odds ratios
    age_ratios_70 = measure_bias_with_odds_and_log_odds_ratios(
        age_bias_metrics_model_70)
    gender_ratios_70 = measure_bias_with_odds_and_log_odds_ratios(
        gender_bias_metrics_model_70)

    print("Ratios for Age Subgroups in Model On Seventy Percent Cold Data:")
    print(age_ratios_70)

    print("Ratios for Gender Subgroups in Model On Seventy Percent Cold Data:")
    print(gender_ratios_70)

    # -------------------------Evaluation for Model_1-------------------------------------------------------------
    model_1_best.eval()
    with torch.no_grad():
        all_predictions_model_1 = []  # Empty list to hold all predictions
        for data in tqdm(test_dataloader_1):
            data = [x.to('cpu') for x in data]
            seqs, labels = data
            test_predictions_model_1 = model_1_best.predict(seqs, labels)
            all_predictions_model_1.append(test_predictions_model_1)

    # Concatenate all predictions into a single tensor
    all_predictions_model_1 = torch.cat(all_predictions_model_1, dim=0)

    # Convert the predictions to class labels (assuming multi-class classification)
    all_test_predictions_labels_model_1 = torch.argmax(
        all_predictions_model_1, dim=1).numpy()

    # Calculate bias metrics for age and gender subgroups
    age_bias_metrics_model_1 = calculate_bias_metrics(
        all_test_predictions_labels_model_1, y_test_for_bias_later[3], x_test_for_bias_later[3]['age'])
    gender_bias_metrics_model_1 = calculate_bias_metrics(
        all_test_predictions_labels_model_1, y_test_for_bias_later[3], x_test_for_bias_later[3]['gender'])

    print("Bias Metrics for Age Subgroups in Model On Hundred Percent Cold Data:")
    print(age_bias_metrics_model_1)

    print("Bias Metrics for Gender Subgroups in Model On Hundred Percent Cold Data:")
    print(gender_bias_metrics_model_1)

    # Measure bias using odds ratios
    age_ratios_1 = measure_bias_with_odds_and_log_odds_ratios(
        age_bias_metrics_model_1)
    gender_ratios_1 = measure_bias_with_odds_and_log_odds_ratios(
        gender_bias_metrics_model_1)

    print("Ratios for Age Subgroups in Model On Hundred Percent Cold Data:")
    print(age_ratios_1)

    print("Ratios for Gender Subgroups in Model On Hundred Percent Cold Data:")
    print(gender_ratios_1)
