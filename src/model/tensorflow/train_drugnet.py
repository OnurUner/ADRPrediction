import pandas as pd
from DrugNet import DrugNet
from math import floor
from sklearn.metrics import roc_auc_score
import numpy as np

data_path = '/media/onur/LinuxHDD/ONUR/Bilkent/DrugSideEffects/data/phase1/data.pkl'
label_path = '/media/onur/LinuxHDD/ONUR/Bilkent/DrugSideEffects/data/phase1/label.pkl'

train_size = 0.9
prune_count = 1


def save_mean_auc(auc_scores, save_path):
    mean_auc_scores = dict()
    for label_index in auc_scores:
        mean_auc_scores[label_index] = np.mean(auc_scores[label_index])

    sorted_means = sorted(mean_auc_scores, key=mean_auc_scores.get, reverse=True)

    f = open(save_path, "w")
    for i in sorted_means:
        f.write(str(i) + " " + str(mean_auc_scores[i]) + "\n")
    f.close()


if __name__ == '__main__':
    auc_scores = dict()
    data_df = pd.read_pickle(data_path)
    label_df = pd.read_pickle(label_path)

    train_cnt = int(floor(data_df.shape[0] * train_size))
    x_train = data_df.iloc[0:train_cnt].values
    y_train = label_df.iloc[0:train_cnt].values
    x_test = data_df.iloc[train_cnt:].values
    y_test = label_df.iloc[train_cnt:].values

    x_train = (x_train - np.min(x_train)) / (np.max(x_train) - np.min(x_train))
    x_test = (x_test - np.min(x_test)) / (np.max(x_test) - np.min(x_test))

    print np.min(x_train)
    print np.max(x_train)
    print np.min(x_test)
    print np.max(x_test)

    print x_train[5,5]

    print "Before filter:", y_train.shape, y_test.shape
    label_indexes = []
    for i in range(label_df.shape[1]):
        if np.sum(y_train[:, i]) >= prune_count and np.sum(y_test[:, i]) >= prune_count:
            label_indexes.append(i)

    y_train = y_train[:, label_indexes]
    y_test = y_test[:, label_indexes]

    print "After filter:", y_train.shape, y_test.shape
    print len(label_indexes)

    net = DrugNet(x_train.shape[1], y_train.shape[1])
    net.fit(x_train, y_train, epochs=50, batch_size=128, is_balanced=True)
    print "Training Finished."
    y_probs = net.predict_proba(x_test)
    print "Prediction Completed."
    scores = roc_auc_score(y_test, y_probs, average=None)

    for i, label_index in enumerate(label_indexes):
        side_effect = label_df.columns.values[label_index]
        if side_effect not in auc_scores:
            auc_scores[side_effect] = []
        auc_scores[side_effect].append(scores[i])
    save_mean_auc(auc_scores, "results.txt")