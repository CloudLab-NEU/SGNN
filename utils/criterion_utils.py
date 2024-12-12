# -*- coding: utf-8 -*-


from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score


def get_indicators(y, y_pred):
    acc = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average=None)
    F1 = f1_score(y, y_pred, average=None)
    recall = recall_score(y, y_pred, average=None)
    # C = confusion_matrix(y, y_pred, labels=[0, 1])
    auc = roc_auc_score(y, y_pred)
    # return acc, precision[1], precision[0], recall[1], recall[0], F1[1], F1[0], auc
    return acc, precision, recall, F1, auc