import numpy as np
import sklearn.metrics as metrics
import sys
from scipy.optimize import linear_sum_assignment


def classification_metric(y_true, y_pred, average='macro', verbose=True, decimals=4):
    # confusion matrix
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    # ACC
    accuracy = metrics.accuracy_score(y_true, y_pred)
    accuracy = np.round(accuracy, decimals)
    # accuracy = calculate_acc(y_true, y_pred)

    # precision
    precision = metrics.precision_score(y_true, y_pred, average=average, zero_division=0)
    precision = np.round(precision, decimals)

    # recall
    recall = metrics.recall_score(y_true, y_pred, average=average, zero_division=0)
    recall = np.round(recall, decimals)

    # F-score
    f_score = metrics.f1_score(y_true, y_pred, average=average, zero_division=0)
    f_score = np.round(f_score, decimals)

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f_measure': f_score}, confusion_matrix


def _encode_contiguous(labels):
    """Map arbitrary label ids to contiguous ids without changing label identity."""
    labels = np.asarray(labels).reshape(-1)
    _, encoded = np.unique(labels, return_inverse=True)
    return encoded.astype(np.int64)


def _contingency_matrix(y_true, y_pred):
    n_true = int(np.max(y_true) + 1) if y_true.size else 0
    n_pred = int(np.max(y_pred) + 1) if y_pred.size else 0
    matrix = np.zeros((n_true, n_pred), dtype=np.int64)
    if y_true.size and y_pred.size:
        np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def get_y_preds(y_true, cluster_assignments, n_clusters=None):
    """Compute predicted labels after Hungarian matching to true labels.

        Args:
            cluster_assignments: array of labels, outputted by kmeans
            y_true:              true labels
            n_clusters:          number of clusters in the dataset

        Returns:
            a tuple containing the accuracy and confusion matrix, in that order
    """
    y_true = _encode_contiguous(y_true)
    cluster_assignments = _encode_contiguous(cluster_assignments)
    contingency = _contingency_matrix(y_true, cluster_assignments)

    n_true, n_pred = contingency.shape
    if n_true == 0 or n_pred == 0:
        return np.zeros_like(y_true)

    expected_clusters = int(n_clusters) if n_clusters is not None else 0
    matrix_size = max(n_true, n_pred, expected_clusters)
    padded = np.zeros((matrix_size, matrix_size), dtype=np.int64)
    padded[:n_true, :n_pred] = contingency

    # Maximize agreement between predicted clusters (columns) and labels (rows).
    row_ind, col_ind = linear_sum_assignment(-padded)
    cluster_to_label = {
        int(col): int(row)
        for row, col in zip(row_ind, col_ind)
        if row < n_true and col < n_pred
    }

    y_pred = np.full_like(cluster_assignments, fill_value=-1)
    for cluster_id, label_id in cluster_to_label.items():
        y_pred[cluster_assignments == cluster_id] = label_id
    return y_pred


def clustering_metric(y_true, y_pred, n_clusters=None, verbose=True, decimals=4):
    y_true = _encode_contiguous(y_true)
    y_pred = _encode_contiguous(y_pred)
    y_pred_ajusted = get_y_preds(y_true, y_pred, n_clusters)

    classification_metrics, confusion_matrix = classification_metric(y_true, y_pred_ajusted)

    # AMI
    ami = metrics.adjusted_mutual_info_score(y_true, y_pred)
    ami = np.round(ami, decimals)
    # NMI
    nmi = metrics.normalized_mutual_info_score(y_true, y_pred)
    nmi = np.round(nmi, decimals)
    # ARI
    ari = metrics.adjusted_rand_score(y_true, y_pred)
    ari = np.round(ari, decimals)

    # PUR (cluster purity)
    cm_for_purity = _contingency_matrix(y_true, y_pred)
    total = np.sum(cm_for_purity)
    purity = np.sum(np.max(cm_for_purity, axis=0)) / total if total > 0 else 0.0
    purity = np.round(purity, decimals)

    return dict({'AMI': ami, 'NMI': nmi, 'ARI': ari, 'PUR': purity}, **classification_metrics), confusion_matrix


def get_cluster_sols(x, cluster_obj=None, ClusterClass=None, n_clusters=None, init_args={}):
    """Using either a newly instantiated ClusterClass or a provided cluster_obj, generates
        cluster assignments based on input data.

        Args:
            x: the points with which to perform clustering
            cluster_obj: a pre-fitted instance of a clustering class,如初始化好的kmeans类
            ClusterClass: a reference to the sklearn clustering class, necessary
              if instantiating a new clustering class
            n_clusters: number of clusters in the dataset, necessary
                        if instantiating new clustering class
            init_args: any initialization arguments passed to ClusterClass

        Returns:
            a tuple containing the label assignments and the clustering object
    """
    # if provided_cluster_obj is None, we must have both ClusterClass and n_clusters
    assert not (cluster_obj is None and (ClusterClass is None or n_clusters is None))
    if cluster_obj is None:
        cluster_obj = ClusterClass(n_clusters, **init_args)
        for _ in range(10):
            try:
                cluster_obj.fit(x)
                break
            except:
                print("Unexpected error:", sys.exc_info())
        else:
            return np.zeros((len(x),)), cluster_obj

    cluster_assignments = cluster_obj.predict(x)
    return cluster_assignments, cluster_obj


def evaluation(y_pred, y_true, accumulated_metrics=None, n_clusters=None):
    if n_clusters is None:
        n_clusters = max(np.size(np.unique(y_true)), np.size(np.unique(y_pred)))
    scores, _ = clustering_metric(y_true, y_pred, n_clusters)
    if accumulated_metrics is not None:
        accumulated_metrics['acc'].append(scores['accuracy'])
        accumulated_metrics['nmi'].append(scores['NMI'])
        accumulated_metrics['ARI'].append(scores['ARI'])
        accumulated_metrics['f-mea'].append(scores['f_measure'])
    return scores
