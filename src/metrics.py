import numpy as np
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    roc_auc_score,
)
from sklearn.preprocessing import MultiLabelBinarizer

def compute_metrics(dataset, labels, threshold=0.2, exclude_label="unsafe"):
    #Binarize true labels
    labels = [label for label in labels if label != exclude_label]
    mlb = MultiLabelBinarizer(classes=labels)
    y_true = mlb.fit_transform([x["ground_labels"] for x in dataset])

    if "probs" in dataset[0]:
        y_pred = mlb.transform(
            [
                [
                    label
                    for label, score in x["probs"].items()
                    if score >= threshold and label != exclude_label
                ]
                for x in dataset
            ]
        )
    else:
        y_pred = mlb.transform([x["pred_labels"] for x in dataset])

    f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    hamming = hamming_loss(y_true, y_pred)
    
    if "probs" in dataset[0]:
        prob_matrix = np.array(
            [
                [x["probs"].get(label, 0.0) for label in labels]
                for x in dataset
            ]
        )
        auc_micro = roc_auc_score(y_true, prob_matrix, average="micro")
    else:
        auc_micro = np.nan
    
    return {
        "F1": f1,
        "Hamming Loss": hamming,
        "ROC AUC Micro": auc_micro,
    }