import torch
from transformers import AutoTokenizer

import sklearn.metrics as metrics


def get_predictions(model: torch.nn.Module, tokenizer: AutoTokenizer, sentence: str,
                    dynamic_input: bool = False, gpu: bool = False):
    """
    Function that returns predictions for a given input.
		
		Args: 
            model (torch.nn.Module): deep learning model
            tokenizer (transformers.AutoTokenizer): pre-trained tokenizer
            sentence (str): raw text input
            dynamic_input (bool): If True, sequences will not be padded
            gpu (bool): If True, torch GPU tensors are enabled

		Returns: 
			Predicted probabilities
    """

    sentence = tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        max_length=model.max_length,
        return_token_type_ids=True,
        padding=False if dynamic_input else "max_length",
        return_attention_mask=True,
        return_tensors="pt",
        truncation=True
    )

    if gpu:
        return model(input_ids=sentence["input_ids"].cuda(), attention_mask=sentence["attention_mask"].cuda())
    else:
        return model(input_ids=sentence["input_ids"].to(torch.device("cpu")),
                     attention_mask=sentence["attention_mask"].to(torch.device("cpu"))).tolist()[0]


def metric_accuracy(y_true, y_pred, pos_label=True):
    """
    Function that returns accuracy score.
		
		Args: 
            y_true (numpy.array): encoded target labels
            y_pred (numpy.array): predicted labels
            pos_label (bool): If True, positive class is reported

		Returns: 
			Accuracy score
    """
    return metrics.accuracy_score(y_true, y_pred)

def metric_f1(y_true, y_pred, pos_label=True):
    """
    Function that returns f1-score.
		
		Args: 
            y_true (numpy.array): encoded target labels
            y_pred (numpy.array): predicted labels
            pos_label (bool): If True, positive class is reported

		Returns: 
			F1-score
    """
    return metrics.f1_score(y_true, y_pred, average="weighted", pos_label=pos_label)

def metric_f1_micro(y_true, y_pred, pos_label=True):
    """
    Function that returns micro f1-score.
		
		Args: 
            y_true (numpy.array): encoded target labels
            y_pred (numpy.array): predicted labels
            pos_label (bool): If True, positive class is reported

		Returns: 
			Micro F1-score
    """
    return metrics.f1_score(y_true, y_pred, average="micro", pos_label=pos_label)

def metric_f1_macro(y_true, y_pred, pos_label=True):
    """
    Function that returns macro f1-score.
		
		Args: 
            y_true (numpy.array): encoded target labels
            y_pred (numpy.array): predicted labels
            pos_label (bool): If True, positive class is reported

		Returns: 
			Macro F1-score
    """
    return metrics.f1_score(y_true, y_pred, average="macro", pos_label=pos_label)

def metric_precision(y_true, y_pred, pos_label=True):
    """
    Function that returns precision score.
		
		Args: 
            y_true (numpy.array): encoded target labels
            y_pred (numpy.array): predicted labels
            pos_label (bool): If True, positive class is reported

		Returns: 
			Precision score
    """
    return metrics.precision_score(y_true, y_pred, average="binary", pos_label=pos_label)

def metric_recall(y_true, y_pred, pos_label=True):
    """
    Function that returns recall score.
		
		Args: 
            y_true (numpy.array): encoded target labels
            y_pred (numpy.array): predicted labels
            pos_label (bool): If True, positive class is reported

		Returns: 
			Recall score
    """
    return metrics.recall_score(y_true, y_pred, average="binary", pos_label=pos_label)

def metric_auc(y_true, y_score, pos_label=True):
    """
    Function that returns ROC-AUC score.
		
		Args: 
            y_true (numpy.array): encoded target labels
            y_score (numpy.array): predicted probabilities
            pos_label (bool): If True, positive class is reported

		Returns: 
			ROC-AUC score
    """
    try:
        fpr, tpr, _ = metrics.roc_curve(
            y_true,
            y_score,
            pos_label=pos_label
        )
        return metrics.auc(fpr, tpr)
    except:
        return None

def metric_kappa(y_true, y_pred):
    """
    Function that returns kappa metrics.
		
		Args: 
            y_true (numpy.array): encoded target labels
            y_pred (numpy.array): predicted labels
            pos_label (bool): If True, positive class is reported

		Returns: 
			Kappa metric
    """
    return metrics.cohen_kappa_score(y_true, y_pred)

def confusion_matrix(y_true, y_pred):
    """
    Function that returns confusion matrix components.
		
		Args: 
            y_true (numpy.array): encoded target labels
            y_pred (numpy.array): predicted labels
            pos_label (bool): If True, positive class is reported

		Returns: 
			Tuple with (TNs, FPs, FNs, TPs)
    """
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    return int(tn), int(fp), int(fn), int(tp)

def evaluate_classification(y_true, y_pred, y_score):
    """
    Function that returns dictionary containing all metrics and their score
		
		Args: 
            y_true (numpy.array): encoded target labels
            y_pred (numpy.array): predicted labels
            y_score (numpy.array): predicted probabilities

		Returns: 
			Dictionary with metrics as keys and their scores as values
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred)

    metrics_dict = {
        "ACCURACY": metric_accuracy(y_true, y_pred),
        "F1-SCORE": metric_f1(y_true, y_pred),
        "F1-MICRO":metric_f1_micro(y_true, y_pred) ,
        "F1-MACRO": metric_f1_macro(y_true, y_pred),
        "PRECISION": metric_precision(y_true, y_pred),
        "RECALL": metric_recall(y_true, y_pred),
        "AUC": metric_auc(y_true, y_score),
        "KAPPA": metric_kappa(y_true, y_pred),
        "TRUE_POSITIVE": tp,
        "TRUE_NEGATIVE": tn,
        "FALSE_POSITIVE": fp,
        "FALSE_NEGATIVE": fn
    }

    return {k:v for k, v in metrics_dict.items() if v is not None}