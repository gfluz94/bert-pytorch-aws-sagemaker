from typing import List
import torch
import torch.nn as nn
from transformers import AutoModel


EMBEDDING_MODEL_PATH = "/root/.cache/huggingface/transformers/distilbert-base-pt-cased/"
SEED = 99


class BERTClassifier(nn.Module):

    def __init__(self, n_classes: int, embedding_model: AutoModel, max_length: int, hidden_size: List[int],
                dropout_ratio: float = 0.2, embeddings_grad: bool = False):
        """
        Custom BERT classifier which uses transfer learning in order to achieve a different task.
	
        Attributes:
            n_classes (int) representing the number of classes to be identified in the task
            embedding_model (transformers.AutoModel) representing a pre-trained BERT model
            max_length (int) representing the max length to pad the sequences accordingly
            hidden_size (list of int) representing the number of neurons in each top hidden layer
            dropout_ratio (float) representing the ratio at which neurons are going to be dropped during training time
            embeddings_grad (bool) representing whether or not fine-tuning will be performed
                
        """
        super(BERTClassifier, self).__init__()

        self.__n_classes = n_classes
        self.__embedding_model = embedding_model
        self.__max_length = max_length
        self.__embeddings_grad = embeddings_grad
        self.__embedding_model.requires_grad = self.__embeddings_grad
        self.__hidden_size = [self.__embedding_model.config.hidden_size] + hidden_size
        self.__dropout_ratio = dropout_ratio

        if len(self.__hidden_size) > 1:
            layers = [
                nn.Sequential(
                    nn.Linear(in_units, out_units),
                    nn.ReLU()
                ) for in_units, out_units in zip(self.__hidden_size, self.__hidden_size[1:])
            ]
            self.__fc = nn.Sequential(*layers)
        else:
            self.__fc = None   

        if self.__dropout_ratio is None:
            layers = nn.Sequential(
                nn.Linear(self.__hidden_size[-1], self.__n_classes)
            )
        else:
            layers = nn.Sequential(
                nn.Dropout(p=self.__dropout_ratio),
                nn.Linear(self.__hidden_size[-1], self.__n_classes)
            )

        if self.__n_classes > 1:
            self.__fc_out = nn.Sequential(*layers, nn.Softmax())
        else:
            self.__fc_out = nn.Sequential(*layers, nn.Sigmoid())

    @property
    def n_classes(self) -> int:
        return self.__n_classes

    @property
    def max_length(self) -> int:
        return self.__max_length

    @property
    def hidden_size(self) -> List[int]:
        return self.__hidden_size

    @property
    def dropout_ratio(self) -> float:
        return self.__dropout_ratio

    @property
    def embeddings_grad(self) -> bool:
        return self.__embeddings_grad

    def forward(self, input_ids, attention_mask):
        x = self.__embedding_model(input_ids=input_ids, attention_mask=attention_mask)

        h = x[0]
        x = h[:, 0]

        if self.__fc is not None:
            x = self.__fc(x)

        return self.__fc_out(x)

    def architecture(self) -> dict:

        return {
            "n_classes": self.__n_classes,
            "dropout_ratio": self.__dropout_ratio,
            "max_length": self.__max_length,
            "embeddings_grad": self.__embeddings_grad,
            "hidden_size": " | ".join(map(str, self.__hidden_size[1:]))
        }


def save_model_state(model: torch.nn.Module) -> dict:
    """
    Function that generates the content of a JSON file to save a trained model.
		
		Args: 
            model (torch.nn.Module): deep learning model

		Returns: 
			Dictionary containing model's architecture and weights
    """
    arch = model.architecture()
    arch["state_dict"] = model.state_dict()
    return arch


def load_model_state(checkpoint: dict, embedding_model: AutoModel) -> nn.Module:
    """
    Function that loads a saved model.
		
		Args: 
            checkpoint (dict): dictionary containing model's architecture and weights
            embedding_model (transformers.AutoModel) representing a pre-trained BERT model

		Returns: 
			Deep learning model in evaluation mode
    """
    state_dict = checkpoint["state_dict"]
    del checkpoint["state_dict"]

    checkpoint["hidden_size"] = [int(x) for x in checkpoint["hidden_size"].split(" | ")]
    model = BERTClassifier(**checkpoint, embedding_model=embedding_model)

    model.load_state_dict(state_dict)
    model.eval()

    return model