import os
from typing import List
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
import unidecode
from html import unescape

from model import EMBEDDING_MODEL_PATH, SEED, load_model_state
from evaluation_utils import get_predictions


np.random.seed(SEED)
torch.manual_seed(SEED)


class PredictorService(object):

    def __init__(self, model_path: str):
        """
        Trained model wrapper to predict instances during inferece serving time.
	
        Attributes:
            model_path (str) representing filepath to model artifact
                
        """
        self.__model_path = model_path

    @property
    def model_path(self):
        return self.__model_path

    @property
    def n_classes(self):
        if not hasattr(self, "__model"):
            return None
        return self.__model.n_classes

    def start(self):
        """
        Loading and instantiation of the trained model
        """
        if not hasattr(self, "__model"):
            self.__model, self.__embedding_tokenizer = self.__get_model()
        torch.set_num_threads(1)

    def __get_model(self) -> torch.nn.Module:
        checkpoint = torch.load(os.path.join(self.__model_path, "model.pth"))

        tokenizer_model = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_PATH)
        embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL_PATH)
        model = load_model_state(checkpoint, embedding_model)
        model.eval()
        return model, tokenizer_model

    def predict(self, inputs: str) -> List[float]:
        """
        Function that generates prediction according to a given text input
            
            Args: 
                inputs (str): input raw text

            Returns: 
                List of predicted probabilities
        """
        if not hasattr(self, "__model"):
            self.__model, self.__embedding_tokenizer = self.__get_model()
        inputs = unescape(unidecode.unidecode(inputs).lower())
        return get_predictions(
            model=self.__model, tokenizer=self.__embedding_tokenizer,
            sentence=inputs, dynamic_input=True, gpu=False
        )