### Training and Deploying a BERT Model using AWS Sagemaker

Have you ever wondered how you can train and deploy a custom model using AWS Sagemaker? If so, this repository helps you achieve your goal.

The basic idea here is to use transfer learning, in order to train a BERT model responsible for predicting whether a given portuguese sentence is humorous or not.

AWS Sagemaker supports PyTorch flavor to deploy your model, however it is a trade-off: you have to let it go some of the flexibility when developing your model. Since data scientist love to design their own solution because it really allows for creativity, it is important to manage to train and deploy your custom algorithm to AWS Sagemaker.

Hence, the following steps are essential to this process:

1. **Containerization**: `Dockerfile` allows us to build our container, installing all packages specified in `requirements.txt`, downloading our pre-trained BERT model and setting up the environment for AWS Sagemaker. Once our Docker image is properly built, we can push it to AWS ECR, from where we will read it when training the model;  

2. **Inference Server**:

* `nginx` is a soft layer which deals with HTTP requests and manages I/O inside and outside of the container in an efficient way;  

* `gunicorn` is a WSGI work server which runs several copies of the application and balances load between them;  

* `flask` is a simples web structure used in the inferece application. It allows us to respond to requests `/ping` and `/invocations` without the need to write too many lines of code.  

3. **Application**: When AWS Sagemaker runs a container, it is invoked with an argument of **train** or **serve**, depending on the nature of the task - training or deploy. 

* `api.py`: Algorithm's inference server;  

* `train`: Main train program, where model is trained and then its artifacts are properly saved;  

* `serve`: Wrapper containing inference server (this file can be applied to a variety of contexts);  

* `wsgi.py`: Initialization server for the inidividual workers;  

* `predictor.py`: Model's wrapper for inference, according to use cases and business rules;  

* `nginx.conf`: Master configuration server that manages several workers.


____________


@inproceedings{smallermdistilbert,
  title={Load What You Need: Smaller Versions of Mutlilingual BERT},
  author={Abdaoui, Amine and Pradel, Camille and Sigel, Grégoire},
  booktitle={SustaiNLP / EMNLP},
  year={2020}
}
