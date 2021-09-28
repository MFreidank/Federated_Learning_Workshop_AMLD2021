# Federated Learning Workshop AMLD2021

## Preparations

We've tried to keep this workshop light on the setup/preparations side by using 
[Google Colab](https://colab.research.google.com/) for a majority of the materials and exercises that we will cover. 

Therefore, the main requirement is access to a Google account which would be required 
to access the notebooks we prepared. 

Please ensure you have access to a google account you can use, or [create one](https://accounts.google.com/SignUp?hl=en)
in preparation for the session. 

Don't worry about studying the notebooks below, these will be covered as 
integral part of the workshop and will remain available to you afterwards as well.


## 1. Interactive Examples: torch Federated Learning
### 1.1 Simulating Federated Learning using torch

This example introduces basic components that are part of the Federated Learning workflow.
In the simulation we:
1. Train a baseline model on the CIFAR10 dataset.
2. Simulate a server & client and send a model from server to client to be trained.
3. Simulate a server & multiple clients and train a global model on multiple clients and perform FedAvg.

Please use the google colab notebook links below to access the relevant materials:

[PyTorch FL Simulation Notebook](https://colab.research.google.com/drive/1a1Ekw5jFs8eYOxAhsBRwFAt3WxEBsyf9)

## 2. Interactive Examples: Syft

### 2.1 Simple Syft Duet Intro

This example introduces basic primitives of a successful PySyft Duet Exchange: 
* establishing a duet connection between data owner and client
* safely providing information about available datasets as a data owner
* issuing access requests as a client/data scientist
* approving/denying requests for data access as a data owner

Please use the google colab notebook links below to access the relevant materials:

* Data Owner:
[Data Owner Notebook](https://colab.research.google.com/drive/1lPa95bboyd_4GTljn_PJtAgxra3l7T7e?usp=sharing)

* Data Scientist:
[Data Scientist Notebook](https://colab.research.google.com/drive/107zodT2X6rogAoYQSUprOQu-mkk7BPs3?usp=sharing)

### 2.2 Syft FL Example
This example showcases a basic federated learning exchange (with a data scientist modeling against 
data held by two separate data owners) using PySyft Duet. 

Please use the google colab notebook links below to access the relevant materials:

* Data Owner 1:
[Data Owner 1 Notebook](https://colab.research.google.com/drive/12pEcshA3eH55LeAWO_dmg5EflDQWpTlD?usp=sharing)

* Data Owner 2:
[Data Owner 2 Notebook](https://colab.research.google.com/drive/1c_O_4TfkKT2jKl5EtYKnxAwISEBefXzK?usp=sharing)

* Data Scientist:
[Data Scientist Notebook](https://colab.research.google.com/drive/1o8wOkrprb8ecKkkiU9AdL2NHxZwA6g4F?usp=sharing)
