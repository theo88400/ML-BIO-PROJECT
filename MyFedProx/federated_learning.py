from copy import deepcopy
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

"""
This file contains all the methods used for the federated learning with pytorch model
"""


def difference_models_norm_2(model_1, model_2) -> float:
    """Return the norm 2 difference between the two model parameters
    """
    tensor_1=list(model_1.parameters())
    tensor_2=list(model_2.parameters())
    
    norm=sum([torch.sum((tensor_1[i]-tensor_2[i])**2) 
        for i in range(len(tensor_1))])
    
    return norm

def train_step(model, model_0, mu:int, optimizer, train_data, loss_f):
    """Train `model` on one epoch of `train_data`"""

    total_loss=0

    for idx, (features,labels) in enumerate(train_data):

        optimizer.zero_grad()

        predictions= model(features)

        loss=loss_f(predictions,labels)
        loss+=mu/2*difference_models_norm_2(model,model_0)
        total_loss+=loss

        loss.backward()
        optimizer.step()

    return total_loss/(idx+1)

def local_learning(model, mu:float, optimizer, train_data, epochs:int, loss_f):

    ### Copy model to a new variable ###
    model_cpy = deepcopy(model)

    for epoch in range(epochs):
        local_loss=train_step(model, model_cpy, mu, optimizer, train_data, loss_f)


    return float(local_loss.detach().numpy())

def loss_classifier(predictions,labels):

    m = nn.LogSoftmax(dim=1)
    loss = nn.NLLLoss(reduction="mean")

    return loss(m(predictions) ,labels.view(-1))


def loss_dataset(model, dataset, loss_f):
    """Compute the loss of `model` on `dataset`"""
    loss=0

    for idx,(features,labels) in enumerate(dataset):

        predictions= model(features)
        loss+=loss_f(predictions,labels)

    loss/=idx+1
    return loss


def accuracy_dataset(model, dataset):
    """Compute the accuracy of `model` on `dataset`"""

    correct=0

    for features,labels in iter(dataset):

        predictions= model(features)

        _,predicted=predictions.max(1,keepdim=True)

        correct+=torch.sum(predicted.view(-1,1)==labels.view(-1, 1)).item()

    accuracy = 100*correct/len(dataset.dataset)

    return accuracy

def set_to_zero_model_weights(model):
    """Set all the parameters of a model to 0"""

    for layer_weigths in model.parameters():
        layer_weigths.data.sub_(layer_weigths.data)

def average_models(model, clients_models_hist:list , weights:list):
    """Creates the new model of a given iteration with the models of the other
    clients"""

    new_model=deepcopy(model)
    set_to_zero_model_weights(new_model)

    for k,client_hist in enumerate(clients_models_hist):

        for idx, layer_weights in enumerate(new_model.parameters()):

            contribution=client_hist[idx].data*weights[k]
            layer_weights.data.add_(contribution)

    return new_model

def classical_training(model, train_set, test_set, n_iter:int, lr=10**-2, decay=1):
    """
    Perfom a classical training on the model with Binary Cross Entropy loss
    Args:
        - `model`: the model to train
        - `train_set`: the training set
        - `test_set`: the testing set
        - `n_iter`: number of iterations (epochs)
        - `lr`: learning rate
        - `decay`: to change the learning rate at each iteration

    Returns:
        - `model`: the trained model
        - `train_acc_hist`: the training accuracy history
        - `test_acc_hist`: the testing accuracy history
        - `loss_hist`: the loss history
    """

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_f=nn.BCELoss()

    train_acc_hist=[]
    test_acc_hist=[]
    loss_hist=[]

    for i in range(n_iter):
        with tqdm(train_set, unit="batch") as tepoch:
            for idx, (features,labels) in enumerate(train_set):

                optimizer.zero_grad()

                predictions= model(features)
                predictions = predictions.reshape(-1)
                loss=loss_f(predictions,labels.float())

                loss.backward()
                optimizer.step()
                tepoch.update(1)

        lr*=decay
        print(f'====> i: {i+1} Loss: {loss.item()}')
        train_acc = accuracy_dataset(model, train_set)
        print(f'====> i: {i+1} Train Accuracy: {train_acc}')
        test_acc = accuracy_dataset(model, test_set)
        print(f'====> i: {i+1} Test Accuracy: {test_acc}')

        train_acc_hist.append(train_acc)
        test_acc_hist.append(test_acc)
        loss_hist.append(loss.item())

    return model, train_acc_hist, test_acc_hist, loss_hist

def FedProx(model, training_sets:list, n_iter:int, testing_sets:list, mu=0,
    file_name="test", epochs=5, lr=10**-2, decay=1):
    """ all the clients are considered in this implementation of FedProx
    Parameters:
        - `model`: common structure used by the clients and the server
        - `training_sets`: list of the training sets. At each index is the
            training set of client "index"
        - `n_iter`: number of iterations the server will run
        - `testing_set`: list of the testing sets. If [], then the testing
            accuracy is not computed
        - `mu`: regularization term for FedProx. mu=0 for FedAvg
        - `epochs`: number of epochs each client is running
        - `lr`: learning rate of the optimizer
        - `decay`: to change the learning rate at each iteration

    returns :
        - `model`: the final global model
    """

    loss_f=loss_classifier

    #Variables initialization
    K=len(training_sets) #number of clients
    n_samples=sum([len(db.dataset) for db in training_sets])
    weights=([len(db.dataset)/n_samples for db in training_sets])
    print("Clients' weights:",weights)


    loss_hist=[[float(loss_dataset(model, dl, loss_f).detach())
        for dl in training_sets]]
    acc_hist=[[accuracy_dataset(model, dl) for dl in testing_sets]]
    server_hist=[[tens_param.detach().numpy()
        for tens_param in list(model.parameters())]]
    models_hist = []


    server_loss=sum([weights[i]*loss_hist[-1][i] for i in range(len(weights))])
    server_acc=sum([weights[i]*acc_hist[-1][i] for i in range(len(weights))])
    print(f'====> i: 0 Loss: {server_loss} Server Test Accuracy: {server_acc}')

    for i in range(n_iter):

        clients_params=[]
        clients_models=[]
        clients_losses=[]

        for k in range(K):

            local_model=deepcopy(model)
            # Define optimizer for local_model, don't forget the learning rate !
            local_optimizer=torch.optim.Adam(local_model.parameters(), lr=lr)
            # compute local_loss by performing learning steps on the current model
            local_loss= local_learning(local_model, mu, local_optimizer, training_sets[k], epochs, loss_f)

            clients_losses.append(local_loss)

            #GET THE PARAMETER TENSORS OF THE MODEL
            list_params=list(local_model.parameters())
            list_params=[tens_param.detach() for tens_param in list_params]
            clients_params.append(list_params)
            clients_models.append(deepcopy(local_model))


        #CREATE THE NEW GLOBAL MODEL
        # Create new global model by avering all locals models
        model = average_models(model, clients_params, weights)
        models_hist.append(clients_models)

        #COMPUTE THE LOSS/ACCURACY OF THE DIFFERENT CLIENTS WITH THE NEW MODEL
        loss_hist+=[[float(loss_dataset(model, dl, loss_f).detach())
            for dl in training_sets]]
        acc_hist+=[[accuracy_dataset(model, dl) for dl in testing_sets]]

        server_loss=sum([weights[i]*loss_hist[-1][i] for i in range(len(weights))])
        server_acc=sum([weights[i]*acc_hist[-1][i] for i in range(len(weights))])

        print(f'====> i: {i+1} Loss: {server_loss} Server Test Accuracy: {server_acc}')


        server_hist.append([tens_param.detach().cpu().numpy()
            for tens_param in list(model.parameters())])

        #DECREASING THE LEARNING RATE AT EACH SERVER ITERATION
        lr*=decay

    return model, loss_hist, acc_hist

def plot_acc_loss(title:str, acc_train_hist:list, acc_test_hist:list, loss_hist:list):
  """
  Plot history of loss and accuracy
  Args:
        - title: title of the plot
        - acc_train_hist: list of the training accuracy
        - acc_test_hist: list of the testing accuracy
        - loss_hist: list of the loss
  """
  fig, axs = plt.subplots(1, 2, figsize=(10, 5))
  axs[0].plot(loss_hist)
  axs[0].set_xlabel("iterations")
  axs[0].set_ylabel("loss")
  axs[1].plot(acc_train_hist, label="train")
  axs[1].plot(acc_test_hist, label="test")
  axs[1].legend()
  axs[1].set_xlabel("iterations")
  axs[1].set_ylabel("accuracy")
  fig.suptitle(title)
  fig.tight_layout()
  fig.show()
