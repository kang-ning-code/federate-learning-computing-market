#!/usr/bin/python3


from brownie import ModelTrain, accounts

train_setting = {
     "batch_size": 10,
     "learning_rate": "0.01",
     "epochs": 2,
     "n_participator":5,
     "n_vote":3,
}
task_setting = {
    "task_description":"Federate Learning Computing Market Task",
    "model_description":"mnist_2nn",
    # "model_description":"emnist_2nn",
    "dataset_description":"MNIST",
}
def deploy_contract():
    # setting_list = [v for v in deploy_setting.values()]
    train = [v for v in train_setting.values()]
    task = [v for v in task_setting.values()]
    contract = ModelTrainTask.deploy(
        train,
        task,
        {"from":accounts[0]}
    )
    return contract
