#!/usr/bin/python3


from brownie import ComputingMarket, accounts

deploy_setting = {
    "batch_size": 10,
    "learning_rate": "0.01",
    "epochs": 2,
    "n_participator": 5,
    # "model_name":"resnet18",
    "model_name":"emnist_2nn",
    # "model_name":"mnist_cnn",
    "n_vote":3,
}


def deploy_contract():
    setting_list = [v for v in deploy_setting.values()]
    contract = ComputingMarket.deploy(
        "Test Model Name", setting_list, {'from': accounts[0]})
    return contract
