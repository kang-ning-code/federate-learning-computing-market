#!/usr/bin/python3


from brownie import ComputingMarket, accounts

deploy_setting = {
    "batch_size": 10,
    "learning_rate": "0.01",
    "epochs": 5,
    "n_participators": 10,
}


def deploy_contract():
    setting_list = [v for v in deploy_setting.values()]
    contract = ComputingMarket.deploy(
        "Test Model Name", setting_list, {'from': accounts[0]})
    return contract
