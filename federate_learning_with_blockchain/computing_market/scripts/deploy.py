#!/usr/bin/python3

from brownie import ComputingMarket, accounts

train_setting = {
        "batch_size":20,
        "learning_rate":"0.01",
        "epochs":5,
        "n_participators":10,
    }
def deploy_contract():
    setting_list = [v for v in train_setting.values()]
    contract =  ComputingMarket.deploy("Test Model Name",setting_list,{'from': accounts[0]})
    return contract


