#!/usr/bin/python3

from brownie import ComputingMarket, accounts


def deploy_contract():
    setting = {
        "batchSize":20,
        "learningRate":"0.01",
        "epochs":5,
        "nParticipators":10,
    }
    setting_list = [v for v in setting.values()]
    contract =  ComputingMarket.deploy("Test Model Name",setting_list,{'from': accounts[0]})
    print('call main')
    return contract


