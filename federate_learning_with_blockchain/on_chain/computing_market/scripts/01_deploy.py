#!/usr/bin/python3

from brownie import ComputingMarket, accounts


def main():
    setting = {
        "batchSize":20,
        "learningRate":"0.01",
        "epochs":5,
        "nParticipators":10,
    }
    setting_list = [v for v in setting.values()]
    contract =  ComputingMarket.deploy("Test Model Name",setting_list,{'from': accounts[0]})
    return contract


