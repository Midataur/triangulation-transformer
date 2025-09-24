from torch import argmax
import torch
import os

VOCABULARY = ['\n', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

def tokenize(item, category=None):
    return VOCABULARY.index(item)

def tokenize_list(cat_data, category=None):
    return [tokenize(x, category) for x in cat_data]

def calculate_accuracy(output, target):
    # targets is a (B) tensor of integers that have the index of the correct class
    # we need to see if the max logit is at the right index

    # cross entropy case
    if len(output.shape) > 1:
        return (argmax(output, dim=1) == target).float().mean()
    
    # bce case
    return (output.round() == target).float().mean()

# returns a list of all the data
def load_data(config):
    data = []

    for filename in os.listdir(f"{config.PATH}/data"):
        with open(f"{config.PATH}/data/{filename}") as file:
            data += file.readlines()

    return data

def reshape_outputs(outputs, targets):
    B, T, C = outputs.shape
    
    return outputs.reshape(B*T, C), targets.reshape(B*T)