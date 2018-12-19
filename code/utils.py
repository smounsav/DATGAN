import torch


def logger(directory, filename, string):
    with open(directory + '/' + filename, 'a') as f:
        f.write(string + '\n')

def toOneHot(input, nclasses):
    onehotlabelssupport = torch.zeros(input.size(0), nclasses).to(input.device)
    onehottrainrealunlbllabels = onehotlabelssupport.scatter_(1, input.unsqueeze(1), 1)
    return onehottrainrealunlbllabels
