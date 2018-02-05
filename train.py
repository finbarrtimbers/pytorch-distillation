import argparse
import utils
import networks
import torch
from torch import autograd
import torch.optim as optim
import torch.nn.functional as F

def get_flags():
    arg_parser = argparse.ArgumentParser(
        description="Parser for distillation experiment.")
    arg_parser.add_argument("--dataset", type=str, default="Cifar10")
    arg_parser.add_argument("--cuda", type=bool, default=False)
    arg_parser.add_argument("--epochs", type=int, default=10)
    arg_parser.add_argument("--network", type=str, default="deep")
    args = arg_parser.parse_args()
    return args

def train(epoch, model, optimizer, train_loader,
          cuda=False, log_interval=10):
    model.train()
    train_len = len(train_loader.dataset)
    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = autograd.Variable(data), autograd.Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            train_perc = batch_idx * len(data) / len(train_loader) * 100.
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{train_len} "\
                  f"({train_perc}%)]\tLoss: {loss.data[0]:.6f}")

def eval(model, test_loader, cuda=False):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    perc_correct = 100. * correct / len(test_loader.dataset)
    print(f"\n {dataset_name} set: Average loss: {test_loss:.4f}, Accuracy: "\
          f"{correct}/{len(test_loader.dataset)} ({perc_correct}%)\n")

def main():
    flags = get_flags()
    if flags.network == "deep":
        net = networks.deep_resnet
    elif flags.network == "shallow":
        net = networks.shallow_resnet
    train_loader, val_loader, test_loader = utils.get_datasets(flags.dataset)
    if flags.cuda:
        net.cuda(), train.cuda(), val.cuda(), test.cuda()
    optimizer = optim.Adam(net.parameters(), lr=1e-5)
    for epoch in range(1, flags.epochs + 1):
        train(epoch, net, optimizer, train_loader, flags.cuda)
        eval(model, val_loader, "Validation", flags.cuda)
    eval(model, val, "Test", flags.cuda)
    torch.save(model ,"trained-deep-model.pt")

if __name__ == "__main__":
    main()
