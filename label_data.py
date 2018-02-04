import cPickle as pickle
import torch

checkpoint = torch.load("checkpoint.pth.tar")
model.load_state_dict(checkpoint["state_dict"])
labels = {datapoint: model(datapoint) for datapoint in data}
with open("distilled_labels.pickle", "wb") as distilled_label_file:
    pickle.dump(labels, distilled_label_file, protocol=pickle.HIGHEST_PROTOCOL)
