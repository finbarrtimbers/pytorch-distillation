import cPickle as pickle
import torch

def label_data(data_loader, checkpoint_filename):
    checkpoint = torch.load(checkpoint_filename)
    model.load_state_dict(checkpoint["state_dict"])
    labels = {i: model(datapoint) for i, datapoint in enumerate(data_loader)}
    with open(data_loader_name + "_distilled.pickle",
              "wb") as distilled_label_file:
        pickle.dump(labels, distilled_label_file,
                    protocol=pickle.HIGHEST_PROTOCOL)
