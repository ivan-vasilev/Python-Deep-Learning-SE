import os

import torch

from ch10.behavioral_cloning.nn_agent import nn_agent_play
from ch10.behavioral_cloning.train import \
    DATA_DIR, \
    MODEL_FILE, \
    build_network, \
    train

if __name__ == '__main__':
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_network()

    restore = False
    if restore:
        model_path = os.path.join(DATA_DIR, MODEL_FILE)
        model.load_state_dict(torch.load(model_path))

    model.eval()
    model = model.to(dev)
    train(model, dev)
    nn_agent_play(model, dev)
