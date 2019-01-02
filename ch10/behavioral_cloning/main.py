import os

import torch

from ch10.behavioral_cloning.nn_agent import nn_agent_play
from ch10.behavioral_cloning.train import DATA_DIR, MODEL_FILE, build_network, train

if __name__ == '__main__':
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    m = build_network()
    m.load_state_dict(torch.load(os.path.join(DATA_DIR, MODEL_FILE)))
    m.eval()
    m = m.to(dev)
    train(m, dev)
    nn_agent_play(m, dev)
