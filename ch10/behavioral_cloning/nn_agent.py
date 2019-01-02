#!/usr/bin/env python
from __future__ import print_function

import os

import gym
import numpy as np
import torch

from ch10.behavioral_cloning.train \
    import \
    data_transform, \
    available_actions, \
    build_network, \
    DATA_DIR, MODEL_FILE


def nn_agent_play(model, device):
    """Let the agent play"""

    env = gym.make('CarRacing-v0')

    # use ESC to exit
    global human_wants_exit
    human_wants_exit = False

    def key_press(key, mod):
        """Capture ESC key"""
        global human_wants_exit
        if key == 0xff1b:  # escape
            human_wants_exit = True

    # initialize environment
    state = env.reset()
    env.unwrapped.viewer.window.on_key_press = key_press

    while 1:
        env.render()

        state = np.moveaxis(state, 2, 0)  # channel first image
        state = torch.from_numpy(np.flip(state, axis=0).copy())  # numpy to tensor
        state = data_transform(state)  # apply transformations
        state = state.unsqueeze(0)  # add additional dimension
        state = state.to(device)  # transfer to GPU

        # forward
        with torch.set_grad_enabled(False):
            outputs = model(state)

        normalized = torch.nn.functional.softmax(outputs, dim=1)  # softmax

        # translate from net output to env action
        max_action = np.argmax(normalized.cpu().numpy()[0])
        action = available_actions[max_action]

        # adjust brake power
        if action[2] != 0:
            action[2] = 0.4

        state, _, terminal, _ = env.step(action)  # one step

        if terminal:
            state = env.reset()

        if human_wants_exit:
            env.close()
            return
