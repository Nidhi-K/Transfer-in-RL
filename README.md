# Transfer in Reinforcement Learning

The complete project details are in `Report.pdf`. 

## Abstract
In this project we attempt transfer learning between different input environments for Reinforcement Learning (RL) Agents in Atari Games. By learning a mapping from RGB screen image inputs to the Atari machine RAM input, we show that we take less than half the time to train an RL agent than on the original RGB screen input. We experiment with feed-forward neural networks, Convolutional neural networks, as well as LSTMs to learn the mappings and apply trained RAM input RL agents under the RGB environments by feeding the mapped inputs to the agents. 

## OpenAI Gym
The OpenAI Gym provides a set of Atari games environments with two input settings: screen frames in the form of RGB images, and RAM (a 256 byte array containing the state of the game). 
There is no direct way to transform one input to another, since RAM potentially contains state information that is not 
represented in the pixels. We are interested in developing methods that transfer the policies learned 
from one input setting to the same game with different input setting. 
Essentially we want to model a mapping between the two settings such that transferring learned knowledge is possible. 
We will be using the openAI gym to get the Atari games (https://gym.openai.com/).

![a](https://github.com/Nidhi-K/Blockchain-for-home-services/blob/master/blockchain%20workflow.png)

## Installation

- Keras-RL: `pip install keras-rl`
- Pillow: `pip install Pillow`
- Atari module for gym: `pip install gym[atari]`

## Usage

1. `new_dqn.py` defines an dqn agent that records both RGB and RAM in callbacks during training. We support the Breakout and Seaquest games.

    - Run `python examples/save_atari_observations.py [--game Breakout]` to train a model on RGB and save observations.
    - To specify the frequency of saving observations, run with options: `python examples/save_atari_observations.py --save_every_episode 10 --save_every_step 20`
    
 2. Run `python examples/new_dqn_atari.py --help` to see how to use transfer learning models to train RL agent.
    eg. `python3 examples/new_dqn_atari.py --mode transfer --game Breakout --transfer_model ff`
 
 3. Go to rgb2ram directory, run `python3 train.py` to learn mapping between RGB and RAM

## References 
This project extends the `keras-rl` repository (https://github.com/keras-rl/keras-rl), which has implemented several state-of-the art deep reinforcement learning algorithms in Python with keras. We use the Deep Q Learning (DQN) algorithm in this project.

