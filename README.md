# Transfer in Reinforcement Learning

In this project we attempt transfer learning between different input environments for Reinforcement Learning (RL) Agents in Atari Games. By learning a mapping from RGB screen image inputs to the Atari machine RAM input, we show that we take less than half the time to train an RL agent than on the original RGB screen input. We experiment with feed-forward neural networks, Convolutional neu- ral networks, as well as LSTMs to learn the mappings and get a best mean squared error of 0.37% of the RAM mappings using a Con- volutional LSTM. We apply trained RAM input RL agents under the RGB environments by feeding the mapped inputs to the agents. The best agent accumulates 3.29 average rewards per episode using the CNN mapping, which is 27.5% of the rewards accumulated under the RAM environments.

The OpenAI Gym provides a set of Atari games environments with two input settings: screen frames in the form of RGB images, and RAM (a 256 byte array containing the state of the game). 
There is no direct way to transform one input to another, since RAM potentially contains state information that is not 
represented in the pixels. We are interested in developing methods that transfer the policies learned 
from one input setting to the same game with different input setting. 
Essentially we want to model a mapping between the two settings such that transferring learned knowledge is possible. 
We will be using the openAI gym to get the Atari games (https://gym.openai.com/). One game we are considering trying transfer in is Amidar v0 to Amidar - ram v0.

The complete project details are in `Report.pdf`. 

## Implementation and Usages

1. `new_dqn.py` defines an dqn agent that records both RGB and RAM in callbacks during training.

    - Run `python examples/save_atari_observations.py [--game Breakout]` to train a model on RGB and save observations.
    
    - To specify the frequency of saving observations, run with options:
    
        `python examples/save_atari_observations.py --save_every_episode 10 --save_every_step 20`
        
    - Currently we only support Breakout and Seaquest
 
 2. Run `python examples/new_dqn_atari.py --help` to see how to use transfer learning models to train RL agent.
    
    eg. `python3 examples/new_dqn_atari.py --mode transfer --game Breakout --transfer_model ff`
 
 3. Go to rgb2ram directory, run `python3 train.py` to learn mapping between RGB and RAM
    
# Deep Reinforcement Learning for Keras

[![Build Status](https://api.travis-ci.org/keras-rl/keras-rl.svg?branch=master)](https://travis-ci.org/keras-rl/keras-rl)
[![Documentation](https://readthedocs.org/projects/keras-rl/badge/)](http://keras-rl.readthedocs.io/)
[![License](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/keras-rl/keras-rl/blob/master/LICENSE)
[![Join the chat at https://gitter.im/keras-rl/Lobby](https://badges.gitter.im/keras-rl/Lobby.svg)](https://gitter.im/keras-rl/Lobby)

<table>
  <tr>
    <td><img src="/assets/breakout.gif?raw=true" width="200"></td>
    <td><img src="/assets/cartpole.gif?raw=true" width="200"></td>
    <td><img src="/assets/pendulum.gif?raw=true" width="200"></td>
  </tr>
</table>

## What is it?

`keras-rl` implements some state-of-the art deep reinforcement learning algorithms in Python and seamlessly integrates with the deep learning library [Keras](http://keras.io).

Furthermore, `keras-rl` works with [OpenAI Gym](https://gym.openai.com/) out of the box. This means that evaluating and playing around with different algorithms is easy.

Of course you can extend `keras-rl` according to your own needs. You can use built-in Keras callbacks and metrics or define your own.
Even more so, it is easy to implement your own environments and even algorithms by simply extending some simple abstract classes. Documentation is available [online](http://keras-rl.readthedocs.org).


## Installation

- Install Keras-RL from Pypi (recommended):

```
pip install keras-rl
```

- Install from Github source:

```
git clone https://github.com/keras-rl/keras-rl.git
cd keras-rl
python setup.py install
```

For atari example you will also need:

- **Pillow**: `pip install Pillow`
- **gym[atari]**: Atari module for gym. Use `pip install gym[atari]`


## Citing

If you use `keras-rl` in your research, you can cite it as follows:

```bibtex
@misc{plappert2016kerasrl,
    author = {Matthias Plappert},
    title = {keras-rl},
    year = {2016},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/keras-rl/keras-rl}},
}
```

## References

1. _Playing Atari with Deep Reinforcement Learning_, Mnih et al., 2013
2. _Human-level control through deep reinforcement learning_, Mnih et al., 2015
3. _Deep Reinforcement Learning with Double Q-learning_, van Hasselt et al., 2015
4. _Continuous control with deep reinforcement learning_, Lillicrap et al., 2015
5. _Asynchronous Methods for Deep Reinforcement Learning_, Mnih et al., 2016
6. _Continuous Deep Q-Learning with Model-based Acceleration_, Gu et al., 2016
7. _Learning Tetris Using the Noisy Cross-Entropy Method_, Szita et al., 2006
8. _Deep Reinforcement Learning (MLSS lecture notes)_, Schulman, 2016
9. _Dueling Network Architectures for Deep Reinforcement Learning_, Wang et al., 2016
10. _Reinforcement learning: An introduction_, Sutton and Barto, 2011
11. _Proximal Policy Optimization Algorithms_, Schulman et al., 2017
