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

## References 
This projct was extends the `keras-rl` repository (https://github.com/keras-rl/keras-rl), which has implemented several state-of-the art deep reinforcement learning algorithms in Python with keras. We use the Deep Q Learning (DQN) algorithm for our Transfer Learning in Atari games project.

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
