# Based on AlphaZero, an AI Gomoku player
[Chinese](readme.md) English


![](https://pic4.zhimg.com/80/v2-1320f6469f11f9d5b72cc9f8fb65ec6b_720w.webp)

## Overview

Training Environment: AIStudio GPU 16G  
Framework: PaddlePaddle 2.0  
Language: Python 3.8  
Strategy Value Network: Consists of Convolutional Layers and Fully Connected Layers. More details can be found in the code.

### Function Overview

* This project is an implementation of the AlphaZero algorithm.
* Gomoku is relatively simple in terms of rules and has a small search space, so it does not require the use of residual networks used in AlphaGo Zero. Instead, it only uses convolutional layers and fully connected layers.
* To start training your own AI model, run "python train.py". To play against the AI or let the AI play against itself, run "python human_play.py". The demonstration of the 15x15 board with moves in the upper left 9x9 area is shown below:

![](https://ai-studio-static-online.cdn.bcebos.com/92e8a5e8b9824133ba63e27cb761ed4ee5a2d11766b34e6c89dd82b57b1770d2)

* The complete effect is provided in the package, which can be used in a PowerPoint presentation.

### Introduction

* AlphaZero: An algorithm proposed by DeepMind, trained through self-play without the need for expert data.
* Process: Self-play learning -> Neural network training -> Network evaluation
* Self-play generates a large number of game samples for neural network training through Monte Carlo Tree Search (MCTS).
* The training primarily focuses on the Strategy Value Network.

![](http://5b0988e595225.cdn.sohucs.com/images/20171024/9e3bb5aca2634e7f8f19dae40bb0a101)

* Structure of the Strategy Value Network: Consists of the Common Network Layer, Action Policy Network Layer, and State Value Network Layer.

* Related APIs can be found on the website [PaddleAPI](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html#:~:text=API%20%E6%96%87%E6%A1%A3%20%E6%AC%A2%E8%BF%8E%E4%BD%BF%E7%94%A8%E9%A3%9E%E6%A1%A8%E6%A1%86%E6%9E%B6,%28PaddlePaddle%29%2C%20PaddlePaddle%20%E6%98%AF%E4%B8%80%E4%B8%AA%E6%98%93%E7%94%A8%E3%80%81%E9%AB%98%E6%95%88%E3%80%81%E7%81%B5%E6%B4%BB%E3%80%81%E5%8F%AF%E6%89%A9%E5%B1%95%E7%9A%84%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A1%86%E6%9E%B6%EF%BC%8C%E8%87%B4%E5%8A%9B%E4%BA%8E%E8%AE%A9%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%8A%80%E6%9C%AF%E7%9A%84%E5%88%9B%E6%96%B0%E4%B8%8E%E5%BA%94%E7%94%A8%E6%9B%B4%E7%AE%80%E5%8D%95%E3%80%82)  

## Preparation:

  ### 1 MCTS based on Neural Network:

  General MCTS: Given a game state, perform N simulations. The main steps of the search process are Selection, Expansion, Simulation, and Backpropagation.

* The first step is Selection: This step starts from the root node and selects a "most promising child node" each time, generally using the UCT selection score, until it reaches a node with "unexpanded child nodes".

* The second step is Expansion: At the node with unexpanded child nodes found in the previous step, add an unexplored child node and initialize it.

* The third step is Simulation: Starting from the unexplored child node, use a simple strategy such as a quick move policy (Rollout policy) to play out the game until the end, resulting in a win or loss. The quick move policy is generally suitable for strategies that are fast

 but not very accurate. If this policy is slow, the result may be more accurate, but it will consume more time, leading to fewer simulations per unit time, which may not necessarily improve the strength.

* The fourth step is Backpropagation: It updates the MCTS tree structure with the cumulative result obtained from the Simulation. Note that in addition to updating the MCTS tree, the new node also needs to be updated with the win/loss history.

  ![](https://ai-studio-static-online.cdn.bcebos.com/73384055df364b44a49e7e206a9015790be7b3c0aa1942d0a4e57aa617fad087)
  Introduce some parameters:
  N(s, a): Record the number of visits of the edge;  
  W(s, a): Total action value;  
  Q(s, a): Average action value;  
  P(s, a): Prior probability of choosing this edge;  
  
* For the Selection part, MCTS searches along the tree, selecting the child branch with the highest Q + U value, until it reaches the end of the game or a leaf node of the MCTS that has not been visited.

$$
U(s, a) = c_{\rm{puct}} \cdot P(s, a) \cdot \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)}\\
a_t = \arg \max_a (Q(s_t, a) + U(s_t, a))
$$

* For the Expansion and Simulation parts, the neural network is used to predict the probabilities and values of possible child nodes at the leaf node state s. These possible new nodes are created in the MCTS, and their branch information is initialized as follows:

$$
N(s_L, a) = 0, W(s_L, a) = 0, Q(s_L, a) = 0, P(s_L, a) = P_a
$$

* For the Backpropagation part, the information of the new leaf node branch is accumulated and updated on the ancestor node branches. Starting from leaf node L and backpropagating to the root node, update the upper-level branch data in sequence:

$$
\displaylines{
N(s_t, a_t) = N(s_t, a_t) + 1\\
W(s_t, a_t) = W(s_t, a_t) + v\\
Q(s_t, a_t) = \frac{W(s_t, a_t)}{N(s_t, a_t)}}
$$

* After the search is completed, the model can select the MCTS move at the root node s.

$$
\pi(a|s) = \frac{N(s, a)^{\frac{1}{\tau}}}{\sum_b N(s, b)^{\frac{1}{\tau}}}
$$
  

$\tau$ is between 0 and 1. The closer it is to 1, the closer it is to the original MCTS sampling, and the closer it is to 0, the closer it is to a greedy strategy (the action with the maximum number of visits N). To avoid numerical abnormalities due to direct access to $\tau$, we can take the natural logarithm first and then use softmax to restore the probability.

```python
def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs
```

#### Key Points

* Through each simulation, MCTS relies on the neural network to explore along the most promising path (i.e., the path with the highest upper confidence bound). It accumulates values (Q), prior probabilities of moves (P), and the frequency of visiting nodes in this path, and uses the combination of these numbers to make decisions.
* In each simulation, MCTS explores as deep as possible until it encounters a game state that it has never seen before. In this case, it evaluates the state using the neural network.
* MCTS cleverly combines the search tree and the neural network, optimizing the neural network parameters through the MCTS search tree, and then using the optimized neural network to guide MCTS search.
Code: `mcts_pure.py`

### 2 Strategy Value Network
Code: `policy_value_net_paddlepaddle.py`

### 3 User Interface
An elegant UI is created for the game, inspired by open-source projects, using the pygame library.
For more details, please refer to the content of game.py, which includes some screenshots and comments.

### 4 Training

Main File: `train.py`

### 5 Actual Effect

As mentioned earlier, enter the following command under the Jupyter Notebook Code:

```shell
!pip install pygame
# File Name
%cd AlphaZero_Demo
!python train.py
```

The AI will perform self-play and gradually improve its performance. After training for a while, switch to human_play.py to play against the AI. The AI will be a friendly player and not lose easily (as shown in the above image).

This completes the entire project, which is suitable for beginners. The network is not too complicated, mainly based on the AlphaZero algorithm, with convolutional and fully connected layers, and ReLU activation.

## Project Links

<https://github.com/Git-lys/NNHomework>  
<https://aistudio.baidu.com/aistudio/projectdetail/4516264>

## References

<https://github.com/junxiaosong/AlphaZero_Gomoku>


$$
loss=(z-v)^2-\pi^T*log(p)+c||\theta||^2
$$

