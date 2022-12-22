# 基于AlphaZero的AI五子棋机器人

![](https://pic4.zhimg.com/80/v2-1320f6469f11f9d5b72cc9f8fb65ec6b_720w.webp)

## 概述

训练环境：AIStudio GPU 16G  
框架：paddlepaddle 2.0  
语言：python3.8  
策略价值网络 本质为卷积层+全连接层 后面会详细看Code

### 功能概述

* 本项目是AlphaZero算法的一个实现
* 五子棋的规则较为简单，落子空间也比较小，因此没有用到AlphaGo Zero中大量使用的残差网络，只使用了卷积层和全连接层
* 开始训练自己的AI模型，请运行“python train.py”；开始人机对战或者AI互搏，请运行“python human_play.py”，15x15棋盘左上角9x9范围下棋的效果展示：
* 图示效果单台电脑训练时间共耗时大概两天 若选用6*6 4子连直线获胜，大概需要训练两个小时左右 此处出于可观性没有保留

![](https://ai-studio-static-online.cdn.bcebos.com/92e8a5e8b9824133ba63e27cb761ed4ee5a2d11766b34e6c89dd82b57b1770d2)

* 效果图在包里有 可放到PPT里

### 简介

* AlphaZero：DeepMind提出的一种算法，在无需专家数据的前提下，通过自博弈的方式训练  
* 流程：自我博弈学习->神经网络训练->评估网络
* 通过蒙脱卡洛树完成自我博弈，并产生大量的棋局样本用于神经网络的训练
* 训练主要通过策略价值网络实现

![](http://5b0988e595225.cdn.sohucs.com/images/20171024/9e3bb5aca2634e7f8f19dae40bb0a101)

* 策略价值网络的结构：由公共网络层、行动策略网络层和状态价值网络层构成。
* 相关API可在网站查看<https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html#:~:text=API%20%E6%96%87%E6%A1%A3%20%E6%AC%A2%E8%BF%8E%E4%BD%BF%E7%94%A8%E9%A3%9E%E6%A1%A8%E6%A1%86%E6%9E%B6,%28PaddlePaddle%29%2C%20PaddlePaddle%20%E6%98%AF%E4%B8%80%E4%B8%AA%E6%98%93%E7%94%A8%E3%80%81%E9%AB%98%E6%95%88%E3%80%81%E7%81%B5%E6%B4%BB%E3%80%81%E5%8F%AF%E6%89%A9%E5%B1%95%E7%9A%84%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A1%86%E6%9E%B6%EF%BC%8C%E8%87%B4%E5%8A%9B%E4%BA%8E%E8%AE%A9%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%8A%80%E6%9C%AF%E7%9A%84%E5%88%9B%E6%96%B0%E4%B8%8E%E5%BA%94%E7%94%A8%E6%9B%B4%E7%AE%80%E5%8D%95%E3%80%82>  
Preparation:
1.基于神经网络的MCTS：
一般的MCTS：给定一个棋面，进行N次模拟，主要搜索过程：选择、拓展、仿真、回溯
* 第一步是选择(Selection):这一步会从根节点开始，每次都选一个“最值得搜索的子节点”，一般使用UCT选择分数最高的节点，直到来到一个“存在未扩展的子节点”的节点
* 第二步是扩展(Expansion)，在这个搜索到的存在未扩展的子节点，加上一个没有历史记录的子节点，初始化子节点
* 第三步是仿真(simulation)，从上面这个没有试过的着法开始，用一个简单策略比如快速走子策略（Rollout policy）走到底，得到一个胜负结果。快速走子策略一般适合选择走子很快可能不是很精确的策略。因为如果这个策略走得慢，结果虽然会更准确，但由于耗时多了，在单位时间内的模拟次数就少了，所以不一定会棋力更强，有可能会更弱。这也是为什么我们一般只模拟一次，因为如果模拟多次，虽然更准确，但更慢。
* 第四步是回溯(backpropagation), 将我们最后得到的胜负结果回溯加到MCTS树结构上。注意除了之前的MCTS树要回溯外，新加入的节点也要加上一次胜负历史记录
  ![](https://ai-studio-static-online.cdn.bcebos.com/73384055df364b44a49e7e206a9015790be7b3c0aa1942d0a4e57aa617fad087)
  引入一些参数：
  N(s,a) :记录边的访问次数；  
  W(s,a):  合计行动价值；  
  Q(s,a) :平均行动价值；  
  P(s,a) :选择该条边的先验概率；  
* 对于选择部分，通过UCT选择子分支 最终选择Q+U最大的子分支，一直走到棋局结束，或者走到了没有走到终局的MCTS的叶子节点。

$$
U(s,a)=c{put}P(s,a)\frac{\sqrt{\sum_b N(s,b)}}{1+N(s,a)}\\
a_t=arg_amax(Q(s_t,a)+U(s_t,a))
$$

* 对于扩展和仿真部分 对于叶子节点状态s,利用神经网络对于叶子节点做预测，得到当前叶子节点的各个可能的子节点位置的概率和对应的价值v,对于这些可能的新节点在MCTS中创建 并初始化其分支上保留的信息为：

$$
{N(s_L,a)=0,W(s_L,a)=0,Q(s_L,a)=0,P(s_L,a)=P_a}
$$

* 对于回溯部分 将新叶子节点分支的信息回溯累加到祖先节点分支上 从叶子节点L依次向根节点回溯，并依次更新上层分支数据：

$$
N(s_t,a_t)=N(s_t,a_t)+1\\
W(s_t,a_t)=W(s_t,a_t)+v\\
Q(s_t,a_t)=\frac{W(s_t,a_t)}{N(s_t,a_t)}
$$

* 完成搜索后，建立的模型可以在根节点s选择MCTS行棋分支
  $$
  \pi(a|s)=\frac{N(s,a)^{\frac{1}{\tau}}}{\sum_b N(s,b)^{\frac{1}{\tau}}}
  $$
  

$\tau$在0,1之间 越接近1表示越接近MCTS原始采样，越接近0表示越接近贪婪策略(最大访问次数N对应的动作) 基于直接访问\tau导致数值异常，可以先去自然对数，再通过softmax还原为概率

```python
def softmax(x):
    probs=np.exp(x-np.max(x))
    probs /=np.sum(probs)
    return probs
```

### !关键点

* 通过每一次模拟，MCTS依靠神经网络， 使用累计价值（Q）、神经网络给出的走法先验概率（P）以及访问对应节点的频率这些数字的组合，沿着最有希望获胜的路径（换句话说，也就是具有最高置信区间上界的路径）进行探索。
* 在每一次模拟中，MCTS会尽可能向纵深进行探索直至遇到它从未见过的盘面状态，在这种情况下，它会通过神经网络来评估该盘面状态的优劣
* 巧妙了使用MCTS搜索树和神经网络一起，通过MCTS搜索树优化神经网络参数，反过来又通过优化的神经网络指导MCTS搜索。
Code:

```python
'''mcts_pure.py'''
#  蒙特卡罗树搜索（MCTS）的纯实现

import numpy as np
import copy
from operator import itemgetter


def rollout_policy_fn(board):
    """在首次展示阶段使用策略方法的粗略,快速的版本."""
    # 初次展示时使用随机方式
    action_probs = np.random.rand(len(board.availables))
    return zip(board.availables, action_probs)


def policy_value_fn(board):
    """
    接受状态并输出（动作，概率）列表的函数元组和状态的分数"""
    # 返回统一概率和0分的纯MCTS
    action_probs = np.ones(len(board.availables)) / len(board.availables)
    return zip(board.availables, action_probs), 0


class TreeNode(object):
    """MCTS树中的节点。 每个节点都跟踪自己的值Q，
       先验概率P及其访问次数调整的先前得分u。
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # 从动作到TreeNode的映射
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        """通过创建新子项来展开树。
        action_priors：一系列动作元组及其先验概率根据策略函数.
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """在子节点中选择能够提供最大行动价值Q的行动加上奖励u（P）。
           return：（action，next_node）的元组
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """从叶节点评估中更新节点值
        leaf_value: 这个子树的评估值来自从当前玩家的视角
        """
        # 统计访问次数
        self._n_visits += 1
        # 更新Q值,取对于所有访问次数的平均数
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """就像调用update（）一样，但是对所有祖先进行递归应用。
        """
        # 如果它不是根节点，则应首先更新此节点的父节点。
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """计算并返回此节点的值。它是叶评估Q和此节点的先验的组合
            调整了访问次数，u。
            c_puct：控制相对影响的（0，inf）中的数字，该节点得分的值Q和先验概率P.
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """
        检查叶节点（即没有扩展的节点）。
        """
        return self._children == {}

    def is_root(self):
        """
        检查根节点
        """
        return self._parent is None


class MCTS(object):
    """对蒙特卡罗树搜索的一个简单实现"""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """
        policy_value_fn：一个接收板状态和输出的函数
          （动作，概率）元组列表以及[-1,1]中的分数
          （即来自当前的最终比赛得分的预期值玩家的观点）对于当前的玩家。
         c_puct：（0，inf）中的数字，用于控制探索的速度 收敛于最大值政策。 更高的价值意味着依靠先前的更多。
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        """
        从根到叶子运行单个播出，获取值
         叶子并通过它的父母传播回来。
         State已就地修改，因此必须提供副本。
        """
        node = self._root
        while (1):
            if node.is_leaf():
                break
            # 贪心算法选择下一步行动
            action, node = node.select(self._c_puct)
            state.do_move(action)

        action_probs, _ = self._policy(state)
        # 查询游戏是否终结
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        # 通过随机的rollout评估叶子结点
        leaf_value = self._evaluate_rollout(state)
        # 在本次遍历中更新节点的值和访问次数
        node.update_recursive(-leaf_value)

    def _evaluate_rollout(self, state, limit=1000):
        """使用推出策略直到游戏结束，
      如果当前玩家获胜则返回+1，如果对手获胜则返回-1，
     如果是平局则为0。
        """
        player = state.get_current_player()
        for i in range(limit):
            end, winner = state.game_end()
            if end:
                break
            action_probs = rollout_policy_fn(state)
            max_action = max(action_probs, key=itemgetter(1))[0]
            state.do_move(max_action)
        else:
            # 如果没有从循环中断，请发出警告。
            print("WARNING: rollout reached move limit")
        if winner == -1:  # tie
            return 0
        else:
            return 1 if winner == player else -1

    def get_move(self, state):
        """按顺序运行所有播出并返回访问量最大的操作。
     state：当前的比赛状态
     return ：所选操作
        """
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)
        return max(self._root._children.items(),
                   key=lambda act_node: act_node[1]._n_visits)[0]

    def update_with_move(self, last_move):
        """保留我们已经知道的关于子树的信息
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    """基于MCTS的AI玩家"""

    def __init__(self, c_puct=5, n_playout=2000):
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board):
        sensible_moves = board.availables
        if len(sensible_moves) > 0:
            move = self.mcts.get_move(board)
            self.mcts.update_with_move(-1)
            return move
        else:
            print("棋盘已满")

    def __str__(self):
        return "MCTS {}".format(self.player)


'''mcts_alphagoZero.py'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  蒙特卡罗树搜索AlphaGo Zero形式，使用策略值网络引导树搜索和评估叶节点

import numpy as np
import copy

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

'''树节点'''
class TreeNode(object):

    #每个节点跟踪其自身的值Q，先验概率P及其访问次数调整的先前得分u。
    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # 从动作到TreeNode的映射
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p
    '''创建新子项的方式展开树'''
    def expand(self, action_priors):
        """通过创建新子项来展开树。
         action_priors：一系列动作元组及其先验概率根据策略函数.
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)
    '''子节点中筛选最大Q给予增益'''
    def select(self, c_puct):
        """在子节点中选择能够提供最大行动价值Q的行动加上奖金u（P）。
        return：（action，next_node）的元组
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))
    '''通过叶节点评估更新节点值'''
    def update(self, leaf_value):
        """从叶节点评估中更新节点值
        leaf_value: 这个子树的评估值来自从当前玩家的视角
        """
        # 统计访问次数
        self._n_visits += 1
        # 更新Q值,取对于所有访问次数的平均数
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits
    '''对所有祖先进行递归update()'''
    def update_recursive(self, leaf_value):
        """就像调用update（）一样，但是对所有祖先进行递归应用。
        """
        # 如果它不是根节点，则应首先更新此节点的父节点。
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)
    '''计算并返回节点值'''
    def get_value(self, c_puct):
        """计算并返回此节点的值。它是叶评估Q和此节点的先验的组合
        调整了访问次数，u。
        c_puct：控制相对影响的（0，inf）中的数字，该节点得分的值Q和先验概率P.
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u
    '''检查叶节点'''
    def is_leaf(self):
        return self._children == {}
    '''检查根节点'''
    def is_root(self):
        return self._parent is None


class MCTS(object):
    """对蒙特卡罗树搜索的一个简单实现"""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):

        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        """从根到叶子运行单个播出，获取值
         叶子并通过它的父母传播回来。
         State已就地修改，因此必须提供副本。
        """
        node = self._root
        while(1):
            if node.is_leaf():
                break
            # 贪心算法选择下一步行动
            action, node = node.select(self._c_puct)
            state.do_move(action)

        # 使用网络评估叶子，该网络输出（动作，概率）元组p的列表以及当前玩家的[-1,1]中的分数v。
        action_probs, leaf_value = self._policy(state)
        # 查看游戏是否结束
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        else:
            # 对于结束状态,将叶子节点的值换成"true"
            if winner == -1:  # tie
                leaf_value = 0.0
            else:
                leaf_value = (
                    1.0 if winner == state.get_current_player() else -1.0
                )

        # 在本次遍历中更新节点的值和访问次数
        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp=1e-3):
        """按顺序运行所有播出并返回可用的操作及其相应的概率。
        state: 当前游戏的状态
        temp: 介于(0,1]之间的临时参数控制探索的概率
        """
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        # 根据根节点处的访问计数来计算移动概率
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move):
        """在当前的树上向前一步，保持我们已经知道的关于子树的一切.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    """基于MCTS的AI玩家"""

    def __init__(self, policy_value_function,
                 c_puct=5, n_playout=2000, is_selfplay=0):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, temp=1e-3, return_prob=0):
        sensible_moves = board.availables
        # 像alphaGo Zero论文一样使用MCTS算法返回的pi向量
        move_probs = np.zeros(board.width*board.height)
        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(board, temp)
            move_probs[list(acts)] = probs
            if self._is_selfplay:
                # 添加Dirichlet Noise进行探索（自我训练所需）
                move = np.random.choice(
                    acts,
                    p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                )
                # 更新根节点并重用搜索树
                self.mcts.update_with_move(move)
            else:
                # 使用默认的temp = 1e-3，它几乎相当于选择具有最高概率的移动
                move = np.random.choice(acts, p=probs)
                # 重置根节点
                self.mcts.update_with_move(-1)

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("棋盘已满")

    def __str__(self):
        return "MCTS {}".format(self.player)

#在训练过程中会分别调用这两个文件下的类的子函数
```

2.策略价值网络
Code：

```python
'''policy_value_net_paddlepaddle.py
    策略价值网络
'''

class Net(paddle.nn.Layer):
    def __init__(self,board_width, board_height):
        super(Net, self).__init__()
        self.board_width = board_width
        self.board_height = board_height
        # 公共网络层
        
        '''
        nn.Conv2D:
        in_channel输入通道数
        out_channel输出通道数
        kernel_size卷积核大小
        padding填充
        '''
        self.conv1 = nn.Conv2D(in_channels=4,out_channels=32,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2D(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.conv3 = nn.Conv2D(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        # 行动策略网络层
        self.act_conv1 = nn.Conv2D(in_channels=128,out_channels=4,kernel_size=1,padding=0)
        self.act_fc1 = nn.Linear(4*self.board_width*self.board_height,
                                 self.board_width*self.board_height)
        #状态价值网络层
        self.val_conv1 = nn.Conv2D(in_channels=128,out_channels=2,kernel_size=1,padding=0)
        self.val_fc1 = nn.Linear(2*self.board_width*self.board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)
    #推理，即激活函数
     def forward(self, inputs):
        # 公共网络层 
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # 行动策略网络层
        x_act = F.relu(self.act_conv1(x))
        x_act = paddle.reshape(
                x_act, [-1, 4 * self.board_height * self.board_width])
        
        x_act  = F.log_softmax(self.act_fc1(x_act))        
        # 状态价值网络层
        x_val  = F.relu(self.val_conv1(x))
        x_val = paddle.reshape(
                x_val, [-1, 2 * self.board_height * self.board_width])
        x_val = F.relu(self.val_fc1(x_val))
        x_val = F.tanh(self.val_fc2(x_val))

        return x_act,x_val

class PolicyValueNet():
    """策略&值网络 """
    def __init__(self, board_width, board_height,
                 model_file=None, use_gpu=True):
        self.use_gpu = use_gpu
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-3  # coef of l2 penalty
        

        self.policy_value_net = Net(self.board_width, self.board_height)        
        
        self.optimizer  = paddle.optimizer.Adam(learning_rate=0.02,
                                parameters=self.policy_value_net.parameters(), weight_decay=self.l2_const)
                                     

        if model_file:
            net_params = paddle.load(model_file)
            self.policy_value_net.set_state_dict(net_params)

     def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        
        """
        legal_positions = board.availables
        current_state = np.ascontiguousarray(board.current_state().reshape(
                -1, 4, self.board_width, self.board_height)).astype("float32")
        
        # print(current_state.shape)
        current_state = paddle.to_tensor(current_state)
        log_act_probs, value = self.policy_value_net(current_state)
        act_probs = np.exp(log_act_probs.numpy().flatten())
        
        act_probs = zip(legal_positions, act_probs[legal_positions])
        # value = value.numpy()
        return act_probs, value.numpy()

    def train_step(self, state_batch, mcts_probs, winner_batch, lr=0.002):
        """perform a training step"""
        # wrap in Variable
        state_batch = paddle.to_tensor(state_batch)
        mcts_probs = paddle.to_tensor(mcts_probs)
        winner_batch = paddle.to_tensor(winner_batch)

        # zero the parameter gradients
        self.optimizer.clear_gradients()
        # set learning rate
        self.optimizer.set_lr(lr)                             
        # forward
        log_act_probs, value = self.policy_value_net(state_batch)
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        value = paddle.reshape(x=value, shape=[-1])
        value_loss = F.mse_loss(input=value, label=winner_batch)
        policy_loss = -paddle.mean(paddle.sum(mcts_probs*log_act_probs, axis=1))
        loss = value_loss + policy_loss
        # backward and optimize
        loss.backward()
        self.optimizer.minimize(loss)
        # calc policy entropy, for monitoring only
        entropy = -paddle.mean(
                paddle.sum(paddle.exp(log_act_probs) * log_act_probs, axis=1)
                )
        return loss.numpy(), entropy.numpy()[0]    

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        paddle.save(net_params, model_file)

```

3.UI界面
为游戏创建了较为elegant的UI界面 借鉴与开源
调用pygame库
详细见game.py内容 文档有部分截图和注释
4.训练
主文件：train.py

```python

#train.py
#  对于五子棋的AlphaZero的训练的实现

from __future__ import print_function
import random
import numpy as np
import os
from collections import defaultdict, deque
from game import Board, Game_UI
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaGoZero import MCTSPlayer
from policy_value_net_paddlepaddle import PolicyValueNet  # paddlepaddle
import paddle


class TrainPipeline():
    def __init__(self, init_model=None, is_shown = 0):
        # 五子棋逻辑和棋盘UI的参数
        self.board_width = 9  ###为了更快的验证算法，可以调整棋盘大小为(8x8) ，(6x6)
        self.board_height = 9
        self.n_in_row = 5
        self.board = Board(width=self.board_width,
                           height=self.board_height,
                           n_in_row=self.n_in_row)
        self.is_shown = is_shown
        self.game = Game_UI(self.board, is_shown)
        # 训练参数
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # 基于KL自适应地调整学习率
        self.temp = 1.0  # 临时变量
        self.n_playout = 400  # 每次移动的模拟次数
        self.c_puct = 5
        self.buffer_size = 10000 #经验池大小 10000
        self.batch_size = 512  # 训练的mini-batch大小 512
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5  # 每次更新的train_steps数量
        self.kl_targ = 0.02
        self.check_freq = 100  #评估模型的频率，可以设置大一些比如500
        self.game_batch_num = 1500
        self.best_win_ratio = 0.0
        # 用于纯粹的mcts的模拟数量，用作评估训练策略的对手
        self.pure_mcts_playout_num = 1000
        if init_model:
            # 从初始的策略价值网开始训练
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height,
                                                   model_file=init_model)
        else:
            # 从新的策略价值网络开始训练
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height)
        # 定义训练机器人
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)

    def get_equi_data(self, play_data):
        """通过旋转和翻转来增加数据集
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                # 逆时针旋转
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_porb.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # 水平翻转
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        """收集自我博弈数据进行训练"""
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player, temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # 增加数据
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

    def policy_update(self):
        """更新策略价值网络"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        
        # print(np.array( state_batch).shape )
        state_batch= np.array( state_batch).astype("float32")
        
        mcts_probs_batch = [data[1] for data in mini_batch]
        mcts_probs_batch= np.array( mcts_probs_batch).astype("float32")
        
        winner_batch = [data[2] for data in mini_batch]
        winner_batch= np.array( winner_batch).astype("float32")
        
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                state_batch,
                mcts_probs_batch,
                winner_batch,
                self.learn_rate * self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                                axis=1)
                         )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # 自适应调节学习率
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))
        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        return loss, entropy

    def policy_evaluate(self, n_games=10):
        """
        通过与纯的MCTS算法对抗来评估训练的策略
        注意：这仅用于监控训练进度
        """
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout)
        pure_mcts_player = MCTS_Pure(c_puct=5,
                                     n_playout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(current_mcts_player,
                                          pure_mcts_player,
                                          start_player=i % 2)
            win_cnt[winner] += 1
        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
            self.pure_mcts_playout_num,
            win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio

    def run(self):
        """开始训练"""
        root = os.getcwd()

        dst_path = os.path.join(root, 'dist')

        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

        try:
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                print("batch i:{}, episode_len:{}".format(
                    i + 1, self.episode_len))
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                    print("loss :{}, entropy:{}".format(loss, entropy))
                if (i + 1) % 50 == 0:
                    self.policy_value_net.save_model(os.path.join(dst_path, 'current_policy_step.model'))
                # 检查当前模型的性能，保存模型的参数
                if (i + 1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i + 1))
                    win_ratio = self.policy_evaluate()
                    self.policy_value_net.save_model(os.path.join(dst_path, 'current_policy.model'))
                    if win_ratio > self.best_win_ratio:
                        print("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        # 更新最好的策略
                        self.policy_value_net.save_model(os.path.join(dst_path, 'best_policy.model'))
                        if (self.best_win_ratio == 1.0 and
                                    self.pure_mcts_playout_num < 8000):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
        device = paddle.get_device()               
        paddle.set_device(device)
        is_shown = 0
        # model_path = 'dist/best_policy.model'
        model_path = 'dist/current_policy.model'

        training_pipeline = TrainPipeline(model_path, is_shown)
        # training_pipeline = TrainPipeline(None, is_shown)
        training_pipeline.run()

```

5.实际效果
正如前文所述 通过在Jupyter Notebook Code下输入如下指令

```
!pip install pygame
#文件名
%cd AlphaZero_Demo
!python train.py
```

实现自我博弈并逐渐训练 每步会自动评估网络并保留参数 截图如下
![](https://ai-studio-static-online.cdn.bcebos.com/f42129e150a44f26adb1f18597347fdeb013709e7e0d412eba49987fe6f3d8a6)

训练一段时间后 改为human_play.py即可实现陪你下棋 并且不会怎么输的小AI机器人(效果图如前面)

```python
#human_play.py
#  人机博弈模块

from __future__ import print_function
from game import Board, Game_UI
from mcts_alphaGoZero import MCTSPlayer
from policy_value_net_paddlepaddle import PolicyValueNet # paddlepaddle
import paddle

class Human(object):
    """
    人类玩家
    """
    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

def run():
    n = 5  # 获胜的条件(5子连成一线)
    width, height = 9, 9  # 棋盘最大可以设置为15x15
    model_file = 'dist/best_policy.model'  # 模型文件名称
    try:
        board = Board(width=width, height=height, n_in_row=n)  # 初始化棋盘
        game = Game_UI(board, is_shown=1)  # 创建游戏对象

        # ############### 人机对弈 ###################
        # 使用paddle加载训练好的模型policy_value_net
        best_policy = PolicyValueNet(width, height, model_file = model_file)
        mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)

        # 人类玩家,使用鼠标落子
        human = Human()

        # 首先为人类设置start_player = 0
        game.start_play_mouse(human, mcts_player, start_player=1, )

        # 机器自己博弈
        game.start_self_play(mcts_player,)
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    device = paddle.get_device()
    paddle.set_device(device)
    run()

```

以上为本项目全部内容 适合初学者 网络没有多复杂 本质为AlphaZero算法 关于神经网络部分为卷积和全连接 激活函数采用reLU()

## 项目链接

<https://github.com/Git-lys/NN_Homework/tree/master>
<https://aistudio.baidu.com/aistudio/projectdetail/4516264>

## 参考文献

<https://github.com/junxiaosong/AlphaZero_Gomoku>

$$
loss=(z-v)^2-\pi^T*log(p)+c||\theta||^2
$$
