# 基于AlphaZero的AI五子棋机器人
中文 [英文](readme_en.md)
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

* 相关API可在网站查看[PaddleAPI](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html#:~:text=API%20%E6%96%87%E6%A1%A3%20%E6%AC%A2%E8%BF%8E%E4%BD%BF%E7%94%A8%E9%A3%9E%E6%A1%A8%E6%A1%86%E6%9E%B6,%28PaddlePaddle%29%2C%20PaddlePaddle%20%E6%98%AF%E4%B8%80%E4%B8%AA%E6%98%93%E7%94%A8%E3%80%81%E9%AB%98%E6%95%88%E3%80%81%E7%81%B5%E6%B4%BB%E3%80%81%E5%8F%AF%E6%89%A9%E5%B1%95%E7%9A%84%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A1%86%E6%9E%B6%EF%BC%8C%E8%87%B4%E5%8A%9B%E4%BA%8E%E8%AE%A9%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%8A%80%E6%9C%AF%E7%9A%84%E5%88%9B%E6%96%B0%E4%B8%8E%E5%BA%94%E7%94%A8%E6%9B%B4%E7%AE%80%E5%8D%95%E3%80%82)  

## Preparation:

  ### 1 基于神经网络的MCTS：

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

#### 关键点

* 通过每一次模拟，MCTS依靠神经网络， 使用累计价值（Q）、神经网络给出的走法先验概率（P）以及访问对应节点的频率这些数字的组合，沿着最有希望获胜的路径（换句话说，也就是具有最高置信区间上界的路径）进行探索。
* 在每一次模拟中，MCTS会尽可能向纵深进行探索直至遇到它从未见过的盘面状态，在这种情况下，它会通过神经网络来评估该盘面状态的优劣
* 巧妙了使用MCTS搜索树和神经网络一起，通过MCTS搜索树优化神经网络参数，反过来又通过优化的神经网络指导MCTS搜索。
Code: `mcts_pure.py`

### 2 策略价值网络
Code：`policy_value_net_paddlepaddle.py`

### 3 UI界面
为游戏创建了较为elegant的UI界面 借鉴与开源
调用pygame库
详细见game.py内容 文档有部分截图和注释

### 4训练

主文件：`train.py`

### 5 实际效果

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

以上为本项目全部内容 适合初学者 网络没有多复杂 本质为AlphaZero算法 关于神经网络部分为卷积和全连接 激活函数采用reLU()

## 项目链接

<https://github.com/Git-lys/NNHomework>  
<https://aistudio.baidu.com/aistudio/projectdetail/4516264>

## 参考文献

<https://github.com/junxiaosong/AlphaZero_Gomoku>


$$
loss=(z-v)^2-\pi^T*log(p)+c||\theta||^2
$$
