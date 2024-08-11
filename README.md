# BERT: TensorRT的推理优化

BERT(Bidirectional Encoder Representations from Transformers)，是由Google AI Language团队在2018年推出的深度学习模型，它能够处理多种NLP任务。以往每个NLP任务通常需要特定的模型来解决，然而BERT的出现改变了这一局面，当时它不仅能够处理超过11种常见的NLP任务，而且在性能上超越了以往的模型。大模型的横空出世之前，基于BERT的模型以及变种是NLP领域的通用解决方案。

下面将分为以下5点来展开讲解BERT以及模型推理优化点，分别是：

## 1. BERT的用途以及如何工作

### 1.1 BERT的用途

BERT，作为一种先进的语言处理模型，在多种语言任务中发挥着重要作用：

- **情绪分析**：BERT能够分析电影评论等文本，判断其情绪倾向是正面还是负面；
- **问答系统**：它可以为聊天机器人提供能力，使其更好地理解并回答用户的问题；
- **文本预测**：在撰写电子邮件等文本时，BERT可以预测用户接下来可能输入的单词或短语，如Gmail的智能撰写功能；
- **文本生成**：BERT还能够基于给定的几句话生成一篇关于特定主题的文章；
- **文本摘要**：它可以快速提取长文本，如法律合同的关键信息，生成摘要；
- **多义词解析**：BERT能够根据上下文准确理解和区分多义词，例如“Bank”可以指金融机构或河岸。

这些只是BERT能力的一部分展示，实际上，BERT和其他NLP技术已经渗透到我们日常生活的许多方面：

- **翻译服务**：谷歌翻译等工具使用NLP技术来提供准确的语言翻译；
- **语音助手**：像Alexa和Siri这样的语音助手依赖NLP来理解和响应用户的语音指令；
- **聊天机器人**：在线客服和聊天机器人使用NLP来解析用户的问题并提供有用的回答；
- **搜索引擎**：谷歌搜索使用NLP技术来理解搜索查询的意图，并提供相关结果；
- **语音导航**：GPS和导航应用程序使用语音控制功能，允许用户通过语音指令进行操作。

<img src="https://xsj-niehen.oss-cn-hangzhou.aliyuncs.com/Docs_assert/BERT-example.png" alt="BERT Google Search Example" width="681" height = "355" />

如上图中Google 翻译，从2020年起BERT 可帮助 Google 更好地显示几乎所有搜索的（英语）结果。

### 1.2 BERT是如何工作的

BERT的工作原理主要基于以下几个关键概念：

1. **Transformer模型**：Transformer是一种深度学习架构，它的核心是注意力机制（Attention Mechanism），这种机制使得模型能够处理序列数据，并且能够捕捉序列中的长距离依赖关系。
2. **双向训练**：传统的语言模型在训练时只考虑文本的一个方向，要么从左到右，要么从右到左。而BERT是双向的，它一次考虑整个文本，这使得BERT能够理解单词的上下文。
3. **预训练和微调**：BERT首先在大量文本数据上进行预训练，然后在特定任务上进行微调。预训练阶段的目标是理解语言的语法和语义（通过两个任务：掩码语言模型和下一句预测）。微调阶段则是为了让模型理解特定任务（例如情感分析或问答）。
4. **词嵌入**：BERT使用WordPiece嵌入，这是一种子词嵌入方法。这意味着BERT可以处理词汇表外的单词，因为它可以将单词分解为已知的子词。

在实际的使用过程中，BERT模型接收一串标记（通常是单词或子词）作为输入，然后输出这些标记的嵌入，这些嵌入是通过考虑上下文（即输入序列中的其他标记）获得的。这些嵌入然后可以用于各种下游任务，如文本分类、命名实体识别或问答。

我们更为关注BERT的推理过程，对于训练以及微调这里就不展开叙述了，最为重要的是Transformer结构以及词嵌入的方式，后面我们将对这两点详细展开。简单来讲，就是文本输入通过词嵌入层转换为tensor，然后执行Transformer结构的Encoder layers，经过多层的layer，得到一个输出tensor，针对特定的任务执行特定的后处理，比如，如果是分类任务，则经过一个线性层以及softmax层等分类模块返回每个类别的概率；如果是翻译类服务，则返回tensor对应的文本信息。

<img src="https://xsj-niehen.oss-cn-hangzhou.aliyuncs.com/Docs_assert/BERT-model-calssification-output-vector-cls.png" alt="img" width="661" height = "400" />

上图为BERT的一个情感分类任务的工作流程，输入为 `a visually stunning rumination on love` 经过Tokenize层、Embedding层、Encoder层得到BERT的输出，再经过一个Logistic Regression层去得到`postive` `negative`的概率，最后得到情感的分类结果是`positive`。

通过这一小节的描述，我们对BERT模型有了一个框架性地理解，但你可能对模型的实现细节还有疑惑。别急，下一小节我们将详细介绍模型的实现细节。

## 2. BERT的模型结构以及大小

BERT模型主要是由Tokenizer层、Embedding层以及Encoder层组成，Encoder层包含Self-Attention层、LayerNorm层、Feed Forward层、残差连接&归一化层。

### 2.1 Tokenizer层

它负责将原始文本转换成BERT模型可以理解和处理的格式。这个转换过程包括以下几个步骤：

1. **分词（Tokenization）**：将原始文本分割成单词或者子词单元（subword units）。BERT使用的是WordPiece算法，它可以将未知或罕见的单词分解成更小的已知子词。
2. **添加特殊标记（Special Tokens）**：例如，每个句子的开始和结束会分别添加特殊的`[CLS]`和`[SEP]`标记。`[CLS]`标记用于分类任务，而`[SEP]`标记用于分隔成对的句子。
3. **转换为ID（Token to ID Conversion）**：将每个分词转换为词汇表中对应的ID。BERT模型的词汇表是固定的，每个词或子词都有一个唯一的索引。
4. **填充和截断（Padding and Truncation）**：为了能够以批处理的方式处理多个句子，`BERT Tokenizer` 会将所有句子填充（或截断）到相同的长度。
5. **创建注意力掩码（Attention Mask）**：这个掩码告诉模型哪些位置是真实的单词，哪些位置是填充的。这样，模型在处理时就可以忽略填充的部分。

使用方式如下：

```python
# 输入文本
text = "Here is some text to encode"
# 使用分词器处理文本 返回tensor
encoded_input = tokenizer(text, return_tensors='pt')
```

在这个例子中，`encoded_input` 将是一个字典，包含了转换后的输入数据，如`input_ids`和`attention_mask`，它们可以直接被BERT模型使用。

### 2.2 Embedding层

在BERT模型中，Embedding层即词嵌入层，是模型的第一层，它负责将输入的token转换成固定维度的向量。这些向量是模型能够理解和处理的数值表示，它们编码了单词的语义信息以及它们在特定上下文中的含义。

具体来说，BERT的词嵌入层包括以下几个部分：

1. **Token Embeddings**：这是将单词或子词token转换为向量的基本嵌入。每个token都被映射到一个高维空间（在BERT的基础模型中是768维）中的一个点。
2. **Segment Embeddings**：BERT可以处理成对的句子（例如，在问答任务中的问题和答案）。Segment embeddings用于区分两个句子，每个句子会被分配一个不同的segment embedding。
3. **Positional Embeddings**：由于BERT使用的是Transformer架构，它不像循环神经网络（RNN）那样自然地处理序列数据。因此，BERT引入了位置嵌入来编码token在句子中的位置信息。

<img src=".\Input.png" alt="Input" style="zoom:80%;" />

这三种嵌入在模型中是逐元素相加的，以产生一个综合的嵌入，它包含了token的语义信息、在句子中的位置信息，以及它属于哪个句子（对于成对句子）。这个综合嵌入随后被传递到BERT的后续层中，用于进一步的处理和学习。

在Torch中的使用

```python
# 初始化
self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
self.segment_embeddings = nn.Embedding(type_vocab_size, hidden_size)
self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
# 使用
# 分词的结果
seq_length = input_ids.size(1)
position_ids = Torch.arange(seq_length, dtype=Torch.long, device=input_ids.device)
position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
# Get Token embeddings from token ids
token_embeds = self.token_embeddings(input_ids)
# Get segment embeddings from segment ids
segment_embeds = self.segment_embeddings(segment_ids)
# Get position embeddings from position ids
position_embeds = self.position_embeddings(position_ids)
```

在内部，`nn.Embedding` 包含一个参数矩阵，其大小为 `(num_embeddings, embedding_dim)`，这个矩阵就是嵌入矩阵。当你传入索引时，PyTorch 会使用这些索引作为嵌入矩阵的行号来检索对应的嵌入向量。可以当成一个词典，根据输入索引去查询得到结果。这个过程在硬件上是非常高效的，尤其是在 GPU 上。

###  2.3 Encoder Layer

Encoder和Decoder是Transformer中的两个重要组成部分，但BERT只使用了Encoder。BERT由多个Encoder layer堆叠而成。下图是一个Encoder 的结构图，包含了Attention层、LayerNorm层、Feed Forward层、残差连接&归一化层等。

<img src="https://xsj-niehen.oss-cn-hangzhou.aliyuncs.com/Docs_assert/c7wD2V.png" alt="img" width="300" height = "300" />

#### 2.3.1 Self-Attention层

**Self-Attention，简单理解就是一个句子内的单词，互相看其他单词对自己的影响力有多大， 也就是说，句子内各单词的注意力应该关注在该句子内其他单词中的哪些单词上。**

 self-attention 的核心组成就是查询（Query）、键（Key）和值（Value）向量。

- **Query ($Q$):** 可以将其视为寻求信息的元素。对于输入序列中的每个词，都会计算一个查询向量。这些查询表示你希望在序列中关注什么。

- **Key ($K$):**  就像路标。它们有助于识别和定位序列中的重要元素。与查询一样，为每个词计算键向量。

- **Value ($V$):** 值携带信息。同样，为每个词计算值向量。这些向量包含我们希望在确定序列中词语重要性时考虑的内容。


1. **Query, Key, and Value 计算:** 对于输入序列中的每个词，我们都会计算查询（Query）、键（Key）和值（Value）向量。这些向量是self- Attention机制运行的基础。 对应这3个向量的计算，对应有3个Linear层，用输入分别跟3个Linear层的权重做矩阵乘就得到了$Q$、$K$、$V$ 向量，如下图中$W^Q$, $W^K$, $W^V$即是模型权重。

   <img src="https://xsj-niehen.oss-cn-hangzhou.aliyuncs.com/Docs_assert/c7wF9x.png" alt="img" width="300" height = "350" />

2. **计算Attention:**  为序列中的每个词对计算注意力分数。查询Query和键Key之间的注意力分数量化了它们的相关性。对于Attention Scores的计算是将上一步的$Q$ 向量和 $K^T$向量相乘，并除以一个参数，然后经过softmax，再乘以 $V$ 就得到了 。
   $$
   Y = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
   $$

   <img src="https://xsj-niehen.oss-cn-hangzhou.aliyuncs.com/Docs_assert/c7wk36.png" alt="img" width="400" height = "120" />

   其中，$QK^T$是两个矩阵的点积；$d_k$ 是Key的dim维度大小，一般情况下与Query 、Value的dim维度一致；$\text{softmax}$ 是沿着最后一个维度按行做softmax操作。

   聊到这里，你可能对Attention机制如何用公式表达已经有了基本的认识，但还是不知道为什么这样做会有效。接下来，我们用通俗易懂的例子形象地理解一下Attention的作用。先来看self-Attention，举个例子，输入序列为“**我 昨天 在 图书馆 遇到了 我的老师**”，那么，“**我**”可能会关注“**昨天**”来获取时间信息；“**图书馆**”可能会关注“**我的老师**”来理解事件发生的地点和人物之间的关系；“**遇到了**”可能会关注“**我**”和“**我的老师**”来理解动作的执行者和对象。

   上述输入序列中各个成分之间存在上述复杂的关系，那么如何将这些复杂关系提取出来呢，这就是self-Attention的作用了。

   我们将包含个词的文本输入序列比作一个班级里有6个学生，这6位学生分别为{“$X_1$: 我”，“$X_2$: 昨天”，“$X_3$: 在”，“$X_4$: 图书馆”，“$X_5$: 遇到了”，“$X_6$: 我的老师”}。为了让班级内6位同学相互助力携手成长，我们设计一次综合测试，包含3个子部分测试，分别考查出每位学生“希望向其他同学学习哪些优秀品质(Query)”、“自己自身具备哪些值得别人学习的优秀品质(Key)”、“培养Key中这些优秀品质的方式方法(Value)”。

   现在学生要向班级内的所有同学（包含自己）去学习，那么应该重点向哪些同学学习呢，取决于各位学生之间的Query与Key的匹配程度。如果学生$X_i$希望学习的优秀品质与学生$X_j$具备的优秀品质恰好匹配，那么学生$X_i$应该重点向学生$X_j$学习，反之学生$X_i$不应该重点向学生$X_j$学习。Attention中用$qk^T$表示序列中词与词的关注程度（是一个标量/数值），对应上述例子来说，就是用$qk^T$表示序列中学生与学生的匹配程度。获得学生$X_i$与班内所有学生的匹配程度之后，学生$X_i$就要开始从班内所有学生身上学习自己想要获得的优秀品质了（吸收培养优秀品质的方式方法），便于提升自己。Attention中用$Y$表示学生向班内所有同学学习并自我进化后的样子。

3. **Muti Head:** 进一步扩展了self Attention，它扩展了模型关注不同位置的能力，它为注意力层提供了多个“表示子空间”，可以让Attention有更丰富的层次。有了多头注意力，拥有多组Query, Key, Value 权重矩阵，会有多个上面的1、2步骤的结果$Z$，那就将多个版本的$Z$拼接称为一个长向量，然后用一个全连接网络，即乘以一个矩阵，就能得到一个短的向量作为输出。

   如下图中$W^{Q}_0$, $W^{K}_0$,   $W^{V}_0$ 以及$W^{Q}_1$, $W^{K}_1$,   $W^{V}_1$ 等都是多头注意力的模型权重，执行的步骤跟上面一致，最后跟$W^{O}$做矩阵乘，得到最后的结果。

   <img src="https://xsj-niehen.oss-cn-hangzhou.aliyuncs.com/Docs_assert/image-20240611225115104-8117476.png" alt="image-20240611225115104" width="600" height = "320" />

multi-head self-Attention，我们也可以直接理解一下它的作用。在解释self-Attention时，我们以班级内同学们之间相互学习成长解释了self-Attention的作用。在那个例子中，我们对班内的同学只做了一次综合测试，去考查同学们的Query、Key、Value，但只有一次综测总会带来误差，因为有些同学可能会发挥失常，**最好的办法当然是设计多次综合测试，尽可能全面准确地考查同学们的Query、Key、Value**。这就是multi-head self-Attention。

Torch实现

```python
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, Q, K, V, attn_mask):
        '''
        Q, K, V: [batch, seq_len, d_model]
        attn_mask: [batch, seq_len, seq_len]
        '''
        batch = Q.size(0)
        '''
        split Q, K, V to per head formula: [batch, seq_len, n_heads, d_k]
        Convenient for matrix multiply opearation later
        q, k, v: [batch, n_heads, seq_len, d_k / d_v]
        '''
        per_Q = self.W_Q(Q).view(batch, -1, n_heads, d_k).transpose(1, 2)
        per_K = self.W_K(K).view(batch, -1, n_heads, d_k).transpose(1, 2)
        per_V = self.W_V(V).view(batch, -1, n_heads, d_v).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        # context: [batch, n_heads, seq_len, d_v]
        context = ScaledDotProductAttention()(per_Q, per_K, per_V, attn_mask)
        context = context.transpose(1, 2).contiguous().view(
            batch, -1, n_heads * d_v)

        # output: [batch, seq_len, d_model]
        output = self.fc(context)
        return output
```

#### 2.3.2 Layer Norm与残差连接

<img src="https://xsj-niehen.oss-cn-hangzhou.aliyuncs.com/Docs_assert/image-20240611232740561-8119662.png" alt="image-20240611232740561" width="400" height = "210"  />

如上图所示，其中Add表示残差连接，Norm表示LayerNorm，在上一步经过 self-Attention 层之后的输出 Self-Attention(Q, K, V)，将这个输出与模型的输入$X_{embedding}$加起来做残差连接，
$$
X_{embedding} + Self-Attention(Q, K, V)
$$
随后将残差连接的结果，进行Layer Normalize。Layer Normalization 的作用是把神经网络中隐藏层归一为标准正态分布，以起到加快训练速度加速收敛的作用

Layer Normalization的计算包含三个步骤：

1. **计算均值$\mu_i$**：
   $$
   \mu_i = \frac{1}{D} \sum_{j=1}^{D} x_{ij}
   $$

​	对于输入 $X$ 的shape为$[B, S, D]$，沿着最后一个维度计算。

2. **计算方差$\sigma_i^2$**：
   $$
   \sigma_i^2 = \frac{1}{D} \sum_{j=1}^{D} (x_{ij} - \mu_i)^2
   $$

3. **归一化每个样本的特征**:
   $$
   \hat{x}_{ij} = \frac{x_{ij} - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}}
   $$

​	 其中$\epsilon$ 是一个很小的数，防止分母为零，通常取 $10^{-5}$或$10^{-6}$。

4. **进行缩放和平移（可学习的参数$ \gamma$ 和 $\beta$​）**：
   $$
   y_{ij} = \gamma \hat{x}_{ij} + \beta
   $$

   其中 $\gamma$ 和 $ \beta $是可学习的参数，它们与输入$X$的维度$D$相同，通过训练过程学习得到，以便网络可以恢复到原始的表示空间，在推理过程中，这两个参数在权重中可以获取。

   最终，$y_{ij} $是层归一化的输出，它将被用作下一层或下一个操作的输入。		

<img src="https://xsj-niehen.oss-cn-hangzhou.aliyuncs.com/Docs_assert/c7wtbQ.png" alt="img" width="500" height = "260"/>

上图为Batch Normalization与Layer Normalize的区别，两者唯一的区别就是Layer Normalize不考虑其他数据，只考虑自己，这样就避免了不同batch size的影响。

在Torch的使用中可以直接调用API

```python
self.norm2 = nn.LayerNorm(d_model)
```

#### 2.3.3 FFN

FFN全称为Feed Forward Neural Network，FFN 的主要目的是引入非线性，帮助模型学习更复杂的表示，主要有以下几个步骤：

1. **线性变换**：输入首先通过一个线性变换，将输入的维度从 `d_model` 映射到一个更大的维度 `d_ff`。一般`d_ff`为`d_model`的四倍。
2. **激活函数**：应用一个非线性激活函数，例如 ReLU 或 GeLU（Gaussian Error Linear Unit），增强模型的表达能力。
3. **第二个线性变换**：激活后的结果再次通过一个线性变换，将维度从 `d_ff` 映射回 `d_model`。

```python
class FeedForwardNetwork(nn.Module):
    def __init__(self):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

        self.gelu = gelu

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x
```

通过这种方式，FFN 可以捕捉到输入数据中的复杂模式和依赖关系。

### 2.4 模型参数及大小

BERT相关参数

```python
max_len = 30
max_vocab = 50
max_pred = 5

d_k = d_v = 64
d_model = 768  # n_heads * d_k
d_ff = d_model * 4

n_heads = 12
n_layers = 6
n_segs = 2

```

- 
  max_len: 输入序列的最大长度；
  
- max_vocab: 字典的最大大小；

- max_pred: Mask时最大的Mask数量；

- d_k, d_v: 自注意力中K和V的维度，Q的维度直接用K的维度代替，因为这二者必须始终相等；

- d_model: Embedding的大小；

- d_ff: 前馈神经网络的隐藏层大小，一般是d_model的四倍；

- n_heads: 多头注意力的头数；

- n_layers: Encoder的堆叠层数；

- n_segs: 输入BERT的句子段数，用于制作Segment Embedding。

  

<img src="https://xsj-niehen.oss-cn-hangzhou.aliyuncs.com/Docs_assert/BERT-size-and-architecture.png" alt="Original BERT models architecture" width="600" height = "286" />

在常见的BERT Base与BERT Large中有不同的参数：

<img src="https://xsj-niehen.oss-cn-hangzhou.aliyuncs.com/Docs_assert/image-20240612000635210.png" alt="image-20240612000635210" style="zoom:67%;" />

## 3. BERT模型的推理优化

### 3.1 Torch推理实现

下面代码原生实现了一个基于BERT的分类网络。通过搭建上面提到的各个模块，然后组成Encoder Layer，最后搭建起来整个网络结构。

前面没有仔细讨论在计算过程中tensor的shape变化情况，在下面代码中都有标注，经过不同的算子会有不同的shape变化，这是其中比较细节的部分：

```python
maxlen = 30  #输入序列的最大长度.
batch_size = 6
max_pred = 5 # max tokens of prediction 字典的最大大小.
n_layers = 6  # Encoder的堆叠层数.
n_heads = 12
d_model = 768  # Embedding的大小 n_heads * d_k
d_ff = 768*4 # 4*d_model, FeedForward dimension 前馈神经网络的隐藏层大小, 一般是d_model的四倍.
d_k = d_v = 64  # dimension of K(=Q), V 自注意力中K和V的维度, Q的维度直接用K的维度代替, 因为这二者必须始终相等.
n_segments = 2 #输入BERT的句子段数. 用于制作Segment Embedding.
class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding
        self.pos_embed = nn.Embedding(maxlen, d_model)  # position embedding
        self.seg_embed = nn.Embedding(n_segments, d_model)  # segment(token type) embedding
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = Torch.arange(seq_len, dtype=Torch.long)
        pos = pos.unsqueeze(0).expand_as(x)  # [seq_len] -> [batch_size, seq_len]
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedding)

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = Torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size, n_heads, seq_len, seq_len]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = Torch.matmul(attn, V)
        return context

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
    def forward(self, Q, K, V, attn_mask):
        # q: [batch_size, seq_len, d_model], k: [batch_size, seq_len, d_model], v: [batch_size, seq_len, d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size, n_heads, seq_len, d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size, n_heads, seq_len, d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size, n_heads, seq_len, d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        # context: [batch_size, n_heads, seq_len, d_v], attn: [batch_size, n_heads, seq_len, seq_len]
        context = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size, seq_len, n_heads, d_v]
        output = nn.Linear(n_heads * d_v, d_model)(context)
        return nn.LayerNorm(d_model)(output + residual) # output: [batch_size, seq_len, d_model]

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.fc2(gelu(self.fc1(x)))
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, seq_len, d_model]
        return enc_outputs

class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.embedding = Embedding()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(0.5),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(d_model, 2)
        self.linear = nn.Linear(d_model, d_model)
        self.activ2 = gelu
        # fc2 is shared with embedding layer
        embed_weight = self.embedding.tok_embed.weight
        self.fc2 = nn.Linear(d_model, vocab_size, bias=False)
        self.fc2.weight = embed_weight

    def forward(self, input_ids, segment_ids, masked_pos):
        output = self.embedding(input_ids, segment_ids) # [bach_size, seq_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids) # [batch_size, maxlen, maxlen]
        for layer in self.layers:
            # output: [batch_size, max_len, d_model]
            output = layer(output, enc_self_attn_mask)
        # it will be decided by first token(CLS)
        h_pooled = self.fc(output[:, 0]) # [batch_size, d_model]
        logits_clsf = self.classifier(h_pooled) # [batch_size, 2] predict isNext

        masked_pos = masked_pos[:, :, None].expand(-1, -1, d_model) # [batch_size, max_pred, d_model]
        h_masked = Torch.gather(output, 1, masked_pos) # masking position [batch_size, max_pred, d_model]
        h_masked = self.activ2(self.linear(h_masked)) # [batch_size, max_pred, d_model]
        logits_lm = self.fc2(h_masked) # [batch_size, max_pred, vocab_size]
        return logits_lm, logits_clsf
model = BERT()
```

以上为原生实现，也有一些库比如`transformers`对BERT进行了封装，调用起来非常的简单，可以直接使用。

```python
from transformers import BERTTokenizer, BERTForMaskedLM
tokenizer = BERTTokenizer.from_pretrained(BERT_PATH)
model = BERTForMaskedLM.from_pretrained(BERT_PATH, return_dict = True)
```

### 3.2 算子执行过程

上面我们分析了BERT的网络结构，其核心就是Transformer的Encoder layer，因此我们也主要针对Encoder layer进行搭建、部署以及优化。

首先对上面Encoder的算子计算过程进行整理：

1. Tokenizer 与Embedding
   $$
   X = Tokenizer(text) \\
   X = PositionalEmbeddings(X) + TokenEmbedding(X) + SegmentEmbedding(X) \\ X.shape = [B, S, D]
   $$


2. Self-Attention
   $$
   Q = Linear_q(X) \\ K = Linear_k(X) \\ V = Linear_v(X)\\ X_{attention} = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
   $$
   （这里如果是Muti-head Attention，在后面还有一个Linear层）

3. 残差连接以及Layer Norm
   $$
   X_{attention} = X_{attention} + X \\ X_{attention} = Layer Norm(X_{attention})
   $$

4. FFN以及残差连接和Layer Norm
   $$
   X_{hidden} = Linear(ReLU(Linear(X_{attention}))) \\ X_{hidden} = X_{attention} + X_{hidden} \\ X_{hidden} = LayerNorm(X_{hidden})
   $$

可以看到，上面包含了5种算子，分别是`Embedding` `Linear` `Softmax` `LayerNorm` `ReLU`，其中最为耗时的就是 `Linear`层也就是矩阵乘运算。

具体的算子执行流程如下图所示：

<img src="https://xsj-niehen.oss-cn-hangzhou.aliyuncs.com/Docs_assert/image-20240612223428972.png" alt="image-20240612223428972" width="300" height = "600" />

GEMM0到GEMM3指的是连续四个矩阵乘法（GEMM全称General Matrix Multiply，即通用矩阵乘法），它们在上图中从GEMM #0数到GEMM #3。另外两个Batched GEMM是Self- Attention的一部分（这里Batch Gemm原因是muti head attention 有多组计算，可以进行拼batch提高效率），因此与softmax一起作为一个整体在上图中被称为MHA（Multi-Head Attention）。

下图展示了两种序列长度（左256，右1024）下Encoder layer的profile。性能分析结果显示，计算密集型的GEMM操作占了两个测试案例总执行时间的61%和40%。Attention模块，包括一个softmax和两个批处理的GEMM，是其中最耗时的部分，随着序列长度从256增加到1024，Attention模块占总执行时间的49%，而其余的访存密集型操作（层归一化、加偏置和激活）只占11%-17%。

<img src="https://xsj-niehen.oss-cn-hangzhou.aliyuncs.com/Docs_assert/image-20240612234601998-8207165.png" alt="image-20240612234601998" style="zoom:50%;" />



鉴于上述分析，我们对Encoder做推理优化主要有4个点可以做优化，分别是做Batch GEMM、优化MHA、算子融合以及Varlen。下面将详细展开介绍。

### 3.3 优化1：Batch Gemm

在前面的计算过程中, 下面的这三个矩阵乘
$$
Q = Linear_q(X) \\ K = Linear_k(X) \\ V = Linear_v(X)
$$

这三个矩阵都有相同的shape，且输入相同，因此将三个权重矩阵拼接到一起，使用cuBLAS的Bathed GEMM，能够增加带宽，提高计算效率。

通常矩阵乘调用cuBLAS的`cublasGemmEx`，在这里可以调用 `cublasGemmStridedBatchedEx`实现更高效，两者都是 NVIDIA cuBLAS提供的 GEMM（General Matrix Multiply, 通用矩阵乘法）接口。两者之间的主要区别如下：

1. `cublasGemmEx`：

该函数用于计算通用矩阵乘法，即 $C = α × A × B + β × C$。此函数可以在不同精度和数据类型的输入矩阵上执行计算。它通常用于执行单个矩阵乘法操作。该接口对使用不同精度和混合数据类型的矩阵运算具有优势，支持高性能计算。

2. `cublasGemmStridedBatchedEx`:

该函数用于执行批量矩阵乘法，即在给定一组矩阵（称为批次）时一次性完成所有矩阵的乘法运算。每个批次的矩阵乘法按照 $C_i = α × A_i × B_i + β × C_i$ 计算，其中 $A_i$、$B_i$ 和 $C_i$ 分别为每个批次中的输入和输出矩阵。使用一种称为“strided”内存布局的机制。这意味着输入矩阵 $A_i$, $B_i$ 和输出矩阵 $C_i$ 在内存中存储为连续的块。每个连续块之间都具有固定跨度（stride）。这通常可以提高内存访问和计算性能。

该处的优化需要将$W^Q$, $W^K$, $W^V$即模型权重拼接在一起，在显存上保证是连续的。

### 3.4 优化2：MHA

在self-Attention那里，也可以做算子融合。将以下几个操作全部融合为一个算子。
$$
Q = Linear_q(X) \\ K = Linear_k(X) \\ V = Linear_v(X)\\ X_{attention} = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$
这里做优化的原因有两个：一个是Attention非常占用显存，$QK^T$​ 的shape为$[B, N, S, S]$，当$S$很大时，这部分非常占用显存；另一个原因是这里的操作是memory bound，不能充分利用计算资源。优化的方式有多种，一种是TensorRT的`MHA`, 一种是`Flash-Attention`，一种是xformers的 `memory_efficient_attention`，后两者是有相关的论文和开源代码的，下面对这三种Attention优化进行简单介绍。

这三种Attention机制都是为了优化Transformer模型中的多头注意力（Multi-Head Attention, MHA）计算而设计的。它们各自采用不同的方法来提高效率、减少内存占用或加速计算。

1. TensorRT的`MHA`:

   - TensorRT中的MHA优化可能包括内核融合、精度校准、层自动调整等技术，以减少在执行多头自注意力时的延迟和内存占用。
   - TensorRT的MHA有多个版本，但都以一种二进制文件的开源方式，根据不同的机器和需求，编译了众多二进制文件，放入Plugin，看不到原生实现。
   - 使用方式的话可以编译该TensorRT的Plugin，插入到项目中。
   - <img src="https://xsj-niehen.oss-cn-hangzhou.aliyuncs.com/Docs_assert/image-20240613214918841.png" alt="image-20240613214918841" />

2. `Flash-Attention`:

   - Flash-Attention是一种用于加速Transformer模型中自注意力计算的技术。它通过减少全局同步点和优化内存访问模式来提高效率。Flash-Attention特别关注减少在大型模型和长序列上的注意力计算的内存占用，使得在有限的硬件资源上运行大型模型成为可能。它是为了在NVIDIA的GPU上实现高效的自注意力计算而设计的。

   - 代码：https://github.com/Dao-AILab/flash-attention  论文：https://arxiv.org/abs/2205.14135

   - 简单来讲就是将Flash- Attention充分利用了GPU的内存体系结构，将输入不断tiling，放入更快的shared memory以及寄存器，让数据不断往访存更快的存储设备上去，另外将 $QK^T$ 与对行计算Softmax的操作进行融合，这样一方面充分利用了计算能力，另一方面将大尺寸的attention scores $[B, N, S, S]$融化掉了，意味着显存的使用降为了线性增长，能够进行更长的seq_length的训练和推理。

   - Flash-Attention有两个版本，分别是Flash-Attention1与Flash-Attention2，版本2相比于1从计算方式上进行优化，速度更快了。（更为详细的讲解可以看知乎上的文章：https://zhuanlan.zhihu.com/p/645376942，这里不详细展开）

   - 使用方式可以直接从开源仓库进行调用或者修改，或者使用Torch2.0以上版本，Torch的`Torch.nn.functional.scaled_dot_product_attention`已经实现了Flash-Attention2，直接调用即可，或者从源仓库编译为TensorRT Plugin，插入项目中。

     ![image-20240613215337965](https://xsj-niehen.oss-cn-hangzhou.aliyuncs.com/Docs_assert/image-20240613215337965.png)

     

3. xformers的`memory_efficient_attention`:

   - xformers是一个模块化的Transformer库，它提供了多种用于提高Transformer模型效率的组件。`memory_efficient_attention`是xformers中的一个特性，它实现了一种内存高效的注意力机制。这种机制通过使用重新计算（recomputation）策略和优化的数据布局来减少在执行注意力操作时的内存占用。这对于训练大型模型特别有用，因为它可以减少显存的使用，从而允许在单个GPU上训练更大的模型或使用更长的序列。
   - 代码：https://github.com/facebookresearch/xformers 论文：https://arxiv.org/pdf/2112.05682
   - 在SD 、SVD等方向xformer的`memory_efficient_attention`通常会更好。
   - 使用方式可以从xformers中调用，也可以在Torch的`Torch.nn.functional.scaled_dot_product_attention`中调用（不过与xformers中实现不太一样），该函数内部实现了3种Attention，会根据输入的shape自动选择attention实现。

### 3.5 优化3：Kernel fuse

这里的Kernel Fuse主要有两种，一种是 Add bias与Layer norm的融合，一种是Gemm add bias与 Activation的融合。算子融合的本质在于降低访存的耗时，通过计算方式的融合，来减少访存的次数或者数据放在更快的存储设备上计算。

**Add bias & Layer norm**

从上面算子的执行图中可以看到，两个layernorm前面都有矩阵乘，而矩阵乘会有bias，通过算子融合将add bias和layer norm一起实现，能够更加的高效。

从上面的profile中可以看到这些操作分别占到了序列长度为256和1024时总执行时间的10%和6%。一般的实现引入了两轮内存访问来加载和存储张量。实现算子融合内核后，它只需要一轮全局内存访问就可以完成层归一化和加偏置。这个子内核的内核融合提高了性能61%，相应地，对于序列长度在128到1024之间的单层BERT变换器性能平均提高了3.2%。（数据来源于ByteTransformer）

**GEMM with add bias & Activation**

添加偏置和激活函数：对于序列长度分别为256和1024，这些操作分别占总执行时间的7%和5%。通过矩阵乘法进行投影后，结果张量将与输入张量相加，并使用GELU激活函数进行逐元素激活。这里的融合实现不是将GEMM（通用矩阵乘法）的输出存储到全局内存然后再次加载它以进行添加偏置和激活，而是通过实现一个定制的融合CUTLASS ，在寄存器级别重用GEMM结果矩阵。这里GEMM完美地隐藏了偏置和GELU的内存延迟进入GEMM。在这一步之后，提高了单层BERT的性能约3.8%。



下图为进行算子融合后的算子执行图，可以看到有两个**add bias & Layer norm**，1个**GEMM with add bias & activation**，分别使用了一个算子实现，而不是之前的多个算子。

<img src="https://xsj-niehen.oss-cn-hangzhou.aliyuncs.com/Docs_assert/image-20240612232856129.png" alt="image-20240612232856129"  width="300" height = "600" />

### 3.6 优化4：Varlen

该优化针对服务场景有众多的请求，而请求的输入大小有比较大的差异，一般处理都会打Batch，而对于一批数据通常采用padding的方式将这批数据的$S$维度pad到输入最长的数据。如果输入长度的方差比较大，这种情况下就带来了冗余计算，Varlen就是解决这种场景下的问题。

这里有两种做法，一种是将所有输入拼接到一条上，将Batch设置为1，使用数据长度的前缀和进行标记，另一种做法是多条Batch，但标记更为复杂些。

**TensorRT**

下图为第一种做法的示意图，该方案在TensorRT以及TCB中都有提到，对于Transformer的Encoder Layer中，在计算过程中大部分算子都依赖于最后一个维度$D$，而这里的修改了前两个维度$[B, S]$。因此只需要修改掉前两个维度会影响到的算子即可，这里受到影响的是Attention处的Mask以及softmax操作，这里的Mask跟原本实现的不一样了，看下图可以看出mask的区域发生了变化，Softmax的操作也会又些变化。其他基本一致。

<img src="https://xsj-niehen.oss-cn-hangzhou.aliyuncs.com/Docs_assert/5-Figure6-1.png" alt="img" width="334" height = "380" />

**ByteTransformer**

另一种实现是ByteTransformer提出的，实现更为复杂些，但性能更好。从下图的padding机制可以看出，需要一个数组来标记出来每个输入的位置。相比于第一种直接拼到一个维度，后续的处理也会不同。上面有提到对于Encoder Layer中受到影响的是Attention实现，第一种方法只需要修改mask以及softmax的处理即可。而ByteTransformer的方式明显不能通过修改maks来实现。

ByteTransformer有两个版本，第一个版本是在处理Attention时将拼接的输出恢复到padding的模式，在进行处理，这种方式效率很低，还是会有很多冗余计算。第二个版本对这里进行了优化，在Cutlass的Group gemm基础上进行了优化，能够实现多个不同尺寸的矩阵乘同时进行。这里面内容很多，就不详细展开了。

<img src="https://xsj-niehen.oss-cn-hangzhou.aliyuncs.com/Docs_assert/image-20240613223803438-8289487.png" alt="image-20240613223803438" width="420" height = "340" />



## 4. BERT-TensorRT的推理优化

TensorRT是NVIDIA提供的一个高性能深度学习推理（inference）平台，它专门用于生产环境中。TensorRT可以显著提高深度学习模型在NVIDIA GPU上的推理速度、效率和性能。TensorRT包含一个优化器和一个运行时环境，可以将训练好的深度学习模型转换为优化的推理引擎。这个过程涉及层和张量的融合、内核自动调整、精度校准等多种优化策略，TensorRT支持FP32、FP16和INT8这三种精度模式，并提供了精度校准工具，以确保即使在降低精度以提高性能的情况下，也能保持模型的准确性，并且TensorRT支持动态输入，即可以处理可变大小的输入张量，这对于处理不同分辨率的图像或可变长度的序列特别有用。另外如果标准层不足以覆盖某些特定的模型需求，TensorRT提供了插件API，允许开发者自定义层。

可惜的是，目前TensorRT只能在NVIDIA的GPU上使用，无法适配应用于其他厂家的GPU。各大厂家也纷纷推出了针对自己GPU的推理平台。

下面将使用TensorRT对BERT进行推理优化。这里介绍两种将模型转换为TensorRT的方式，一种是使用TensorRT API搭建，另一种是使用ONNX进行转换；另外介绍下如何编写TensorRT Plugin，以及如何在网络中使用；最后介绍下TensorRT使用FP16、INT8进行加速的方式。

### 4.1 使用TensorRT的简要流程

构建TensorRT大概步骤有以下9步

step1：创建logger
step2：创建builder
step3：创建network
step4：向network中添加网络层
step5：设置并标记输出
step6：创建config并设置最大batchsize和最大工作空间
step7：创建engine
step8：序列化保存engine
step9：释放资源

如下面代码所示

```python
import TensorRT as trt

logger = trt.Logger(trt.Logger.ERROR)

builder = trt.Builder(logger)
builder.reset()  # reset Builder as default, not required

network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
config = builder.create_builder_config()
inputTensor = network.add_input("inputT0", trt.float32, [3, 4, 5])
identityLayer = network.add_identity(inputTensor)
network.mark_output(identityLayer.get_output(0))
builder.max_batch_size = 8
builder.max_workspace_size = 1 << 30
engineString = builder.build_serialized_network(network, config)
```

其中工作量最大的是构建网络，针对不同的网络，需要使用对应的算子去搭建，随后去按照流程去build engine。

搭建网络一般有两种方式，分别是使用API搭建以及使用ONNX进行转换。每种方法都有其优缺点，具体如下：

**使用 TensorRT API 直接搭建网络**：

优点：

- 直接使用 TensorRT API 可以充分利用 TensorRT 的所有优化功能，包括层融合、精度混合（FP32、FP16、INT8）等；
- 可以手动调整和优化每一层的参数和配置，以获得最佳性能；

- 可以完全控制网络的每一层和每一个操作，适合需要高度定制化的应用场景；
- 可以直接使用 TensorRT 提供的各种高级功能和插件；

- 可以精细地控制内存分配、数据流和计算图的执行顺序，适合对性能要求极高的应用。

缺点：

- 需要深入了解 TensorRT API 和底层实现，开发和调试的复杂度较高；
- 对于复杂的网络结构，手动编写代码可能会非常繁琐和容易出错；

- 直接使用 TensorRT API 编写的代码通常与特定的硬件和软件环境绑定，移植到其他平台可能需要大量修改；

- 由于代码高度定制化，后期的维护和更新成本较高。

**使用 ONNX 解析 TensorRT 网络**

优点：

- 可以使用高层次的深度学习框架（如 PyTorch、TensorFlow）来构建和训练模型，然后导出为 ONNX 格式；
- ONNX 模型可以直接导入 TensorRT，简化了开发流程；

- ONNX 是一个开放的标准格式，支持多种深度学习框架和硬件平台；
- 使用 ONNX 可以更容易地在不同平台之间移植模型；

- ONNX 有广泛的社区支持和丰富的工具生态系统，可以利用现有的工具进行模型转换、优化和部署；

- 使用高层次框架构建和训练模型，代码更简洁，维护成本较低。

缺点：

- 虽然 TensorRT 对 ONNX 模型进行了优化，但可能无法达到手动优化的性能；
- 某些高级优化和自定义操作可能无法通过 ONNX 表达，需要额外的插件或手动调整；

- 需要确保所使用的深度学习框架和 TensorRT 都支持 ONNX 格式；
- 某些新特性或自定义层可能在 ONNX 中不支持，导致模型转换失败或性能下降；

- 如果在导入 ONNX 模型时遇到问题，调试可能会比较复杂，需要了解 ONNX 格式和 TensorRT 的内部实现。

简单来讲就是：

- **使用 TensorRT API 直接搭建网络** 适合需要高度定制化和性能优化的应用场景，但开发和维护成本较高。
- **使用 ONNX 解析 TensorRT 网络** 适合希望简化开发流程、提高跨平台兼容性和降低维护成本的应用场景，但可能在性能和灵活性上有所妥协。

选择哪种方式取决于具体的应用需求、开发资源和性能要求。对于大多数应用，使用 ONNX 解析 TensorRT 网络是一个更为简便和高效的选择，而对于需要极致性能优化的应用，可以考虑直接使用 TensorRT API。

### 4.2 使用API搭建BERT

通常搭建网络都是使用Torch，而使用TensorRT进行推理就需要用它的API去搭建一套网络，两者API差别还是比较大的。

下面以几个例子来介绍TensorRT的API

**Relu**

```python
def addReLU(self, layer, x, layer_name=None, precision=None):
    trt_layer = self.network.add_activation(x, type=trt.ActivationType.RELU)

    if layer_name is None:
        layer_name = "nn.ReLU"

    self.layer_post_process(trt_layer, layer_name, precision)

    x = trt_layer.get_output(0)
    return x
```

上面代码通过调用TRT的API `self.network.add_activation` 添加了一个激活层，设置type为`trt.ActivationType.RELU` ，这里支持多种激活函数如`trt.ActivationType.TANH`等，通过设置不同的type进行选择。

随后`layer_post_process`函数去设置layer.name（在进行debug以及build过程中会有用处），并对shape进行了检查，最后使用`trt_layer.get_output(0)`获取到这层的输出并返回。

**Reshape**

```python
def addReshape(self, x, reshape_dims, layer_name=None, precision=None):
    trt_layer = self.network.add_shuffle(x)
    trt_layer.reshape_dims = reshape_dims

    if layer_name is None:
        layer_name = "Torch.reshape"
    else:
        layer_name = "Torch.reshape." + layer_name

    self.layer_post_process(trt_layer, layer_name, None)

    x = trt_layer.get_output(0)
    return x
```

Reshape在TensorRT中通过`self.network.add_shuffle` API 来实现，并且需要设置目标shape，通过`trt_layer.reshape_dims` 来进行设置，这点跟Torch的使用方法很不同。后续是类似的操作，进行后处理返回该层的输出。

**Linear**

```python
def addLinear(self, x, weight, bias, layer_name=None, precision=None):
    input_len = len(x.shape)
    if input_len < 3:
        raise RuntimeError("addLinear x.shape.size must >= 3")

    if layer_name is None:
        layer_name = "nn.Linear"

    # calc pre_reshape_dims and after_reshape_dims
    pre_reshape_dims = trt.Dims()
    after_reshape_dims = trt.Dims()
    if input_len == 3:
        pre_reshape_dims = (0, 0, 0, 1, 1)
        after_reshape_dims = (0, 0, 0)
    elif input_len == 4:
        pre_reshape_dims = (0, 0, 0, 0, 1, 1)
        after_reshape_dims = (0, 0, 0, 0)
    elif input_len == 5:
        pre_reshape_dims = (0, 0, 0, 0, 0, 1, 1)
        after_reshape_dims = (0, 0, 0, 0, 0)
    else:
        raise RuntimeError("addLinear x.shape.size >5 not support!")

    # add pre_reshape layer
    trt_layer = self.network.add_shuffle(x)
    trt_layer.reshape_dims = pre_reshape_dims

    self.layer_post_process(trt_layer, layer_name+"_pre_reshape", precision)

    x = trt_layer.get_output(0)

    # add Linear layer
    out_features = weight.shape[1]
    weight = trt.Weights(weight)
    if bias is not None:
        bias = trt.Weights(bias)

    trt_layer = self.network.add_fully_connected(x, out_features, weight, bias)
    self.layer_post_process(trt_layer, layer_name, precision)
    x = trt_layer.get_output(0)

    # add after_reshape layer
    trt_layer = self.network.add_shuffle(x)
    trt_layer.reshape_dims = after_reshape_dims
    self.layer_post_process(trt_layer, layer_name+"_after_reshape", precision)
    x = trt_layer.get_output(0)

    return x
```

这里首先检查输入张量$x$的维度是否至少是3维的，这是因为全连接层至少需要一个批次维度和两个特征维度（例如，批次大小、特征数）。如果没有提供层名称，它会使用默认的`"nn.Linear"`。使用`trt.Dims()`获取输入的维度信息，根据输入张量的维度，函数计算了在全连接层之前和之后需要的reshape维度。这是为了确保全连接层可以正确地处理输入数据。在添加全连接层之前，代码首先添加了一个shuffle层（`self.network.add_shuffle`）来改变输入张量的形状，使其适合全连接层的输入要求。然后，函数添加了一个全连接层(`self.network.add_fully_connected`)，其中使用提供的权重和偏置参数。全连接层之后，再次添加了一个shuffle层来将输出张量的形状改变回原来的维度（或者是适合后续操作的维度）。

后续还有Softmax Add等算子，实现思路是一致的，可以参考TensorRT的API进行调用，传入或者设置对应的参数。

这里我们再以上面的算子来搭建一个Block，以Encoder中的Self-Attention为例。

**Self-Attention Block**

```python
def self_attention_layer(network_helper, prefix, config, weights_dict, input_tensor, imask):
    num_heads = config.num_attention_heads
    head_size = config.head_size

    q_w = weights_dict[prefix + "attention_self_query_kernel"]
    q_b = weights_dict[prefix + "attention_self_query_bias"]
    q = network_helper.addLinear(input_tensor, q_w, q_b)
    q = network_helper.addShuffle(q, None, (0, -1, num_heads, head_size), (0, 2, 1, 3), "att_q_view_transpose")

    k_w = weights_dict[prefix + "attention_self_key_kernel"]
    k_b = weights_dict[prefix + "attention_self_key_bias"]
    k = network_helper.addLinear(input_tensor, k_w, k_b)
    k = network_helper.addShuffle(k, None, (0, -1, num_heads, head_size), (0, 2, 3, 1), "att_k_view_and transpose")
    # k = network_helper.addShuffle(k, None, (0, -1, self.h, self.d_k), (0, 2, 3, 1), "att_k_view_and transpose")

    v_w = weights_dict[prefix + "attention_self_value_kernel"]
    v_b = weights_dict[prefix + "attention_self_value_bias"]
    v = network_helper.addLinear(input_tensor, v_w, v_b)
    v = network_helper.addShuffle(v, None, (0, -1, num_heads, head_size), (0, 2, 1, 3), "att_v_view_and transpose")

    scores = network_helper.addMatMul(q, k, "q_mul_k")

    scores = network_helper.addScale(scores, 1/math.sqrt(head_size))

    attn = network_helper.addSoftmax(scores, dim=-1)

    attn = network_helper.addMatMul(attn, v, "matmul(p_attn, value)")

    attn = network_helper.addShuffle(attn, (0, 2, 1, 3), (0, -1, num_heads * head_size), None, "attn_transpose_and_reshape")

    return attn
```

1. 代码首先使用`addLinear`函数来生成$Q$、$K$、$V$矩阵。这些矩阵是通过将输入张量与权重矩阵相乘并添加偏置来计算的，其中权重矩阵是从模型权重中加载而来，放到weights_dict这个字典中，使用对应的key获取权重，另外TensorRT的权重一般使用numpy数组来加载。
2. 使用`addShuffle`函数（来调整$Q$、$K$和$V$矩阵的形状，以便它们可以分割成多个头部（heads），这是多头注意力机制的一部分。这里的重塑操作是将每个矩阵分割成`num_heads`个头部，每个头部的大小是`head_size`。
3. 代码使用`addMatMul`函数）来计算$Q$和$K$的点积，得到分数（scores）。这个分数表示输入序列中每个元素对其他元素的注意力权重，并通过除以`head_size`的平方根进行缩放。然后使用`addSoftmax`函数。再使用`addMatMul`函数将注意力权重（attn）与值（V）矩阵相乘，得到加权的值表示。
4. 最后，使用`addShuffle`函数将输出转换回原始输入张量的形状。

上面代码并没有完全组成一个Attention_layer, 因为还缺少一个线性层以及残差和layernorm层。下面代码完善了一个Attention_layer。

```python
def self_output_layer(network_helper, prefix, config, weights_dict, hidden_states, input_tensor):
    out_w = weights_dict[prefix + "attention_output_dense_kernel"]
    out_b = weights_dict[prefix + "attention_output_dense_bias"]
    out = network_helper.addLinear(hidden_states, out_w, out_b)
    out = network_helper.addAdd(out, input_tensor)
    gamma = weights_dict[prefix + "attention_output_layernorm_gamma"]
    beta = weights_dict[prefix + "attention_output_layernorm_beta"]
    out = network_helper.addLayerNorm(out, gamma, beta)
    return out
def attention_layer(network_helper, prefix, config, weights_dict, input_tensor, imask):
    attn = self_attention_layer(network_helper, prefix, config, weights_dict, input_tensor, imask)
    out = self_output_layer(network_helper, prefix, config, weights_dict, attn, input_tensor)
    return out
```

`self_output_layer` 函数构建了自注意力层的输出部分。与上面一样，`out_w` 和 `out_b` 是从`weights_dict`字典中获取的权重和偏置，用于线性变换（通常是全连接层）。随后将将线性变换的结果与`input_tensor`（残差连接的一部分）相加。最后读取参数`gamma` 和 `beta`，是从`weights_dict`获取的层归一化参数，添加LayerNorm层，并返回输出。

`attention_layer`这个函数构建了完整的自注意力层，其中包括自注意力计算和一个输出层。

当然这只是Transformer Encoder中的一部分，并没有将完全的代码实现展示出来，展示了最为核心的Attention，其他模块的实现方式也是一致的。

最后网络搭建完成之后，设置build的参数，进行build，最后进行序列化保存到本地，在使用前进行反序列化，并设置输入进行推理就得到了输出。

### 4.3 使用ONNX搭建BERT

使用ONNX来搭建BERT，相比于API，能够更为简单。这里有两个步骤：一是将Torch模型转换为ONNX，二是将ONNX转为TensorRT engine。

**model2ONNX**

```python
from transformers import BERTTokenizer, BERTForMaskedLM
tokenizer = BERTTokenizer.from_pretrained(BERT_PATH)
model = BERTForMaskedLM.from_pretrained(BERT_PATH, return_dict = True)
text = "The capital of France, " + tokenizer.mask_token + ", contains the Eiffel Tower."
encoded_input = tokenizer.encode_plus(text, return_tensors = "pt")
model.eval()
export_model_path = BERT_PATH + "/model.ONNX"
opset_version = 12
symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
Torch.ONNX.export(model,                                            # model being run
                  args=tuple(encoded_input.values()),                 # model input (or a tuple for multiple inputs)
                  f=export_model_path,                         # where to save the model (can be a file or file-like object)
                  opset_version=opset_version,                      # the ONNX version to export the model to
                  do_constant_folding=False,                         # whether to execute constant folding for optimization
                  input_names=['input_ids',                         # the model's input names
                               'attention_mask',
                               'token_type_ids'],
                output_names=['logits'],                    # the model's output names
                dynamic_axes={'input_ids': symbolic_names,        # variable length axes
                              'attention_mask' : symbolic_names,
                              'token_type_ids' : symbolic_names,
                              'logits' : symbolic_names})
print("Model exported at ", export_model_path)
```

这里最核心的是`Torch.ONNX.export`, 调用该函数将Torch模型转换为ONNX模型。

对以上代码该函数的参数进行以下说明：

- `model`：要导出的BERT模型。
- `args`：模型的输入参数。这里使用 `tuple(encoded_input.values())`，表示模型的输入是一个包含多个输入张量的元组。
- `f`：导出模型的保存路径。
- `opset_version`：导出的 ONNX 模型的操作集版本。
- `do_constant_folding`：是否执行常量折叠优化。常量折叠是指在导出过程中将计算图中的常量表达式进行预计算，以优化模型。
- `input_names`：模型的输入名称列表。
- `output_names`：模型的输出名称列表。
- `dynamic_axes`：动态轴的定义，用于表示输入和输出张量的可变长度。

导出的ONNX可以使用Netron对节点进行可视化。下图为BERT模型其中一层的Self-Attention的部分算子。另外ONNX中算子与Torch中以及TensorRT中算子定义都不同，以至于导出和转换会有区别。

<img src="https://xsj-niehen.oss-cn-hangzhou.aliyuncs.com/Docs_assert/image-20240614224838204-8376520.png" alt="image-20240614224838204" width="520" height = "680" />

**ONNX2TensorRT**

这里有两种实现方式，一种是使用TensorRT自带的trtexec工具来转换，另一种是使用TensorRT API来实现。

1. **使用 `trtexec` 命令**：

   `trtexec` 是 TensorRT 提供的一个命令行工具，用于快速将 ONNX 模型转换为 TensorRT 引擎，并进行推理测试。

   ```
   trtexec --ONNX=BERT.ONNX --saveEngine=BERT.engine --explicitBatch  \
           --minShapes=input_ids:1x6,token_type_ids:1x6,input_mask:1x6 \
           --optShapes=input_ids:1x64,token_type_ids:1x64,input_mask:1x64 \
           --maxShapes=input_ids:1x256,token_type_ids:1x256,input_mask:1x256
   ```

   其中：

   - `--ONNX=BERT.ONNX`：指定输入的 ONNX 模型文件。
   - `--saveEngine=BERT.engine`：指定输出的 TensorRT 引擎文件。
   - `--explicitBatch`：启用显式批次模式。
   - `--minShapes` `--optShapes` `--maxShapes` 将为输入张量 `input_ids`、`token_type_ids` 和 `input_mask` 设置最小、最优和最大形状。

   **其他常用参数**：

   - `--fp16`：启用 FP16 精度。
   - `--int8`：启用 INT8 精度（需要校准数据）。
   - `--workspace=N`：设置最大 GPU 内存工作空间大小（以 MB 为单位）。
   - `--batch=N`：设置批次大小。

   例如，启用 FP16 精度并设置最大工作空间为 4096 MB：

   ```
   trtexec --ONNX=BERT.ONNX --saveEngine=BERT.engine --explicitBatch --workspace=4096 --fp16 \
           --minShapes=input_ids:1x6,token_type_ids:1x6,input_mask:1x6 \
           --optShapes=input_ids:1x64,token_type_ids:1x64,input_mask:1x64 \
           --maxShapes=input_ids:1x256,token_type_ids:1x256,input_mask:1x256
   ```

2. **使用TensorRT API**

   ```python
   def ONNX2trt(ONNXFile, plan_name):
       logger = trt.Logger(trt.Logger.VERBOSE)
   
       builder = trt.Builder(logger)
       config = builder.create_builder_config()
       profile = builder.create_optimization_profile()
       network = builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
       config.max_workspace_size = 3<<30
   
       parser = trt.ONNXParser(network, logger)
       print("Succeeded finding ONNX file!")
       with open(ONNXFile, 'rb') as model:
           if not parser.parse(model.read()):
               print("Failed parsing ONNX file!")
               for error in range(parser.num_errors):
                   print(parser.get_error(error))
               exit()
       print("Succeeded parsing ONNX file!")
   
       input_ids = network.get_input(0)
       token_type_ids = network.get_input(1)
       input_mask = network.get_input(2)
       profile.set_shape(input_ids.name, (1, 6), (1, 64), (1, 256))
       profile.set_shape(token_type_ids.name, (1, 6), (1, 64), (1, 256))
       profile.set_shape(input_mask.name, (1, 6), (1, 64), (1, 256))
       config.add_optimization_profile(profile)
   
       engine = builder.build_engine(network, config)
       if not engine:
           raise RuntimeError("build_engine failed")
       print("Succeeded building engine!")
   
       print("Serializing Engine...")
       serialized_engine = engine.serialize()
       if serialized_engine is None:
           raise RuntimeError("serialize failed")
   
       with open(plan_name, "wb") as fout:
           fout.write(serialized_engine
   ```

主要步骤如下：

1. 创建 TensorRT 的核心对象，包括日志记录器、构建器、配置对象、优化配置文件和网络对象。
2. 解析指定的 ONNX 文件，并将其转换为 TensorRT 网络表示。
3. 设置优化配置文件，定义输入张量的最小、最优和最大形状。
4. 构建 TensorRT 引擎，并将其序列化为字节流。
5. 将序列化的引擎保存到指定的文件路径。

### 4.4 使用TensorRT推理及测试

以下`InferHelper` 类是一个帮助类，用于加载 TensorRT 引擎并执行推理。

```python
class InferHelper():
    """"""
    def __init__(self, plan_name, trt_logger):
        """"""
        self.logger = trt_logger
        self.runtime = trt.Runtime(trt_logger)
        with open(plan_name, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()
            self.context.active_optimization_profile = 0

    def infer(self, inputs: list):
        nInput = len(inputs)

        bufferD = []
        # alloc memory
        for i in range(nInput):
            bufferD.append(cuda.mem_alloc(inputs[i].nbytes))
            cuda.memcpy_htod(bufferD[i], inputs[i].ravel())
            self.context.set_binding_shape(i, tuple(inputs[i].shape))
        outputs = []
        for i in range(len(inputs), self.engine.num_bindings):
            outputs.append(np.zeros(self.context.get_binding_shape(i)).astype(np.float32))

        nOutput = len(outputs)
        for i in range(nOutput):
            bufferD.append(cuda.mem_alloc(outputs[i].nbytes))
            # print(outputs[i].nbytes)

        for i in range(len(inputs), self.engine.num_bindings):
            trt_output_shape = self.context.get_binding_shape(i)
            output_idx = i - len(inputs)
            if not (list(trt_output_shape) == list(outputs[output_idx].shape)):
                self.logger.log(trt.Logger.ERROR, "[Infer] output shape is error!")
                self.logger.log(trt.Logger.ERROR, "trt_output.shape = " + str(trt_output_shape))
                self.logger.log(trt.Logger.ERROR, "base_output.shape = " + str(outputs[output_idx].shape))
                assert(0)

        for i in range(nInput, nInput + nOutput):
            cuda.memcpy_dtoh(outputs[i - nInput].ravel(), bufferD[i])

        for i in range(0, len(outputs)):
            print("outputs.shape:" + str(outputs[i].shape))
            print("outputs.sum:" + str(outputs[i].sum()))
        return outputs
```

`InferHelper` 类包含两个函数，一个是初始化函数，一个是推理函数。

初始化函数输入`plan_name` 是 TensorRT 引擎文件的路径， `trt_logger` 是 TensorRT 的日志记录器对象。处理步骤如下：

1. 创建 TensorRT 运行时对象 `self.runtime`。
2. 打开并读取 TensorRT 引擎文件。
3. 反序列化引擎文件，创建引擎对象 `self.engine`。
4. 创建执行上下文 `self.context`。
5. 设置活动优化配置文件为 0。

`infer` 方法用于执行推理，接受一个输入张量列表 `inputs`。

推理函数步骤如下：

1. 获取输入张量的数量 `nInput`。
2. 为每个输入张量分配 GPU 内存，并将数据从主机内存复制到设备内存。
3. 设置每个输入张量的形状。
4. 创建输出张量列表 `outputs`，并根据推理上下文中的绑定形状初始化输出张量。
5. 为每个输出张量分配 GPU 内存。
6. 检查推理上下文中的输出形状是否与预期的输出形状一致。如果不一致，记录错误日志并断言失败。
7. 将推理结果从设备内存复制回主机内存。
8. 打印输出张量的形状和元素和。
9. 返回输出张量列表。

通过这个类，可以方便地使用 TensorRT 引擎进行推理，并处理输入和输出张量的内存管理。



BERT使用`InferHelper` 的方法如下：

```
tokenizer = BERTTokenizer.from_pretrained('BERT-base-uncased')
text = "The capital of France, " + tokenizer.mask_token + ", contains the Eiffel Tower."
encoded_input = tokenizer.encode_plus(text, return_tensors = "pt")
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
infer_helper = InferHelper(plan_name, TRT_LOGGER)
input_list = [encoded_input['input_ids'].detach().numpy(), encoded_input['token_type_ids'].detach().numpy(), encoded_input['attention_mask'].detach().numpy()]
output = infer_helper.infer(input_list)
print(output)
```

以上函数这段代码，首先使用预训练的 BERT tokenizer 对输入文本进行编码。然后初始化 `InferHelper` 对象，加载 TensorRT 引擎。紧接着准备输入数据，将 PyTorch 张量转换为 NumPy 数组。最后使用 `InferHelper` 对象执行推理，并打印推理结果。

**性能测试**

`trtexec` 提供了许多参数来控制性能测试的行为。以下是一些常用参数：

- `--iterations=N`：设置要运行的推理次数。
- `--duration=N`：设置测试的持续时间（以秒为单位）。
- `--warmUp=N`：设置预热时间（以秒为单位）。
- `--batch=N`：设置批次大小。

使用方式：

```
trtexec --loadEngine=BERT.engine --batch=1 --shapes=input_ids:1x6,token_type_ids:1x6,input_mask:1x6 --fp16 --duration=60 --warmUp=10 --workspace=4096
```

运行 `trtexec` 命令后，会看到一系列的输出，包括推理的平均延迟、吞吐量等性能指标。以下是一些关键指标的解释：

- **Average latency**：平均延迟，表示每次推理的平均时间。
- **Throughput**：吞吐量，表示每秒处理的推理次数。
- **Host Walltime**：主机墙时间，表示整个测试过程的总时间。
- **GPU Compute Time**：GPU 计算时间，表示在 GPU 上执行推理的总时间。

### 4.5 实现TensorRT Plugin

在 TensorRT 中，Plugin是一个非常强大的功能，用于扩展和自定义 TensorRT 的能力。插件允许用户定义自定义的层（layer）或操作（operation），以便在 TensorRT 优化和推理过程中使用。这对于那些在标准 TensorRT 操作集中找不到的特殊操作或自定义操作特别有用。

- Plugin允许用户定义自定义的操作，这些操作可能在标准的 TensorRT 操作集中不存在。例如，某些特定的激活函数、归一化操作或其他复杂的计算。

- Plugin可以用来优化特定操作的性能。通过编写高效的 CUDA 代码，用户可以实现比标准 TensorRT 操作更高效的计算。

- Plugin使得 TensorRT 能够支持新的模型架构和操作。随着深度学习领域的快速发展，新模型和操作不断涌现，插件提供了一种灵活的方式来支持这些新特性。

编写以及使用TensorRT Plugin的流程

- 用户需要定义一个继承自 `IPluginV2` 或 `IPluginV2DynamicExt` 的类，并实现其虚函数。这些函数包括插件的初始化、执行、序列化和反序列化等。
- 定义好插件类后，需要将其注册到 TensorRT 中。可以使用 `IPluginCreator` 接口来实现插件的注册。
- 在构建 TensorRT 引擎时，可以通过 `INetworkDefinition` 接口将自定义插件层添加到网络中。

TensorRT的使用API有C++、Python，但编写Plugin只有Ç++，另外Kernel的实现函数需要用CUDA编写。

#### 4.5.1 TensorRT定义Plugin类

在 TensorRT 中定义插件类时，必须实现一系列关键的虚函数。这些函数负责插件的初始化、执行、序列化和反序列化等操作。通过实现这些函数，用户可以创建自定义的操作，并将其集成到 TensorRT 的优化和推理过程中。以下为一个Plugin类的一系列关键的虚函数。

```c++
#include "NvInfer.h"
#include "NvInferPlugin.h"

using namespace nvinfer1;

class MyCustomPlugin : public IPluginV2DynamicExt {
public:
    MyCustomPlugin() {}
    MyCustomPlugin(const void* data, size_t length) {}

    // Implement required virtual functions
    int getNbOutputs() const override { return 1; }
    DimsExprs getOutputDimensions(int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) override {
        return inputs[0];
    }
    int initialize() override { return 0; }
    void terminate() override {}
    size_t getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const override { return 0; }
    int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) override {
        // Implement your custom operation here
        return 0;
    }
    size_t getSerializationSize() const override { return 0; }
    void serialize(void* buffer) const override {}
    void destroy() override { delete this; }
    const char* getPluginType() const override { return "MyCustomPlugin"; }
    const char* getPluginVersion() const override { return "1"; }
    IPluginV2DynamicExt* clone() const override { return new MyCustomPlugin(); }
    void setPluginNamespace(const char* PluginNamespace) override {}
    const char* getPluginNamespace() const override { return ""; }
};
```

对于以上部分函数需要进行继承并实现。对部分重点函数进行介绍：

1. `getNbOutputs()`

```
int getNbOutputs() const override;
```

- **作用**：返回插件的输出数量。
- **返回值**：输出张量的数量。

2. `getOutputDimensions()`

```
DimsExprs getOutputDimensions(int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) override;
```

- **作用**：根据输入张量的维度，计算并返回输出张量的维度。TensorRT的输入输出shape必须要是已知或者可推导出的。
- **参数**
  - `outputIndex`：输出张量的索引。
  - `inputs`：输入张量的维度数组。
  - `nbInputs`：输入张量的数量。
  - `exprBuilder`：用于构建维度表达式的工具。
- **返回值**：输出张量的维度。

3. `getWorkspaceSize()`

```
size_t getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const override;
```

- **作用**：返回插件执行时所需的工作空间大小（以字节为单位）。在推理过程中如果需要申请显存，就在设置进行申请，在推理时使用
- **参数**
  - `inputs`：输入张量的描述数组。
  - `nbInputs`：输入张量的数量。
  - `outputs`：输出张量的描述数组。
  - `nbOutputs`：输出张量的数量。
- **返回值**：工作空间的大小（以字节为单位）。

4. `enqueue()`

```
int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) override;
```

- **作用**：执行插件的计算逻辑。这是最为重要的一个函数，是该Plugin实现前向推理的逻辑过程。
- **参数**
  - `inputDesc`：输入张量的描述数组。
  - `outputDesc`：输出张量的描述数组。
  - `inputs`：输入张量的数据指针数组。
  - `outputs`：输出张量的数据指针数组。
  - `workspace`：工作空间指针。
  - `stream`：CUDA 流。
- **返回值**：返回 0 表示成功，非 0 表示失败。

5. `getSerializationSize()`

```
size_t getSerializationSize() const override;
```

- **作用**：返回插件序列化所需的字节数。
- **返回值**：序列化所需的字节数。



#### 4.5.2 注册以及使用Plguin

另外需要进行注册，`MyCustomPlugin` 是这个Plugin的名次， `1` 是版本号，这个要是唯一的，不能跟已有的Plugin重复，不然会冲突。

```c++
class MyCustomPluginCreator : public IPluginCreator {
public:
    const char* getPluginName() const override { return "MyCustomPlugin"; }
    const char* getPluginVersion() const override { return "1"; }
    const PluginFieldCollection* getFieldNames() override { return nullptr; }
    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) override { return new MyCustomPlugin(); }
    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override { return new MyCustomPlugin(serialData, serialLength); }
    void setPluginNamespace(const char* PluginNamespace) override {}
    const char* getPluginNamespace() const override { return ""; }
};

REGISTER_TensorRT_Plugin(MyCustomPluginCreator);
```

通过调用`REGISTER_TensorRT_Plugin`将该Plugin的信息放入一个全局变量，在使用的时候会根据PluginName和PluginVersion进行匹配。

在完成Plugin的函数实现以及注册后需要编译为动态链接库，以便于后续的使用。

**使用Plugin**

```python
INetworkDefinition* network = builder->createNetworkV2(0);
ITensor* input = network->addInput("input", DataType::kFLOAT, Dims3{1, 28, 28});
IPluginV2Layer* customLayer = network->addPluginV2(&input, 1, MyCustomPlugin());
network->markOutput(*customLayer->getOutput(0));
```

以上代码为使用c++在使用API搭建网络时，使用Plugin作为一层Layer，放入到网络中。另外在编译时需要链接上一步编译好的Plugin的动态链接库。

#### 4.5.3 LayerNorm Plugin

在TensorRT7版本中对Transformer支持的还不够好，其中对LayerNorm算子并不支持，在这种情况如果使用TensorRT进行BERT推理，就需要编写LayerNorm的Plugin。

在前面我们已经搞清楚了编写Plugin的步骤，以及LayerNorm的计算过程。下面以该算子为例，实际展示一下如何实现该Plugin。

常量定义，定义了插件的版本和名称。

```cpp
constexpr const char* LAYER_NORM_VERSION{"1"};
constexpr const char* LAYER_NORM_NAME{"LayerNormPluginDynamic"};
```
构造函数

```cpp
LayerNormPlugin::LayerNormPlugin(const std::string& name, const nvinfer1::DataType type, const size_t dim, const float eps)
    : layer_name_(name), dim_(dim), data_type_(type), eps_(eps) {}

LayerNormPlugin::LayerNormPlugin(const std::string& name, const void* data, size_t length) : layer_name_(name) {
  // Deserialize in the same order as serialization
  deserialize_value(&data, &length, &data_type_);
  deserialize_value(&data, &length, &dim_);
  deserialize_value(&data, &length, &eps_);
}
```
- 第一个构造函数用于初始化插件的名称、数据类型、维度和 epsilon 值。
- 第二个构造函数用于反序列化插件的状态，从而恢复插件的内部状态。

`IPluginV2DynamicExt` 方法实现

1. `clone`

```cpp
IPluginV2DynamicExt* LayerNormPlugin::clone() const TRTNOEXCEPT {
  auto ret = new LayerNormPlugin(layer_name_, data_type_, dim_, eps_);
  return ret;
}
```
这个方法用于克隆插件对象，返回一个新的 `LayerNormPlugin` 实例。

2. `getOutputDimensions`

```cpp
DimsExprs LayerNormPlugin::getOutputDimensions(int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) TRTNOEXCEPT {
  assert(nbInputs == 3);
  return inputs[0];
}
```
这个方法用于获取输出张量的维度。它假设输入张量的数量为 3，并返回第一个输入张量的维度作为输出维度。

2. `supportsFormatCombination`

```cpp
bool LayerNormPlugin::supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) TRTNOEXCEPT {
  assert(nbInputs == 3);
  assert(nbOutputs == 1);

  const PluginTensorDesc& in_out = inOut[pos];
  return (in_out.type == data_type_) && (in_out.format == TensorFormat::kLINEAR);
}
```
这个方法用于检查插件是否支持特定的数据格式和类型。它假设输入张量的数量为 3，输出张量的数量为 1，并检查数据类型和格式是否匹配。

4. `configurePlugin`

```cpp
void LayerNormPlugin::configurePlugin(const DynamicPluginTensorDesc* inputs, int nbInputs, const DynamicPluginTensorDesc* outputs, int nbOutputs) TRTNOEXCEPT {
  // Validate input arguments
  assert(nbInputs == 3);
  assert(nbOutputs == 1);
  assert(data_type_ == inputs[0].desc.type);
}
```
这个方法用于配置插件，验证输入和输出张量的数量和数据类型。

`getWorkspaceSize`

```cpp
size_t LayerNormPlugin::getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const TRTNOEXCEPT {
  return 0;
}
```
这个方法返回插件所需的工作空间大小。在这个例子中，工作空间大小为 0。

5. `enqueue`

   这个方法是插件的核心计算部分。它根据数据类型（`kFLOAT` 或 `kHALF`）选择不同的计算路径，并调用 `compute_layer_norm` 函数来执行实际的 Layer Normalization 操作。

```cpp
int LayerNormPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) TRTNOEXCEPT {
  const int input_volume = volume(inputDesc[0].dims);
  const int S = input_volume / dim_;
  int status = -1;

  const size_t word_size = getElementSize(data_type_);

  if (data_type_ == DataType::kFLOAT) {
    // Our Plugin outputs only one tensor
    const float* input = static_cast<const float*>(inputs[0]);
    const float* gamma_ptr = static_cast<const float*>(inputs[1]);
    const float* beta_ptr = static_cast<const float*>(inputs[2]);
    float* output = static_cast<float*>(outputs[0]);

    // status = compute_layer_norm(stream, dim_, input_volume, input, gamma_ptr, beta_ptr, output);
    status = compute_layer_norm(stream, S, dim_, input, gamma_ptr, beta_ptr, output);

  } else if (data_type_ == DataType::kHALF) {
    // Our Plugin outputs only one tensor
    const half* input = static_cast<const half*>(inputs[0]);
    const half* gamma_ptr = static_cast<const half*>(inputs[1]);
    const half* beta_ptr = static_cast<const half*>(inputs[2]);
    half* output = static_cast<half*>(outputs[0]);

    // status = compute_layer_norm(stream, dim_, input_volume, input, gamma_ptr, beta_ptr, output);
    status = compute_layer_norm(stream, S, dim_, input, gamma_ptr, beta_ptr, output);

  } else {
    assert(false);
  }

  return status;
}
```
在调用enque之后会去调用CUDA代码实现LayerNorm的前向过程， 会根据输入维度的大小，设置不同的grid、block、thread并采用不同的CUDA函数实现，这里展示并简单讲解其中的一种`layer_norm_kernel_small`

```C++
template <typename T, typename OP_T, int TPB>
__global__ void layer_norm_kernel_small(const int nHiddenDimension, const T* input, const T* gamma, const T* beta,
                                        T* output) {
  const int index = blockIdx.x * nHiddenDimension + threadIdx.x;
  const T denominator = T(1) / T(nHiddenDimension);
  OP_T val = 0;
  kvp<OP_T> threadData(0, 0);

  if (threadIdx.x < nHiddenDimension) {
    val = input[index] * denominator;
    OP_T tmp0 = val * (OP_T)denominator, tmp1 = val * tmp0;
    threadData = mySum<OP_T>()(threadData, kvp<OP_T>(tmp0, tmp1));
  }

  using WarpReduce = cub::WarpReduce<kvp<OP_T>, TPB>;
  __shared__ typename WarpReduce::TempStorage temp;
  __shared__ OP_T mu, rsigma;

  const auto sumKV = WarpReduce(temp).Reduce(threadData, mySum<OP_T>());

  if (threadIdx.x == 0) {
    mu = sumKV.key;
    rsigma = rsqrt(sumKV.value - mu * mu);
  }
  __syncthreads();

  if (threadIdx.x < nHiddenDimension) {
    const OP_T g = gamma[threadIdx.x], b = beta[threadIdx.x];
    output[index] = (val - mu) * rsigma * g + b;
  }
}
```

这个核函数主要实现过程如下，这也是LayerNorm的实际计算过程（前面有公式计算过程）

1. 计算每个线程的输入数据值和中间结果。
2. 使用 Warp Reduce 计算所有线程的均值和方差。
3. 计算均值和反标准差。
4. 计算归一化后的输出值。



其中使用 CUB 库的 Warp Reduce 功能来计算均值和方差。计算均值和方差

```
using WarpReduce = cub::WarpReduce<kvp<OP_T>, TPB>;
__shared__ typename WarpReduce::TempStorage temp;
__shared__ OP_T mu, rsigma;

const auto sumKV = WarpReduce(temp).Reduce(threadData, mySum<OP_T>());
```

- `temp`：共享内存，用于存储 Warp Reduce 的临时数据。
- `mu` 和 `rsigma`：共享内存变量，用于存储均值和反标准差。
- `sumKV`：使用 Warp Reduce 计算所有线程的中间结果的总和。

#### 4.5.4 将LayerNorm Plugin插入到网络中

```python
handle = ctypes.CDLL("LayerNorm.so", mode=ctypes.RTLD_GLOBAL)
def addLayerNorm(network, layer, x, layer_name=None, precision=None):
    gamma = layer.weight
    beta = layer.bias

    plg_creator = plg_registry.get_Plugin_creator("LayerNorm", "1", "")
    if not plg_creator:
        raise RuntimeError("Could not find LayerNorm")

    # pfc = trt.PluginFieldCollection([data_type, dim, eps, gamma_w, beta_w])
    pfc = trt.PluginFieldCollection([])
    Plugin = plg_creator.create_Plugin("LayerNorm", pfc)
    if not Plugin:
        raise RuntimeError("Could not create_Plugin LayerNormPluginDynamic")

    gamma = network.add_constant(gamma.shape, trt.Weights(layer.weight.detach().numpy())).get_output(0)
    beta = network.add_constant(beta.shape, trt.Weights(layer.bias.detach().numpy()) ).get_output(0)

    trt_layer = network.add_Plugin_v2([x, gamma, beta], Plugin)

    return trt_layer.get_output(0)
```

这段代码的主要功能是将一个LayerNorm层添加到TensorRT网络中。

代码使用`ctypes`库加载一个名为`LayerNorm.so`的共享库，并将其设置为全局加载模式。这通常是为了确保库中的符号可以被其他库或模块访问。
```python
handle = ctypes.CDLL("LayerNorm.so", mode=ctypes.RTLD_GLOBAL)
```
1. **获取Plugin creator**：

```python
plg_creator = plg_registry.get_Plugin_creator("LayerNorm", "1", "")
if not plg_creator:
    raise RuntimeError("Could not find LayerNorm")
```
使用Plugin注册表获取名为`LayerNorm`的插件创建器。如果找不到该插件创建器，则抛出运行时错误。

2. **创建插件字段集合**：

```python
pfc = trt.PluginFieldCollection([])
```
创建一个空的插件字段集合。这个集合可以用来传递插件所需的参数，但在这个例子中没有传递参数。

3. **创建插件**：

```python
Plugin = plg_creator.create_Plugin("LayerNorm", pfc)
if not Plugin:
    raise RuntimeError("Could not create_Plugin LayerNormPluginDynamic")
```
使用插件创建器创建名为`LayerNorm`的插件。如果创建失败，则抛出运行时错误。

4. **添加常量层**：

```python
gamma = network.add_constant(gamma.shape, trt.Weights(layer.weight.detach().numpy())).get_output(0)
beta = network.add_constant(beta.shape, trt.Weights(layer.bias.detach().numpy())).get_output(0)
```
将权重和偏置转换为TensorRT的常量层，并获取它们的输出。

5. **添加插件层**：

```python
trt_layer = network.add_Plugin_v2([x, gamma, beta], Plugin)
```
使用`add_Plugin_v2`方法将插件层添加到网络中，输入包括原始输入张量`x`、权重`gamma`和偏置`beta`。

### 4.6 使用TensorRT量化

在深度学习模型压缩中，模型量化是一种常用的技术，用于减少模型的大小和计算复杂度，同时尽量保持模型的性能。量化是将模型的权重和激活从高精度（如 FP32）转换为低精度（如 FP16 或 INT8）的过程。低精度计算通常比高精度计算更快，因为它们需要的计算资源更少。 NVIDIA GPU对低精度计算进行了优化，能够显著提高推理速度。另外低精度数据类型占用的内存更少，例如，FP16 占用的内存是 FP32 的一半，INT8 占用的内存是 FP32 的四分之一。如果想要系统学习模型量化的知识，尤其是大模型量化，可以学习深蓝学院深度学习模型压缩的课程，课程中会系统讲解模型量化的概念以及代码实践。

FP32、FP16 、 TF32和INT8 是三种不同的浮点数数据格式，它们在表示范围、精度和存储需求上各有不同。

FP32 是标准的 32 位浮点数格式，符合 IEEE 754 标准。它由三个部分组成：符号位、指数位和尾数位。

- **符号位（1 位）**：表示数值的正负。
- **指数位（8 位）**：表示数值的范围。
- **尾数位（23 位）**：表示数值的精度。
- **表示范围**：约为 1.4×$10^{-45}$ 到 3.4×$10^{38}$

FP16 是 16 位浮点数格式，也符合 IEEE 754 标准。它同样由三个部分组成：符号位、指数位和尾数位。

- **符号位（1 位）**：表示数值的正负。
- **指数位（5 位）**：表示数值的范围。
- **尾数位（10 位）**：表示数值的精度。
- **表示范围**：约为 6.1×$10^{-5}$ 到 6.5×$10^{4}$

TF32 是 NVIDIA 提出的专用于深度学习的浮点数格式，旨在在保持较高精度的同时提高计算性能。TF32 结合了 FP32 和 FP16 的特点。

- **符号位（1 位）**：表示数值的正负。
- **指数位（8 位）**：与 FP32 相同，表示数值的范围。
- **尾数位（10 位）**：与 FP16 相同，表示数值的精度。
- **表示范围**：与 FP32 相同，约为 1.4×$10^{-45}$ 到 3.4×$10^{38}$。

INT8 是一种 8 位整数格式，可以表示有符号整数或无符号整数。

- **符号位（1 位）**：表示数值的正负。
- **数值位（7 位）**：表示数值的大小。
- **表示范围**：-128 到 127。

<img src="https://xsj-niehen.oss-cn-hangzhou.aliyuncs.com/Docs_assert/numerical-formats-supported-by-ampere-gpus-1.png" alt="Accelerating TensorFlow on NVIDIA A100 GPUs | NVIDIA Technical Blog" style="zoom:75%;" />

#### 4.6.1 在TensorRT中使用FP16

在前面也有提到过，使用FP16比较简单。

使用API时在config中进行设置 `config->setFlag(nvinfer1::BuilderFlag::kFP16);` (C++) `builder_config.set_flag(trt.BuilderFlag.FP16)` (Python)

使用trtexec进行ONNX转换时加入参数 `-fp16`

#### 4.6.2 在TensorRT中使用INT8

在INT8量化中，关键的一步是确定合适的阈值（scale），这个阈值决定了如何将浮点数的激活值映射到8位整数。

首先，通过对模型中每一层的激活值进行统计分析，可以观察到激活值的分布特点：大部分激活值集中在较低的范围内，而较大的激活值非常少，几乎可以忽略。这一观察表明，使用饱和量化（即将最大激活值映射到127）可能不是最优的选择，因为这会将时间浪费在大量的量化范围在极少数的大激活值上。

<img src="https://xsj-niehen.oss-cn-hangzhou.aliyuncs.com/Docs_assert/distribution-of-different-layers.png.png" alt="img" style="zoom:67%;" />

为了更有效地利用量化范围，TensorRT采用一种动态确定阈值的方法。具体做法是：
1. 将每层特征图的激活值分到2048个bins中，每个bin代表一个激活值区间；
2. 从第127个bin开始，尝试将每个bin的中间值作为阈值，并计算相应的量化结果；
3. 对于每个尝试的阈值，将低于该阈值的激活值映射到0-127之间的整数，高于阈值的激活值则映射到127；
4. 生成一个128维的分布向量，表示每个量化值的元素个数；
5. 通过计算原始分布（2048个bins）和量化后分布（128维向量）之间的相似度（使用KL散度），评估每个阈值的合理性；
6. 选择使得KL散度最小的阈值作为最终的scale。

<img src="https://xsj-niehen.oss-cn-hangzhou.aliyuncs.com/Docs_assert/satuation_int8_quantization.png" alt="img" style="zoom:67%;" />

这种方法的核心在于通过动态调整阈值，使得量化后的分布尽可能接近原始分布，从而减少量化带来的信息损失。通过这种方式，可以在保持模型性能的同时，有效地减少模型的计算和存储需求。另外这种做法需要使用一个校准数据集进行量化参数。

这里依旧使用BERT为例子，讲解如何使用int8量化，相比于FP16，需要准备校准数据集，并定义一个自定义的校准器类，继承自`trt.IInt8EntropyCalibrator`。

**准备数据集**

跟上面使用到的得到BERT输入的API基本一致，是将输入的文本转换为适合模型输入的格式。具体来说，它使用一个分词器（tokenizer）对文本进行编码，并生成模型所需的输入张量。

```
def text2inputs(tokenizer, text):
    encoded_input = tokenizer.encode_plus(text, return_tensors = "pt")

    input_ids = encoded_input['input_ids'].int().detach().numpy()
    token_type_ids = encoded_input['token_type_ids'].int().detach().numpy()
    # position_ids = Torch.arange(0, encoded_input['input_ids'].shape[1]).int().view(1, -1).numpy()
    seq_len = encoded_input['input_ids'].shape[1]
    position_ids = np.arange(seq_len, dtype = np.int32).reshape(1, -1)
    input_list = [input_ids, token_type_ids, position_ids]

    return input_list
```

**实现一个校准类**

`BERTCalibrator` 类用于在 TensorRT 中进行 BERT 模型的 INT8 量化校准，并继承 `trt.IInt8EntropyCalibrator`。它通过读取文本数据并将其转换为模型输入格式，然后在校准过程中提供这些输入数据。该类还实现了读取和写入校准缓存的功能，以便在多次运行时复用校准数据。

```python
class BERTCalibrator(trt.IInt8LegacyCalibrator):
    def __init__(self, data_txt, BERT_path, cache_file, batch_size, max_seq_length, num_inputs):
        trt.IInt8LegacyCalibrator.__init__(self)

        tokenizer = BERTTokenizer.from_pretrained(BERT_path)
        self.tokenizer = BERTTokenizer.from_pretrained(BERT_path)

        self.input_ids_list = []
        self.token_type_ids_list = []
        self.position_ids_list = []
        with open(data_txt, "r") as f:
            lines = f.readlines()
            for i in range(0, num_inputs):
                inputs = text2inputs(tokenizer, lines[i])
                self.input_ids_list.append(inputs[0])
                self.token_type_ids_list.append(inputs[1])
                self.position_ids_list.append(inputs[2])
                if i % 10 == 0:
                    print("text2inputs:" + lines[i])

        self.cache_file = cache_file
        # Every time get_batch is called, the next batch of size batch_size will be copied to the device and returned.
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.current_index = 0
        if num_inputs > len(self.input_ids_list):
            self.num_inputs = len(self.input_ids_list)
        else:
            self.num_inputs = num_inputs
        self.doc_stride = 128
        self.max_query_length = 64

        # Allocate enough memory for a whole batch.
        self.device_inputs = [cuda.mem_alloc(self.max_seq_length * trt.int32.itemsize * self.batch_size) for binding in range(3)]

    def free(self):
        for dinput in self.device_inputs:
            dinput.free()

    def get_batch_size(self):
        return self.batch_size
    def get_batch(self, names):
        if self.current_index + self.batch_size > self.num_inputs:
            print("Calibrating index {:} batch size {:} exceed max input limit {:} sentences".format(self.current_index, self.batch_size, self.num_inputs))
            return None

        current_batch = int(self.current_index / self.batch_size)
        if current_batch % 10 == 0:
            print("Calibrating batch {:}, containing {:} sentences".format(current_batch, self.batch_size))

        input_ids = self.input_ids_list[self.current_index]
        token_type_ids = self.token_type_ids_list[self.current_index]
        position_ids = self.position_ids_list[self.current_index]

        seq_len = input_ids.shape[1]
        if seq_len > self.max_seq_length:
            print(seq_len)
            print(input_ids.shape)
            input_ids = input_ids[:, :self.max_seq_length]
            token_type_ids = token_type_ids[:, :self.max_seq_length]
            position_ids = position_ids[:, :self.max_seq_length]
            print(input_ids.shape)

        cuda.memcpy_htod(self.device_inputs[0], input_ids.ravel())
        cuda.memcpy_htod(self.device_inputs[1], token_type_ids.ravel())
        cuda.memcpy_htod(self.device_inputs[2], position_ids.ravel())

        self.current_index += self.batch_size
        return self.device_inputs

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
            f.flush()
            os.fsync(f)

    def get_quantile(self):
        return 0.9999

    def get_regression_cutoff(self):
        return 1.0

    def read_histogram_cache(self, length):
        return None

    def write_histogram_cache(self, ptr, length):
        return None
```

在初始化的时候有以下步骤：

- 调用父类的构造函数 `trt.IInt8LegacyCalibrator.__init__(self)`。
- 使用 `BERTTokenizer` 从预训练的 BERT 模型路径初始化分词器。
- 从文本文件中读取数据，并使用 `text2inputs` 函数将文本转换为 BERT 模型的输入格式（`input_ids`, `token_type_ids`, `position_ids`）。
- 设置校准所需的参数，如缓存文件路径、批处理大小、最大序列长度等。
- 分配足够的 GPU 内存用于存储整个批次的数据。

对一些需要重点注意的，需要实现的函数，进行简单介绍：

-  `get_batch_size()` 函数， 用于返回批处理大小。
- `get_batch` 方法根据当前索引获取下一个批次的输入数据，并将其复制到 GPU 内存中。如果当前索引加上批处理大小超过输入数据的数量，则返回 `None`

- `read_calibration_cache` 方法检查是否存在校准缓存文件，如果存在则读取缓存。
- `write_calibration_cache` 方法将校准缓存写入文件。
- `get_quantile` 和 `get_regression_cutoff` 方法返回量化所需的参数。
- `read_histogram_cache` 和 `write_histogram_cache` 方法用于读取和写入直方图缓存。

## 5. 参考

https://huggingface.co/blog/BERT-101

https://arxiv.org/pdf/1810.04805

https://jalammar.github.io/a-visual-guide-to-using-BERT-for-the-first-time/

https://wmathor.com/index.php/archives/1438/

https://luweikxy.gitbook.io/machine-learning-notes/self-attention-and-transformer#encoder-mo-kuai

https://armanasq.github.io/nlp/self-attention/

https://adaning.github.io/posts/52648.html#toc-heading-1

https://github.com/NVIDIA/FasterTransformer/blob/main/docs/BERT_guide.md#run-fastertransformer-BERT-on-c

https://github.com/NVIDIA/TensorRT/tree/release/10.0/demo/BERT

https://arxiv.org/pdf/2210.03052

https://arleyzhang.github.io/articles/923e2c40/
