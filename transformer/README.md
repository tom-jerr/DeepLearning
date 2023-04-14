# Transformer

- 主体是一个`Encoder-Decoder`模型

  ![](https://github.com/tom-jerr/MyblogImg/blob/main/object_detection/encoder_decoder.png)
  
  ## 网络结构
  
  ![](https://github.com/tom-jerr/MyblogImg/blob/main/object_detection/Transformer.png)

- 其中，编码组件由多层编码器（Encoder）组成（在论文中作者使用了 6 层编码器，在实际使用过程中你可以尝试其他层数）。解码组件也是由相同层数的解码器（Decoder）组成（在论文也使用了 6 层）。

  ![](https://github.com/tom-jerr/MyblogImg/blob/main/object_detection/Transformer_1.png)

- 每个编码器由两个子层组成：[Self-Attention](https://so.csdn.net/so/search?q=Self-Attention&spm=1001.2101.3001.7020) 层（自注意力层）和 Position-wise Feed Forward Network（前馈网络，缩写为 FFN）

- 解码器也有编码器中这两层，但是它们之间还有一个注意力层（即 Encoder-Decoder Attention），其用来帮忙解码器关注输入句子的相关部分（类似于 seq2seq 模型中的注意力）。

  ![](https://github.com/tom-jerr/MyblogImg/blob/main/object_detection/Transformer_2.png)

## Self-Attention机制

![](https://github.com/tom-jerr/MyblogImg/blob/main/object_detection/self-attention.png)

- 对于 Self Attention 来讲，Q（Query），K（Key）和 V（Value）三个矩阵均来自同一输入，并按照以下步骤计算：

  - 首先计算 Q 和 K 之间的点积，为了防止其结果过大，会除以 $ \sqrt{d_{k}} $，其中$d_{k}$为 Key 向量的维度。
  - 然后利用 Softmax 操作将其结果归一化为概率分布，再乘以矩阵 V 就得到权重求和的表示。

  整个计算过程可以表示为：
  $$
  Attention(Q,K,V)=softmax(\frac{QK^{T}}{\sqrt{d_{k}}})
  $$
  ![](https://github.com/tom-jerr/MyblogImg/blob/main/object_detection/self-attention_1.png)

### 矩阵计算Self-Attention

![](https://github.com/tom-jerr/MyblogImg/blob/main/object_detection/self-attention_2.png)

![](https://github.com/tom-jerr/MyblogImg/blob/main/object_detection/self-attention_3.png)

## Multi-head Attention

![](https://github.com/tom-jerr/MyblogImg/blob/main/object_detection/multi-head_attention.png)
$$
MultiHead(Q,K,V)=Concat(head_{1},\cdots,head_{h})W^{O}
$$

$$
head_i=Attention(QW^{Q}_{i},KW^{K}_{i},VW^{V}_{i})
$$
![](https://github.com/tom-jerr/MyblogImg/blob/main/object_detection/multi-head_attention_matrix.png)

## 位置前馈网络（Position-wise Feed-Forward Networks）

- 位置前馈网络就是一个全连接前馈网络，每个位置的词都单独经过这个完全相同的前馈神经网络。其由两个线性变换组成，即两个全连接层组成，第一个全连接层的激活函数为 ReLU 激活函数。可以表示为：

$$
FFN(x)=max(0,xW_{1}+b_{1})W_{2}+b_{2}
$$

## 残差连接和LayerNorm

$$
sub\_layer\_output=LayerNorm(x+SubLayer(x))
$$

![](https://github.com/tom-jerr/MyblogImg/blob/main/object_detection/sublayerout.png)

- 为了方便进行残差连接，编码器和解码器中的所有子层和嵌入层的输出维度需要保持一致，在 Transformer 论文中 $d_{model}=512$
  

## 位置编码

- Transformer 模型为每个输入的词嵌入向量添加一个向量。这些向量遵循模型学习的特定模式，有助于模型确定每个词的位置，或序列中不同词之间的距离。

![](https://github.com/tom-jerr/MyblogImg/blob/main/object_detection/position_code.png)

- **公式：**$pos$表示位置，$i$表示维度

$$
PE_{(pos,2i)}=sin(pos/10000^{wi/d_{model}})
$$

$$
PE_{(pos,2i+1)}=cos(pos/10000^{wi/d_{model}})
$$

## Mask(掩码)

- Padding Mask
  - 因为每个批次输入序列的长度是不一样的，所以我们要对输入序列进行对齐。具体来说，就是在较短的序列后面填充 0（但是如果输入的序列太长，则是截断，把多余的直接舍弃）。
  - 把这些位置的值加上一个非常大的负数（负无穷），这样的话，经过 Softmax 后，这些位置的概率就会接近 0。

- Sequence Mask
  - Sequence Mask 是为了使得 Decoder 不能看见未来的信息。
  - 产生一个上三角矩阵，上三角的值全为 0。把这个矩阵作用在每个序列上，就可以达到我们的目的。

##  最后的线性层和 Softmax 层

- 现在假设我们的模型有 10000 个英文单词（模型的输出词汇表）。因此 logits 向量有 10000 个数字，每个数表示一个单词的分数。
- 然后，Softmax 层会把这些分数转换为概率（把所有的分数转换为正数，并且加起来等于 1）。最后选择最高概率所对应的单词，作为这个时间步的输出。

![](https://github.com/tom-jerr/MyblogImg/blob/main/object_detection/Transformer_last.png)

## 贪心解码

- 采用贪心算法来进行解码，即在每个时间步选择概率最大的输出作为下一个输入。该过程一直持续到生成的序列以 "." 结尾。在每个时间步中，输入序列是由之前生成的目标单词构成的。每次将生成的单词添加到输入序列的末尾，并使用编码器和解码器来计算输出。接着使用模型的投影层将解码器的输出转换为目标序列的概率分布，并选择概率最大的单词作为下一个输入。

~~~python
def greedy_decoder(model, enc_input, start_symbol):
    """
    For simplicity, a Greedy Decoder is Beam search when K=1. This is necessary for inference as we don't know the
    target sequence input. Therefore we try to generate the target input word by word, then feed it into the transformer.
    Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
    :param model: Transformer Model
    :param enc_input: The encoder input (batch_size, seq_len)
    :param start_symbol: The start symbol. In this example it is 'S' which corresponds to index 4
    :return: The target input
    """
    enc_outputs, enc_self_attns = model.encoder(enc_input)
    dec_input = torch.zeros(1, 0).type_as(enc_input.data)
    terminal = False
    next_symbol = start_symbol
    # 在每个时间步中，将生成的单词添加到输入序列的末尾，通过 torch.cat 函数将上一个生成的单词添加到输入序列中。
    while not terminal:
        # (1, seq_len) -> (1, seq_len + 1)         
        dec_input = torch.cat([dec_input.detach(),torch.tensor([[next_symbol]],dtype=enc_input.dtype).cuda()],-1)
        dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
        # (1, seq_len, tgt_vocab_size)
        projected = model.projection(dec_outputs)
        # prob:(1, )		(seq_len, tgt_vocab_size) -> (1, )
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        # 将张量中的唯一元素取出，并转换为标量
        next_word = prob.data[-1]
        next_symbol = next_word
        if next_symbol == tgt_vocab["."]:
            terminal = True
        #print(next_word)            
    return dec_input
~~~

