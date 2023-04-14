# ConvLSTM

## ConvLSTM网络设计

- 全卷积LSTM主要公式
  ![CLSTM_dynamics](https://user-images.githubusercontent.com/7113894/59357391-15c73e00-8d2b-11e9-8234-9d51a90be5dc.png)

### 实现

~~~python
self.Wxi = nn.Conv2d(in_channels=self.input_channels,
                     out_channels=self.hidden_channels,
                     kernel_size=self.kernel_size,
                     stride=self.stride,
                     padding=self.padding,
                     bias=False)
self.Whi = nn.Conv2d(in_channels=self.hidden_channels,
                     out_channels=self.hidden_channels,
                     kernel_size=self.kernel_size,
                     stride=self.stride,
                     padding=self.padding,
                     bias=False)
self.Wci = torch.tensor(torch.zeros(1, self.hidden_channels, self.shape[0], self.shape[1]))
# parameters of forget gate
self.Wxf = nn.Conv2d(in_channels=self.input_channels,
                     out_channels=self.hidden_channels,
                     kernel_size=self.kernel_size,
                     stride=self.stride,
                     padding=self.padding,
                     bias=False)
self.Whf = nn.Conv2d(in_channels=self.hidden_channels,
                     out_channels=self.hidden_channels,
                     kernel_size=self.kernel_size,
                     stride=self.stride,
                     padding=self.padding,
                     bias=False)
self.Wcf = torch.tensor(torch.zeros(1, self.hidden_channels, self.shape[0], self.shape[1]))
# parameters of output gate
self.Wxo = nn.Conv2d(in_channels=self.input_channels,
                     out_channels=self.hidden_channels,
                     kernel_size=self.kernel_size,
                     stride=self.stride,
                     padding=self.padding,
                     bias=False)
self.Who = nn.Conv2d(in_channels=self.hidden_channels,
                     out_channels=self.hidden_channels,
                     kernel_size=self.kernel_size,
                     stride=self.stride,
                     padding=self.padding,
                     bias=False)
self.Wco = torch.tensor(torch.zeros(1, self.hidden_channels, self.shape[0], self.shape[1]))

# parameters of hidden state
self.Wxc = nn.Conv2d(in_channels=self.input_channels,
                     out_channels=self.hidden_channels,
                     kernel_size=self.kernel_size,
                     stride=self.stride,
                     padding=self.padding,
                     bias=False)
self.Whc = nn.Conv2d(in_channels=self.hidden_channels,
                     out_channels=self.hidden_channels,
                     kernel_size=self.kernel_size,
                     stride=self.stride,
                     padding=self.padding,
                     bias=True)
~~~

***

## 解决问题

- 解决时空预测问题，解决时间上的依赖问题

~~~python
for t in range(step_numbers):
    x = input[t,:,:,:,:]
    for i in range(self.layer_numbers):
        name = 'cell{}'.format(i)
        x, hidden_states[i] = getattr(self, name)(x, hidden_states[i])
    output.append(x)
    state.append(tuple(hidden_states))
output = torch.cat(output, 0).view(step_numbers, batch_size, self.hidden_channels[i], self.shape[0], self.shape[1])
~~~

***

## 网络模型的构建

### Encoding

- 使用了ConvLSTM层进行构造，只向Forcasting层传入隐藏层的输出

~~~python
class Encoding(nn.Module):
    def __init__(self, layer_numbers, input_channels, hidden_channels, shape, kernel_size = (3, 3), stride = 1):
        super(Encoding, self).__init__()
        self.layer_numbers = layer_numbers
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.shape = shape
        self.kernel_size = kernel_size
        self.stride =stride
        assert self.layer_numbers == len(self.hidden_channels)

        self.layers = ConvLSTM(self.layer_numbers, self.input_channels, self.hidden_channels, self.shape, self.kernel_size, self.stride)

    def forward(self, input, hidden_states = None):
        output, all_hidden_states = self.layers(input, hidden_states)

        return all_hidden_states
~~~

### Forcasting

- 由ConvLSTM层构成，在预测中需要传入给定的时间区间来预测下一时间点的状态；

  输出隐藏层状态和得到的结果

~~~python
class Forcasting(nn.Module):
    def __init__(self, layer_numbers, input_channels, hidden_channels, shape, kernel_size=(3, 3), stride=1):
        super(Forcasting, self).__init__()
        self.layer_numbers = layer_numbers
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.shape = shape
        self.kernel_size = kernel_size
        self.stride = stride
        assert self.layer_numbers == len(self.hidden_channels)

        self.layers = ConvLSTM(self.layer_numbers, self.input_channels, self.hidden_channels, self.shape,
                               self.kernel_size, self.stride)

    def forward(self, output_stepnumber, hidden_states):
        assert hidden_states != None
        input = torch.tensor(torch.zeros(output_stepnumber, hidden_states[0][0].size()[0], hidden_states[0][0].size()[1], hidden_states[0][0].size()[2], hidden_states[0][0].size()[3]))
        output, all_hidden_states = self.layers(input, hidden_states)

        return output, all_hidden_states
~~~

### EncodingForcasting

- 将forcasting的最后一层状态传入预测层，输出为encoding输入的维度

- *最后将forcasting层的所有输出全卷积为输入的维度（将FC-LSTM中的全连接变成全卷积）*

***

## 损失函数和优化

- 使用交叉熵作为损失函数；SGD作为优化条件，学习率为1e-2

~~~python
criterion = MyCrossEntropyLoss()
optimizer = torch.optim.SGD(convLSTM.parameters(), lr = 1e-2)

class MyCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MyCrossEntropyLoss, self).__init__()

    def forward(self, predit, truth):
        out = - torch.mean(truth * torch.log(predit) + (1 - truth) * torch.log(1 - predit))
        return out
~~~

