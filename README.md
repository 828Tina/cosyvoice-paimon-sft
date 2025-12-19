# CosyVoice2 微调派蒙语音

## 简介📖

一直以来我们的教程基本都是自然语言领域的训练，多模态领域也基本就是图像相关，很少研究音频模型，本次教程我们就研究研究如何来训练一波音频模型。

模型我们选择通义实验室发布的CosyVoice2模型来微调原神中的派蒙语音，我们会非常详细地按照步骤教会大家如何训练CosyVoice。并且作者也会写出自己对于CosyVoice原理的理解，希望能帮到各位读者。

> 特此声明：本次教程仅作为AI模型训练，数据集均来自开源数据集。

**详细教程和SwanLab观测结果链接如下：**

[![知乎](https://img.shields.io/static/v1?label=📖&message=教程&color=blue)](https://zhuanlan.zhihu.com/p/1984999416596276696)
[![SwanLab](https://img.shields.io/static/v1?label=📈&message=SwanLab&color=green)](https://swanlab.cn/@LiXinYu/cosyvoice-sft/overview)

## 环境安装⚙️

- 克隆代码

```bash
git clone --recursive https://github.com/828Tina/cosyvoice-paimon-sft.git
cd cosyvoice-paimon-sft
git submodule update --init --recursive
```

- 安装环境

```bash
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
```

- 要求：
1. $5090个数 \ge 2$
2. `Pytorch` $\ge$ 2.7，CUDA适应自己的版本，我的是12.8


> 如果有问题，请参考[教程](https://zhuanlan.zhihu.com/p/1984999416596276696)中的环境安装bug汇总

## 数据处理📊

你需要准备的数据集样式：

```python
├── your data_dir/
│   ├── test/
│   │   ├── 1_1.wav
│   │   ├── 1_1.normalized.txt
│   │   ├── 1_2.wav
│   │   ├── 1_2.normalized.txt
│   │   ├── 1_3.wav
│   │   ├── 1_3.normalized.txt
│   │   └── ...
│   └── train/
│       ├── 1_1.wav
│       ├── 1_1.normalized.txt
│       ├── 1_2.wav
│       ├── 1_2.normalized.txt
│       ├── 1_3.wav
│       ├── 1_3.normalized.txt
│       └── ...
```

其中`test`和`train`分别是验证集和训练集，名字其实并不是必要的，但是要和脚本`for x in test train; do`中的名字保持一致。

**当然最重要的是，`.wav`和`.normalized.txt`内容一定要对应，否则训练的时候可能会出现乱码。**

然后修改`run.sh`中的进程：

```bash
stage=0
stop_stage=4
```

运行下面的代码：

```bash
bash run.sh
```

## 训练启动🎬

根据自己需要配置超参数设置`conf/cosyvoice2.yaml`

然后修改`run.sh`中的进程：

```bash
stage=5
stop_stage=5
```

运行下面的代码：

```bash
bash run.sh
```

## 结果展示📈


本次实验我们总共进行了两个模型的训练，分别是`llm`和`flow`模型，然后在训练`flow`模型的时候，除了展示本来`flow`模型的音频效果，还补充了`llm`和`flow`训练后权重组合的音频效果。

可以直接看完整结果，并且可以在线听👉[SwanLab](https://swanlab.cn/@LiXinYu/cosyvoice-sft/overview)

**仅llm训练的曲线图和音频结果**

<div style="display:flex;justify-content:center;">
  <figure style="text-align:center;margin:0;">
    <img src="./examples/pictures/paimon13.png" style="width:100%">
  </figure>
</div>

<div style="display:flex;justify-content:center;">
  <figure style="text-align:center;margin:0;">
    <img src="./examples/pictures/paimon14.png" style="width:100%">
  </figure>
</div>

<div style="display:flex;justify-content:center;">
  <figure style="text-align:center;margin:0;">
    <img src="./examples/pictures/paimon15.png" style="width:100%">
  </figure>
</div>

**flow训练的曲线图和音频结果**

<div style="display:flex;justify-content:center;">
  <figure style="text-align:center;margin:0;">
    <img src="./examples/pictures/paimon16.png" style="width:100%">
  </figure>
</div>

<div style="display:flex;justify-content:center;">
  <figure style="text-align:center;margin:0;">
    <img src="./examples/pictures/paimon17.png" style="width:100%">
  </figure>
</div>

其中`Train`就是训练集训练的过程，`CV`可以理解为每个`epoch`完成后进行验证集做前向传播的过程。

这里需要注意几点⚠️：

1. 对于准确度`acc`，只有`llm`才有，`flow`是没有的，这是因为`llm`其实是根据输入的文本预测对应语音的speech tokens，这个tokens其实就可以理解为字典里的编号，那么离散的编号数字当然可以求准确度，同时也保证在反向传播过程中结合loss提升模型训练的效果。`flow`模型理论上是根据speech tokens生成对应的梅尔频谱图，这种非离散状态的输出无法做准确度计算，因此只有loss。
2. `flow`里有个`llm_flow`的效果展示，这是我做的一点小改动🤏，用于在`llm`训练好后，在`flow`训练的时候可以同时对比只有`flow`微调以及两个都微调后音频效果，方便随时观察音频效果变化。
3. `example: same as the last one`这个是我在训练数据中随便找了一个数据作为参考，因为可能有些读者并没有玩过原神，不清楚派蒙的声音，因此做了个对比。

