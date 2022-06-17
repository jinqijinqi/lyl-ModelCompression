简述：本仓库是基于贝叶斯模型压缩思想对目标分类网络VGG和目标检测网络SSD进行模型压缩，相关代码存在SourceCode文件夹下。

# Tutorial_BayesianCompressionForDL-master

该部分是贝叶斯模型压缩基础代码，代码相关细节请查看https://github.com/Lyken17/Bayesian-Compression-for-Deep-Learning，实现细节参考论文https://arxiv.org/abs/1705.08665.

# bayesian_vgg_lenet

## 1.构造贝叶斯网络

### 1.1原始vgg网络vgg.py脚本介绍

1.1.1 make_layers(cfg, batch_norm=False)函数（vgg.py第40-43行）

vgg.py位于models文件夹下。其中make_layers(cfg, batch_norm=False)函数用于从cfgs字典（第56-61行）中读取配置参数进行模型构造，batch_norm参数用于指定是否使用BN层。cfgs字典中A,B,D,E分别指代vgg11、vgg13、vgg16、vgg19。以A为例子，其字典键值列表：[64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']中的数字代表每一层的输出通道数，'M'代表池化层。

1.1.2 vgg11～vgg19_bn函数（第63-85行）

此类函数用于调用make_layers函数得到网络各层，并且传入VGG（）这个类用于将类实例化，vgg19_bn即返回构建好的vgg19网络，网络使用BN层。

1.1.3 VGG(nn.Module)类

第10~16行：用于完成VGG网络的初始化

第20~24行：VGG网络的前向传播

第27~38行：pytorch官方的权值初始化方法



### 1.2 **贝叶斯VGG的构建**

1.2.1复制vgg.py脚本

由于贝叶斯vgg网络是基于原始vgg网络构造的，所以从代码实现上可以参考vgg.py。将vgg.py拷贝一份至同级目录下，并将拷贝脚本命名为vgg_bayesian.py，用于构建贝叶斯vgg网络。

1.2.2导入所需要的新的库（vgg_bayesian.py第3~6行）

其中第3行的Conv2dGroupNJ，LinearGroupNJ即为贝叶斯网络二维卷积以及全连接层。

1.2.3修改make_layers函数（vgg_bayesian.py第101~119行）

主要是将vgg_bayesian.py中第47行的普通的二维卷积层nn.Conv2d 改成贝叶斯网络模块Conv2dGroupNJ 。另外，对于不同输出通道数的Conv2dGroupNJ，clip_var的值大小不一样，这个值直接根据论文中来定。所以需要将vgg_bayesian.py中第47行展开写成vgg_bayesian.py第108~113行的写法。

1.2.4 self._initialize_weights()的删除：

将第16行的self._initialize_weights()注释掉或者删掉，贝叶斯VGG网络的权值初始化不使用pytorch官方用法，其初始化在Conv2dGroupNJ，LinearGroupNJ中默认完成。同时将原始VGG里的_initialize_weights方法注释掉或者删除（在vgg_bayesian.py第26~38行）

1.2.5 kl_list属性的增加：

在vgg_bayesian.py的第18行增加self.kl_list属性，就是将贝叶斯vgg里features和classifier里的Conv2dGroupNJ 和 LinearGroupNJ加到kl_list里，用于进行训练时候clip_variances()与测试时的压缩率计算。

1.2.6 draw_logalpha_hist方法的增加：

在vgg_bayesian.py中的第41～49行，在原始VGG网络的基础上加入draw_logalpha_hist方法，用于画出log_var的直方图，以便之后选定剪枝的阈值。图片将会保存在上级目录的vgg_figs里，0.png的图像为第一层的z的方差除以均值的平方的对数化取值，可以用于筛选阈值，这个值的物理意义在 节介绍。

1.2.7 get_masks方法的增加：

在vgg_bayesian.py中的第52～105行，在原始VGG网络的基础上加入get_masks方法，根据筛选阈值得到用于剪枝的mask，同时会打印出剪枝前/后每一层对应的通道数，用于对比剪枝前以及设计剪枝后的网络结构。返回的第一个列表为每一层输入与输出通道的剪枝掩码，第二个列表为每一层输出通道的剪枝掩码。

1.2.8 kl_divergence方法的增加：

在vgg_bayesian.py中的第108～112行，在原始VGG网络的基础上加入kl_divergence方法，用于在训练中计算kl散度来优化贝叶斯网络，以达到剪枝的目的。

 

## 2. **贝叶斯网络训练**

### 2.1 **原始vgg网络训练脚本vgg_train.py介绍**

2.1.1 数据读取：vgg_train.py第21～45行：进行训练以及测试数据的读取

2.1.2 模型初始化：vgg_train.py第48～51行

2.1.3 优化器选择：vgg_train.py第54行

2.1.4 损失函数：vgg_train.py第56~60行。损失函数选择交叉熵，将其写进objective函数里是为了与贝叶斯VGG网络的损失函数做对比。

2.1.5 模型训练函数：vgg_train.py第62~78行，在训练时，每个epoch结束后打印训练与测试损失以及在测试时的准确率。

2.1.6 模型测试函数：vgg_train.py第80~94行，计算测试损失以及准确率。

2.1.7 进行模型的训练以及测试：vgg_train.py第97~104行，运行train与test函数进行训练以及测试，在训练至198个epoch时保存模型的权值。

2.1.8 指定训练超参数：vgg_train.py第108~113行，指定训练epoch以及batchsize，使用当前默认超参数可达到报告中的精度。



### 2.2 **贝叶斯vgg网络训练：**

2.2.1复制vgg_train.py脚本：

由于贝叶斯VGG网络训练流程与原始VGG网络训练流程几乎一致，所以从代码实现上可以参考vgg_train.py。将vgg_train.py拷贝一份至同级目录下，并将拷贝脚本命名为vgg_bayesian_train.py，用于训练贝叶斯vgg网络。

2.2.2 更改导入的VGG模型：

vgg_bayesian_train.py第12行，将之前的from models.vgg import vgg19_bn更改为

from models.vgg_bayesian import vgg19_bn，即将导入的VGG模型更改为贝叶斯VGG。

2.2.3 加入训练集样本大小：

vgg_bayesian_train.py第12行，对于CIFAR10来说，训练集样本数量为50000，这个数量在vgg_bayesian_train.py第59行计算KL时要用到。

2.2.4 修改训练损失函数：

vgg_bayesian_train.py第57～60行，首先在objective函数的形参里加入kl_divergence以传kl散度。接着，在第58行将原始VGG的discrimination_error乘以10，如果不乘以10，那么分类损失和kl之间会有一个明显的不平衡，这会影响网络的精度。然后，计算变分下界variational_bound = discrimination_error + kl_divergence / N，注意最后一项kl_divergence要除以训练集样本数N。最后，第60行返回值从discrimination_error变成了variational_bound。

2.2.5 训练时进行kl计算：

vgg_bayesian_train.py第69行，将模型计算的kl传入objective函数以进行总loss的计算。vgg_bayesian_train.py第73～74行，在每次模型权值更新后，对贝叶斯网络每一层使用clip_variances()函数进行方差限制，否则可能会出现梯度爆炸的情况。

2.2.6更改保存权值路径：

vgg_bayesian_train.py第103行，将保存权值命名为vgg_bay.pth，以便与原始vgg的权值区分开。

 

## 3. **贝叶斯网络测试**

### 3.1 **原始vgg网络测试脚本vgg_test.py介绍**

测试脚本只使用test函数，其他内容与vgg_train.py的一致，需要注意在第70行要指定加载预训练模型的路径。

 

### 3.2 **贝叶斯vgg网络测试**

3.2.1复制vgg_test.py脚本：

由于贝叶斯VGG网络测试流程与原始VGG网络训练流程几乎一致，所以从代码实现上可以参考vgg_test.py。将vgg_test.py拷贝一份至同级目录下，并将拷贝脚本命名为vgg_bayesian_test.py，用于测试贝叶斯vgg网络。

3.2.2导入计算压缩率以及量化权值的函数：

在vgg_bayesian_test.py第8行加入from utils.compression import compute_compression_rate, compute_reduced_weights，其中compute_compression_rate用于计算模型剪枝以及量化后的压缩率，compute_reduced_weights用于在数值上对模型权值进行量化。

3.2.3 更改导入的VGG模型：

vgg_bayesian_test.py第9行，将之前的from models.vgg import vgg19_bn更改为

from models.vgg_bayesian import vgg19_bn，即将导入的VGG模型更改为贝叶斯VGG。

3.2.4画出以![img](F:\project\ProjectTemplate-master\ProjectTemplate-master\images\README.assets\wps1.jpg)为横坐标的每层卷积核通输出道数直方图以确定剪枝阈值：

vgg_bayesian_test.py第34行加入model.draw_logalpha_hist()，图像保存在vgg_figs文件夹里。阈值的选取以下图为例

![image-20220616134728455](F:\project\ProjectTemplate-master\ProjectTemplate-master\assets\image-20220616134728455.png)

上图为vgg第8层（编号从0开始）的直方图。横坐标为logα的取值，纵坐标为在该logα的取值区间下，对应的卷积核输出通道数目，即每一层的每一组（每一个输出通道）卷积核都对应着一个logα区间，这个直方图的直观意义就是看卷积核输出通道数随着logα值变化的聚集情况。其中logα的计算公式如下：

![img](F:\project\ProjectTemplate-master\ProjectTemplate-master\assets\wps26.jpg)

其中z为模型权值的一个缩放量\****，卷积核每个输出通道对应个一个z，若卷积核形状为[out_channel,in_channel,h,w],则对应的z形状为[out_channel,1,1,1]，z是一个随机变量，其服从一定的统计分布（论文假设为正态分布），所以其有均值平方![img](F:\project\ProjectTemplate-master\ProjectTemplate-master\assets\wps27.jpg)和方差![img](F:\project\ProjectTemplate-master\ProjectTemplate-master\assets\wps29.jpg)。根据论文https://arxiv.org/abs/1705.08665，在推断阶段，最终的权值w由z的均值![img](F:\project\ProjectTemplate-master\ProjectTemplate-master\assets\wps30.jpg)与w的均值![img](F:\project\ProjectTemplate-master\ProjectTemplate-master\assets\wps31.jpg)以及剪枝掩码m决定，即

​																														 ![img](F:\project\ProjectTemplate-master\ProjectTemplate-master\assets\wps28.jpg)

而对于z而言，其方差![img](F:\project\ProjectTemplate-master\ProjectTemplate-master\assets\wps32.jpg)越大则说明z的取值越不确定，即z有可能有很多种取法，说明这个通道的z对应卷积核W提取到的特征相对来说并不重要，一个重要的特征相对来说取值应该是稳定的，则此通道对应的权值更应被剪去。同时，z的均值![img](F:\project\ProjectTemplate-master\ProjectTemplate-master\assets\wps33.jpg)越小则说明最后对应的这个通道的W越小，则此时W对于卷积结果的贡献就较小，此通道对应的权值更应被剪去。

综合以上，方差![img](F:\project\ProjectTemplate-master\ProjectTemplate-master\assets\wps34.jpg)越大、![img](F:\project\ProjectTemplate-master\ProjectTemplate-master\assets\wps35.jpg)越小则该通道卷积核越应被剪去，即![img](F:\project\ProjectTemplate-master\ProjectTemplate-master\assets\wps36.jpg)越大则该通道越应被剪去，越小则越应被保留。经过贝叶斯压缩算法优化后的网络卷积核数目随logα值变化往往呈两端堆靠的情况。如上图，往右边（logα较大区域）堆靠的那些卷积核属于无用卷积核，可以被剪去。有左边（logα较小区域)堆靠的卷积核为有效卷积核，应该被保留，此时阈值可选定在两堆中间或者中间稍低的位置。例如这层阈值可选定在-5.5。被剪去后，滤波器的舍弃与保留情况应该如下：

 ![image-20220616141135891](F:\project\ProjectTemplate-master\ProjectTemplate-master\assets\image-20220616141135891.png)

注意，在网络的前几层，卷积核数目随logα变化的直方图往往比较集中，如下图，此时阈值需要选到比较高，例如-5

 ![image-20220616141202690](F:\project\ProjectTemplate-master\ProjectTemplate-master\assets\image-20220616141202690.png)

 

选取的阈值在vgg_bayesian_test.py的第73～74行指定。

3.2.5 计算压缩率

vgg_bayesian_test.py第52～54行分别计算剪枝后的压缩率和剪枝+位编码后的压缩率，同时打印出剪枝前后每一层的卷积核输出通道数。对于任何网络而言，只要定义好了model.kl_list, model.get_masks以及剪枝的阈值，compute_compression_rate都不用修改。运行后会打印出：

"Compressing the architecture will degrease the model by a factor of xx"   

"Making use of weight uncertainty can reduce the model by a factor of xx" 

第一行即为模型剪枝获得的压缩率，第二行为模型剪枝+权值位编码后的压缩率，注意这里的剪枝只是将应当被剪去的滤波器权值置0，并未真正剪枝，真正剪枝的执行会在vgg_bayesian_prune.py执行。

3.2.6 获得剪枝+位编码后的权值

vgg_bayesian_test.py第56-60行compute_reduced_weights获得剪枝+位编码后的权值，将网络的后验权值（推断阶段使用后验权值）post_weight_mu设置为剪枝+位编码后的权值（第58行），并且通过layer.deterministic = True将网络切换为推断模式。

 

3.2.7测试剪枝+位编码后网络的精度

vgg_bayesian_test.py第63行，运行后输出：

"Test error after with reduced bit precision:××"即为打印出的准确率。

## 4. **贝叶斯网络剪枝**

### 4.1 **贝叶斯vgg网络权值剪枝**

4.1.1 复制vgg.py

由于剪枝后的vgg网络是基于原始vgg网络构造的，所以从代码实现上可以参考vgg.py。将vgg.py拷贝一份至同级目录下，并将拷贝脚本命名为vgg_pruned.py，用于构建剪枝后的vgg网络。

4.1.2 增加剪枝后模型的配置

vgg_pruned.py第61行，在cfgs字典里增加剪枝后的vgg每层输出通道数目的配置，对应的键命名为‘F’，每行具体的通道数目由之前运行vgg_bayesian_test.py打印出的信息来填写。

 

4.1.3增加剪枝后vgg网络的导出函数

vgg_pruned.py第87行，增加剪枝后vgg网络的导出函数并且命名为vgg19_bn_pruned。

 

4.1.4 复制vgg_bayesian_test.py脚本：

贝叶斯VGG网络剪枝流程与其测试流程大体一致，所以从代码实现上可以参考vgg_bayesian_test.py。将vgg_bayesian_test.py拷贝一份至同级目录下，并将拷贝脚本命名为vgg_bayesian_prune.py，用于对贝叶斯vgg网络剪枝。

 

4.1.5导入剪枝所需模型以及函数

vgg_bayesian_prune.py第8~10行，分别导入剪枝后的模型、原始贝叶斯vgg模型、剪枝所用函数prunning_weights。

 

4.1.6初始化剪枝后的模型

vgg_bayesian_prune.py第34行，model_pruned = vgg19_bn_pruned()初始化剪枝后的vgg模型。

 

4.1.7对原始贝叶斯模型权值进行剪枝以及保存

vgg_bayesian_prune.py第54~56行，根据 原始贝叶斯网络的权值 以及 剪枝掩码（model.get_masks(thresholds)获得） 与 剪枝后网络的结构 来提取剪枝后vgg的权值，并将权值保存在checkpoints目录下。

 

4.1.8 测试剪枝后vgg的精度

vgg_bayesian_prune.py第58~61行，首先将剪枝后的权值导入模型，接着再做测试。



# ssd_bayesian_v3_supp

## 1. **构造贝叶斯SSD**

### 1.1 **原始SSD脚本ssd_vgg.py介绍**

位于examples/object_detection/my_models下的ssd_vgg.py。

第26~28行BASE_NUM_OUTPUTS为backbone（VGG16）的配置，字典键300与512代表网络输入图像的宽高大小。列表里的数字为每一层的输出通道数，‘M’或‘C’为池化层。

第30行的EXTRAS_NUM_OUTPUTS为SSD在VGG的基础上搭建的额外特征层。

第35与40行的BASE_OUTPUT_INDICES和EXTRA_OUTPUT_INDICES为SSD输出层的索引，这个设置与SSD原论文保持一致，不用修改。

第49～77行为SSD的初始化以及前向过程，第79到87行为SSD的权值加载方法，不用修改。

第90~105行的make_ssd_vgg_layer函数为SSD基础模块的搭建函数，其分别被第108行的build_vgg_ssd_layers与第137行的build_vgg_ssd_extra调用以创建SSD的backbone（VGG16）以及额外特征层。

第108～134行build_vgg_ssd_layers创建SSD的backbone（VGG16）。注意其中的第129与130行创建的是backbone与额外特征层间的过度层，其也可以被剪枝，即可能会被修改，需要注意。

第137~157行为额外特征层的创建，其所有层通过EXTRAS_NUM_OUTPUTS与调用make_ssd_vgg_laye进行创建，因此不用修改。

第160行开始的build_ssd_vgg为这个SSD网络的创建以及预训练模型加载函数。

### 1.2 **创建贝叶斯SSD**

1.2.1复制ssd_vgg.py脚本

贝叶斯SSD创建过程与原始的SSD大部分一致，可以在my_models文件夹下直接拷贝ssd_vgg.py并且命名为ssd_vgg_bayesian.py用于贝叶斯SSD的搭建。

1.2.2导入所需要的新的库

ssd_vgg_bayesian.py第17、22、24行分别导入numpy、Conv2dGroupNJ以及matplotlib用于贝叶斯VGG的应用

1.2.3贝叶斯网络新属性以及函数的添加

ssd_vgg_bayesian.py第64行增加kl_list用于之后计算kl散度。

ssd_vgg_bayesian.py第79行增加get_masks函数用于获得剪枝掩码。

ssd_vgg_bayesian.py第129行增加draw_t_hist函数用于画出阈值直方图，直方图的物理意义以及阈值选取方式可见贝叶斯VGG以及Lenet压缩文档。

ssd_vgg_bayesian.py第139行增加kl_divergence函数用于计算kl散度。

1.2.4修改make_ssd_vgg_layer函数

ssd_vgg_bayesian.py第168～176行，即将ssd_vgg.py中第100行的卷积层添加做了修改。修改有两个，第一个修改，将原来的普通卷积nn.Conv2d替换成了贝叶斯二维卷积Conv2dGroupNJ，因为SSD中没有用到全连接层，这里不处理全连接层。第二个修改，由于贝叶斯压缩原论文中对不同输出通道数的卷积做了不同的方差限制，所以这里将贝叶斯卷积层的添加写成了三种情况（ssd_vgg_bayesian.py第168、171、174行）。

 

1.2.5修改build_ssd_vgg函数

在ssd_vgg_bayesian.py中将原始vgg中的build_ssd_vgg函数（ssd_vgg.py中第160行）更名为build_ssd_vgg_bayesian以方便区分。同时在ssd_vgg_bayesian.py第244~247行中对导入的在Imagenet上预训练的VGG模型权值做更名处理，否则贝叶斯网络无法导入预训练模型，这会影响贝叶斯SSD的精度。

## 2. **贝叶斯SSD的训练以及测试**

### 2.1 **原始SSD训练以及测试与main.py脚本介绍**

2.1.1训练sh脚本命令参数说明

终端命令行运行train_ssd.sh脚本即可进行原始ssd的训练。脚本中的命令如下：

python main.py -m train 		--config ./examples/object_detection/configs/ssd300_vgg_voc.json 		--data ./data/voc/VOCdevkit/ 

--log-dir=./results/quantization/ssd300 		

--multiprocessing-distributed

 

其中的-m代表运行模式，可选train或者test，

--config即ssd配置文件的目录，这个配置文件可以指定ssd训练的batchsize、初始学习率、vgg预训练权值目录等等。

--data指定数据存放目录

--log-dir指定模型训练中输出的日志以及保存权值的目录，保存目录地址为results/ssd300/ssd_vgg_voc/下，根据运行脚本的时间保存。例如若在2021年04月12_号9点18分45秒运行脚本，则保存目录为results/ssd300/ssd_vgg_voc/2021-04-12__09-18-45，目录下的intermediate_checkpoints文件夹保存一定epoch间隔（默认10个epoch）的模型权值。

--multiprocessing-distributed 指定分布式训练

 

 

2.1.2测试sh脚本命令参数说明

终端命令行运行test_ssd.sh脚本即可进行原始ssd的测试。脚本中的命令如下：

 

python main.py -m test 

--config ./examples/object_detection/configs/ssd300_vgg_voc.json 

--data ./data/voc/VOCdevkit/ 

--resume ./examples/object_detection/ckpt/ssd300_vgg_voc.pth

--resume参数指定加载的训练好的ssd模型的目录。

脚本运行后会对每个batch的图片做运行时间测试，同时最后做所有测试集mAP的测试。

补充：终端输入touch xx.sh便可创建sh脚本，接着就可以通过vim或者gedit编写shell命令。终端输入bash xx.sh即可运行。

 

2.1.3 main.py脚本介绍

 

main.py用于对原始ssd进行训练以及测试。第74行起的main_worker函数为主要工作函数，其通过调用create_dataloaders函数（第114行）来创造训练以及测试集的dataloader。通过调用create_model函数（第124行）创建模型。第142~148行为在测试模式下对模型进行mAP测试。若是在训练模式下，则第150行调用train函数对模型进行训练。

第161行起的create_dataloaders加载数据并且创造dataloader。

第195行起的create_model创造模型，其中第208～210行是在测试模式下，加载模型训练好的权值，然后保存模型权值到/examples/object_detection/ckpt中，这样可以看到剪枝前模型的权重大小。而训练中保存的pth文件，如/examples/object_detection/ckpt/ssd300_vgg_voc.pth，不仅包含模型权值，还包含训练中优化器的参数，所以会导致ssd300_vgg_voc.pth的文件大小比模型真正的权值要大。

第226行起的train_step函数完成为单个batch的前传以及梯度方向传播过程，且返回单个batch的loss。

第255行起的train函数读取config中的训练超参数，并且汇总所有batch的loss。

### 2.2 **贝叶斯SSD训练以及测试与main_bayesian.py脚本介绍**

2.2.1贝叶斯ssd训练以及测试脚本说明

训练以及测试脚本分别为train_ssd_bayesian.sh和test_ssd_bayesian.sh。脚本中的节选参数项与train_ssd.sh和test_ssd.sh一致，只不过导入的模型以及配置文件为贝叶斯ssd的。

运行test_ssd_bayesian.sh后，不仅会测试模型运行时间与精度，还会打印出剪枝前后的模型输出通道数，可提供于设计剪枝后的模型，同时打印模型剪枝以及剪枝+量化后的压缩率。***\*注意这里的剪枝只是将权值置0，真正的剪枝在之后（第三节）完成。\****

“Compressing the architecture will degrease the model by a factor of x”为模型剪枝获得的压缩率。

“Making use of weight uncertainty can reduce the model by a factor of x”为模型剪枝+量化获得的压缩率。

2.2.2 创建main_bayesian.py 脚本

将main.py拷贝至同级目录，重命名为main_bayesian.py用于进行贝叶斯ssd的训练以及测试。

2.2.3 导入相关函数

main_bayesian.py第40行导compute_compression_rate, compute_reduced_weights分别用于压缩率以及剪枝+量化后模型的获取。

 

2.2.4 修改测试步骤

main_bayesian.py第145~157行，先获取阈值直方图，图像在ssd_figs文件夹中，以此来确定剪枝阈值。阈值选取方式可见贝叶斯vgg压缩使用说明文档。第149行计算压缩率，第150行获得剪枝+量化后的权值，第152行将贝叶斯网络的后验权值置为剪枝+量化后的权值，贝叶斯模型在推断阶段使用的是后验权值。第153行即将贝叶斯模型设置为推断模式。

2.2.5修改训练loss

main_bayesian.py第244行，计算kl损失用于优化贝叶斯模型。第245行在原有loss的基础上加入kl_loss。第250行，返回的loss中加入kl_loss。第304行，train_step返回的loss中加入kl_loss以汇总整个数据集的kl_loss。第333～335行，打印的loss中加入kl_loss。

## 3. **贝叶斯SSD剪枝**

### 3.1 **SSD剪枝prune_ssd.sh脚本介绍**

因为SSD剪枝需要导入贝叶斯SSD模型，所以其配置文件以及模型权值均应参考test_ssd_bayesian.sh，只不过把其中具体运行的py脚本改为了prune_ssd.py。运行prune_ssd.sh可获得剪枝后的模型保存在/examples/object_detection/ckpt目录下的ssd_pruned.pth。并且测试剪枝后模型的精度与速度。

### 3.2 **SSD剪枝流程**

3.2.1 创建ssd_vgg_pruned.py

因为剪枝前后模型每层具体输出通道数已经有了变化，需要重新搭建一个SSD。在examples/object_detection/my_models目录下拷贝ssd_vgg.py并命名为ssd_vgg_pruned.py用于创建剪枝后的模型。

3.2.2修改SSD每一层的输出通道

ssd_vgg_pruned.py第27行与31行，根据运行test_ssd_bayesian.sh打印出的剪枝后模型每一层的输出通道数情况，修改BASE_NUM_OUTPUTS与EXTRAS_NUM_OUTPUTS中列表的数字，这些数字即对应的每一层的输出通道数。注意运行test_ssd_bayesian.sh打印出来的layer0~layer12即为BASE_NUM_OUTPUTS中的各个卷积层。Layer15~layer22对应的EXTRAS_NUM_OUTPUTS中的各层。layer13与layer14对应的ssd_vgg_pruned.py第130行与131行的中间过渡层。

3.2.3模型剪枝prune_ssd.py脚本的创建

复制main.py至同级目录下，并命名为prune_ssd.py。在prune_ssd.py第23行导入模型剪枝函数prunning_weights。

3.2.4修改create_model函数

prune_ssd.py第140～141行首先创建贝叶斯SSD模型并导入训练好后得到的权值。第143行创建剪枝后的SSD模型，第145行调用prunning_weights函数得到剪枝后的权值。第146行保存剪枝后的权值至/examples/object_detection/ckpt/ssd_pruned.pth。第147行，将剪枝后的权值导入到模型中，以便在prune_ssd.py第107～110行做测试。

 

### 3.3 **剪枝后的SSD测试**

得到剪枝后得到权值后，若想直接测试剪枝后的模型而不再重复剪枝过程，可以运行test_ssd_pruned.sh脚本直接测试。

 
