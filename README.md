# 3DReconstruction_Cartoon-emoticons
Facial expression capture and cartoon expression reconstruction based on deep learning

# 基于深度学习的人脸表情捕捉及卡通表情重建

# 目录

[TOC]

# 一、背景介绍

## 1. 项目概述

**项目主题**是《基于深度学习的人脸表情捕捉及卡通表情重建》。

**功能目标**是人脸表情捕捉及卡通表情重建，即从视频中捕捉人脸表情，并重建为卡通表情。

**应用场景**是虚拟偶像、动画制作等。



## 2. 未来市场潜力

近年来，随着VR、元宇宙等概念的火热，虚拟角色越来越多的出现在影视制作、视频聊天、网络游戏、广告制作等领域，使得虚拟角色动画在生产生活中得到了广泛的应用。而传统的面部捕捉设备及解决方案成本较高，难以让每一个人都用上，故需要一种廉价且好用的解决方案。



# 二、成果说明

本项目实现了头部姿态、嘴部、眉毛、眼睛等关键部位的捕捉和重建，已初步跑通了整个流程。且整个项目涉及到的所有软件均为开源的，不涉及商业软件。

此次实践采用的软件是开源软件Blender，但也可以将此方法移植到其他主流三维软件中，例如3Dmax、Maya、Unity、UE4等。本项目的代码使用了前后端分离的设计，即如果需要从Blender切换到3DMax，只需要前端做一些工作即可，后端不需要改变。从而为后续兼容更多的三维动画软件留下了空间。

本项目为了调试方便，自制了面捕头盔（成本100元左右）。

![成果说明](assets/成果说明.png)



# 二、方案说明

## 1. 设备使用情况

| 编号 | 设备                       | 说明                         |
| ---- | -------------------------- | ---------------------------- |
| 1    | 奥比中光的Astra+ S深度相机 | 使用，采集视频流的传感器     |
| 2    | 英伟达的Jetson Nano开发板  | 使用，处理数据的计算设备     |
| 3    | 显示器                     | 使用，可视化卡通表情重建结果 |
| 4    | 自制面捕头盔               | 使用，固定深度相机的支架     |
| 5    | 手机自拍杆（三脚架版）     | 使用，固定深度相机的支架     |



## 2. 系统架构

系统架构分为硬件和软件，如下图所示：

- 硬件包括：
  - RGBD相机：采集视频流
  - 开发板：处理数据
  - 显示器：显示结果
- 软件分为：
  - 后端：由Python实现，从视频流中检测并跟踪人脸关键点，转换为卡通表情重建需要的参数
  - 前端：驱动Blender，控制卡通表情，并渲染出最终效果

![系统架构图](assets/系统架构图.png)



## 3. 关键技术创新点

实现卡通表情重建，有两种技术路线：

| 技术路线   | 优点                     | 缺点                                           | 案例            |
| ---------- | ------------------------ | ---------------------------------------------- | --------------- |
| Blendshape | 通用性强、效果稳定       | 美术端工作量稍大                               | IPhone的Animoji |
| 关键点映射 | 可控性高、美术端工作量低 | 对前期数据的采集要求高、需要模型的一些基础绑定 |                 |

本项目采用第2种技术路线，即"关键点映射"，使用**PNP**来解决头部姿态估计问题、使用**关键点映射**来解决脸部各个部位的控制。



## 4. 算法设计

### 概述

本项目最终的效果取决于3个层面：

- 硬件的性能：相机的成像质量，视频流是最原始的输入
- 数据的处理：算法的处理效果，很大程度上决定了速度和精度
- 模型的精度：建模的精细程度，决定了表情细节的上限



### 人脸关键点的检测与跟踪

人脸关键点检测常用的开源工具有：dlib的68关键点、opencv（自带的lbf模型）的68关键点、openpose的70关键点、mediapipe的468关键点等。

在对比检测精度和稳定性后，本项目选择使用[mediapipe](https://google.github.io/mediapipe/solutions/face_mesh#python-solution-api)的468关键点。

人脸关键点检测结果如下图所示。

| 检测到的所有关键点         | 实际使用到的关键点         |
| -------------------------- | -------------------------- |
| ![示例1](assets/示例1.png) | ![示例2](assets/示例2.png) |

但是，逐帧检测，人脸的关键点会有明显的跳动，影响卡通表情的稳定性，为此使用**卡尔曼滤波**和**移动均值滤波**。

如下图所示，4条曲线为某段视频右眼的特征距离，说明如下表。综合考虑滤波效果和实时性，本项目使用红色（`kalmanFilter`）。

| 曲线颜色                      | 曲线说明                         | 效果说明                             |
| ----------------------------- | -------------------------------- | ------------------------------------ |
| 黑色（`origin`）              | 原始的结果                       | 有很多噪点                           |
| 红色（`kalmanFilter`）        | 卡尔曼滤波的结果                 | 滤波效果明显，且延时较小             |
| 绿色（`meanFilter`）          | 移动均值滤波的结果               | 滤波效果明显，但实时性不如卡尔曼滤波 |
| 蓝色（`kalmanAndMeanFilter`） | 先卡尔曼滤波再移动均值滤波的结果 | 滤波效果明显，但实时性不如卡尔曼滤波 |

![卡尔曼滤波的折线图](assets/卡尔曼滤波的折线图.png)



### 卡通表情的建模与控制

在人脸动画中，影响最终面部表情效果的有以下几项：

- 模型点的数量
- 参与绑定的骨骼的数量
- "Blendshape"技术路线中，Blendshape的数量与精度
- "关键点映射"技术路线中，算法及控制器的合理性
- 角色贴图的精度



通常在人脸动画中，通过三个层级来控制面部表情：

- 点：点包含法线和位置，影响渲染
- 骨骼：控制一堆点，可以设置点的权重，决定了模型细节的上限
- 控制器：控制一堆骨骼，可以设置骨骼的权重

从越高的层级来控制模型，越容易迁移和复用，故本项目从**控制器**的层级来控制卡通模型。



### 人脸关键点与卡通表情的映射

把关键点分为两部分来分别处理：

- 头部刚性运动：使用6个面部关键点的PNP来估算整个头部的位姿，6个关键点的3D点通过深度相机采集人脸获取点云，并从点云中拾取关键点来获得（如下图所示）。一个细节是，返回给控制器的是两个位姿（obj2cam）之间的相对变化关系，而非简单的欧拉角相减。
- 面部柔性运动：结合卡通建模结果，实现面部关键点与嘴部、眉毛、眼睛等关键部位的映射，映射通过计算关键点的相对位置、归一化到[0,1]、缩放范围来实现。（详见：[四、开发过程 6. 映射关系获取](# 四、开发过程)）

| PNP原理示意图                              | 从点云中拾取PNP的3D点                                      | 头部刚性运动和面部柔性运动 |
| ------------------------------------------ | ---------------------------------------------------------- | -------------------------- |
| ![PNP原理示意图](assets/PNP原理示意图.png) | ![从点云中拾取PNP的3D点](assets/从点云中拾取PNP的3D点.png) | ![示例3](assets/示例3.png) |



# 四、开发过程

## 1. Astra+ S上手

本项目使用的深度相机是奥比中光的[Astra+ S深度相机](https://developer.orbbec.com.cn/module.html?id=3)。原理是基于激光散斑的单目结构光，基线距为75mm，深度范围为0.4-2m，相对精度为±3mm @ 1m。

<img src="assets/Astra+S.png" alt="Astra+ S" style="zoom:33%;" />

本项目中使用到的主要功能是：获取视频流并保存为图片。

可以借助的调试工具是[Orbbec Viewer_v1.1.1](https://developer.orbbec.com.cn/download.html?id=31)。



## 2. Jetson Nano上手

本项目使用的开发板是英伟达的[Jetson Nano开发者套件（2GB）](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit)。CPU是四核ARM A57 @ 1.43GHz，GPU为128核Maxwell，显存为2GB 64位LPDDR4，存储是自配的64GB A2速度的闪迪microSD。

<img src="assets/JetsonNano开发板.png" alt="Jetson Nano开发板" style="zoom:33%;" />

参考英伟达官方的[Jetson Nano 开发者套件入门](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#write)教程和[NVIDIA Jetson NANO 2GB新手入门教程汇总（持续更新）](https://zhuanlan.zhihu.com/p/357438016)，下载[镜像](https://developer.nvidia.com/jetson-nano-2gb-sd-card-image)文件`jetson-nano-2gb-sd-card-image`（6.08GB）、使用`SD Memory Card Formatter`格式化microSD卡、使用`Etcher`把镜像写入microSD卡。

注意： Jetson Nano 4GB 与 Jetson Nano 2GB 的**镜像文件不兼容**，也就是说如果错误地下载了 Jetson Nano 4GB 的镜像包，在 Jetson Nano 2GB 上会出现错误，主要表现为卡在印有"NVIDIA"LOGO的画面而进不去。



系统为`Ubuntu 18.04.5 LTS`，已经预装了`CUDA 10.2`，但需要自己加环境变量。

参考：[Jetson Nano查看CPU\GPU\内存使用率](https://blog.csdn.net/biubiubiu617/article/details/107984931)

```shell
# 先更新
sudo apt-get update
sudo apt-get upgrade

# 常用命令：查看系统版本、CPU、内存、硬盘、CUDA版本
cat /etc/issue
lscpu
free -m
df -lh
nvcc -V

# 把cuda加入环境变量
sudo gedit  ~/.bashrc
# 在最后添加
export CUBA_HOME=/usr/local/cuda-10
export LD_LIBRARY_PATH=/usr/local/cuda-10/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-10/bin:$PATH
# 保存退出
source ~/.bashrc

# 安装常用软件
sudo apt-get install htop
```

 安装python环境

参考：[Jetson nano 的使用和安装 (Archiconda+OpenCV)](https://verimake.com/topics/114)、[Archiconda](https://github.com/Archiconda/build-tools/releases)

```shell
# jetson nano 不能使用 Anaconda，使用 Archiconda 代替
bash Archiconda3-0.2.3-Linux-aarch64.sh

# 新建虚拟环境
conda create -n mediapipe_py37 python=3.7
conda activate mediapipe_py37
pip install --upgrade pip
pip install ipython
pip install mediapipe
pip install opencv_contrib_python -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com # 安装的版本是4.5.4
```

安装blender

```shell
# 安装blender
sudo add-apt-repository ppa:thomas-schiex/blender
sudo apt-get update
sudo apt-get install blender

# 启动blender
blender
```

如果需要自己编译blender，可以参考：[Building Blender on Ubuntu](https://wiki.blender.org/wiki/Building_Blender/Linux/Ubuntu)



| 格式化SD卡                                 | 向SD卡内写入镜像                           |
| ------------------------------------------ | ------------------------------------------ |
| ![1638718356833](assets/jeston_nano_1.png) | ![1638718232808](assets/jeston_nano_2.png) |
| 第一次启动Jeston Nano的Ubuntu              | blender安装成功                            |
| ![1638718232808](assets/jeston_nano_3.png) | ![1638718232808](assets/jeston_nano_4.png) |



## 3. 面捕头盔制作

面捕头盔的好处是，在获取视频流时，保持相机与头部的相对静止。便于采集标准数据，用于测试。

但淘宝现成的面捕头盔过于昂贵（2000元左右），所以通过买普通头盔、3D打印支架的方式自制面捕头盔（100元左右）。



面捕头盔由4部分构成：

- 头盔：淘宝购买
- 固定支架：3D打印
- 固定铁丝：淘宝购买
- 相机支架：淘宝购买

| 面捕头盔的参照                               | 自制面捕头盔正面                         | 自制头盔反面                             |
| -------------------------------------------- | ---------------------------------------- | ---------------------------------------- |
| ![面捕头盔的参照](assets/面捕头盔的参照.png) | ![面捕头盔正面](assets/面捕头盔正面.png) | ![面捕头盔反面](assets/面捕头盔反面.png) |



## 4. 人脸关键点检测调试

本项目把人脸关键点分为**头部刚性运动**和**面部柔性运动**来处理，分别调试，再综合调试。

- 先使用**自制面捕头盔**采集视频，保证整个头部的位姿不变，调试嘴部、眉毛、眼睛等关键部位的面部关键点。
- 再使用**手机自拍杆（三脚架版）**采集视频，控制整个头部的位姿变化，只调试头部位姿，忽略面部表情。
- 继续使用**手机自拍杆（三脚架版）**采集视频，控制整个头部的位姿变化，同时面部表情变化。

| 基于 自制面捕头盔 采集视频                     | 基于 手机自拍杆（三脚架版） 采集视频           |
| ---------------------------------------------- | ---------------------------------------------- |
| ![采集视频-头盔.png](assets/采集视频-头盔.png) | ![采集视频-三脚架](assets/采集视频-三脚架.png) |



## 5. 卡通表情控制点调试

出于版权等的考虑，本项目使用Blender官网提供的免费卡通模型[Vincent](https://studio.blender.org/characters/5718a967c379cf04929a4247/v1/)。

仔细研究后，有8个关键控制器，如下表所示。

参考：[blender手册](https://docs.blender.org/manual/en/2.93/)、[blender python API](https://docs.blender.org/api/2.93/)

| blender中的控制器名称 | 作用       |
| --------------------- | ---------- |
| head_fk               | 整个头部   |
| mouth_ctrl            | 嘴部       |
| brow_ctrl_L           | 左眉毛     |
| brow_ctrl_R           | 右眉毛     |
| eyelid_up_ctrl_L      | 左眼皮上部 |
| eyelid_low_ctrl_L     | 左眼皮下部 |
| eyelid_up_ctrl_R      | 右眼皮上部 |
| eyelid_low_ctrl_R     | 右眼皮下部 |



## 6. 映射关系获取

通过实际操作模型的控制点，根据实测效果，获取各个控制点的限界范围如下表。

| 控制部位   | 整个头部                  | 嘴部                                        | 左眉毛                          | 右眉毛                          |
| ---------- | ------------------------- | ------------------------------------------- | ------------------------------- | ------------------------------- |
| 控制器名称 | head_fk                   | mouth_ctrl                                  | brow_ctrl_L                     | brow_ctrl_R                     |
| Location X | \                         | [横向张大嘴, 横向闭起嘴]<br />[-0.02, 0.02] | \                               | \                               |
| Location Y | \                         | \                                           | \                               | \                               |
| Location Z | \                         | [纵向张大嘴, 常态]<br />[-0.06, 0]          | [降眉, 挑眉]<br />[-0.02, 0.02] | [降眉, 挑眉]<br />[-0.02, 0.02] |
| Rotation X | 上下点头<br />[-30°, 30°] | \                                           | \                               | \                               |
| Rotation Y | 左右摇头<br />[-30°, 30°] | \                                           | \                               | \                               |
| Rotation Z | 旋转歪头<br />[-30°, 30°] | \                                           | \                               | \                               |

| 控制部位   | 左眼皮上部                                       | 左眼皮下部                                       | 右眼皮上部                                       | 右眼皮下部                                       |
| ---------- | ------------------------------------------------ | ------------------------------------------------ | ------------------------------------------------ | ------------------------------------------------ |
| 控制器名称 | eyelid_up_ctrl_L                                 | eyelid_low_ctrl_L                                | eyelid_up_ctrl_R                                 | eyelid_low_ctrl_R                                |
| Location X | \                                                | \                                                | \                                                | \                                                |
| Location Y | \                                                | \                                                | \                                                | \                                                |
| Location Z | [上眼皮闭到眼睛的一半, 常态]<br />[-0.02, 0.005] | [常态, 下眼皮闭到眼睛的一半]<br />[-0.005, 0.02] | [上眼皮闭到眼睛的一半, 常态]<br />[-0.02, 0.005] | [常态, 下眼皮闭到眼睛的一半]<br />[-0.005, 0.02] |
| Rotation X | \                                                | \                                                | \                                                | \                                                |
| Rotation Y | \                                                | \                                                | \                                                | \                                                |
| Rotation Z | \                                                | \                                                | \                                                | \                                                |



各个部位的控制器的效果图如下。

### 头部姿态（head_fk）

| [RX,RY,RZ] | [-30°,0°,0°]                              | [30°,0°,0°]                              | [0°,0°,0°]                                    |
| ---------- | ----------------------------------------- | ---------------------------------------- | --------------------------------------------- |
| 效果图     | ![1](assets/head_fk_rx_-30_ry_0_rz_0.png) | ![2](assets/head_fk_rx_30_ry_0_rz_0.png) | ![3](assets/head_fk_rx_0_ry_0_rz_0.png)       |
| [RX,RY,RZ] | [0°,-30°,0°]                              | [0°,30°,0°]                              | [-30°,-30°,-30°]                              |
| 效果图     | ![4](assets/head_fk_rx_0_ry_-30_rz_0.png) | ![5](assets/head_fk_rx_0_ry_30_rz_0.png) | ![6](assets/head_fk_rx_-30_ry_-30_rz_-30.png) |
| [RX,RY,RZ] | [0°,0°,-30°]                              | [0°,0°,30°]                              | [30°,30°,30°]                                 |
| 效果图     | ![7](assets/head_fk_rx_0_ry_0_rz_-30.png) | ![8](assets/head_fk_rx_0_ry_0_rz_30.png) | ![9](assets/head_fk_rx_30_ry_30_rz_30.png)    |



### 嘴部（mouth_ctrl）

| [X,Z]  | [-0.02, -0.06]                              | [-0.02, 0]                              | [-0.02, 0.06]                              |
| ------ | ------------------------------------------- | --------------------------------------- | ------------------------------------------ |
| 效果图 | ![1](assets/mouth_ctrl_x_-0.02_z_-0.06.png) | ![2](assets/mouth_ctrl_x_-0.02_z_0.png) | ![3](assets/mouth_ctrl_x_-0.02_z_0.06.png) |
| [X,Z]  | [0, -0.06]                                  | [0, 0]                                  | [0, 0.06]                                  |
| 效果图 | ![4](assets/mouth_ctrl_x_0_z_-0.06.png)     | ![5](assets/mouth_ctrl_x_0_z_0.png)     | ![6](assets/mouth_ctrl_x_0_z_0.06.png)     |
| [X,Z]  | [0.02, -0.06]                               | [0.02, 0]                               | [0.02, 0.06]                               |
| 效果图 | ![7](assets/mouth_ctrl_x_0.02_z_-0.06.png)  | ![8](assets/mouth_ctrl_x_0.02_z_0.png)  | ![9](assets/mouth_ctrl_x_0.02_z_0.06.png)  |



### 左眉毛（brow_ctrl_L）、右眉毛（brow_ctrl_R）

| （brow_ctrl_L）[Z] | [-0.02]                              | [0]                              | [0.02]                              |
| ------------------ | ------------------------------------ | -------------------------------- | ----------------------------------- |
| 效果图             | ![1](assets/brow_ctrl_L_z_-0.02.png) | ![2](assets/brow_ctrl_L_z_0.png) | ![3](assets/brow_ctrl_L_z_0.02.png) |
| （brow_ctrl_R）[Z] | [-0.02]                              | [0]                              | [0.02]                              |
| 效果图             | ![4](assets/brow_ctrl_R_z_-0.02.png) | ![5](assets/brow_ctrl_R_z_0.png) | ![6](assets/brow_ctrl_R_z_0.02.png) |



### 左眼皮上部（eyelid_up_ctrl_L）、左眼皮下部（eyelid_low_ctrl_L）

| （eyelid_up_ctrl_L）[Z], （eyelid_low_ctrl_L）[Z] | [-0.02,-0.005]                                               | [-0.02,0]                                                   | [-0.02,0.02]                                                 |
| ------------------------------------------------- | ------------------------------------------------------------ | ----------------------------------------------------------- | ------------------------------------------------------------ |
| 效果图                                            | ![1](assets/eyelid_up_ctrl_L_-0.02_eyelid_low_ctrl_L_-0.005.png) | ![2](assets/eyelid_up_ctrl_L_-0.02_eyelid_low_ctrl_L_0.png) | ![3](assets/eyelid_up_ctrl_L_-0.02_eyelid_low_ctrl_L_0.02.png) |
| （eyelid_up_ctrl_L）[Z], （eyelid_low_ctrl_L）[Z] | [0,-0.005]                                                   | [0,0]                                                       | [0,0.02]                                                     |
| 效果图                                            | ![4](assets/eyelid_up_ctrl_L_0_eyelid_low_ctrl_L_-0.005.png) | ![5](assets/eyelid_up_ctrl_L_0_eyelid_low_ctrl_L_0.png)     | ![6](assets/eyelid_up_ctrl_L_0_eyelid_low_ctrl_L_0.02.png)   |
| （eyelid_up_ctrl_L）[Z], （eyelid_low_ctrl_L）[Z] | [0.005,-0.005]                                               | [0.005,0]                                                   | [0.005,0.02]                                                 |
| 效果图                                            | ![7](assets/eyelid_up_ctrl_L_0.005_eyelid_low_ctrl_L_-0.005.png) | ![8](assets/eyelid_up_ctrl_L_0.005_eyelid_low_ctrl_L_0.png) | ![9](assets/eyelid_up_ctrl_L_0.005_eyelid_low_ctrl_L_0.02.png) |



### 右眼皮上部（eyelid_up_ctrl_R）、右眼皮下部（eyelid_low_ctrl_R）

| （eyelid_up_ctrl_R）[Z], （eyelid_low_ctrl_R）[Z] | [-0.02,-0.005]                                               | [-0.02,0]                                                   | [-0.02,0.02]                                                 |
| ------------------------------------------------- | ------------------------------------------------------------ | ----------------------------------------------------------- | ------------------------------------------------------------ |
| 效果图                                            | ![1](assets/eyelid_up_ctrl_R_-0.02_eyelid_low_ctrl_R_-0.005.png) | ![2](assets/eyelid_up_ctrl_R_-0.02_eyelid_low_ctrl_R_0.png) | ![3](assets/eyelid_up_ctrl_R_-0.02_eyelid_low_ctrl_R_0.02.png) |
| （eyelid_up_ctrl_R）[Z], （eyelid_low_ctrl_R）[Z] | [0,-0.005]                                                   | [0,0]                                                       | [0,0.02]                                                     |
| 效果图                                            | ![4](assets/eyelid_up_ctrl_R_0_eyelid_low_ctrl_R_-0.005.png) | ![5](assets/eyelid_up_ctrl_R_0_eyelid_low_ctrl_R_0.png)     | ![6](assets/eyelid_up_ctrl_R_0_eyelid_low_ctrl_R_0.02.png)   |
| （eyelid_up_ctrl_R）[Z], （eyelid_low_ctrl_R）[Z] | [0.005,-0.005]                                               | [0.005,0]                                                   | [0.005,0.02]                                                 |
| 效果图                                            | ![7](assets/eyelid_up_ctrl_R_0.005_eyelid_low_ctrl_R_-0.005.png) | ![8](assets/eyelid_up_ctrl_R_0.005_eyelid_low_ctrl_R_0.png) | ![9](assets/eyelid_up_ctrl_R_0.005_eyelid_low_ctrl_R_0.02.png) |



# 五、测试说明

相关代码和数据见`3代码&数据`，该文件夹下的`*.py`是源码，`data`文件夹下的是图片和面部关键点数据，`vincent.blend`是blender的文件。

测试过程如下：

1. 采集视频流为图片，运行`main1_capture_astra.py`。

2. 从图片集中提取人脸关键点，运行`main2_meidapipe_gen_npy.py`。

3. 使用卡尔曼滤波生成新的人脸关键点，运行`main3_kalmanfilter_read_npy_gen_npy.py`。

4. 打开blender

   1. 切换到`Script`标签页，把`Script`中的`OpenCVAnimOperator.py`中的变量`FOLDER_ROOT_DATA`修改为`data`文件夹的绝对路径。启动脚本`OpenCVAnimOperator.py`、`OpenCVAnim.py`

      ![测试说明1](assets/测试说明1.png)

      ![测试说明2](assets/测试说明2.png)

   2. 切换到`Layout`标签页，点击`Capture`，开始卡通表情重建。

      ![测试说明3](assets/测试说明3.png)



# 六、参考

本项目在总体方案、细节处理等方面，参考了如下项目，向这些项目的作者表示衷心的感谢。

- [FacialMotionCapture_v2](https://github.com/jkirsons/FacialMotionCapture_v2)
- [OpenPose-to-Blender-Facial-Capture-Transfer](https://github.com/nkeeline/OpenPose-to-Blender-Facial-Capture-Transfer)
- [OpenVHead](https://github.com/TianxingWu/OpenVHead)
- [mediapipe](https://github.com/google/mediapipe)

