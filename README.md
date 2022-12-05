# testopencv
使用opencv进行图片质量评测

## 文件说明

`allmodel` 为图片QA质量模型

`brisque_model_live.yml` 和 `brisque_range_live.yml` 为 opencv-python 质量检测函数 `cv2.quality.QualityBRISQUE_compute` 入参

`img\` 目录下是手机拍的测试图片用于测试，由于测试图片太少，又用了 [IQA-Dataset](https://github.com/icbcbicc/IQA-Dataset) 作为测试的数据集

## 环境准备

环境中确保安装好了 python3，如果没有，可以用brew进行安装(mac)，windows可以自行安装python3，并配置到环境变量

安装python虚拟隔离环境工具，`pip3 install virtualenv`

切换到当前项目目录，执行 `virtualenv .` 在当前目录初始化虚拟环境

执行 `source bin/activate` 使当前环境生效，此时在当前项目目录下，`python` 和 `pip` 命令均为 `./bin` 目录下的命令

如果用的 `pyCharm` IDE，在项目设置中，将python解释器改为 `./bin` 下面的 `python`，这样ide才能识别第三方库的代码

`pip install -r requirements.txt` 安装依赖


## IQA-Dataset 使用说明
先 clone 项目到本地，修改 `demo.py`
找到下面这个代码
```py
dataset = load_dataset("LIVE", dataset_root="data", download=True)
```
`LIVE`为数据集名称，可以修改为别的数据集，然后运行此文件，开始数据集下载，下载的数据会在项目 `./data` 目录中
