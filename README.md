# 狗品种分类问题

## 环境依赖

首先需要python和conda正确安装，conda是一个python的库管理工具，
如果未安装可以先去安装一下，https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
推荐直接装 anaconda，里面包括conda，还有很多python科学计算的库，下载链接在
https://www.anaconda.com/distribution/ 注意找到对应的版本。

项目基于 Pytorch 1.2，别的版本可能也可以正常运行，但是未经过测试，
如果不满足环境条件，建议新建一个环境。

conda 创建一个叫做 torch 的 python3.6 新环境。

```shell script
conda create -n torch python=3.6
source activate torch
pip install -r requirements.txt
```

## 程序运行方式

推荐使用 Jupyter Notebook 打开教程文件，在命令行中运行

```shell script
jupyter notebook
```

然后浏览器里会有当前目录的所有文件，打开 Tutorial.ipynb，然后跟随教程一个个执行就可以了。

或者直接运行代码

```shell script
python3 main.py
```



