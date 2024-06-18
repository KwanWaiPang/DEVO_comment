[comment]: <> (# DEVO)

<!-- PROJECT LOGO -->

<p align="center">

  <h1 align="center"> DEVO (复现及中文注释版~仅供个人学习记录用)
  </h1>

[comment]: <> (  <h2 align="center">PAPER</h2>)
  <h3 align="center">
  <a href="https://arxiv.org/pdf/2312.09800">Paper</a> 
  | <a href="https://github.com/tum-vision/DEVO">Original Github Page</a>
  | <a href="https://blog.csdn.net/gwplovekimi/article/details/139436796?spm=1001.2014.3001.5501">CSDN DPVO的配置教程</a>
  </h3>
  <div align="center"></div>

<p align="center">
  <img width="90%" src="assets/devo.svg">
</p>

<br>

# DEVO配置记录
~~~
<!-- 创建conda环境 -->
conda env create -f environment.yml
conda activate devo

<!-- 安装eigen -->
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip -d thirdparty

<!-- 安装DEVO -->
# install DEVO
pip install .
~~~

# 测试作者已训练好的模型
~~~
./download_model.sh
~~~
* 下载测试数据，并且对测试数据集进行处理（以HKU数据集为例）
~~~
conda activate nerf-ngp
bypy list
bypy download [remotepath] [localpath]  #注意要指定一下下载的路径~

python scripts/pp_hku.py
~~~
* 运行测试代码
~~~
python evals/eval_evs/eval_XXX_evs.py --datapath=<path to xxx dataset> --weights="DEVO.pth" --stride=1 --trials=1 --expname=<your name>
~~~


# 训练记录
* 下载[TartanAir](https://theairlab.org/tartanair-dataset/)中所有的数据,采用[工具](https://github.com/castacks/tartanair_tools)
~~~
cd thirdparty/tartanair_tools/
pip install boto3 #需要安装依赖~
python download_training.py --output-dir ../../datasets --rgb --depth --only-left
~~~

* 采用[vid2e/ESIM](https://github.com/KwanWaiPang/ESIM_comment)实现将video变成event