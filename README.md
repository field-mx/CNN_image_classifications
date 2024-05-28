<注意>master这个分支为实际代码部分
程序说明：

<环境>
cuda11.6——>3050TI 

pytorch——>1.12.0

<1>image_pre_process.py会在data中构建一个output文件夹，按照721的比例划分训练集、验证集合与测试集合

<2>Customdata文件是用于生成csv数据集的，然而似乎并不需要

<3>image_process:更新了生成csv的功能，但是本网络似乎是吃图像的

<4>clear：用于清除数据集中的csv文件，以及空的文件夹

<5>connect：用于连接数据集与CNN网络

<6>block：CNN网络结构

<7>hw：参考例程，但是输入数据集为csv格式的，参考价值不是很大，容易走弯路
