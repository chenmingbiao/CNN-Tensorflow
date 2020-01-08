#!/usr/bin/python
#coding:utf-8

import inspect #检查运行信息的模块
import os
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt

VGG_MEAN = [103.939, 116.779, 123.68] #样本rgb平均值

class Vgg16():
    def __init__(self, vgg16_path=None):
        if vgg16_path is None:
            vgg16_path = os.path.join(os.getcwd(), "vgg16.npy")#加入路径 ，os.getcwd()用于返回当前工作目录
            self.data_dict = np.load(vgg16_path, encoding='latin1').item()#遍历，模型参数读入字典 

    def forward(self, images):
        
        print("build model started")
        start_time = time.time() #获取前项传播的开始时间
        rgb_scaled = images * 255.0 #逐像素乘上255.0
        #从GRB转换色道，到BGR
        red, green, blue = tf.split(rgb_scaled,3,3) 
        bgr = tf.concat([     
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2]],3)
        #逐像素减去像素平均值，可以移除平均亮度，常用于灰度图像上
        
        #接下来构建VGG的16层网络（包括5段卷积，3层全连接）并逐层根据网络空间读取网络参数
        #第一段卷积，有两个卷积层，加最大池化，用来缩小图片尺寸
        
        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        #传入name，获取卷积核和偏置，并卷积运算，经过激活函数后返回。
        
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        
        #池化
        self.pool1 = self.max_pool_2x2(self.conv1_2, "pool1")
        
        #第二段卷积，和第一段相同
        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool_2x2(self.conv2_2, "pool2")
        
        #第三段卷积，三个卷积层，一个最大池化

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool_2x2(self.conv3_3, "pool3")
        
        #第四段卷积，三个卷积层，一个最大池化
        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool_2x2(self.conv4_3, "pool4")
        #第五段卷积，三个卷积层，一个最大池化
        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.max_pool_2x2(self.conv5_3, "pool5")
        #第六段全连接层
        self.fc6 = self.fc_layer(self.pool5, "fc6")#根据命名空间fc6，做加权求和 
        self.relu6 = tf.nn.relu(self.fc6) #激活
        #第七段全连接层
        self.fc7 = self.fc_layer(self.relu6, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)
        #第八层全连接
        self.fc8 = self.fc_layer(self.relu7, "fc8")
        self.prob = tf.nn.softmax(self.fc8, name="prob")#softmax分类，得到属于各个类别的概率。
        
        end_time = time.time() #获取结束时间
        print(("time consuming: %f" % (end_time-start_time)))#耗时

        self.data_dict = None #清空本次读到地模型字典
        
        
        
        #卷积计算的相关定义，
    def conv_layer(self, x, name):
        with tf.variable_scope(name):#根据命名空间找到网络参数。 
            w = self.get_conv_filter(name) #读取卷积核
            conv = tf.nn.conv2d(x, w, [1, 1, 1, 1], padding='SAME')卷积计算 
            conv_biases = self.get_bias(name) #读取偏置
            result = tf.nn.relu(tf.nn.bias_add(conv, conv_biases)) #加上偏置，并激活
            return result
    
    def get_conv_filter(self, name):#其中获取卷积核的定义
        return tf.constant(self.data_dict[name][0], name="filter") 
    
    def get_bias(self, name):#其中获取偏置的定义
        return tf.constant(self.data_dict[name][1], name="biases")
    
    def max_pool_2x2(self, x, name):#其中最大池的定义
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
    
    #定义全连接层的前项传播计算
    def fc_layer(self, x, name):
        with tf.variable_scope(name):#根据命名空间做计算 
            shape = x.get_shape().as_list() #该层维度信息
            dim = 1
            for i in shape[1:]:
                dim *= i #每层的维度相乘
            x = tf.reshape(x, [-1, dim])#改变特征图形状，多维度特征的拉伸操作，只第六层用
            w = self.get_fc_weight(name) #读取w
            b = self.get_bias(name) #读取b
                
            result = tf.nn.bias_add(tf.matmul(x, w), b)#加权求和加偏置 
            return result
    #定义获取权重
    def get_fc_weight(self, name):  
        return tf.constant(self.data_dict[name][0], name="weights")