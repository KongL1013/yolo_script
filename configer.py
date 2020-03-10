# -*- coding: utf-8 -*-
# @Author: lushujie
import os


class configer():
    def __init__(self):
        # 三种模式retrain train test,图片格式无特殊要求，保持一致即可
        mode = ['train', 'test', 'retrain']
        self.mode = mode[1]  # 三种模式:训练，测试，接着上次的继续训练

        self.test_pic_path = '/home/dd/darknet/data/yolo_data/pic/0.jpg'  # 如果是测试模式，请输入测试图片地址
        self.testing_weight_path = '/home/dd/darknet/backup/yolov3-tiny_final.weights'  # 测试模式的权重地址，在darknet/backup文件夹下，保存有训练权重结果
        self.re_training_path = '/home/dd/darknet/backup/yolov3-tiny_800.weights'  # 如果是retrian模式,请输入darknet/backup/下最后保存的权重，接着上次训练结束地方继续训练(仅限前1000次），

        self.if_tiny_yolo = True  # True->tiny-yolo False->yolo
        self.classes = ["door", "table", "doc", "pipe"]  # 分类类别

        # 这只是部分常用参数设置，具体设置在.cfg文件中
        # 参数含义参考 https://www.cnblogs.com/shierlou-123/p/11152623.html
        self.learning_rate = 0.001
        self.max_batches = 4000  # 训练次数
        self.steps = [1000, 2000]  # 训练多少次时衰减学习率
        self.scales = [.1, .1]  # 每次衰减的学习率规模
        self.batch_size = 32
        self.subdivisions = 4

        # 数据增强,数据增强具体参数请在enhance.py中修改
        # #官方说明文档 https://imgaug.readthedocs.io/en/latest/,有很多增强方法
        self.if_enhance = False  # 是否进行数据增强,
        if self.mode == 'retrain':
            self.if_enhance = False  # 如果重新开始训练mode='retrain'，关闭数据增强选项,否则会将所有图片再次增强
        self.if_oneof = True  # True,每张图下面3个滤波函数中你选择那几个的随机挑一执行。 False 每张图逐个执行下面3个滤波函数中你选择的那几个
        self.GaussianBlur = True
        self.AverageBlur = False
        self.MedianBlur = True

        # 下面的增强方法逐一执行
        self.Multiply = False  # change the brightness of the whole image (sometimes per channel)
        self.Add = False  # change brightness of images of original value
        self.AddToHueAndSaturation = True  # change their color
        self.ContrastNormalization = True  # Strengthen or weaken the contrast in each image.
        self.AdditiveGaussianNoise = True  # # add gaussian noise to images
        self.Affine = False  # 旋转操作，不建议执行,并且该操作我加了sometimes(0.3,..)，即使执行也只有30%图片旋转

        ##########################请勿随意更改下列设置##############################
        self.darknet_path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
        self.yolo_tiny_train_weights_path = os.path.join(self.darknet_path, 'yolov3-tiny.weights')  # yolo-tiny训练权重地址
        self.yolo_train_weights_path = os.path.join(self.darknet_path, 'yolov3.weights')  # yolo weights

        # 训练支撑文件地址
        self.cfg_path = os.path.join(self.darknet_path, 'cfg')
        self.voc_data_path = os.path.join(self.cfg_path, 'coco.data')  # yolo默认指向coco.data
        self.voc_names_path = os.path.join(self.cfg_path, 'voc.names')
        self.yolov3_cfg_path = os.path.join(self.cfg_path, 'yolov3.cfg')
        self.yolov3_tiny_cfg_path = os.path.join(self.cfg_path, 'yolov3-tiny.cfg')

        # pic和xml地址 需要绝对路径
        cur_path = os.getcwd()
        self.yolo_data_path = os.path.dirname(cur_path)
        self.pic_path = os.path.join(self.yolo_data_path, 'pic')
        self.xml_path = os.path.join(self.yolo_data_path, 'xml')
        self.txt_path = os.path.join(self.yolo_data_path, 'txt')
        self.labels_path = os.path.join(self.yolo_data_path, 'pic')
        pic_type = os.listdir(self.pic_path)[0].split('.')[-1]
        self.pic_type = pic_type  # 图片格式 jpg,png,其他图片格式均可
