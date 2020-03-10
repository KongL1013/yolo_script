# -*- coding: utf-8 -*-
# @Author: lushujie
import os
import sys
from configer import configer
from enhance import start_img_enhance
from convert_to_txt import covert_to_txt
from voc_label import voc_label


class train_yolo():
    def __init__(self, config):
        self.config = config

    def file_check(self):
        img_total = os.listdir(self.config.pic_path)
        for i in img_total:
            img_name = i.split('.')[0]
            xml = os.path.join(self.config.xml_path, str(img_name) + '.xml')
            if not os.path.exists(xml):
                pic_path = os.path.join(self.config.pic_path, i)
                os.remove(pic_path)
                print(img_name, '.xml文件不存在:', ' 我已经把这张图删了')

    def change_voc_data(self):
        text = []
        text.append('classes = {}'.format(len(self.config.classes)))
        text.append('train = {}'.format(os.path.join(self.config.txt_path, 'final_train.txt')))
        text.append('valid = {}'.format(os.path.join(self.config.txt_path, 'final_test.txt')))
        cu = os.getcwd()
        text.append('names = {}'.format(self.config.voc_names_path))
        text.append('backup = backup')
        with open(self.config.voc_data_path, 'w') as f:
            for i in range(len(text)):
                f.write(text[i] + '\n')

    def change_voc_names(self):
        with open(self.config.voc_names_path, 'w') as f:
            for i in range(len(self.config.classes)):
                f.write(self.config.classes[i] + '\n')

    def change_yolov3_cfg(self):
        if not os.path.exists(self.config.yolov3_cfg_path):
            print("darknet/cfg/文件夹下无yolov3.cfg，请复制一个过来0.0")
            sys.exit()
        with open(self.config.yolov3_cfg_path, 'r+') as f:
            yolov3_cfg = f.readlines()
            len_class = len(self.config.classes)
            filters = 3 * (len_class + 5)

            yolov3_cfg[17] = 'learning_rate={}\n'.format(self.config.learning_rate)
            yolov3_cfg[19] = 'max_batches = {}\n'.format(self.config.max_batches)
            yolov3_cfg[21] = 'steps={},{}\n'.format(self.config.steps[0], self.config.steps[1])
            yolov3_cfg[22] = 'scales={},{}\n'.format(self.config.scales[0], self.config.scales[1])

            yolov3_cfg[602] = 'filters={}\n'.format(filters)
            yolov3_cfg[609] = 'classes={}\n'.format(len_class)

            yolov3_cfg[688] = 'filters={}\n'.format(filters)
            yolov3_cfg[695] = 'classes={}\n'.format(len_class)

            yolov3_cfg[775] = 'filters={}\n'.format(filters)
            yolov3_cfg[782] = 'classes={}\n'.format(len_class)
        with open(self.config.yolov3_cfg_path, 'w') as f:
            for lines in yolov3_cfg:
                f.write(lines)

    def change_yolov3_tiny_cfg(self):
        if not os.path.exists(self.config.yolov3_tiny_cfg_path):
            print("darknet/cfg/文件夹下无yolov3-tiny.cfg，请复制一个过来0.0")
            sys.exit()
        with open(self.config.yolov3_tiny_cfg_path, 'r+') as f:
            yolov3_cfg = f.readlines()
            len_class = len(self.config.classes)
            filters = 3 * (len_class + 5)

            yolov3_cfg[17] = 'learning_rate={}\n'.format(self.config.learning_rate)
            yolov3_cfg[19] = 'max_batches = {}\n'.format(self.config.max_batches)
            yolov3_cfg[21] = 'steps={},{}\n'.format(self.config.steps[0], self.config.steps[1])
            yolov3_cfg[22] = 'scales={},{}\n'.format(self.config.scales[0], self.config.scales[1])

            yolov3_cfg[126] = 'filters={}\n'.format(filters)
            yolov3_cfg[134] = 'classes={}\n'.format(len_class)

            yolov3_cfg[170] = 'filters={}\n'.format(filters)
            yolov3_cfg[176] = 'classes={}\n'.format(len_class)
        with open(self.config.yolov3_tiny_cfg_path, 'w') as f:
            for lines in yolov3_cfg:
                f.write(lines)

    def change_train_cfg(self):
        if self.config.if_tiny_yolo:
            cfg_path = self.config.yolov3_tiny_cfg_path
        else:
            cfg_path = self.config.yolov3_cfg_path
        with open(cfg_path, 'r+') as f:
            cfg = f.readlines()
            cfg[2] = '#batch=1\n'
            cfg[3] = '#subdivisions=1\n'
            cfg[5] = 'batch={}\n'.format(self.config.batch_size)
            cfg[6] = 'subdivisions={}\n'.format(self.config.subdivisions)
        with open(cfg_path, 'w') as f:
            for lines in cfg:
                f.write(lines)

    def change_test_cfg(self):
        if self.config.if_tiny_yolo:
            cfg_path = self.config.yolov3_tiny_cfg_path
        else:
            cfg_path = self.config.yolov3_cfg_path
        with open(cfg_path, 'r+') as f:
            cfg = f.readlines()
            cfg[2] = 'batch=1\n'
            cfg[3] = 'subdivisions=1\n'
            cfg[5] = '#batch={}\n'.format(self.config.batch_size)
            cfg[6] = '#subdivisions={}\n'.format(self.config.subdivisions)
        with open(cfg_path, 'w') as f:
            for lines in cfg:
                f.write(lines)


def yolo_train(config):
    train = train_yolo(config)
    train.file_check()
    if config.if_enhance:
        start_img_enhance()
    covert_to_txt()
    voc_label()

    train.change_voc_data()
    train.change_voc_names()
    if config.if_tiny_yolo:
        train.change_yolov3_tiny_cfg()
        train.change_train_cfg()

        if not os.path.exists(config.yolo_tiny_train_weights_path):
            print("请下载yolo权重到darknet文件夹下")
            print("加载权重方法\nyolo-tiny:wget https://pjreddie.com/media/files/yolov3-tiny.weights")
            sys.exit()
        else:
            command1 = "cd ../../../;{}/darknet partial {} {} {}/yolov3-tiny.conv.15 15".format(config.darknet_path,
                                                                                                config.yolov3_tiny_cfg_path,
                                                                                                config.yolo_tiny_train_weights_path,
                                                                                                config.darknet_path)
            os.system(command1)
            command2 = "cd ../../../;{}/darknet detector train {} {} {}".format(config.darknet_path,
                                                                  config.voc_data_path,
                                                                  config.yolov3_tiny_cfg_path,
                                                                  os.path.join(config.darknet_path,
                                                                               'yolov3-tiny.conv.15'))
            os.system(command2)
            # os.system('echo %s|sudo -S %s' % (config.sudo_password, command2))

    if not config.if_tiny_yolo:
        train.change_yolov3_cfg()
        train.change_train_cfg()

        if not os.path.exists(config.yolo_train_weights_path):
            print("请下载yolo权重到darknet文件夹下")
            print("加载权重方法\nyolo:wget https://pjreddie.com/media/files/darknet53.conv.74")
            sys.exit()
        else:
            # ./darknet detector train cfg/voc.data cfg/yolov3-voc.cfg darknet53.conv.74
            os.system("cd ../../../;{}/darknet detector train {} {} {}".format(config.darknet_path,
                                                                               config.voc_data_path,
                                                                               config.yolov3_cfg_path,
                                                                               os.path.join(config.darknet_path,
                                                                                            "darknet53.conv.74")))


def yolo_test(config):
    test = train_yolo(config)
    test.change_test_cfg()
    if config.if_tiny_yolo:
        # ./darknet detect cfg/yolov3-tiny.cfg yolov3-tiny.weights data/horses.jpg
        os.system('cd ../../../;{}/darknet detect  {} {} {}'.format(config.darknet_path,
                                                                    config.yolov3_tiny_cfg_path,
                                                                    config.testing_weight_path,
                                                                    config.test_pic_path))
    else:
        # ./darknet detect cfg/yolov3-tiny.cfg yolov3-tiny.weights data/dog.jpg
        os.system('cd ../../../;{}/darknet detect {} {} {}'.format(config.darknet_path,
                                                                   config.yolov3_cfg_path,
                                                                   config.testing_weight_path,
                                                                   config.test_pic_path))


def yolo_retrain(config):
    # sudo ./darknet detector train cfg/voc.data cfg/yolov3-tiny.cfg backup/yolov3-tiny_200.weights
    train = train_yolo(config)
    if not os.path.exists(config.re_training_path):
        print("请输入darknet/backup文件夹下最后保存的权重地址")
        sys.exit()
    if config.if_tiny_yolo:
        train.change_train_cfg()
        os.system("cd ../../../;{}/darknet detector train {} {} {}".format(config.darknet_path,
                                                                           config.voc_data_path,
                                                                           config.yolov3_tiny_cfg_path,
                                                                           config.re_training_path))
    else:
        train.change_train_cfg()
        os.system("cd ../../../;{}/darknet detector train {} {} {}".format(config.darknet_path,
                                                                           config.voc_data_path,
                                                                           config.yolov3_cfg_path,
                                                                           config.re_training_path))


if __name__ == '__main__':
    config = configer()

    if config.mode == 'retrain':
        yolo_retrain(config)

    if config.mode == 'train':
        yolo_train(config)

    if config.mode == 'test':
        yolo_test(config)
