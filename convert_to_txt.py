# -*- coding: utf-8 -*-
# @Author: lushujie
import os
import random
from configer import configer
def covert_to_txt():
    print("start covert_to_txt")
    trainval_percent = 0.05
    train_percent = 0.95

    confige = configer()

    curent_path = os.getcwd()
    dir_path = os.path.dirname(curent_path)
    xmlfilepath = confige.xml_path
    txtsavepath = confige.txt_path
    if not os.path.exists(txtsavepath):
        os.makedirs(txtsavepath)
    total_xml = os.listdir(xmlfilepath)

    num = len(total_xml)
    list = range(num)
    tv = int(num * trainval_percent)
    tr = int(tv * train_percent)
    trainval = random.sample(list, tv)
    train = random.sample(trainval, tr)

    ftrainval = open('%s/trainval.txt' % (txtsavepath), 'w')
    ftest = open('%s/test.txt' % (txtsavepath), 'w')
    ftrain = open('%s/train.txt' % (txtsavepath), 'w')
    fval = open('%s/val.txt' % (txtsavepath), 'w')

    for i in list:
        name = total_xml[i][:-4] + '\n'
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftest.write(name)
            else:
                fval.write(name)
        else:
            ftrain.write(name)

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()
