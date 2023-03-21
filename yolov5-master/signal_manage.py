# coding:utf-8
'''
f_name为所读xx.txt文件
输出为：文件最后一行
'''
import os
import time
import matplotlib.pyplot as plt
import numpy
import csv

signal_data=numpy.zeros(440)

# 读取信号函数
def read_signal():
    txt_local = "E:\\Ultrasonic-point\\sdk包\\SuperDect-Demo\\Project\\SuperDect\\signal.txt"  # txt文件地址
    with open(txt_local, 'r') as f:    # 打开文件
        lines = f.readlines()       # 读所有行
        for i in range(-440,0):     # 取最后440个数据作为信号
            signal_data[i + 440] = int(lines[i])
    return signal_data

# 实时显示信号波形图函数-间隔0.5s
def show_signal(data):
    x=range(440)
    plt.plot(x,data[x])
    plt.draw()
    plt.pause(0.5)
    plt.clf()

# 保存信号到csv中
def save_signal(data):
    csv_file = open('signal.csv', 'a', newline='', encoding='gbk') # 调用open()函数打开csv文件，传入参数
    writer = csv.writer(csv_file)           # 用csv.writer()函数创建一个writer对象。
    writer.writerow(data)                   # 写入data数据到csv文件中
    csv_file.close()                        # 关闭文件


if __name__ == '__main__':

    os.remove('signal.csv')     # 清除残留的csv文件
    while True:
        data=read_signal()      # 读取信号
        # show_signal(data)       # 显示信号
        save_signal(data)       # 保存信号
        print(data[300])

        time.sleep(0.01)