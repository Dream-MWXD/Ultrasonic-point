# -*- coding: utf-8 -*-
# 导入依赖
import random
import argparse
import os
import numpy as np
import time
import matplotlib.pyplot as plt
import csv
import sys
from JAKA_Zu import jkrc

robot = jkrc.RC("10.5.5.100")  # 返回一个机器人对象
robot.login()
robot.power_on()
robot.enable_robot()
PI=3.1415926
signal_data = np.zeros(1)
xyz=[]

# 读取信号函数
def read_signal():
    txt_local = "E:\\Ultrasonic-point\\data\\ultrasonic_coding.txt"  # txt文件地址
    with open(txt_local, 'r') as f:    # 打开文件
        lines = f.readlines()       # 读所有行
        signal_data[0] = lines[-1]
        # for i in range(-440,0):     # 取最后440个数据作为信号
        #     signal_data[i + 440] = int(lines[i])
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
    csv_file = open('E:\\Ultrasonic-point\\data\\ultrasonic_robot.csv', 'a', newline='', encoding='gbk') # 调用open()函数打开csv文件，传入参数
    writer = csv.writer(csv_file)           # 用csv.writer()函数创建一个writer对象。
    writer.writerow(data)                   # 写入data数据到csv文件中
    csv_file.close()                        # 关闭文件

if __name__ == '__main__':
    ret = robot.program_load("Sxingquxian")  # 加载通过APP编写的脚本 program_test需要自己编写
    ret = robot.get_loaded_program()
    file = "E:\\Ultrasonic-point\\data\\ultrasonic_robot.csv"
    if (os.path.exists(file) and os.path.isfile(file)):
        os.remove(file)
    print("the loaded program is:", ret[1])
    robot.program_run()
    time.sleep(2)  # 机械臂运动到起始位置的时间内不进行坐标输出

    for i in range(200000):  # 持续输出TCP坐标
        ultrasonic=[]
        ret = robot.get_robot_status()  # 由于机械臂新旧版本解释器问题，采用获得机械臂状态指令来替代机械臂的TCP坐标获取指令
        if ret[0] == 0:
            txt_local = "E:\\Ultrasonic-point\\data\\ultrasonic_coding.txt"  # txt文件地址
            with open(txt_local, 'r') as f:  # 打开文件
                lines = f.readlines()  # 读所有行
            ultrasonic.extend(ret[1][18])
            xyz.append(ret[1][18][0] + ret[1][18][1] + ret[1][18][2])
            signal_data[0]=lines[-1]
            ultrasonic.extend(signal_data)
            save_signal(ultrasonic)  # 保存数据到csv
        time.sleep(0.11)  # 写入TCP坐标的时间间隔
        if i>20000 and xyz[i] == xyz[i-10]:
            break
    robot.logout()  # 登出


    # # ret = robot.program_load("Sxingquxian")#加载通过APP编写的脚本 program_test需要自己编写
    # ret = robot.get_loaded_program()
    # print("the loaded program is:", ret[1])
    # robot.program_run()
    # time.sleep(4)  # 机械臂运动到起始位置的时间内不进行坐标输出
    #
    # while (1):
    #     ret = robot.get_robot_status()  # 由于机械臂新旧版本解释器问题，采用获得机械臂状态指令来替代机械臂的TCP坐标获取指令
    #     if ret[0] == 0:
    #         ultrasonic=[]
    #         print("the tcp position is :", ret[1][18])
    #         signal = read_signal()  # 读取信号
    #         ultrasonic.extend(ret[1][18])
    #         ultrasonic.extend(signal)
    #         # show_signal(signal)         # 显示信号
    #         save_signal(ultrasonic)  # 保存数据到csv
    #     else:
    #         print("some things happend,the errcode is: ", ret[0])
    #     time.sleep(1)  # 写入TCP坐标的时间间隔
    #
    # # robot.logout()  # 登出