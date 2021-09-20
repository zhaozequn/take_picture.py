# -*- coding: utf-8 -*-
import datetime
import os
import sys
import time
from threading import Thread

import cv2
import numpy as np
import paddle
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import gxipy as gx
from Gui import Ui_Form
import glob
import matplotlib.image as mpimg

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

paddle.enable_static()


class MyThread(Thread):
    def __init__(self, func, args):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None


class Adjust():
    def __init__(self):
        cv2.namedWindow("Adjust Window")
        cv2.resizeWindow("Adjust Window", 640, 340)
        cv2.createTrackbar("Hue Min", "Adjust Window", 0, 179, self.hsv_adjust)
        cv2.createTrackbar("Hue Max", "Adjust Window", 19, 179, self.hsv_adjust)
        cv2.createTrackbar("Sat Min", "Adjust Window", 110, 255, self.hsv_adjust)
        cv2.createTrackbar("Sat Max", "Adjust Window", 240, 255, self.hsv_adjust)
        cv2.createTrackbar("Val Min", "Adjust Window", 153, 255, self.hsv_adjust)
        cv2.createTrackbar("Val Max", "Adjust Window", 255, 255, self.hsv_adjust)
        cv2.createTrackbar("RGB Threshold", "Adjust Window", 2, 255, self.hsv_adjust)
        cv2.createTrackbar("Gray", "Adjust Window", 0, 255, self.hsv_adjust)

    def hsv_adjust(self, a):
        h_min = cv2.getTrackbarPos("Hue Min", "Adjust Window")
        h_max = cv2.getTrackbarPos("Hue Max", "Adjust Window")
        s_min = cv2.getTrackbarPos("Sat Min", "Adjust Window")
        s_max = cv2.getTrackbarPos("Sat Max", "Adjust Window")
        v_min = cv2.getTrackbarPos("Val Min", "Adjust Window")
        v_max = cv2.getTrackbarPos("Val Max", "Adjust Window")
        rgb_threshold = cv2.getTrackbarPos("RGB Threshold", "Adjust Window")
        gray = cv2.getTrackbarPos("Gray", "Adjust Window")

        # print(h_min, h_max, s_min, s_max, v_min, v_max)
        return h_min, h_max, s_min, s_max, v_min, v_max, rgb_threshold, gray


class SploceGui(Ui_Form, QMainWindow):

    def __init__(self):
        super(SploceGui, self).__init__()
        self.setupUi(self)
        # self.pushButton_OpenCamera.clicked.connect(self.open_frame_sdk)
        self.pushButton_OpenCamera.clicked.connect(self.fruit_kind)
        self.pushButton_clear.clicked.connect(self.clear)
        self.pushButton_stop.clicked.connect(self.stop)
        self.speed = 30
        self.statusBar().showMessage("Ready")


    def fruit_kind(self):
        # 开始
        global start
        start = True
        # 选择水果种类
        self.fruit_kinds = self.comboBox_fruit_kinds.currentText()
        if self.fruit_kinds == "黄桃":
            self.pushButton_OpenCamera.clicked.connect(self.open_frame_yellowpeach)
        if self.fruit_kinds == "大桃":
            self.pushButton_OpenCamera.clicked.connect(self.open_frame_peach)
        if self.fruit_kinds == "苹果":
            self.pushButton_OpenCamera.clicked.connect(self.open_frame_apple)
        if self.fruit_kinds == "番茄":
            self.pushButton_OpenCamera.clicked.connect(self.open_frame_tomato)

    def stop(self):

        global start
        start = False

        self.statusBar().showMessage("Stop")

        self.black = np.zeros((480, 640, 1), dtype='uint8')
        self.show_on_gui(self.black, self.label_frame1)
        self.show_on_gui(self.black, self.label_frame2)
        self.show_on_gui(self.black, self.label_frame3)
        self.show_on_gui(self.black, self.label_frame4)
        self.show_on_gui(self.black, self.label_frame5)
        self.show_on_gui(self.black, self.label_frame6)
        self.show_on_gui(self.black, self.label_frame7)
        self.show_on_gui(self.black, self.label_frame8)

        self.num_pinjie = 0
        self.label_save_tips.setText("图像保存")
        self.label_text_framenum.setText("程序已停止")

    def clear(self):
        self.cropped_list = np.zeros((260, 1, 3), dtype='uint8')
        self.show_on_gui(self.cropped_list, self.label_frame7)
        self.num_pinjie = 0
        self.label_save_tips.setText("图像保存")

    def threshold_value(self, value):
        global threshold_value
        threshold_value = value
        print(value)

    def cal_calibrate_params(self,file_paths):
        object_points = []  # 三维空间中的点：3D
        image_points = []  # 图像空间中的点：2d
        # 2.1 生成真实的交点坐标：类似(0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)的三维点
        objp = np.zeros((11 * 8, 3), np.float32)
        objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
        # 2.2 检测每幅图像角点坐标
        for file_path in file_paths:
            img = cv2.imread(file_path)
            # 将图像转换为灰度图
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 自动检测棋盘格内4个棋盘格的角点（2白2黑的交点）
            rect, corners = cv2.findChessboardCorners(gray, (11, 8), None)
            # 若检测到角点，则将其存储到object_points和image_points
            if rect == True:
                object_points.append(objp)
                image_points.append(corners)
        # 2.3 获取相机参数
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)
        return ret, mtx, dist, rvecs, tvecs

    def img_undistort(self,img, mtx, dist):
        """
        图像去畸变
        """
        return cv2.undistort(img, mtx, dist, None, mtx)


    def open_frame_peach(self):
        try:

            global start
            # 读取相机
            cap = cv2.VideoCapture(1)

            # 滑块控制阈值分割的准备
            self.threshold_value(0)

            self.cropped = np.zeros((260, 1, 3), dtype='uint8')

            self.start_1s_time = time.time()

            # 帧数
            frame_num = 0

            # 初始化拼接图像列
            self.cropped_list = np.zeros((260, 1, 3), dtype='uint8')

            # 抓取到目标的次数
            self.num_zhuaqu = 0

            # 完成拼接的次数
            self.num_pinjie = 0

            # statusbar
            self.statusBar().showMessage("Busy")

            # 计算大小
            self.area_all = []

            #相机校正
            file_paths = glob.glob("./picture/11630244875.6.bmp")
            self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = self.cal_calibrate_params(file_paths)

            while start:

                frame_num += 1

                self.start_time = time.time()

                # 读取帧
                ret, self.carmera_img = cap.read()

                if ret != True:
                    self.statusBar().showMessage("Cannot read camera!")

                #校正相机
                self.undistort_img = self.img_undistort(self.carmera_img, self.mtx, self.dist)

                # 白平衡后得到待处理图像
                self.img = self.white_balance(self.undistort_img)

                # 显示
                self.show_on_gui(self.img, self.label_frame1)

                # 步骤1，转变色彩通道与调整阈值,形态学处理
                self.img_mask = self.adjust(self.img)

                # 图像裁剪
                self.img_mask = self.img_mask[140:400, :]
                self.img = self.img[140:400, :]

                # 步骤2，轮廓去噪
                self.output, self.img_roi, = self.roi_frame(self.img_mask, self.img)

                # 轮廓相关
                # self.fill = self.contours(self.output)
                self.peach, self.peach_binary = self.roi_frame_2(self.output, self.img_roi)

                if frame_num % 1 == 0:
                    # 得到了完整的轮廓,找轮廓中心
                    self.cropped = self.findcenter(self.peach, self.peach_binary)

                    # 未抓取到目标
                    if self.cropped == []:

                        # 未抓取到目标时，清空前存拼接图像
                        self.cropped_list = np.zeros((260, 1, 3), dtype='uint8')

                        # 清空本目标抓取次数
                        self.num_zhuaqu = 0
                        self.label_num_zhauqu.setText("当前目标被抓取：" + str(self.num_zhuaqu) + "次")

                        self.label_text_jiance.setText("检测：未抓取到目标")
                        self.label_text_pinjie.setText("拼接：未完成拼接")


                    # 抓取到目标
                    else:
                        # 计算目标的面积
                        self.area_one = self.area_count(self.peach_binary)
                        self.area_all.append(self.area_one)

                        self.cropped_list = np.append(self.cropped_list, self.cropped, axis=1)

                        self.label_text_jiance.setText("检测：抓取到目标!")
                        self.label_text_pinjie.setText("拼接：未完成拼接")

                        self.num_zhuaqu += 1
                        self.label_num_zhauqu.setText("当前目标被抓取：" + str(self.num_zhuaqu) + "次")

                        if self.cropped_list.shape[1] >= 600:

                            # 计算平均面积
                            self.area_all_cacul = sum(self.area_all)
                            self.n = len(self.area_all)
                            self.area_avg = round(self.area_all_cacul / self.n)
                            self.label_area.setText("大小： " + str(self.area_avg) + "Pixel")
                            self.area_all = []

                            # 取拼接完成后图像去除第一列
                            self.cropped_list = self.cropped_list[:, 1:, :]
                            self.show_on_gui(self.cropped_list, self.label_frame7)

                            # 得到拼接完成后的图像结果
                            self.img_cropped = self.cropped_list

                            # 拼接个数增加
                            self.num_pinjie += 1

                            # 对拼接完成的图像进行后续处理

                            # 是否保存图像
                            if self.checkBox_savepic.isChecked() == True:
                                savetime = datetime.datetime.now().day
                                path = "./picture/" + str(savetime) + "_" + str(self.num_pinjie) + "_" + str(
                                    frame_num) + ".bmp"
                                cv2.imwrite(path, self.img_cropped)
                                self.label_save_tips.setText("图像已保存，序号为" + str(self.num_pinjie) + "_" + str(frame_num))
                            else:
                                self.label_save_tips.setText("图像未保存!")

                            # 已拼接图像做颜色处理
                            self.red_ratio, self.mask_red = self.color_peach(self.img_cropped)
                            self.label_redratio.setText("红色比例：" + str(self.red_ratio))
                            self.show_on_gui(self.mask_red, self.label_frame8)

                            # 显示次数
                            self.label_text_pinjie.setText("拼接：完成拼接!")
                            self.label_num_pinjie.setText("完成拼接：" + str(self.num_pinjie))

                            # 完成一次拼接后将抓取归零

                            self.num_zhuaqu = 0
                            self.cropped_list = np.zeros((260, 1, 3), dtype='uint8')

                            # cv2.imshow("ssss",self.cropped_img)
                            # cv2.waitKey(0)
                        else:
                            pass

                # self.fill = self.floodfill(self.output)
                # 拼接多张图片的中心

                # 计算帧速
                self.end_time = time.time()
                self.seconds = self.end_time - self.start_time
                self.fps = int(1 / self.seconds)

                # 显示FPS
                self.label_text_framenum.setText("FPS:" + str(self.fps))

        except Exception as error:
            self.statusBar().showMessage("Error")
            print("ERROR:", error)

    def open_frame_yellowpeach(self):

        try:

            global start
            # 读取相机
            cap = cv2.VideoCapture(1)

            # 滑块控制阈值分割的准备
            self.threshold_value(0)

            self.cropped = np.zeros((260, 1, 3), dtype='uint8')

            self.start_1s_time = time.time()

            # 帧数
            frame_num = 0

            # 初始化拼接图像列
            self.cropped_list = np.zeros((260, 1, 3), dtype='uint8')

            # 抓取到目标的次数
            self.num_zhuaqu = 0

            # 完成拼接的次数
            self.num_pinjie = 0

            # statusbar
            self.statusBar().showMessage("Busy")

            # 计算大小
            self.area_all = []

            while start:

                frame_num += 1

                self.start_time = time.time()

                # 读取帧
                ret, self.carmera_img = cap.read()

                # 白平衡后得到待处理图像
                self.img = self.white_balance(self.carmera_img)

                # 显示
                self.show_on_gui(self.img, self.label_frame1)

                # 步骤1，转变色彩通道与调整阈值,形态学处理
                self.img_mask = self.adjust(self.img)

                # 图像裁剪
                self.img_mask = self.img_mask[140:400, :]
                self.img = self.img[140:400, :]

                # 步骤2，轮廓去噪
                self.output, self.img_roi, = self.roi_frame(self.img_mask, self.img)

                # 轮廓相关
                # self.fill = self.contours(self.output)
                self.peach, self.peach_binary = self.roi_frame_2(self.output, self.img_roi)

                if frame_num % 1 == 0:

                    # 得到了完整的轮廓,找轮廓中心
                    self.cropped = self.findcenter(self.peach, self.peach_binary)

                    # 未抓取到目标
                    if self.cropped == []:

                        # 未抓取到目标时，清空前存拼接图像
                        self.cropped_list = np.zeros((260, 1, 3), dtype='uint8')

                        # 清空本目标抓取次数
                        self.num_zhuaqu = 0
                        self.label_num_zhauqu.setText("当前目标被抓取：" + str(self.num_zhuaqu) + "次")

                        self.label_text_jiance.setText("检测：未抓取到目标")
                        self.label_text_pinjie.setText("拼接：未完成拼接")


                    # 抓取到目标
                    else:
                        # 计算目标的面积
                        self.area_one = self.area_count(self.peach_binary)
                        self.area_all.append(self.area_one)

                        self.cropped_list = np.append(self.cropped_list, self.cropped, axis=1)

                        self.label_text_jiance.setText("检测：抓取到目标!")
                        self.label_text_pinjie.setText("拼接：未完成拼接")

                        self.num_zhuaqu += 1
                        self.label_num_zhauqu.setText("当前目标被抓取：" + str(self.num_zhuaqu) + "次")

                        if self.cropped_list.shape[1] >= 640:

                            # 计算平均面积
                            self.area_all_cacul = sum(self.area_all)
                            self.n = len(self.area_all)
                            self.area_avg = round(self.area_all_cacul / self.n)
                            self.label_area.setText("大小： " + str(self.area_avg) + "Pixel")
                            self.area_all = []

                            # 取拼接完成后图像去除第一列
                            self.cropped_list = self.cropped_list[:, 1:, :]
                            self.show_on_gui(self.cropped_list, self.label_frame7)

                            # 得到拼接完成后的图像结果
                            self.img_cropped = self.cropped_list

                            # 拼接个数增加
                            self.num_pinjie += 1

                            # 对拼接完成的图像进行后续处理

                            # 是否保存图像
                            if self.checkBox_savepic.isChecked() == True:
                                savetime = datetime.datetime.now().day
                                path = "./picture/" + str(savetime) + "_" + str(self.num_pinjie) + "_" + str(
                                    frame_num) + ".bmp"
                                cv2.imwrite(path, self.img_cropped)
                                self.label_save_tips.setText("图像已保存，序号为" + str(self.num_pinjie) + "_" + str(frame_num))
                            else:
                                self.label_save_tips.setText("图像未保存!")

                            # 已拼接图像做颜色处理
                            self.red_ratio, self.mask_red = self.color_yellowpeach(self.img_cropped)
                            self.label_redratio.setText("红色比例：" + str(self.red_ratio))
                            self.show_on_gui(self.mask_red, self.label_frame8)

                            # 显示次数
                            self.label_text_pinjie.setText("拼接：完成拼接!")
                            self.label_num_pinjie.setText("完成拼接：" + str(self.num_pinjie))

                            # 完成一次拼接后将抓取归零

                            self.num_zhuaqu = 0
                            self.cropped_list = np.zeros((260, 1, 3), dtype='uint8')

                            # cv2.imshow("ssss",self.cropped_img)
                            # cv2.waitKey(0)
                        else:
                            pass

                # self.fill = self.floodfill(self.output)
                # 拼接多张图片的中心

                # 计算帧速
                self.end_time = time.time()
                self.seconds = self.end_time - self.start_time
                self.fps = int(1 / self.seconds)

                # 显示FPS
                self.label_text_framenum.setText("FPS:" + str(self.fps))

        except Exception as error:
            print(error)

    def findcenter(self, img, binary):

        # t = Timer(1,self.findcenter)
        # t.start()
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        self.yilie = np.zeros((260, 1, 3), dtype='uint8')
        self.cX = self.cY = 0
        if contours == []:
            return []
        else:
            # 用i来遍历轮廓
            for i in contours:
                print(cv2.contourArea(i))
                if cv2.contourArea(i) > 12000:
                    M = cv2.moments(i)
                    print('enter')
                    if M["m00"] != 0:  # 图像面积为零
                        self.cX = int(M["m10"] / M["m00"])  # 找到轮廓中心的X坐标
                        self.cY = int(M["m01"] / M["m00"])  # 找到轮廓中心的Y坐标
                    # print(self.cX,self.cY)

                    # 判断坐标位置
                    if self.cX <= 280:
                        self.cx1_zuo = self.cX - 6
                        self.cx1_you = self.cX + 7

                    elif 280 < self.cX < 360:
                        # 得到左右偏移裁剪量
                        self.cx1_zuo = self.cX - 8
                        self.cx1_you = self.cX + 8

                    elif self.cX >= 360:
                        self.cx1_zuo = self.cX - 6
                        self.cx1_you = self.cX + 7

                    # 以轮廓中心为中点提取一定宽度像素矩阵
                    self.cropped = img[:, self.cx1_zuo:self.cx1_you]  # 裁剪坐标为[y0:y1, x0:x1]

                    self.show_on_gui(self.cropped, self.label_frame6)

                    return self.cropped
                else:
                    return []

                # 拼接
                # self.cropped_list = np.append(self.cropped_list,self.cropped,axis=1)

    def contours(self, img):
        contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        n = len(contours)

        j = []

        for i in contours:
            area = cv2.contourArea(i)
            print(area)
            if area > 10000:
                j.append(i)
            else:
                continue
        cv2.fillPoly(img, j, (0, 0, 0))

        self.show_on_gui(img, self.label_frame5)

    def area_count(self, img):
        self.num_labels, self.labels, self.stats, self.centroids = cv2.connectedComponentsWithStats(img,
                                                                                                    connectivity=8)  # 在mask中找到所需要的所有轮廓及信息
        self.area_totall = 0
        # 在循环中找到连通域,将所有的连通域面积计算出来
        for i in range(1, self.num_labels):
            self.area = self.stats[i, 4]
            self.area_totall += self.area

        return self.area_totall

    def color_peach(self, img):
        # 分离色彩通道
        self.b, self.g, self.r = cv2.split(img)

        self.gray = 2 * self.g - self.b - self.r

        ret1, self.mask = cv2.threshold(self.gray, 1, 255, cv2.THRESH_BINARY)  # 阈值分割，黑白对调

        # 双边滤波
        # self.gray = cv2.bilateralFilter(self.gray, 3, 75, 75)

        self.area_all_count = self.area_count(self.mask)

        ret2, self.r_mask = cv2.threshold(self.r, 165, 255, cv2.THRESH_BINARY)  # 阈值分割，黑白对调

        self.area_bad = self.area_count(self.r_mask)

        self.red_ratio = (self.area_all_count - self.area_bad) / self.area_all_count

        self.red_ratio = round(self.red_ratio, 2)

        return self.red_ratio, self.r_mask

    def color_yellowpeach(self, img):  # 对黄桃使用R通道，因为不介意其红色区域
        # 分离色彩通道
        self.b, self.g, self.r = cv2.split(img)

        ret1, self.mask = cv2.threshold(self.r, 1, 255, cv2.THRESH_BINARY)  # 阈值分割，黑白对调

        # 双边滤波
        self.r = cv2.bilateralFilter(self.r, 3, 75, 75)

        self.area_all_count = self.area_count(self.mask)

        ret2, self.r_mask = cv2.threshold(self.r, 110, 255, cv2.THRESH_BINARY)  # 阈值分割，黑白对调

        self.area_bad = self.area_count(self.r_mask)

        self.red_ratio = (self.area_all_count - self.area_bad) / self.area_all_count

        self.red_ratio = round(self.red_ratio, 2)

        return self.red_ratio, self.r_mask

    # 填充小轮廓算法
    def floodfill(self, img):
        im_floodfill = img.copy()
        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels than the image.
        h, w = img.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        # Floodfill from point (0, 0)
        cv2.floodFill(im_floodfill, mask, (0, 0), 255)
        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        # Combine the two images to get the foreground.
        mask = img | im_floodfill_inv

        self.show_on_gui(mask, self.label_frame5)

        return mask

    def open_camera(self):

        # 得到原始图像
        self.raw_image = self.cam.data_stream[0].get_image()

        if self.raw_image is None:
            print("Getting image failed.")

        # 从原始图像得到RGB图像
        self.rgb_image = self.raw_image.convert("RGB")

        # 提高图像质量
        # self.rgb_image.image_improvement(color_correction_param, contrast_lut, gamma_lut)

        # 使用原始图像中的数据创建 numpy 数组
        self.numpy_image = self.rgb_image.get_numpy_array()

        return self.numpy_image

    def open_frame_sdk(self):

        # 得到相机设备号
        self.device_manager = gx.DeviceManager()

        self.dev_num, self.dev_info_list = self.device_manager.update_device_list()

        # 打开设备
        self.cam = self.device_manager.open_device_by_index(1)

        # 设置连续采集
        self.cam.TriggerMode.set(gx.GxSwitchEntry.OFF)

        # 设置曝光
        self.cam.ExposureTime.set(20000.0)

        # 设置增益
        self.cam.Gain.set(100.0)

        """#设置提高图像质量的参数
        if self.cam.GammaParam.is_readable():
            self.gamma_value = self.cam.GammaParam.get()
            self.gamma_lut = gx.Utility.get_gamma_lut(self.gamma_value)
        else:
            self.gamma_lut = None
        if self.cam.ContrastParam.is_readable():
            self.contrast_value = self.cam.ContrastParam.get()
            self.contrast_lut = gx.Utility.get_contrast_lut(self.contrast_value)
        else:
            self.contrast_lut = None
        if self.cam.ColorCorrectionParam.is_readable():
            self.color_correction_param = self.cam.ColorCorrectionParam.get()
        else:
            self.color_correction_param = 0"""

        # 开启流
        self.cam.stream_on()

        # 滑块控制阈值分割的准备
        self.threshold_value(0)

        while 1:
            self.start_time = time.time()
            self.carmera_img = self.open_camera()
            # self.carmera_img = self.open_camera(self.color_correction_param,self.contrast_lut,self.gamma_lut)

            # 白平衡后得到待处理图像
            self.img = self.white_balance(self.carmera_img)

            # 步骤1，转变色彩通道与调整阈值,形态学处理
            self.img_mask = self.adjust(self.img)

            # 步骤2，轮廓去噪
            self.output, self.img_roi, self.area_totall = self.roi_frame(self.img_mask, self.img)

            # 形态学提取中心位置，取适当宽度图像

            # 计算帧速
            self.end_time = time.time()
            self.seconds = self.end_time - self.start_time
            self.fps = int(1 / self.seconds)

            # 显示FPS
            self.label_text_framenum.setText("FPS:" + str(self.fps))

            self.show_on_gui(self.img, self.label_frame1)

    def white_balance(self, img):

        self.b, self.g, self.r = cv2.split(img)  # 将原图分为三通道

        self.r_avg = cv2.mean(self.r)[0]  # 求出每一个通道均值
        self.g_avg = cv2.mean(self.g)[0]
        self.b_avg = cv2.mean(self.b)[0]
        self.k = (self.r_avg + self.g_avg + self.b_avg) / 3

        self.kr = self.k / self.r_avg  # 求每个通道的增益
        self.kg = self.k / self.g_avg
        self.kb = self.k / self.b_avg
        self.r = cv2.addWeighted(src1=self.r, alpha=self.kr, src2=0, beta=0, gamma=0)
        self.g = cv2.addWeighted(src1=self.g, alpha=self.kg, src2=0, beta=0, gamma=0)
        self.b = cv2.addWeighted(src1=self.b, alpha=self.kb, src2=0, beta=0, gamma=0)

        self.frameWhite_balance = cv2.merge([self.b, self.g, self.r])  # 三通道数值合成

        return self.frameWhite_balance

    def show_on_gui(self, img, objectname):

        # 显示图像
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = QImage(img[:], img.shape[1], img.shape[0], img.shape[1] * 3, QImage.Format_RGB888)  # 将图像转化为Qimage
        img = QPixmap(img).scaled(objectname.width(), objectname.height())
        objectname.setPixmap(img)
        QApplication.processEvents()  # 刷新界面显示

    def adjust(self, img):

        # 判断是否使用调节窗口
        if self.checkBox_adjust.isChecked() == True:

            # 判断是否使用HSV通道
            if self.checkBox_RGBorHSV.isChecked() == False:

                # 转到HSV通道
                img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

                h_min, h_max, s_min, s_max, v_min, v_max, rgb_threshold, gray = adjust.hsv_adjust(0)
                # print(h_min, h_max, s_min, s_max, v_min, v_max)

                # 获得指定颜色范围内的掩码
                self.hsv_lower = np.array([h_min, s_min, v_min])
                self.hsv_upper = np.array([h_max, s_max, v_max])
                self.hsv_mask = cv2.inRange(img_hsv, self.hsv_lower, self.hsv_upper)

                # 显示
                self.show_on_gui(self.hsv_mask, self.label_frame2)

                return self.hsv_mask

            else:
                r, g, b = cv2.split(img)
                h_min, h_max, s_min, s_max, v_min, v_max, rgb_threshold, gray = adjust.hsv_adjust(0)

                # 阈值分割
                ret, self.rgb_mask = cv2.threshold(b, rgb_threshold, 255, cv2.THRESH_BINARY)

                # 显示
                self.show_on_gui(self.rgb_mask, self.label_frame2)

                return self.rgb_mask

        # 这里即为不采用调节窗口
        else:
            # 如果HSV按钮未选中，默认HSV算法
            if self.checkBox_RGBorHSV.isChecked() == False:

                # 转到HSV通道
                img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

                # 获得指定颜色范围内的掩码
                self.hsv_lower = np.array([32, 47, 0])
                self.hsv_upper = np.array([179, 255, 255])
                self.hsv_mask = cv2.inRange(img_hsv, self.hsv_lower, self.hsv_upper)

                # 显示
                self.show_on_gui(self.hsv_mask, self.label_frame2)

                return self.hsv_mask

            else:
                r, g, b = cv2.split(img)

                # 阈值
                ret, self.mask = cv2.threshold(r, 50, 255, cv2.THRESH_BINARY)

                # 显示
                self.show_on_gui(self.mask, self.label_frame2)

                return self.mask

    def roi_frame(self, img_in, img_yuantu):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_in,
                                                                                connectivity=8)  # 在mask中找到所需要的所有轮廓及信息
        max_area = 0
        j = []  # 统计符合大小的个数
        self.output = np.zeros((img_in.shape[0], img_in.shape[1], 3), np.uint8)

        for i in range(1, num_labels):  # 小轮廓区域被填充
            self.area = stats[i, 4]
            if self.area >= 10000:
                # max_area[j] = area
                j.append(i)
                # self.area_totall += self.area

        for i in range(0, len(j)):
            img_in = labels == j[i]  # ==优先级更高，先比较labels如果等于j，则img=1，否则为0
            self.output[img_in] = 255

        self.roi = cv2.bitwise_and(img_yuantu, self.output)  # 将白区域与原图与操作

        self.img_gray = cv2.cvtColor(self.roi, cv2.COLOR_RGB2GRAY)

        h_min, h_max, s_min, s_max, v_min, v_max, rgb_threshold, gray = adjust.hsv_adjust(0)

        mat, self.output = cv2.threshold(self.img_gray, 17, 255, cv2.THRESH_BINARY)

        kernel = np.ones((4, 4), np.uint8)
        # closing = cv2.morphologyEx(self.output, cv2.MORPH_CLOSE, kernel)

        self.output = cv2.erode(self.output, kernel, iterations=1)
        # self.output = cv2.dilate(self.output,kernel,iterations=1)

        # 显示
        self.show_on_gui(self.roi, self.label_frame3)
        self.show_on_gui(self.output, self.label_frame4)

        # 返回值中output是黑白二值图，roi是彩图
        return self.output, self.roi

    def roi_frame_2(self, img, roi):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img,
                                                                                connectivity=8)  # 在mask中找到所需要的所有轮廓及信息
        max_area = 0
        j = []  # 统计符合大小的个数
        output = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
        for i in range(1, num_labels):  # 在循环中找到最大的连通域
            area = stats[i, 4]
            if area >= 10000:
                # max_area[j] = area
                j.append(i)
        for i in range(0, len(j)):
            img = labels == j[i]  # ==优先级更高，先比较labels如果等于j，则img=1，否则为0
            output[img] = 255

        roi = cv2.bitwise_and(roi, output)  # 将白区域与原图与操作

        # output1,output2,output = cv2.split(output)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)

        self.show_on_gui(output, self.label_frame5)

        return roi, output


if __name__ == '__main__':
    adjust = Adjust()
    app = QApplication(sys.argv)
    # opencamera = OpenCamera()
    gui = SploceGui()
    gui.show()

    sys.exit(app.exec_())
