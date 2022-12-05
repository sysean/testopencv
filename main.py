# -*-coding=UTF-8-*-
"""
在无参考图下，检测图片质量的方法
"""
import os
import cv2
import time
import re

import numpy as np
import pandas as pd
from datetime import datetime
from skimage import filters
from brisquequality import test_measure_BRISQUE


def write2csv(func):
    """
    装饰器: 测试结果输出到csv文件
    """

    def wrapper(*args, **kw):
        pre, names, score_list = func(*args, **kw)
        dataframe = pd.DataFrame({'picture_name': names, 'score': score_list})
        csv_name = pre + "_" + datetime.now().strftime("%H:%M:%S") + ".csv"
        dataframe.to_csv("./output/" + csv_name, index=False)
        return

    return wrapper


class BlurDetection:
    def __init__(self, strDir):
        print("图片检测对象已经创建...")
        self.strDir = strDir

    def _get_all_img(self):
        """
        根据目录读取所有的图片
        :return:  图片列表
        """
        names = []
        for root, dirs, files in os.walk(self.strDir):
            for file in files:
                if os.path.splitext(file)[1] in ('.jpg', '.jpeg', '.png', '.bmp'):
                    names.append(str(file))

        names = sorted(names, key=lambda info: (int(re.findall(r'(\d+)', info)[0])))
        return names

    def _image_to_matrix(self, image):
        """
        根据名称读取图片对象转化矩阵
        :param image:
        :return: 返回矩阵
        """
        img_mat = np.matrix(image)
        return img_mat

    def _blur_detection(self, img_name):
        """
        Brenner
        """
        # step 1 图像的预处理
        img2gray, re_img = self.pre_img_ops(img_name)
        img_mat = self._image_to_matrix(img2gray) / 255.0
        x, y = img_mat.shape  # shape 为矩阵的长和宽，在这里就是图片的像素长和宽
        score = 0
        for i in range(x - 2):
            for j in range(y - 2):
                score += (img_mat[i + 2, j] - img_mat[i, j]) ** 2
        score = score / 10
        # self._save_img(re_img, score, "/_blurDetection_/", img_name)
        return score

    def _SMD_detection(self, imgName):
        """
        灰度方差
        """
        # step 1 图像的预处理
        img2gray, reImg = self.pre_img_ops(imgName)
        f = self._image_to_matrix(img2gray) / 255.0
        x, y = f.shape
        score = 0
        for i in range(x - 1):
            for j in range(y - 1):
                score += np.abs(f[i + 1, j] - f[i, j]) + np.abs(f[i, j] - f[i + 1, j])
        score = score / 100
        self._save_img(reImg, score, "/_SMDDetection_/", imgName)
        return score

    def _SMD2_detection(self, imgName):
        """
        灰度方差乘积
        """
        img2gray, reImg = self.pre_img_ops(imgName)
        f = self._image_to_matrix(img2gray) / 255.0
        x, y = f.shape
        score = 0
        for i in range(x - 1):
            for j in range(y - 1):
                score += np.abs(f[i + 1, j] - f[i, j]) * np.abs(f[i, j] - f[i, j + 1])
        # strp3: 绘制图片并保存  不应该写在这里  抽象出来   这是共有的部分
        score = score
        # self._save_img(reImg, score, "/_SMD2Detection_/", imgName)
        return score

    def _variance(self, imgName):
        """
        方差
        """
        img2gray, reImg = self.pre_img_ops(imgName)
        f = self._image_to_matrix(img2gray)
        score = np.var(f)
        self._save_img(reImg, score, "/_Variance_/", imgName)
        return score

    def _tenengrad(self, imgName):
        # 图像的预处理
        img2gray, re_img = self.pre_img_ops(imgName)
        f = self._image_to_matrix(img2gray)

        tmp = filters.sobel(f)
        score = np.sum(tmp ** 2)
        score = np.sqrt(score)
        # 绘制图片并保存
        # self._save_img(re_img, score, "/_Tenengrad_/", imgName)
        return score

    def _lapulase_detection(self, imgName):
        # step1: 预处理
        img2gray, reImg = self.pre_img_ops(imgName)
        # step2: laplacian算子 获取评分
        resLap = cv2.Laplacian(img2gray, cv2.CV_64F)
        score = resLap.var()

        # self._save_img(reImg, score, "/_lapulaseDetection_/", imgName)
        return score

    def test_brenner(self):
        """
        1 Brenner 梯度函数
        """
        print("1 Brenner 梯度函数")
        img_list = self._get_all_img()
        score_list = []
        for i in range(len(img_list)):
            start = time.time()
            score = self._blur_detection(img_list[i])
            cost = (time.time() - start) * 1000
            print(str(img_list[i]) + " is " + str(score) + " cost: ", cost)
            score_list.append(score)
        return "brenner", img_list, score_list

    def test_tenengrad(self):
        """
        2 Tenengrad 梯度函数
        """
        print("2 Tenengrad 梯度函数")
        img_list = self._get_all_img()
        score_list = []
        for i in range(len(img_list)):
            start = time.time()
            score = self._tenengrad(img_list[i])
            cost = (time.time() - start) * 1000
            print(str(img_list[i]) + " is " + str(score) + " cost: ", cost)
            score_list.append(score)
        return "tenengrad", img_list, score_list

    @write2csv
    def test_laplacian(self):
        """
        3 Laplacian 梯度函数
        """
        print("3 Laplacian 梯度函数")
        score_list = []
        names = self._get_all_img()
        for i in range(len(names)):
            start = time.time()
            score = self._lapulase_detection(names[i])
            cost = (time.time() - start) * 1000
            print(str(names[i]) + " is " + str(score) + " cost: ", cost)

            score_list.append(score)
        return "laplacian", names, score_list

    def test_SMD(self):
        """
        4 SMD（灰度方差）
        """
        print("4 SMD（灰度方差）")
        img_list = self._get_all_img()
        for i in range(len(img_list)):
            start = time.time()
            score = self._SMD_detection(img_list[i])
            cost = (time.time() - start) * 1000
            print(str(img_list[i]) + " is " + str(score) + " cost: ", cost)
        return

    @write2csv
    def test_SMD2(self):
        """
        5 SMD2 （灰度方差乘积）
        """
        print("5 SMD2 （灰度方差乘积）")
        img_list = self._get_all_img()
        score_list = []
        for i in range(len(img_list)):
            start = time.time()
            score = self._SMD2_detection(img_list[i])
            cost = (time.time() - start) * 1000
            print(str(img_list[i]) + " is " + str(score) + " cost: ", cost)
            score_list.append(score)

        return "smd2", img_list, score_list

    def test_variance(self):
        """
        6 方差函数(TestVariance)
        """
        print("6 方差函数(TestVariance)")
        img_list = self._get_all_img()
        for i in range(len(img_list)):
            start = time.time()
            score = self._variance(img_list[i])
            cost = (time.time() - start) * 1000
            print(str(img_list[i]) + " is " + str(score) + " cost: ", cost)

    @write2csv
    def test_MSCN(self):
        """
        7 MSCN Mean Substracted Contrast Normalization
        平均减去对比归一化
        """
        print("7 MSCN 平均减去对比归一化")
        score_list = []
        img_list = self._get_all_img()
        for i in range(len(img_list)):
            start = time.time()
            strPath = self.strDir + '/' + img_list[i]
            score = test_measure_BRISQUE(strPath)
            cost = (time.time() - start) * 1000
            print(str(img_list[i]) + " is " + str(score) + " cost: ", cost)

            score_list.append(score)
        return "mscn", img_list, score_list

    @write2csv
    def test_BRISQUE(self):
        print("8 BRISQUE")
        score_list = []
        img_list = self._get_all_img()
        for i in range(len(img_list)):
            start = time.time()
            strPath = self.strDir + '/' + img_list[i]
            img = cv2.imread(strPath, 1)
            score = cv2.quality.QualityBRISQUE_compute(img, "brisque_model_live.yml", "brisque_range_live.yml")
            cost = (time.time() - start) * 1000
            score = score[0]
            print(str(img_list[i]) + " is " + str(score) + " cost: ", cost)

            score_list.append(score)
        return "BRISQUE", img_list, score_list

    def pre_img_ops(self, imgName):
        """
        图像的预处理操作
        :param imgName: 图像的而明朝
        :return: 灰度化和resize之后的图片对象
        """
        strPath = self.strDir + '/' + imgName

        img = cv2.imread(strPath)  # 读取图片
        cv2.moveWindow("", 1000, 100)
        # cv2.imshow("原始图", img)
        # 预处理操作
        # img = cv2.resize(img, (800, 900), interpolation=cv2.INTER_CUBIC)
        img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图片压缩为单通道的灰度图
        return img2gray, img

    def _draw_img_fonts(self, img, strContent):
        """
        绘制图像
        :param img: cv下的图片对象
        :param strContent: 书写的图片内容
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontSize = 5
        # 照片 添加的文字    /左上角坐标   字体   字体大小   颜色        字体粗细
        cv2.putText(img, strContent, (0, 200), font, fontSize, (0, 255, 0), 6)

        return img

    def _save_img(self, re_img, score, path_name, img_name):
        """
        保存图片
        :param re_img 保存的图片数据 ndarray 类型
        :param score 分数
        :param 在 ./img 下的 dir 名字
        :param img_name 图片名称
        """
        newImg = self._draw_img_fonts(re_img, str(score))
        newDir = self.strDir + path_name
        if not os.path.exists(newDir):
            os.makedirs(newDir)
        newPath = newDir + img_name
        # 显示
        cv2.imwrite(newPath, newImg)  # 保存图片


if __name__ == "__main__":
    # 输出结果的目录
    out_dir = "output"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    str1 = "~/code/mypy/IQA-Dataset-master/data/LIVE/jpeg"
    str2 = "./img"
    str3 = "./img/test1"
    str4 = "~/code/mypy/IQA-Dataset-master/data/VCLFER/vcl_fer"

    bd = BlurDetection(strDir=str2)
    # bd.test_brenner()
    # bd.test_tenengrad()
    # bd.test_laplacian()
    # bd.test_SMD()
    # bd.test_SMD2()
    # bd.test_variance()
    # bd.test_MSCN()
    bd.test_BRISQUE()
