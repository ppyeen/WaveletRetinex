# -*- coding: utf-8 -*-
"""
Time:     2024/3/11 15:22
Author:   ppyeen
Version:  1.0.0
File:     retinex.py
Describe: 
"""
import cv2
import numpy as np


def ssr(image, sigma):
    """
    single_scale_retinex 单尺度retinex
    :param image:
    :param sigma:
    :return:
    """
    # 转换范围，所有图像元素增加1.0,保证能正常取对数
    double_image = image.astype(np.float64) + 1.0
    # 对图像进行高斯模糊
    gaussian_image = cv2.GaussianBlur(double_image, (sigma, sigma), 0)
    cv2.imshow("blurred", gaussian_image.astype(np.uint8))
    cv2.waitKey()

    # 计算图像的对数变换
    log_image = np.log(double_image.astype(np.float32))
    log_gaussian = np.log(gaussian_image.astype(np.float32))

    # 计算单路Retinex增强图像
    retinex = log_image - log_gaussian

    # 将增强图像进行归一化处理
    enhanced_image = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX)
    enhanced_image = enhanced_image.astype(np.uint8)

    return enhanced_image
