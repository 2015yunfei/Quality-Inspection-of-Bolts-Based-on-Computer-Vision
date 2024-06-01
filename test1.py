import cv2 as cv
import numpy as np


# 转二进制图像
def to_binary(img):
    # 1、灰度图
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('imgray', imgray)

    # 2、二进制图像
    ret, binary = cv.threshold(imgray, 127, 255, cv.THRESH_BINARY)
    cv.imshow('binary', binary)
    return binary


# 提取轮廓并计算旋转角度
def get_contours_and_angles(binary):
    contours, _ = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    angles = []
    for contour in contours:
        if len(contour) > 5:  # 确保轮廓点数足够计算旋转角度
            rect = cv.minAreaRect(contour)
            angles.append(rect[2])  # 获取旋转角度
        else:
            angles.append(0)  # 获取旋转角度
    return angles


# 比较两组角度，判断是否发生旋转
def has_rotation_occurred(angles1, angles2, threshold=5):
    if len(angles1) != len(angles2):
        return True  # 轮廓数量不同，认为发生了变化
    for angle1, angle2 in zip(angles1, angles2):
        if abs(angle1 - angle2) > threshold:
            return True  # 角度变化超过阈值，认为发生了旋转
    return False


if __name__ == '__main__':
    img1 = cv.imread('first1.jpg')
    img2 = cv.imread('first2.jpg')
    # cv.imshow('img1', img1)
    # cv.imshow('img2', img2)

    binary1 = to_binary(img1)
    binary2 = to_binary(img2)

    angles1 = get_contours_and_angles(binary1)
    angles2 = get_contours_and_angles(binary2)

    if has_rotation_occurred(angles1, angles2):
        print("The structures have rotated.")
    else:
        print("The structures have not rotated.")

    cv.waitKey(0)
    cv.destroyAllWindows()
