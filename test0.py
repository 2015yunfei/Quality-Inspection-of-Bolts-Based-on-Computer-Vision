import cv2 as cv
import numpy as np


# 转二进制图像
def to_binary(img):
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(imgray, 127, 255, cv.THRESH_BINARY)
    return binary


# 提取轮廓
def get_contours(binary):
    contours, _ = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return contours


# 计算最小外接矩形的旋转角度
def get_rotation_angle(contour):
    rect = cv.minAreaRect(contour)
    angle = rect[-1]
    if angle < -45:
        angle += 90
    return angle


# 判断旋转变化
def has_rotation_change(img1, img2):
    binary1 = to_binary(img1)
    binary2 = to_binary(img2)

    contours1 = get_contours(binary1)
    contours2 = get_contours(binary2)

    dst1 = cv.drawContours(img1, contours1, -1, (0, 0, 255), 3)
    #   轮廓第几个(默认-1：所有)颜色线条厚度
    cv.imshow('dst1', dst1)

    dst2 = cv.drawContours(img2, contours2, -1, (0, 0, 255), 3)
    #   轮廓第几个(默认-1：所有)颜色线条厚度
    cv.imshow('dst2', dst2)

    cv.waitKey(0)

    if len(contours1) != 2 or len(contours2) != 2:
        raise ValueError("Each image must contain exactly one hexagon and one line segment")

    # Assuming the larger contour is the hexagon and the smaller is the line segment
    areas1 = [cv.contourArea(c) for c in contours1]
    areas2 = [cv.contourArea(c) for c in contours2]

    hexagon1 = contours1[np.argmax(areas1)]
    line1 = contours1[np.argmin(areas1)]

    hexagon2 = contours2[np.argmax(areas2)]
    line2 = contours2[np.argmin(areas2)]

    hexagon_angle1 = get_rotation_angle(hexagon1)
    line_angle1 = get_rotation_angle(line1)

    hexagon_angle2 = get_rotation_angle(hexagon2)
    line_angle2 = get_rotation_angle(line2)

    # If either the hexagon or the line segment has rotated, we consider it as a rotation change
    if hexagon_angle1 != hexagon_angle2 or line_angle1 != line_angle2:
        return True
    return False


if __name__ == '__main__':
    img1 = cv.imread('first1.jpg')
    img2 = cv.imread('first2.jpg')

    if has_rotation_change(img1, img2):
        print("Rotation change detected")
    else:
        print("No rotation change detected")
