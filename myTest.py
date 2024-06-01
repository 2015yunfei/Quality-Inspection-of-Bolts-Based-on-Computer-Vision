import cv2
import numpy as np


# 转二进制图像
def to_binary(img):
    # 1、灰度图
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('imgray', imgray)

    # 2、二进制图像
    ret, binary = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow('binary', binary)
    return imgray


# 提取轮廓并计算旋转角度
def get_contours_and_angles(binary, img=None):
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    angles = []

    # 进一步筛选大小符合要求的图形
    filtered_contours = []
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        if rect[1][0] > 10 and rect[1][1] > 10:  # 排除过小的轮廓
            filtered_contours.append(contour)

    # 做一个简要的判断，是否筛出了噪点
    if len(contours) == len(filtered_contours):
        print("未过滤出任何噪点")
    else:
        print("过滤出部分噪点")

    for contour in filtered_contours:
        if len(contour) > 5:  # 确保轮廓点数足够计算旋转角度
            rect = cv2.minAreaRect(contour)
            print("rect[2]:" + str(rect[2]))
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            if img is not None:
                cv2.drawContours(img, [box], 0, (0, 255, 0), 2)  # 绘制外接矩形
            angles.append(rect[2])  # 获取旋转角度
        else:
            angles.append(0)  # 获取旋转角度
    if img is not None:
        cv2.imshow('Contours and Min Area Rects', img)
        cv2.waitKey(0)
    return angles


# 比较两组角度，判断是否发生旋转
def has_rotation_occurred(angles1, angles2, threshold=5):
    if len(angles1) != len(angles2):
        print(str(angles1) + " 轮廓数量不同，认为发生了变化 " + str(angles2))
        return True  # 轮廓数量不同，认为发生了变化
    for angle1, angle2 in zip(angles1, angles2):
        if abs(angle1 - angle2) > threshold:
            print("角度变化超过阈值，认为发生了旋转")
            return True  # 角度变化超过阈值，认为发生了旋转
    return False


if __name__ == '__main__':
    img1 = cv2.imread('test0.jpg')
    img2 = cv2.imread('test1.jpg')
    # cv.imshow('img1', img1)
    # cv.imshow('img2', img2)

    binary1 = to_binary(img1)
    binary2 = to_binary(img2)

    angles1 = get_contours_and_angles(binary1, img1)
    angles2 = get_contours_and_angles(binary2, img2)

    # exit()

    if has_rotation_occurred(angles1, angles2):
        print("The structures have rotated.")
    else:
        print("The structures have not rotated.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
