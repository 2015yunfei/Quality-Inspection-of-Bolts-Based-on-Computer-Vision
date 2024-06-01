# coding=utf-8
# 导入相应的python包
import cv2
import numpy as np

# 读取输入图片
img = cv2.imread('test3.jpg')
# 将彩色图片灰度化
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 使用OpenCV的imshow方法显示灰度图片
cv2.imshow('Gray Image', gray)

# 等待用户按键，之后关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()

# 使用Canny边缘检测
edges = cv2.Canny(gray, 50, 200, apertureSize=3)
# 进行Hough_line直线检测
lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
print(lines)
# 遍历每一个r和theta
for i in range(len(lines)):
    r, theta = lines[i, 0, 0], lines[i, 0, 1]
    # 存储cos(theta)的值
    a = np.cos(theta)
    # 存储sin(theta)的值
    b = np.sin(theta)
    # 存储rcos(theta)的值
    x0 = a * r
    # 存储rsin(theta)的值
    y0 = b * r
    # 存储(rcos(theta)-1000sin(theta))的值
    x1 = int(x0 + 1000 * (-b))
    # 存储(rsin(theta)+1000cos(theta))的值
    y1 = int(y0 + 1000 * (a))
    # 存储(rcos(theta)+1000sin(theta))的值
    x2 = int(x0 - 1000 * (-b))
    # 存储(rsin(theta)-1000cos(theta))的值
    y2 = int(y0 - 1000 * (a))
    # 绘制直线结果
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
# 保存结果
cv2.imwrite('test3_r.jpg', img)
cv2.imshow("result", img)
cv2.waitKey(0)

