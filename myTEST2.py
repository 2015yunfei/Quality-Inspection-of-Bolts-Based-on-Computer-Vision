import cv2 as cv
import numpy as np


# 高斯模糊处理
def blur_image(img):
    blurred = cv.GaussianBlur(img, (5, 5), 0)
    # cv.imshow('blurred', blurred)
    return blurred


# 根据颜色分割提取每个元素的掩码
def color_segmentation(img):
    # 转换为HSV颜色空间
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # 定义颜色范围（需要根据具体图像调整）
    lower_color1 = np.array([90, 105, 200])
    upper_color1 = np.array([120, 135, 240])
    lower_color2 = np.array([5, 185, 215])
    upper_color2 = np.array([35, 215, 250])
    lower_color3 = np.array([185, 195, 110])
    upper_color3 = np.array([215, 230, 150])
    lower_color0 = np.array([0, 0, 235])
    upper_color0 = np.array([20, 20, 255])

    # 创建掩码
    mask0 = cv.inRange(hsv, lower_color0, upper_color0)
    mask1 = cv.inRange(hsv, lower_color1, upper_color1)
    mask2 = cv.inRange(hsv, lower_color2, upper_color2)
    mask3 = cv.inRange(hsv, lower_color3, upper_color3)

    # 合并掩码
    mask = cv.bitwise_or(mask1, mask2)
    mask = cv.bitwise_or(mask, mask3)
    mask = cv.bitwise_or(mask, mask0)

    # cv.imshow('mask', mask)
    return mask


# 提取轮廓并筛选主要图形
def get_key_contours(mask):
    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    key_contours = []
    for contour in contours:
        if len(contour) > 5:  # 确保轮廓点数足够
            epsilon = 0.02 * cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, epsilon, True)
            if len(approx) in [4, 6]:  # 筛选出矩形和六边形
                key_contours.append(contour)

    # 进一步筛选符合大小要求的图形
    min_px = 60
    filtered_contours = []
    for contour in key_contours:
        rect = cv.minAreaRect(contour)
        if rect[1][0] > min_px and rect[1][1] > min_px:  # 排除过小的轮廓
            filtered_contours.append(contour)
    if len(key_contours) != len(filtered_contours):
        print("删除了一些较小的矩形")
    return filtered_contours


# 提取轮廓并计算旋转角度
def get_contours_and_angles(contours):
    angles = []
    for contour in contours:
        if len(contour) > 5:  # 确保轮廓点数足够计算旋转角度
            rect = cv.minAreaRect(contour)
            angle = rect[2]
            if rect[1][0] < rect[1][1]:
                angle -= 90
            else:
                angle = angle
            angles.append(angle)  # 获取旋转角度
    return angles


# 比较两组角度，判断是否发生旋转
def has_rotation_occurred(angles1, angles2, threshold=5):
    if len(angles1) != len(angles2):
        print(str(angles1) + " 轮廓数量不同，认为发生了变化 " + str(angles2))
        return True  # 轮廓数量不同，认为发生了变化
    for angle1, angle2 in zip(angles1, angles2):
        print("角度对比： |  " + str(angle1) + "  |  " + str(angle2) + "  |  ")
        if abs(angle1 - angle2) > threshold:
            print("角度变化超过阈值，认为发生了旋转")
            return True  # 角度变化超过阈值，认为发生了旋转

    print("角度大致一致！没发生旋转！！！")
    return False


# 调整图像大小
def resize_images(img1, img2, dst1, dst2):
    # 计算最大宽度和高度
    max_width = max(img1.shape[1], img2.shape[1], dst1.shape[1], dst2.shape[1])
    max_height = max(img1.shape[0], img2.shape[0], dst1.shape[0], dst2.shape[0])

    # 调整图像大小
    img1_resized = cv.resize(img1, (max_width, max_height))
    img2_resized = cv.resize(img2, (max_width, max_height))
    dst1_resized = cv.resize(dst1, (max_width, max_height))
    dst2_resized = cv.resize(dst2, (max_width, max_height))

    return img1_resized, img2_resized, dst1_resized, dst2_resized


# 缩放图像
def scale_image(image, scale_factor):
    width = int(image.shape[1] * scale_factor)
    height = int(image.shape[0] * scale_factor)
    return cv.resize(image, (width, height))


def run(img1, img2):
    mask1 = color_segmentation(blur_image(img1))
    mask2 = color_segmentation(blur_image(img2))
    # cv.imshow('mask', mask)
    key_contours1 = get_key_contours(mask1)
    key_contours2 = get_key_contours(mask2)

    angles1 = get_contours_and_angles(key_contours1)
    angles2 = get_contours_and_angles(key_contours2)

    if has_rotation_occurred(angles1, angles2):
        print("The structures have rotated.")
    else:
        print("The structures have not rotated.")

    # 画出关键轮廓
    dst1 = cv.drawContours(img1.copy(), key_contours1, -1, (0, 0, 255), 2)
    dst2 = cv.drawContours(img2.copy(), key_contours2, -1, (0, 0, 255), 2)

    # 调整图像大小
    img1_resized, img2_resized, dst1_resized, dst2_resized = resize_images(img1, img2, dst1, dst2)

    # 水平拼接两个图像
    concatenated0 = cv.hconcat([img1_resized, img2_resized])
    # 水平拼接两个图像
    concatenated1 = cv.hconcat([dst1_resized, dst2_resized])
    # 垂直拼接两个图像
    concatenated = cv.vconcat([concatenated0, concatenated1])

    # 缩放拼接后的图像
    scale_factor = 0.5  # 缩放因子，根据屏幕大小调整
    concatenated_scaled = scale_image(concatenated, scale_factor)

    # 显示拼接后的图像
    cv.imshow('Images', concatenated_scaled)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    img0 = cv.imread('new0.jpg')
    img1 = cv.imread('new1.jpg')
    img2 = cv.imread('new2.jpg')
    img3 = cv.imread('new3.jpg')
    img4 = cv.imread('new4.jpg')

    print("")
    run(img0, img1)
    print("")
    run(img0, img2)
    print("")
    run(img0, img3)
    print("")
    run(img0, img4)
