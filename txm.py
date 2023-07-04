import cv2
import numpy as np
import math

# 读取图像
img = cv2.imread("E:/test/tp1.jpg")
# 灰度处理
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 对灰度图像进行高斯模糊
blur = cv2.GaussianBlur(gray, (5, 5), 0)
# 使用Sobel算子进行边缘检测
sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0)
sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1)
sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
sobel = np.uint8(sobel)
# 均值滤波消除高频噪声
blur = cv2.blur(sobel, (5, 5))

# 自适应阈值处理
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 5)
# 闭运算
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
clo = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
# 膨胀
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
dilation = cv2.dilate(clo, kernel, iterations=1)
# 腐蚀
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
erosion = cv2.erode(dilation, kernel, iterations=1)
# 寻找轮廓
contours, hierarchy = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# 绘制最小外接矩形并筛选
for contour in contours:
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    area = rect[1][0] * rect[1][1]

    # 计算矩形框中黑色像素的比例
    x, y, w, h = cv2.boundingRect(contour)
    black_pixels = 0
    for i in range(y, y + h):
        for j in range(x, x + w):
            if erosion[i][j] == 0:
                black_pixels += 1
    prop = black_pixels / (w * h) * 100

    # 判断是否保留矩形框
    if area > 30000 and prop < 80 and prop > 20 \
            and 2 <= max(rect[1][0], rect[1][1]) / min(rect[1][0], rect[1][1]) < 5:
        # 获取矩形边长
        min_side = min(rect[1][0], rect[1][1])
        # 创建一个LSD对象
        lsd = cv2.createLineSegmentDetector(0)
        lines = lsd.detect(blur)

        # 存储直线斜率的列表
        slopes = []
        # 创建一个空列表来存储直线组
        line_groups = []
        # 绘制检测结果
        for line in lines[0]:
            x0 = int(round(line[0][0]))
            y0 = int(round(line[0][1]))
            x1 = int(round(line[0][2]))
            y1 = int(round(line[0][3]))

            # 长度
            len_2 = (x1 - x0) ** 2 + (y1 - y0) ** 2
            len_1 = math.sqrt(len_2)

            # 筛选
            if len_1 > min_side * 0.3 and len_1 < min_side * 1:
                cv2.line(img, (x0, y0), (x1, y1), (0, 255, 0), 3, cv2.LINE_AA)
                box = np.int0(box)
                cv2.drawContours(img, [box], 0, (0, 0, 255), 3)

                # 计算直线斜率
                if x1 != x0:
                    slope = (y1 - y0) / (x1 - x0)
                    slopes.append(slope)

        # 判断直线组数量和斜率之差
        if len(slopes) < 4:
            slope_diff = max(slopes) - min(slopes)
            if slope_diff < 0.1:
                cv2.line(img, (x0, y0), (x1, y1), (0, 255, 0), 3, cv2.LINE_AA)
                box = np.int0(box)
                cv2.drawContours(img, [box], 0, (0, 0, 255), 3)

# 显示结果
cv2.namedWindow('image', 0)
cv2.resizeWindow('image', 600, 800)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
