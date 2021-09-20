import cv2


class Process():
    def __init__(self):
        for i in range(1, 12):
            path = "D:\\PingGuPeach\\picture\\blue(" + str(i) + ").bmp"
            img = cv2.imread(path)
            cv2.imshow('img',img)
            red_ratio, g_mask = self.color_yellowpeach(img)
            print(red_ratio)
            cv2.imshow("mask",g_mask)
            cv2.waitKey()




    def color_yellowpeach(self, img):#对黄桃使用R通道，因为不介意其红色区域
        # 分离色彩通道
        self.b, self.g, self.r = cv2.split(img)

        cv2.imshow("b",self.b)
        cv2.imshow("g", self.g)
        cv2.imshow("r", self.r)

        self.gray = 2*self.g - self.b - self.r
        cv2.imshow("gray",self.gray)

        ret1, self.mask = cv2.threshold(self.gray, 1, 255, cv2.THRESH_BINARY)  # 阈值分割，黑白对调

        # 双边滤波
        #self.gray = cv2.bilateralFilter(self.gray, 3, 75, 75)

        self.area_all = self.area_count(self.mask)

        ret2, self.r_mask = cv2.threshold(self.r, 155, 255, cv2.THRESH_BINARY)  # 阈值分割，黑白对调

        self.area_bad = self.area_count(self.r_mask)

        self.red_ratio = (self.area_all - self.area_bad) / self.area_all

        return self.red_ratio, self.r_mask

    def area_count(self,img):
        self.num_labels, self.labels, self.stats, self.centroids = cv2.connectedComponentsWithStats(img,
                                                                                connectivity=8)  # 在mask中找到所需要的所有轮廓及信息
        self.area_totall = 0
        # 在循环中找到连通域,将所有的连通域面积计算出来
        for i in range(1, self.num_labels):
            self.area = self.stats[i, 4]
            self.area_totall += self.area

        return self.area_totall


a = Process()

