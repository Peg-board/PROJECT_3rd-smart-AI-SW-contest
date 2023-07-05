class ImgDTO:
    # cv2.imwrite('./savedImages/channel_%d(%d, %d, %d).png' % (i, int(header[0]), int(header[1]), int(header[2])), img)
    def __init__(self,jindex, iindex, b1, b2, b3, img):
        self.jindex = jindex # 이미지 인덱스:j
        self.iindex = iindex # 이미지 인덱스:i
        self.b1 = b1 # 비콘1번의 RSSI
        self.b2 = b2 # 비콘2번의 RSSI
        self.b3 = b3 # 비콘3번의 RSSI
        self.img = img # CV2Img 객체