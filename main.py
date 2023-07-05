import csv
import numpy as np
import cv2
import imgDTO
import Learning

class ImgConverter:

    ### 생성자를 이용한 클레스 변수 초기화(첫 번째 인자인 self는 생성되는 인스턴스를 의미, 가로넓이, 세로 높이)
    def __init__(self, width, height):
        self.imgArr = [] #IMGDTO 함수 객체를 저장하기 위한 배열
        self.width = width
        self.height = height
        # 색상 변수를 선언한다. 3개의 원소를 갖는 튜플
        self.R, self.G, self.B = (0, 0, 255), (0, 255, 0), (255, 0, 0)
        self.imgArr0 = []
        self.imgArr1 = []
        self.imgArr2 = []
        self.imgArr3 = []

    def campus(self):
        # 800 x 800의 크기의 3채널 np.uint8형(생략가능)으로 행렬을 생성
        img = np.zeros((self.width, self.height, 3), np.uint8)
        # 슬라이스를 이용하여 모든 행렬의 화소를 흰색으로 지정
        img[:] = (255, 255, 255)
        return img

    def secDivider(self):
        secA = []
        secB = []
        # 1번영역
        secA.append((0, 0))
        secB.append((int(self.width / 2), int(self.height / 2)))
        # 2번 영역
        secA.append((int(self.width/2), 0))
        secB.append((self.width, int(self.height/2)))
        # 3번 영역
        secA.append((0,int(self.height / 2)))
        secB.append((int(self.width/2), self.height))
        # 4번 영역
        secA.append((int(self.width / 2), int(self.height / 2)))
        secB.append((self.width, self.height))

        return secA, secB


    def drawing(self, img, secA, secB):
        MAX = 20

        for j in range(4):
            fd1 = open("./data_%d.CSV" % j, "r", encoding="ms932")
            print(j)
            # delimiter: 필드간을 분할하기 위해 사용하는 문자 지정
            # lineterminator: writer을 사용할때 각 행의 끝을 표시하기 위해 사용되는 문자이다. reader에서는 '\r'1) 혹은 '\n'울 행의 끝으로 지정하도록 하드 코딩되어 있기 때문에 관계없음
            # skipinitialspace true에경우, delimiter의 직후에 있는 공백은 무시된다.

            # 1) /r: 개행 문자 (캐리지 리턴(CarriageReturn), 커서를 행의 맨 앞으로 이동시킴(잘 사용되지 않음)
            fr1 = csv.reader(fd1, delimiter=",", lineterminator="\r\n", skipinitialspace=True)

            for i in range(MAX):
                # next는 다음 행으로 넘어가기 위해 사용ex)(-90,-80,-70)
                header = next(fr1)

                # 각 비콘의 rssi 값이 -100보다 작을경우 -100으로 처리
                if (abs(int(header[0])) > 100): header[0] = -100
                if (abs(int(header[1])) > 100): header[1] = -100
                if (abs(int(header[2])) > 100): header[2] = -100

                # 배경 위에 각 rssi값을 이용해 사각형으로 칠함, ( rssi값이 작을수록 어두움B, G, R), 두께는 크기를 전부 채우도록 설정 )
                cv2.rectangle(img, secA[0], secB[0],
                              (255 + 2.55 * int(header[0]), 255 + 2.55 * int(header[0]), 255 + 2.55 * int(header[0])),
                              -1)  # 우측상단 좌측하단 좌표, 색, 테두리 두께
                cv2.rectangle(img, secA[1], secB[1],
                              (255 + 2.55 * int(header[1]), 255 + 2.55 * int(header[1]), 255 + 2.55 * int(header[1])),
                              -1)  # 우측상단 좌측하단 좌표, 색, 테두리 두께
                cv2.rectangle(img, secA[2], secB[2],
                              (255 + 2.55 * int(header[2]), 255 + 2.55 * int(header[2]), 255 + 2.55 * int(header[2])),
                              -1)  # 우측상단 좌측하단 좌표, 색, 테두리 두께
                # cv2.rectangle(img, secA[3], secB[3], self.G, -1)  # 우측상단 좌측하단 좌표, 색, 테두리 두께

                # bgr로 만들어진 img를 grasycale로 변환
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # imgDto.py 있는 ImgDto 클래스 객체에 이미지 및 각종 변수를 저장 (타입 캐스팅으로 코드가 너무 길어짐)(i,rssi1,rssi2,rssi3,이미지)
                dto = imgDTO.ImgDTO(j ,i, int(header[0]), int(header[1]), int(header[2]), img_gray)
                # dto(이미지 및 출력에 필요한 변수)를 imgArr에 저장
                self.imgArr.append(dto)

                # cv2.imshow("channel_%d(%d, %d, %d) )" %(i, int(header[0]), int(header[1]), int(header[2])), img2)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                if j == 0:
                    self.imgArr0.append(dto)
                elif j == 1:
                    self.imgArr1.append(dto)
                elif j == 2:
                    self.imgArr2.append(dto)
                elif j == 3:
                    self.imgArr3.append(dto)

            # file descriptor(파일 기술자) 를 닫아줌

        fd1.close()



    def saveToFile(self):
        # cv2.imwrite('./savedImages/channel_%d(%d, %d, %d).png' % (i, int(header[0]), int(header[1]), int(header[2])), img)
        #self.imgArr 만큼(100번) 반복
        for cv2img in self.imgArr:
            j = cv2img.jindex # j값 추출하여 저장
            i = cv2img.iindex # i값 추출하여 저장
            b1 = cv2img.b1 # rssi1 추출하여 저장
            b2 = cv2img.b2 # rssi1 추출하여 저장
            b3 = cv2img.b3 # rssi1 추출하여 저장
            img = cv2img.img # 이미지 객체 추출하여 저장
            cv2.imwrite('./savedImages/channel%d_%d(%d, %d, %d).png' % (j, i, b1, b2, b3), img)

    def convertCV2ImgTo2DArray(self):
        imgArr = []
        areaNo = []

        #dto 객체 배열(imgArr0)에서 dto 객체 추출(cv2img)
        #cv2img 에서 img 객체 추출후 np.array로 객체 배열 추출후 tolist()로 list배열로 변경
        for cv2img in self.imgArr0:
            imgArr.append(cv2img.img)
            areaNo.append(0)
        for cv2img in self.imgArr1:
            imgArr.append(cv2img.img)
            areaNo.append(1)
        for cv2img in self.imgArr2:
            imgArr.append(cv2img.img)
            areaNo.append(2)
        for cv2img in self.imgArr3:
            imgArr.append(cv2img.img)
            areaNo.append(3)

        # 일반 배열을 numpy배열(ndarray)로 변경하여준다.(.reshape)를 사용하기 위하여
        # img는 이미 np.zeros로 생성하였으므로 numpy 객체 이다.
        return np.array(imgArr), np.array(areaNo)


width = 400
height = 400
ic = ImgConverter(width, height)
img = ic.campus()
secA, secB = ic.secDivider()
ic.drawing(img, secA, secB)
ic.saveToFile()
train_images, train_labels = ic.convertCV2ImgTo2DArray()
#################################################################################

obj1 = Learning.ML(train_images, train_labels, width, height)
obj1.create()
obj1.train()
