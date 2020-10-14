from PyQt5.QtWidgets import (QWidget, QPushButton, QLineEdit,
                             QInputDialog, QApplication, QFileDialog, QLabel)
import sys
from collections import Counter, defaultdict
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon, QPixmap, QImage
import cv2
from PyQt5.QtCore import QThread, pyqtSignal, Qt
import datetime
import numpy as np
import urllib.request

import face_recognition
import sqlite3

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

conn = sqlite3.connect('db/info.db')
c = conn.cursor()
face_cascade = cv2.CascadeClassifier('C:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:\\opencv\\build\\etc\\haarcascades\\haarcascade_eye.xml')


def mail2(email,image):
    fromaddr = "msrsn123@gmail.com"
    toaddr = email

    conn = sqlite3.connect('db/info.db')
    c = conn.cursor()
    sql = 'SELECT * FROM person where email like "'+email+'"'

    c.execute(sql)
    rows = c.fetchall()
    name=rows[0][2]

    msg = MIMEMultipart()

    msg['From'] = fromaddr
    msg['To'] = toaddr
    msg['Subject'] = "EMERGENCY  ALERT"

    body = "Hello "+name+", \n \nYour Bag has been detected abandoned in suryabinayak bus stop. \n " \
           "Please immediately respond. \n\n Thank you,\n MSRSN Team"

    msg.attach(MIMEText(body, 'plain'))

    filename = "a.jpg"
    attachment = open(image, "rb")

    part = MIMEBase('application', 'octet-stream')
    part.set_payload((attachment).read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', "attachment; filename= %s" % filename)

    msg.attach(part)

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(fromaddr, "school12345")
    text = msg.as_string()
    server.sendmail(fromaddr, toaddr, text)
    server.quit()

def mail1(image):
    fromaddr = "msrsn123@gmail.com"
    toaddr = "snabinbikram@gmail.com"

    msg = MIMEMultipart()

    msg['From'] = fromaddr
    msg['To'] = toaddr
    msg['Subject'] = "EMERGENCY  ALERT"

    body = "Hello Police Department, \n \nAn unknown bag has been abandoned in suryabinayak bus stop. The person who left object is not the citizen of this city. \n " \
           "Please immediately respond. The photo of host is attached below. \n\n Thank you,\n MSRSN Team"

    msg.attach(MIMEText(body, 'plain'))

    filename = "suspect.jpg"
    attachment = open(image, "rb")

    part = MIMEBase('application', 'octet-stream')
    part.set_payload((attachment).read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', "attachment; filename= %s" % filename)

    msg.attach(part)

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(fromaddr, "school12345")
    text = msg.as_string()
    server.sendmail(fromaddr, toaddr, text)
    server.quit()

def mail3(image):
    fromaddr = "msrsn123@gmail.com"
    toaddr = "snabinbikram@gmail.com"

    msg = MIMEMultipart()

    msg['From'] = fromaddr
    msg['To'] = toaddr
    msg['Subject'] = "EMERGENCY  ALERT"

    body = "Dear sir, \n \nAn unknown bag has been abandoned in suryabinayak bus stop. \n " \
           "Please immediately respond. No person Detected who left the object. \n\n Thank you,\n MSRSN Team"

    msg.attach(MIMEText(body, 'plain'))

    filename = "suspect.jpg"
    attachment = open(image, "rb")

    part = MIMEBase('application', 'octet-stream')
    part.set_payload((attachment).read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', "attachment; filename= %s" % filename)

    msg.attach(part)

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(fromaddr, "school12345")
    text = msg.as_string()
    server.sendmail(fromaddr, toaddr, text)
    server.quit()

class BackGroundSubtractor:

    def __init__(self, alpha, firstFrame):
        self.alpha = alpha
        self.backGroundModel = firstFrame

    def getForeground(self, frame):
        self.backGroundModel = frame * self.alpha + self.backGroundModel * (1 - self.alpha)
        return cv2.absdiff(self.backGroundModel.astype(np.uint8), frame)

def denoise(frame):
    frame = cv2.medianBlur(frame, 5)
    frame = cv2.GaussianBlur(frame, (5, 5), 1)

    return frame

def gammacorrection(frame, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(frame, table)

def build_filters():
    filters = []
    ksize = 21
    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 2.0, 0, ktype=cv2.CV_32F)
        kern /= 1.5 * kern.sum()
        filters.append(kern)
    return filters

def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum

def facedetect(image):
    face_locations = face_recognition.face_locations(image)
    if face_locations == []:
        return False
    else:
        return True

def compare(image):

    # image from database load garne
    conn = sqlite3.connect('db/info.db')
    c = conn.cursor()
    sql = 'SELECT * FROM person'
    c.execute(sql)
    rows = c.fetchall()
    i=0
    result=[False]
    email="abcd" #initialized
    for row in rows:
        rimage = rows[i][1]
        rimage = "image/"+rimage
        email = rows[i][4]
        known_image = face_recognition.load_image_file(rimage)
        # captured image
        unknown_image = face_recognition.load_image_file(image)

        biden_encoding = face_recognition.face_encodings(known_image)[0]

        unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

        # image compare garne
        results = face_recognition.compare_faces([biden_encoding], unknown_encoding)
        if results == [True]:
            result=email
            break
        else:
            result="abcd"
    conn.close()
    return result

class live(QWidget):
    global path
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setStyleSheet("background-color:rgbrgb(2, 15, 45);")                        #background ko color

        self.th = Thread(self)

        # ********message label ko start***********
        self.message = QLabel(self)  # link aaune thau
        self.message.setGeometry(0, 30, 335, 330)   # browse labe ko geometry size
        self.message.setStyleSheet("background-color:lavenderblush;")
        self.message.move(10, 400)
        # ********message label ko start***********


        # ******* message_text label start********
        self.message_text = QLabel(self)  # link aaune thau
        self.message_text.setGeometry(0, 30, 335, 40)  # browse labe ko geometry size
        self.message_text.setStyleSheet("background:rgb(209, 216, 4);")
        self.message_text.move(10, 390)
        self.message_text.setText('           Live Message')
        font = QtGui.QFont()
        font.setFamily("MV Boli")
        font.setPointSize(12)
        self.message_text.setFont(font)
        # ******* message text label end********



        # ********border label ko start***********
        self.border = QLabel(self)  # link aaune thau
        self.border.setGeometry(0, 0, 2, 350)  # browse labe ko geometry size
        self.border.setStyleSheet("background-color:rgb(224, 232, 9);")
        self.border.move(10, 10)
        # ********border label ko start***********

        # ********border1 label ko start***********
        self.border1 = QLabel(self)  # link aaune thau
        self.border1.setGeometry(0, 0, 335, 2)  # browse labe ko geometry size
        self.border1.setStyleSheet("background-color:rgb(224, 232, 9);")
        self.border1.move(10, 10)
        # ********border1 label ko start***********

        # ********border2 label ko start***********
        self.border2 = QLabel(self)  # link aaune thau
        self.border2.setGeometry(0, 0, 2, 350)  # browse labe ko geometry size
        self.border2.setStyleSheet("background-color:rgb(224, 232, 9);")
        self.border2.move(344, 10)
        # ********border2 label ko start***********

        # ********border3 label ko start***********
        self.border3 = QLabel(self)  # link aaune thau
        self.border3.setGeometry(0, 0, 335, 2)  # browse labe ko geometry size
        self.border3.setStyleSheet("background-color:rgb(224, 232, 9);")
        self.border3.move(10, 360)
        # ********border1 label ko start***********


         # ******start ko start button start*********
        self.start_btn = QPushButton('Start', self)
        self.start_btn.clicked.connect(self.onclick_load_start)
        font = QtGui.QFont()
        font.setFamily("MV Boli")
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.start_btn.setGeometry(0, 30, 180, 40)
        self.start_btn.setFont(font)
        self.start_btn.setStyleSheet("\n"
                                     " QPushButton {\n"
                                     "\n"
                                     "color:#000;\n"
                                     "border: 2px solid #555;\n"
                                     "border-radius: 11px;\n"
                                     "\n"
                                     "padding: 5px;\n"
                                     "background: qradialgradient(cx: 0.3, cy: -0.4,\n"
                                     "fx: 0.3, fy: -0.4,\n"
                                     "radius: 1.35, stop: 0 #fff, stop: 1 rgb(173, 165, 162));\n"
                                     "min-width: 80px;\n"
                                     "}\n"
                                     "\n"
                                     " QPushButton:hover {\n"
                                     "background: qradialgradient(cx: 0.3, cy: -0.4,\n"
                                     "fx: 0.3, fy: -0.4,\n"
                                     "radius: 1.35, stop: 0 #fff, stop: 1  rgb(68, 112, 78));\n"
                                     "}\n"
                                     "\n"
                                     " QPushButton:pressed {\n"
                                     "background: qradialgradient(cx: 0.4, cy: -0.1,\n"
                                     "fx: 0.4, fy: -0.1,\n"
                                     "radius: 1.35, stop: 0 #fff, stop: 1 rgb(68, 112, 78));\n"
                                     "}")

        self.start_btn.move(80, 90)  # browse button ko position
        # ******start ko start button end*********

        #******* reset ko reset button ko start*********

        self.reset_btn = QPushButton('Reset', self)
        self.reset_btn.clicked.connect(self.onclick_reset)
       # self.reset_btn.clicked.connect(self.lbl.clear)
        font = QtGui.QFont()
        font.setFamily("MV Boli")
        font.setPointSize(15)
        font.setBold(True)
        self.reset_btn.setGeometry(0, 30, 180, 40)
        self.reset_btn.setFont(font)
        self.reset_btn.setStyleSheet("\n"
                                     " QPushButton {\n"
                                     "\n"
                                     "color:  #000;\n"
                                     "border: 2px solid #555;\n"
                                     "border-radius: 11px;\n"
                                     "font: 75 15pt \"MV Boli\";\n"
                                     "padding: 5px;\n"
                                     "background: qradialgradient(cx: 0.3, cy: -0.4,\n"
                                     "fx: 0.3, fy: -0.4,\n"
                                     "radius: 1.35, stop: 0 #fff, stop: 1 rgb(173, 165, 162));\n"
                                     "min-width: 80px;\n"
                                     "}\n"
                                     "\n"
                                     " QPushButton:hover {\n"
                                     "background: qradialgradient(cx: 0.3, cy: -0.4,\n"
                                     "fx: 0.3, fy: -0.4,\n"
                                     "radius: 1.35, stop: 0 #fff, stop: 1  rgb(255, 0, 0));\n"
                                     "}\n"
                                     "\n"
                                     " QPushButton:pressed {\n"
                                     "background: qradialgradient(cx: 0.4, cy: -0.1,\n"
                                     "fx: 0.4, fy: -0.1,\n"
                                     "radius: 1.35, stop: 0 #fff, stop: 1 rgb(242, 62, 62));\n"
                                     "}")
        self.reset_btn.move(80, 170)
        # *******  reset button ko end*********

        # ******* back ko button ko start*********

        self.back_btn = QPushButton('Back', self)

        font = QtGui.QFont()
        font.setFamily("MV Boli")
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.back_btn.setGeometry(0, 30, 180, 40)
        self.back_btn.setFont(font)
        self.back_btn.setStyleSheet("\n"
                                     " QPushButton {\n"
                                     "\n"
                                     "color:  #000;\n"
                                     "border: 2px solid #555;\n"
                                     "border-radius: 11px;\n"
                                     "font: 75 15pt \"MV Boli\";\n"
                                     "padding: 5px;\n"
                                     "background: qradialgradient(cx: 0.3, cy: -0.4,\n"
                                     "fx: 0.3, fy: -0.4,\n"
                                     "radius: 1.35, stop: 0 #fff, stop: 1 rgb(173, 165, 162));\n"
                                     "min-width: 80px;\n"
                                     "}\n"
                                     "\n"
                                     " QPushButton:hover {\n"
                                     "background: qradialgradient(cx: 0.3, cy: -0.4,\n"
                                     "fx: 0.3, fy: -0.4,\n"
                                     "radius: 1.35, stop: 0 #fff, stop: 1  rgb(35, 73, 135));\n"
                                     "}\n"
                                     "\n"
                                     " QPushButton:pressed {\n"
                                     "background: qradialgradient(cx: 0.4, cy: -0.1,\n"
                                     "fx: 0.4, fy: -0.1,\n"
                                     "radius: 1.35, stop: 0 #fff, stop: 1 rgb(35, 73, 135));\n"
                                     "}")
        self.back_btn.move(80, 250)
        # ******* back ko button ko end*********




        #*******main background ko*********
        self.setGeometry(20, 30, 355, 740) #main dialog box ko height and width
        self.setWindowTitle('Live Video')
        self.show()
        # *******main background ko*********

    def onclick_load_start(self):
        global path
        # print(path)
        # self.th.changePixmap.connect(self.original.setPixmap)  # display each frame in original
        # self.th1.changePixmapp.connect(self.edge.setPixmap)  # display each frame in original
        # self.th2.changePixmappp.connect(self.threshold.setPixmap)  # display each frame in original
        self.th.changeLabel.connect(self.message.setText)
        self.th.start()
        # self.th1.start()
        # self.th2.start()

    def onclick_reset(self):
        self.close()
        self.initUI()


    def closeEvent(self, event):
        self.th.stop()
        # self.th1.stop1()
        # self.th2.stop2()
        QWidget.closeEvent(self, event)

    def showDialog(self):
        options = QFileDialog.Options()                     # opens the serch box
        fileName, ok = QFileDialog.getOpenFileName(self, "select Video", "",
                                                  "mp4 (*.mp4)", options=options)
        if ok:
          global path
          path = str(fileName)
          self.lbl.setText(path)

#******** original video display start*********
class Thread(QThread):
    # changePixmap = pyqtSignal(QPixmap)
    # changePixmapp = pyqtSignal(QPixmap)
    changeLabel = pyqtSignal(str)

    def __init__(self, parent=None):
        QThread.__init__(self, parent=parent)
        self.isRunning = True

    def run(self, mirror=False):
        url = 'http://192.168.0.100:8080/shot.jpg'
        imgResp = urllib.request.urlopen(url)
        imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
        img = cv2.imdecode(imgNp, -1)

        if mirror:
            img = cv2.flip(img, 1)

        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        back_frame = img2
        # imgfinal=gammacorrection(img2,1)
        # filters = build_filters()
        #
        # res1 = process(img2, filters)
        global backSubtractor
        backSubtractor = BackGroundSubtractor(0.0004, denoise(img2))
        frameno = 0
        track_temp = []

        track_master = []
        track_temp2 = []
        centers = []
        top_contour_dict = defaultdict(int)
        obj_detected_dict = defaultdict(int)
        flg = 0
        point = 0
        k = 0
        l = 0
        a = 0
        fno = 0
        t = 0
        while True:

            imgResp = urllib.request.urlopen(url)
            imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
            img = cv2.imdecode(imgNp, -1)

            if mirror:
                img = cv2.flip(img, 1)
            img3 = img
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            frameno = frameno + 1
            cv2.putText(img, '%s%.f' % ('Frameno:', frameno), (600, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                        2)
            # cv2.imshow('after denoise',denoise(gray))
            # imgfinal = gammacorrection(gray, 1)
            # filters = build_filters()
            #
            # res1 = process(gray, filters)
            # cv2.imshow('filter', res1)

            # Background subtract gareko
            foreGround = backSubtractor.getForeground(denoise(gray))

            # foreGround[foreGround>=120 ]=0
            # cv2.imshow("initial foreground", foreGround)

            # edged = cv2.Canny(foreGround, 30, 100)  # any gradient between 30 and 150 are considered edges
            # cv2.imshow('CannyEdgeDet', edged)

            # morphological operation gareko
            ret, mask = cv2.threshold(foreGround, 27, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dilation = cv2.dilate(mask, kernel, iterations=2)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
            closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
            opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
            dilation = opening
            # contour find garne
            im2, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            consecutiveframe = 20

            mycnts = []
            count = 0

            for c in contours:

                M = cv2.moments(c)
                if M['m00'] == 0:
                    pass
                else:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])

                    # centers.append([cx, cy])
                    # print(centers)
                    if cv2.contourArea(c) < 1000 or cv2.contourArea(c) > 200000:
                        # cv2.drawContours(closing, contours, 0, (0, 0, 0), -1)
                        pass
                    else:
                        mycnts.append(c)
                        (x, y, w, h) = cv2.boundingRect(c)

                        if cv2.contourArea(c) < 5000:
                            count = count + 1

                        print(count)
                        cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
                        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.circle(img, (cx, cy), 3, (140, 58, 8), -1)

                        # if len(centers) >= 2:
                        #     dx = centers[0][0] - centers[1][0]
                        #     dy = centers[0][1] - centers[1][1]
                        #     D = np.sqrt(dx * dx + dy * dy)
                        #     #print(D)
                        #     cv2.putText(img, '%s%.f' % ('Distance:', D), (300, 300),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
                        # cv2.putText(img, "center", (cx - 20, cy - 20),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        # cv2.putText(img, 'C %s,%s,%.0f' % (cx, cy, cx + cy), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        #             (0, 0, 0),
                        #             2)
                        text = "Occupied"

                        # draw the text and timestamp on the frame
                        cv2.putText(img, "Bus Stand Status: {}".format(text), (10, 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.putText(img, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                                    (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

                        sumcxcy = cx + cy

                        track_temp.append([sumcxcy, frameno])
                        track_master.append([sumcxcy, frameno])
                        # print(track_master)
                        countuniqueframe = set(j for i, j in track_master)
                        # print(countuniqueframe)

                        cv2.putText(img, '%s%.f' % ('Uniqueframeno:', len(countuniqueframe)), (300, 450),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                        if len(countuniqueframe) > consecutiveframe:
                            minframeno = min(j for i, j in track_master)
                            for i, j in track_master:
                                if j != minframeno:  # get a new list. omit the those with the minframeno
                                    track_temp2.append([i, j])

                            track_master = list(track_temp2)  # transfer to the master list
                            track_temp2 = []

                        # print ('After', track_master)
                        countcxcy = Counter(i for i, j in track_master)
                        #    print(countcxcy)
                        distance = []

                        for i, j in countcxcy.items():
                            # print(j,consecutiveframe)
                            if j >= consecutiveframe:
                                if count >= 2:
                                    cv2.imwrite('image.png', dilation)
                                top_contour_dict[i] += 1
                                if cv2.contourArea(c) > 5000:
                                    k = cx
                                    l = cy

                        if sumcxcy in top_contour_dict or (sumcxcy - 1) in top_contour_dict or (
                                sumcxcy + 1) in top_contour_dict:

                            cv2.line(img, (cx, cy), (k, l), (150, 0, 0), 2)
                            dx = cx - k
                            dy = cy - l

                            D = np.sqrt(dx * dx + dy * dy)
                            cv2.putText(img, 'Distance: %.0f' % (D), (k, l), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (0, 0, 0), 2)

                            fno = fno + 1

                            if fno % 5 == 0 and a != 4:
                                a = a + 1
                                gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
                                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                                crop = img
                                for (x, y, w, h) in faces:
                                    # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                                    crop = img[(y - 20):(y + h + 20), (x - 20):(x + w + 20)]
                                cv2.imwrite('suspect.png', crop)

                            if D > 400:

                                if top_contour_dict[sumcxcy] > 350 or top_contour_dict[sumcxcy - 1] > 350 or \
                                        top_contour_dict[
                                            sumcxcy + 1] > 350:
                                    cv2.drawContours(img, [c], -1, (255, 0, 0), 3)

                                    # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
                                    cv2.putText(img, '%s' % ('CheckObject'), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                                (255, 255, 255), 2)
                                    flg = flg + 1
                                    if flg == 1:

                                        global mailer
                                        image = facedetect(cv2.imread("suspect.png"))
                                        if image == True:
                                            comp = compare("suspect.png")
                                            if comp == "abcd":
                                                mailer = "email send to police\n\n he is not citizen"
                                                # print("email send to police he is not citizen")
                                                mail1("suspect.png")
                                                # send email to police
                                                break
                                                # check database
                                            else:
                                                mailer = "Email Sent to : " + comp
                                                # print("Email Sent to : " + comp)
                                                mail2(comp, "suspect.png")

                                                break
                                                # check database

                                        else:
                                            mailer = "Email send to police"
                                            # print("email send to police")
                                            mail3("suspect.png")
                                            break
                                            # send email to police

                                        # mail()
                                        # print("mail")
                                        # flg = 2

                                    msg = mailer + '\n\n Detected : ' + str(sumcxcy) + '\n\nFrame no. : ' + str(
                                        frameno) + " \n\n" + str(obj_detected_dict) + " \n\n"
                                    self.changeLabel.emit(str("\n" + msg + "\n"))

                                    print('Detected : ', sumcxcy, frameno, obj_detected_dict)
                                    obj_detected_dict[sumcxcy] = frameno

            for i, j in obj_detected_dict.items():
                if frameno - obj_detected_dict[i] > 200:
                    print('PopBefore', i, obj_detected_dict[i], frameno, obj_detected_dict)
                    print('PopBefore : top_contour :', top_contour_dict)
                    obj_detected_dict.pop(i)

                    # Set the count for eg 448 to zero. because it has not be 'activated' for 200 frames. Likely, to have been removed.
                    top_contour_dict[i] = 0
                    print('PopAfter', i, obj_detected_dict[i], frameno, obj_detected_dict)
                    print('PopAfter : top_contour :', top_contour_dict)

            # cv2.polylines(img, [pts], True, (255, 0, 0), thickness=2)
            # edges = cv2.Canny(erosion, 100, 200)
            # cv2.imshow('mask', dilation)
            # cv2.imshow('canny edge', edges)

            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = img[y:y + h, x:x + w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            cv2.imshow('my webcam', img)

            # rgbImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #
            # convertToQtFormat = QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0], QImage.Format_RGB888)
            # convertToQtFormat = QPixmap.fromImage(convertToQtFormat)
            # p = convertToQtFormat.scaled(2040, 700, Qt.KeepAspectRatio)
            # self.changePixmap.emit(p)


            if cv2.waitKey(1) == 13:
                break  # esc to quit
        cv2.destroyAllWindows()

        top_contours = sorted(top_contour_dict, key=top_contour_dict.get,
                              reverse=True)  # sort based on highest value, its a list.

        for i in top_contours:
            print(i, top_contour_dict[i])  # print out the key, count
        print("Contours recorded :", len(top_contours))


    def stop(self):
        self.isRunning = False
        self.quit()
        self.wait()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = live()
    sys.exit(app.exec_())