from PyQt5.QtWidgets import (QWidget, QPushButton, QLineEdit,
                             QInputDialog, QApplication, QFileDialog, QLabel)
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon, QPixmap, QImage
import cv2
from PyQt5.QtCore import QThread, pyqtSignal, Qt
import datetime

from livewindow import live
from browsesucess import Example

class firstwindow(QWidget):
    global path
    def __init__(self):
        super().__init__()
        window2 = QtWidgets.QMainWindow()
        self.initUI()


    def initUI(self):
        self.setStyleSheet("background-color:rgb(2, 15, 45);")                        #background ko color



        # ********border label ko start***********
        self.border = QLabel(self)  # link aaune thau
        self.border.setGeometry(0, 0, 2, 690)  # browse labe ko geometry size
        self.border.setStyleSheet("background-color:rgb(224, 232, 9);")
        self.border.move(200, 20)
        # ********border label ko start***********



        # ********border2 label ko start***********
        self.border2 = QLabel(self)  # link aaune thau
        self.border2.setGeometry(0, 0, 2, 690)  # browse labe ko geometry size
        self.border2.setStyleSheet("background-color:rgb(224, 232, 9);")
        self.border2.move(1160, 20)
        # ********border2 label ko start***********

        # ********border3 label ko start***********
        self.border3 = QLabel(self)  # link aaune thau
        self.border3.setGeometry(0, 0, 1235, 2)  # browse labe ko geometry size
        self.border3.setStyleSheet("background-color:rgb(224, 232, 9);")
        self.border3.move(50, 600)
        # ********border1 label ko start***********


        # ******* msrsn_text label start********

        self.message_text = QLabel(self)  # link aaune thau
        self.message_text.setGeometry(0, 30, 660, 30)  # browse labe ko geometry size
        self.message_text.setStyleSheet("background:rgb(2, 15, 45);")
        self.message_text.move(360, 200)
        self.message_text.setText('   ABANDONED   OBJECT   DETECTOR')
        font = QtGui.QFont()
        font.setFamily("MS Serif")
        font.setPointSize(23)
        font.setBold(True)
        font.setWeight(75)
        self.message_text.setFont(font)
        self.message_text.setStyleSheet("color:rgb(239, 239, 232)")

        # ******* msrsne text label end********

        # ******* msrsn_text1 label start********

        self.message_text1 = QLabel(self)  # link aaune thau
        self.message_text1.setGeometry(0, 30, 660, 30)  # browse labe ko geometry size
        self.message_text1.setStyleSheet("background:rgb(2, 15, 45);")
        self.message_text1.move(360, 160)
        self.message_text1.setText("   MSRSN's ")
        font = QtGui.QFont()
        font.setFamily("MS Serif")
        font.setPointSize(28)
        font.setBold(True)
        font.setUnderline(False)
        font.setWeight(75)
        self.message_text1.setFont(font)
        self.message_text1.setStyleSheet("color:rgb(153, 136, 52)")

        # ******* msrsne text1 label end********




        # *********live button ko code start***********
        self.live_btn = QPushButton('Live Video', self)
        self.live_btn.clicked.connect(self.onclick_live)
        font = QtGui.QFont()
        font.setFamily("MV Boli")
        font.setPointSize(20)
        self.live_btn.setGeometry(0, 30, 200, 50)
        font.setBold(True)
        self.live_btn.setFont(font)
        self.live_btn.setStyleSheet("\n"
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
                               "radius: 1.35, stop: 0 #fff, stop: 1  rgb(221, 232, 30));\n"
                               "}\n"
                               "\n"
                               " QPushButton:pressed {\n"
                               "background: qradialgradient(cx: 0.4, cy: -0.1,\n"
                               "fx: 0.4, fy: -0.1,\n"
                               "radius: 1.35, stop: 0 #fff, stop: 1 rgb(68, 112, 78));\n"
                               "}")

        self.live_btn.move(583, 300)  # browse button ko position

        # *********browse button ko code end***********


         # ******browse button start*********
        self.browse_btn = QPushButton('Browse Video', self)
        self.browse_btn.clicked.connect(self.onclick_browse)

        font = QtGui.QFont()
        font.setFamily("MV Boli")
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.browse_btn.setGeometry(0, 30, 200, 50)
        self.browse_btn.setFont(font)
        self.browse_btn.setStyleSheet("\n"
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
                               "radius: 1.35, stop: 0 #fff, stop: 1  rgb(221, 232, 30));\n"
                               "}\n"
                               "\n"
                               " QPushButton:pressed {\n"
                               "background: qradialgradient(cx: 0.4, cy: -0.1,\n"
                               "fx: 0.4, fy: -0.1,\n"
                               "radius: 1.35, stop: 0 #fff, stop: 1 rgb(68, 112, 78));\n"
                               "}")

        self.browse_btn.move(583, 400)  # browse button ko position
        # ******start ko start button end*********

        #******* reset ko reset button ko start*********

        self.exit_btn = QPushButton('EXIT', self)
        self.exit_btn.clicked.connect(self.close)
        font = QtGui.QFont()
        font.setFamily("MV Boli")
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.exit_btn.setGeometry(0, 30, 200, 50)
        self.exit_btn.setFont(font)
        self.exit_btn.setStyleSheet("\n"
                                     " QPushButton {\n"
                                     "\n"
                                     "color:  #000;\n"
                                     "border: 2px solid #555;\n"
                                     "border-radius: 11px;\n"
                                     
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
        self.exit_btn.move(583, 500)
        # *******  reset button ko end*********






        #*******main background ko*********
        self.setGeometry(20, 30, 1340, 740) #main dialog box ko height and width
        self.setWindowTitle('main window')
        self.show()
        # *******main background ko*********

    def onclick_browse(self):
           # self.close()
            self.app1 = QApplication(sys.argv)
            self.ui = Example()
            Example.close(self)

    def onclick_live(self):
           # self.close()
            self.app1 = QApplication(sys.argv)
            self.ui = live()
            live.close(self)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = firstwindow()
    sys.exit(app.exec_())