

import sys
import cv2
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QWidget, QLabel
import datetime


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 Video'
        self.left = 100
        self.top = 100
        self.width = 1240
        self.height = 480
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.resize(1800, 1200)
        # create a label
        # create a label
        label = QLabel(self)
        label.move(280, 120) #  image wala label
        label.resize(10400, 480)
        label1 = QLabel(self)
        label1.move(1880, 620)


        self.th = Thread(self)
        self.th.changePixmap.connect(label.setPixmap) #display each frame
        self.th.changeLabel.connect(label1.setText)
        self.th.start()

    def closeEvent(self, event):
        self.th.stop()
        QWidget.closeEvent(self, event)

class Thread(QThread):
    changePixmap = pyqtSignal(QPixmap)
    changeLabel = pyqtSignal(str)

    def __init__(self, parent=None):
        QThread.__init__(self, parent=parent)
        self.isRunning = True

    def run(self):
        cap = cv2.VideoCapture('a.mp4')

        while self.isRunning:

            ret, frame = cap.read()
            rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            convertToQtFormat = QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0], QImage.Format_RGB888)
            convertToQtFormat = QPixmap.fromImage(convertToQtFormat)
            p = convertToQtFormat.scaled(640, 400, Qt.KeepAspectRatio)
            self.changePixmap.emit(p)

            now = datetime.datetime.now()
            sec = now.second
            self.changeLabel.emit(str(sec))

    def stop(self):
        self.isRunning = False
        self.quit()
        self.wait()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())