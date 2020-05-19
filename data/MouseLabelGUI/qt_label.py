from qt_sectmice import Ui_MainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QApplication, QLabel, QMessageBox
from PyQt5.QtCore import QObject, pyqtSignal
from PIL import Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import sys
import scipy.io as sio
import os
import pickle


img = 'lisha.jpg'
im = cv2.imread(img)[...,::-1]
matplotlib.use("Qt5Agg")


# color https://www.cnblogs.com/wutanghua/p/11503835.html
COLORS = ['darkslategray', 'darkgoldenrod', 'dodgerblue', 'limegreen', 'mediumvioletred'] * 20
MICENUMS = 2


def getMiceColor(miceInd):
    return COLORS[miceInd-1]


class MyLabelData(QObject):
    def __init__(self, myFrame, figure, canvas, axes):
        super().__init__()
        self.myFrame   = myFrame
        self.imagFiles = []
        self.imagInd   = None
        self.miceInd   = None
        self.dataROI   = []
        self.axes      = axes

    @property
    def imagNums(self):
        return len(self.imagFiles)

    def clearcurrent(self):
        pass

    def out2file(self):
        # basenames of imagFiles
        dirname  = os.path.dirname(self.imagFiles[0])
        imagenames = [os.path.basename(p) for p in self.imagFiles]

        # make masker
        imagemasks = []
        for imagi in range(self.imagNums):
            image_filename = self.imagFiles[imagi]
            x_len, y_len = Image.open(image_filename).size
            imag_zero = np.zeros((y_len, x_len), dtype=np.uint8)
            for micei in range(MICENUMS):
                d = self.dataROI[imagi, micei]
                x, y = d.get('x', []), d.get('y', [])
                if len(x) > 2:
                    cv2.fillConvexPoly(imag_zero, np.array([x, y], dtype=np.int).T, micei + 1)
            imagemasks.append(imag_zero)

        # save to MAT file
        MAT = {'imagenames': imagenames, 'imagemasks': imagemasks}
        sio.savemat(dirname+'/data.mat', MAT)

        # save pickle file
        output = open(dirname+'/data.pkl', 'wb')
        pickle.dump(MAT, output)

    def flash_ROIs_data(self):
        self.imagInd = self.myFrame.imagInd
        self.dataROI[self.imagInd - 1, :] = {}

    def loadall(self, imagFiles):
        self.imagFiles = imagFiles
        self.dataROI   = np.empty((self.imagNums, MICENUMS), dtype=object)
        self.dataROI[:] = {}
        self.imagInd   = None

    def save_1ROI_to_data(self, imagInd, miceInd, x, y):
        self.imagInd, self.miceInd = imagInd, miceInd
        print("save_1ROI_to_data : ", imagInd, miceInd)
        self.dataROI[imagInd-1, miceInd-1] = {"x":x, "y":y}

    def load_1ROI_from_data(self):
        d = self.dataROI[self.imagInd-1, self.miceInd-1]
        x, y = d.get('x', []), d.get('y', [])
        lineColor = getMiceColor(self.miceInd)
        self.axes.plot(x, y, color=lineColor, picker=20)
        self.axes.fill(x, y, color=lineColor, alpha=.6)

    def load_allROI_from_data(self, imagInd):
        print("load_allROI_from_data : imag=", imagInd)
        self.imagInd = imagInd
        for i in range(MICENUMS):
            miceInd = i + 1
            d = self.dataROI[self.imagInd - 1, miceInd - 1]
            x, y = d.get('x', []), d.get('y', [])
            lineColor = getMiceColor(miceInd)
            self.axes.plot(x, y, color=lineColor, picker=20)
            self.axes.fill(x, y, color=lineColor, alpha=.6)
        if len(x)==0:
            print("empty ROI")


class MyDraw(QObject):
    newROI = pyqtSignal(int, int, list, list) #miceInd, x, y
    def __init__(self, myFrame, figure, canvas, axes):
        super().__init__()
        self.myFrame = myFrame
        self.figure = figure
        self.canvas = canvas
        self.axes   = axes
        self.axisrg = None
        self.h_click  = None
        self.h_move   = None
        self.h_release= None
        self.x        = None
        self.y        = None
        self.line     = None

    @property
    def miceInd(self):
        return self.myFrame.miceInd

    @property
    def imagInd(self):
        return self.myFrame.imagInd

    def restrictRange(self,x,y,axisrg):
        x_min, x_max, y_max, y_min = axisrg
        x = x_min if x < x_min else x_max if x > x_max else x
        y = y_min if y < y_min else y_max if y > y_max else y
        return x, y

    def connectdraw(self):
        self.h_click = self.canvas.mpl_connect('button_press_event', self.onClick)

    def disconnectdraw(self):
        self.canvas.mpl_disconnect(self.h_click)

    def onClick(self, event):
        if event.button == 1:
            Coords1x = event.xdata
            Coords1y = event.ydata
        else:
            return
        self.x = []
        self.y = []
        self.axisrg = self.axes.axis()
        lineColor = getMiceColor(self.miceInd)
        self.line = self.axes.plot(self.x, self.y, color=lineColor, picker=20)[0]
        self.h_move = self.canvas.mpl_connect('motion_notify_event', self.onMouseMotion)
        self.h_release = self.canvas.mpl_connect('button_release_event', self.onRelease)
        print('On Click')

    def onMouseMotion(self, event):
        if event.button == 1:
            coordsX, coordsY = event.xdata, event.ydata
            coordsX, coordsY = self.restrictRange(coordsX, coordsY, self.axisrg)
            self.x.append(coordsX)
            self.y.append(coordsY)
            self.line.set_data(self.x, self.y)
            self.canvas.draw()
        elif event.button == 3:
            pass

    def onRelease(self, event):
        self.canvas.mpl_disconnect(self.h_move)
        self.canvas.mpl_disconnect(self.h_release)
        print('On release')
        x0, y0 = self.line.get_data()
        if len(x0) != 0:
            x2 = np.append(x0, x0[0])
            y2 = np.append(y0, y0[0])
            self.line.set_data(x2, y2)
            self.axes.fill(x2, y2, color=self.line.get_color(), alpha=.6)
            self.canvas.draw()
            self.newROI.emit(self.imagInd, self.miceInd, list(x2), list(y2))
        else:
            print('Empty relase')


class MyFrame(QObject):
    def __init__(self):
        super().__init__()
        self.win = QtWidgets.QMainWindow()
        self.win_sub = Ui_MainWindow()
        self.win_sub.setupUi(self.win)
        self.figure = plt.figure()
        self.axes = self.figure.add_subplot(1, 1, 1)
        self.canvas = self.figure.canvas
        self.toolbar = self.canvas.toolbar

        self.axes.set_title('this is title')
        self.axes.set_axis_off()
        self.axes.imshow(im)
        self.axes.set_position([0,0,1,0.95])

        self.win_sub.verticalLayout_2.addWidget(self.canvas)
        self.win.show()

        self.myLabelData = MyLabelData(self, self.figure, self.canvas, self.axes)
        self.myDraw = MyDraw(self, self.figure, self.canvas, self.axes)
        self.myDraw.connectdraw()
        self.myDraw.newROI.connect(self.myLabelData.save_1ROI_to_data)
        self.myDraw.newROI.connect(self.ui_newROI)

        self.win_sub.btnLoad.clicked.connect(self.ui_load)
        self.win_sub.btnNextImag.clicked.connect(self.ui_nextImag)
        self.win_sub.btnPrevImag.clicked.connect(self.ui_prevImag)
        self.win_sub.btnClear.clicked.connect(self.ui_clear)
        self.win_sub.btnZoom.clicked.connect(self.ui_zoom)
        self.win_sub.btnZoomReset.clicked.connect(self.ui_zoomReset)

        # Status bar #
        self.stbLabelNum = QLabel("Num:", parent = self.win)
        self.stbLabelDone = QLabel("Label:", parent =  self.win)
        self.win_sub.statusbar.addWidget(self.stbLabelNum)
        self.win_sub.statusbar.addWidget(self.stbLabelDone)

        # Data
        self.imagFiles = []
        self.imagInd   = None    # 1 based
        self.imagNow   = None
        self.imagRGB   = None
        self.isZoomLocked = True
        self.axisPre   = (None,None,None,None)

        # Mice Index
        self.miceNums = MICENUMS
        self._miceInd = 1        # 1 based
        self.rdoMices = [self.win_sub.rdoMice1, self.win_sub.rdoMice2]

        for i, rdoControl in enumerate(self.rdoMices):
            rdoControl.clicked.connect(lambda a=0,b=0,ind=i+1: self.ui_rdoClick(ind))

    def ui_rdoClick(self, ind):
        self.miceInd = int(ind)

    def save2mat(self):
        self.myLabelData.out2file()

    @property
    def imagNums(self):
        return len(self.imagFiles)

    @property
    def miceInd(self):
        return self._miceInd

    @miceInd.setter
    def miceInd(self, value):
        assert 1<=value<=self.miceNums
        print("miceInd", value)
        self._miceInd = value
        rodControl = self.rdoMices[self._miceInd - 1]
        if not rodControl.isChecked():
            rodControl.setChecked(True)

    def ui_newROI(self, miceInd, x, y):
        # current ROI save
        pass
        # new ROI
        miceInd_pre = self.miceInd

        print("before", self.miceInd, self.miceNums)
        if miceInd_pre < self.miceNums:
            self.miceInd = miceInd_pre + 1
        else:  #finished current mice
            self.finishCurrentImag()

        print("after", self.miceInd, self.miceNums)

    def finishCurrentImag(self):
        if self.imagInd < self.imagNums:
            self.ui_nextImag()
        else:  #finished All
            self.win_sub.statusbar.showMessage("Finished all!")
            self.save2mat()
            QMessageBox.information(self.win, '完成', '顺利保存所有数据到 data.mat 文件！可关闭程序。')

    def ui_zoom(self):
        if self.win_sub.btnZoom.isChecked():
            self.myDraw.disconnectdraw()
            self.toolbar.zoom()
            self.win_sub.statusbar.showMessage("Zoom On")
        else:
            self.toolbar.zoom()
            self.win_sub.statusbar.showMessage("Zoom Off", 2000)
            self.myDraw.connectdraw()

    def ui_zoomReset(self):
        self.axes.axis('auto')
        self.axes.axis('equal')
        self.canvas.draw_idle()

    def ui_clear(self):
        self.myLabelData.flash_ROIs_data()
        self.refreshImag()

    def ui_load(self):
        print("hello")
        fileNames, filetype = QFileDialog.getOpenFileNames(self.win, "Select Image files", "","Images (*.png *.bmp *.jpg)")
        if fileNames:
            # load data from any storage
            self.imagFiles = fileNames
            self.myLabelData.loadall(self.imagFiles)

            # refresh GUI
            nimag = len(fileNames)
            self.win_sub.statusbar.showMessage(f"Load Succeed! [{nimag}]", 2000)

            self.imagInd = 1
            self.refreshImag()
            self.ui_zoomReset()



        else:
            self.win_sub.statusbar.showMessage("Load Canceled!", 2000)

    def ui_nextImag(self):
        self.imagInd += 1
        print("Change to Image:", self.imagInd)
        self.refreshImag()

    def ui_prevImag(self):
        self.imagInd -= 1
        print("Change to Image:", self.imagInd)
        self.refreshImag()

    def refreshImag(self):
        self.axisPre = self.axes.axis()
        if self.imagInd:
            self.imagNow = self.imagFiles[self.imagInd-1]
        else:
            self.imagInd = 0
            self.imagNow = 'lisha.jpg'

        print(self.imagNow)
        self.imagRGB = cv2.imread(self.imagNow)[..., ::-1]
        self.axes.cla()
        self.axes.imshow(self.imagRGB)
        self.axes.set_axis_off()
        if self.win_sub.ckbZoomLock.isChecked():
            self.axes.axis(self.axisPre)
        self.canvas.draw_idle()

        # GUI hints
        nimagFiles = len(self.imagFiles)
        if self.imagInd:
            self.stbLabelNum.setText("Num:[{} / {}]".format(self.imagInd, nimagFiles))
        else:
            self.stbLabelNum.setText("Num:")

        self.win_sub.btnNextImag.setEnabled(self.imagInd < nimagFiles)
        self.win_sub.btnPrevImag.setEnabled(self.imagInd > 1)

        # Load frame ROI from datastorage
        self.myLabelData.load_allROI_from_data(self.imagInd)

        # Next miceInd
        self.miceInd = 1



app = QApplication(sys.argv)

a = MyFrame()
# a.win.show()
app.exec_()


