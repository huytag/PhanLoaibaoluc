import time
import cv2
import threading
from PIL import Image
from PIL import ImageTk
from Modules import PublicModules as libs
from Modules import LSTM_Config as cf
import random
from HanhViBaoLuc import RealTime_URL as RL


class MyThreadingVideo:
    def __init__(self, lbShow, lbFather, lbShowKetQua, vgg16_model, lstm_model, treeAction):
        self.vgg16_model = vgg16_model
        self.lstm_model = lstm_model
        self.treeAction = treeAction
        self.frames = None
        self._RM = None
        self._10 = None
        self.lbShow = lbShow
        self.lbShowKetQua = lbShowKetQua
        self.lbFather = lbFather
        self.myThread = threading.Thread(target=self.VideoThread)
        self.myThread.setDaemon(True)
###
    def setFrames(self, frames: list):
        self.frames = frames
        # Tinh chinh Frame phu hop trong day
        self._10 = RL.funget10F(self.frames)
        self._RM =  RL.Remove_backgournd_RealTime(self._10)
###
    def isBusy(self):
        return self.myThread.isAlive()

    def startShowVideo(self):
        self.myThread = threading.Thread(target=self.VideoThread)
        self.myThread.setDaemon(True)
        self.myThread.start()
    def VideoThread(self):
        # # Predict cho moi 20Frames Anh tai day
        transfer = cf.fun_getTransferValue_EDIT(pathVideoOrListFrame= self._RM, modelVGG16= self.vgg16_model)
        pre, real = libs.fun_predict(modelLSTM= self.lstm_model, transferValue=transfer)
        conv = cf.VIDEO_NAMES_DETAIL[pre]
        if real > 0.6 and conv != 'no':
            text = 'Predict: {0} [ {1:.4f} ]'.format(conv, real)
        else:
            text = ''
        if self.lbShow is not None:
            # Show thread video
            for frame in self.frames:
                image = libs.fun_cv2_imageArrayToImage(containerFather= self.lbFather, frame= frame.copy(), reSize= 0.8)
                self.lbShow.config(image= image)
                self.lbShow.image = image
        self.lbShowKetQua.config(text= text)

        if text != '':
            self.treeAction.fun_saveVideoDetection(frames= self._RM, fol= cf.VIDEO_NAMES[pre])
