#This will use the OpenCV library to handle camera input, 1 from the main camera, and 2 from the auxillary safety cameras

#The GUI Library used is pyQT
#IP Cameras are streamed usign RTSP

import sys 
import time
import threading
import re

from pathlib import Path

from scipy.interpolate import interp1d
import numpy as np
import math
import csv

from pymodbus.client.tcp import ModbusTcpClient
import asyncio
import numpy as np
import csv

import utils


from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
import cv2
from vidgear.gears import VideoGear
import imutils

from sllurp import llrp
from sllurp.llrp import LLRPReaderConfig, LLRPReaderClient, LLRP_DEFAULT_PORT
import logging


M_Camera_IP = "rtsp://admin:ABEFarmData1!@192.168.0.150/media/video1/multicast"
S_Camera1_IP = "rtsp://admin:ABEFarmData1!@192.168.0.151/media/video1/multicast"
S_Camera2_IP = "rtsp://admin:ABEFarmData1!@192.168.0.152/media/video1/multicast"

STRIDE_IP = "192.168.0.131"
STRIDE_PORT = 502

ROOT_DIR = Path(__file__).parent

modules_dict_list = [
    {"Module Name": "Identifier", "Module RFID": "00000", "Weight": 0, "Lint Moisture":0, "Seed Moisture": 0}
]


TESTMODE = False

class ModuleData:
    name_identifier = ""
    rfid = "00000"
    weight = 0
    LM = 0
    SM = 0
    
    def __init__(self, name_identifier="", rfid = "00000", weight=0):
        # Instance attributes (unique to each instance)
        self.name_identifier = name_identifier
        self.rfid = rfid
        self.weight = weight
        
class InclonometerData:
    angleX=0
    angleY=0

class StrideData():
    analog0 = 0
    analog2 = 0
    analog4 = 0
    analog7 = 0

class repeatFunction(threading.Thread):
    def __init__(self, interval, func, *args, **kwargs):
        super().__init__()
        self.interval = interval
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.stopped = threading.Event()

    def run(self):
        while not self.stopped.wait(self.interval):
            self.func(*self.args, **self.kwargs)
            
    def stop(self):
        self.stopped.set()


ROOT_DIR = Path(__file__).parent

modules_dict_list = [
    {"Module Name": "Identifier", "Module RFID": "00000", "Weight": 0, "Lint Moisture":0, "Seed Moisture": 0}
]

class ModuleData:
    name_identifier = ""
    rfid = "00000"
    weight = 0
    LM = 0
    SM = 0
    
    def __init__(self, name_identifier="", rfid = "00000", weight=0):
        # Instance attributes (unique to each instance)
        self.name_identifier = name_identifier
        self.rfid = rfid
        self.weight = weight
        
class InclonometerData:
    angleX=0
    angleY=0

class MainWindow(QMainWindow): 
    c = None #represents modbus client
    
    #IMPINJ setup
    config = LLRPReaderConfig()
    reader = LLRPReaderClient('192.168.0.100', LLRP_DEFAULT_PORT, config)
    reader.add_tag_report_callback(utils.tag_report_cb)

    reader.connect()

    MFeed_infocusmode = False
    
    # screen_rect = QGuiApplication.primaryScreen().availableGeometry()
    # Window_W = screen_rect.width()
    # Window_H = screen_rect.height()

    Window_W = 1920
    Window_H = 1080


    module0 = ModuleData() #Test Module
    inclodata = InclonometerData()
    stridedata0 = StrideData()
    curmoisturedata = (0,0) #Tuple which store the seed and lint moisture content respectively

    curweightdata = 0

    fTick_retrieveStrideData = None
    
    hint1_popup = None

    def enable_TESTMODE():
        global M_Camera_IP
        global S_Camera1_IP 
        global S_Camera2_IP
        M_Camera_IP = "http://218.219.214.248:50000/nphMotionJpeg?Resolution=640x480"
        S_Camera1_IP = "http://218.219.195.24/nphMotionJpeg?Resolution=640x480"
        S_Camera2_IP = "http://218.219.195.24/nphMotionJpeg?Resolution=640x480"

        
    if(TESTMODE is True):
        enable_TESTMODE()
        #Default the inclonometer readings from the stride to be at the 0 degree current reading
        stridedata0.analog0 = 12000
        stridedata0.analog2 = 12000

    if(not TESTMODE):
        c = ModbusTcpClient(host=STRIDE_IP, port=STRIDE_PORT)
        
    #SETUP QT Window
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("WINDOW TITLE")
        self.setGeometry(0,0,self.Window_W, self.Window_H)
        self.setFixedSize(self.Window_W, self.Window_H)
        self.showFullScreen()
        self.initUI()
        self.initCSV()
        utils.ShowDialogPopup(self,"NOTE:","Press CTRL + ESC to close application when ready.")



    #Custom function to setup UI Elements
    def initUI(self):
        self.mfw = None
        self.emw = None
        self.mawp = None #Module Added popup object initialization

        #UI Settings
        self.M_Camera_W = 1280
        self.M_Camera_H = 720
        self.S_Camera1_W = 640
        self.S_Camera1_H = 480
        self.S_Camera2_W = 640
        self.S_Camera2_H = 480
        
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        #central_widget.resizeEvent = self.OnCentralWidgetResize
        self.resizeEvent = self.OnWindowResize

        self.feed1_lbl = utils.clickableLabel()
        self.feed2_lbl = utils.clickableLabel(self.feed1_lbl,resize=True)
        self.feed3_lbl = utils.clickableLabel(self.feed1_lbl,resize=True)
        self.cameraHLayout_Spacer = QLabel()
        
        self.feed1_lbl.setMinimumWidth(self.Window_W)
        self.feed1_lbl.setMinimumHeight(self.Window_H)
        self.feed1_lbl.resize(self.Window_W, self.Window_H)

        self.feed2_lbl.setMinimumWidth(200)
        self.feed2_lbl.setMinimumHeight(200)
        self.feed3_lbl.setMinimumWidth(200)
        self.feed3_lbl.setMinimumHeight(200)
        
        #region Setup Camera MultiThreading
        #Three(3) Cameras Total 1 Main Camera and 2 Security Cameras

        self.camThread1 = camThread()
        self.camThread1.isMainFeed = True
        self.camThread1.src = M_Camera_IP
        self.camThread1.lbl = self.feed1_lbl
        self.camThread1.Window = self
        self.camThread1.ImageUpdate.connect(self.ImageUpdateSlot)
        self.camThread1.start()

        self.camThread2 = camThread()
        self.camThread2.src = S_Camera1_IP
        self.camThread2.lbl = self.feed2_lbl
        self.camThread2.Window = self
        self.camThread2.ImageUpdate.connect(self.ImageUpdateSlot)
        self.camThread2.start()
       

        self.camThread3 = camThread()
        self.camThread3.src = S_Camera2_IP
        self.camThread3.lbl = self.feed3_lbl
        self.camThread3.Window = self
        self.camThread3.ImageUpdate.connect(self.ImageUpdateSlot)
        self.camThread3.start()

        #END of setting up multithreading of cameras
        #endregion

        #Define other UI Elements


        self.focusview_btn = QPushButton(text="FOCUS MAIN VIEW", parent=self)
        self.focusview_btn.clicked.connect(self.focusMCameraFeed)
        self.focusview_btn.setMinimumWidth(200)
        self.focusview_btn.setMinimumHeight(50)

        self.weight_lbl = QLabel(f"Weight = {round(self.module0.weight,1)} lbs", self)
        self.weight_lbl.setFont(QFont("Arial", 20))
        self.weight_lbl.setStyleSheet("color: white;" "background-color: green; padding: 10px 5px;") #Use CSS properties to modify text
        self.weight_lbl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom )
        self.weight_lbl.setMaximumWidth(300)
        self.sm_lbl = QLabel(f"Seed Moisture = {self.curmoisturedata[0]}%", self)
        self.sm_lbl.setFont(QFont("Arial", 20))
        self.sm_lbl.setStyleSheet("color: white;" "background-color: green; padding: 10px 5px;") #Use CSS properties to modify text
        self.sm_lbl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom )

        self.lm_lbl = QLabel(f"Lint Moisture = {self.curmoisturedata[1]}%", self)
        self.lm_lbl.setFont(QFont("Arial", 20))
        self.lm_lbl.setStyleSheet("color: white;" "background-color: green; padding: 10px 5px;") #Use CSS properties to modify text
        self.lm_lbl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom )

        self.RFID_lbl = QLabel("Module RFID: ", self)
        self.RFID_lbl.setFont(QFont("Arial", 20))
        self.RFID_lbl.setStyleSheet("color: white;" "background-color: green; padding: 10px 5px;")
        self.RFID_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)

        self.feed1_lbl.setStyleSheet("background-color: black;")
        self.feed2_lbl.setStyleSheet("background-color: black;")
        self.feed3_lbl.setStyleSheet("background-color: black;")

        self.feed2_lbl.setMaximumSize(320,240)
        self.feed3_lbl.setMaximumSize(320,240)

        self.pausevideo_btn = QPushButton("Pause Video Feed")
        self.pausevideo_btn.clicked.connect(self.pauseFeeds)
        self.pausevideo_btn.setStyleSheet("color: yellow;" "background-color: blue;")



        self.addModule_btn = QPushButton("ADD MODULE")
        self.addModule_btn.clicked.connect(self.addModule)
        self.addModule_btn.setMinimumHeight(300)
        self.addModule_btn.setMinimumWidth(200)
        
        #STYLING UI 

        self.focusview_btn.setStyleSheet("""
        QPushButton {
            background-color: #4CAF50; /* Green */
            color: white;
            border-radius: 0px;
            padding: 0.8rem 1.5rem;
            font-size: 20px;
            font-weight: 750;
        }
        QPushButton:hover {
            background-color: #45a049; /* Darker green on hover */
        }
        QPushButton:pressed {
            background-color: #1b401c; /* Even darker green when pressed */
        }
    """)
        
        self.addModule_btn.setStyleSheet("""
        QPushButton {
            background-color: #f01313; 
            color: white;
            border-radius: 0px;
            padding: 10px 5px;
            font-size: 24px;
            font-weight: 900;                             
        }
        QPushButton:hover {
            background-color: #b03030; 
        }
        QPushButton:pressed {
            background-color: #ab0000; 
        }
    """)
        BottomBar_H_Layout = QHBoxLayout()
        BottomBar_H_Layout.addWidget(self.weight_lbl,0,Qt.AlignmentFlag.AlignLeft)
        BottomBar_H_Layout.addWidget(self.sm_lbl, 0,Qt.AlignmentFlag.AlignLeft)
        BottomBar_H_Layout.addWidget(self.lm_lbl, 0,Qt.AlignmentFlag.AlignLeft)
        BottomBar_H_Layout.addWidget(self.RFID_lbl,0,Qt.AlignmentFlag.AlignLeft)
        

        
        
        #Create Grid Layout Manager
        grid = QGridLayout()
        grid.addWidget(self.feed1_lbl, 0,0)
        grid.addLayout(BottomBar_H_Layout, 2,0,alignment= Qt.AlignmentFlag.AlignLeft)
        grid.addWidget(self.addModule_btn, 1,4, alignment= Qt.AlignmentFlag.AlignRight)

        #grid.addWidget(self.pausevideo_btn, 1,2)
        central_widget.setLayout(grid)

        #Create Horizontal Layout within the main video feed that has the security camera feeds
        TopBar_H_Layout = QHBoxLayout()
        
        TopBar_H_Layout.addWidget(self.focusview_btn, 100,Qt.AlignmentFlag.AlignTop|Qt.AlignmentFlag.AlignLeft)
        TopBar_H_Layout.addWidget(self.feed2_lbl,25,Qt.AlignmentFlag.AlignTop)
        TopBar_H_Layout.addWidget(self.feed3_lbl,25,Qt.AlignmentFlag.AlignTop)
        
        TopBar_H_Layout.insertStretch(0)

        self.feed1_lbl.setLayout(TopBar_H_Layout)
        
        self.updateDisplay_modData()

    def initializeCameraFeedLbl(camera_lbl, Camera_W, Camera_H):
        camera_lbl.resize(Camera_W, Camera_H)
        camera_lbl.setMinimumWidth(Camera_W)
        camera_lbl.setMinimumHeight(Camera_H)


    #On Image updated
    def ImageUpdateSlot(self, Image, lbl):
        lbl.setPixmap(QPixmap.fromImage(Image))
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.onWindowImageUpdated()
        
    def onWindowImageUpdated(self):
        minweight = 0
        
        if(not TESTMODE):
            if(self.stridedata0 is not None):         
                #Get Weight Data from stride 0-20mA
                self.curweightdata = self.stridedata0.analog4/1000  #Convert the input voltage of the moisture content sensor to %
                #Get MTX-V(Moisture) data from stride 0-10V
                self.curmoisturedata = utils.getMoistureContent_MTXV(self, self.stridedata0.analog7/1000) # Convert voltage from milliVolts
                #print(self.curmoisturedata)
                #Find the weight on the load cells
                    
                self.inclodata = utils.getIncloData(self.stridedata0.analog0/1000,self.stridedata0.analog2/1000)
                self.module0.weight = utils.get_weight_lbs(self,self.curweightdata, self.inclodata.angleX, self.inclodata.angleY, 10000)
                self.updateDisplay_modData()
            pass
    
    def OnWindowResize(self, resizeEvent:QResizeEvent):
        self.M_Camera_W, self.M_Camera_H = self.feed1_lbl.width(), self.feed1_lbl.height()
        pass
    def OnCentralWidgetResize(self, resizeEvent:QResizeEvent):
        pass
    def OnMCameraResize(self, resizeEvent:QResizeEvent):
        self.M_Camera_W, self.M_Camera_H = self.feed1_lbl.width(), self.feed1_lbl.height()
    
    def focusMCameraFeed(self):
        print("FOCUSING MAIN CAMERA VIEW")
        if self.mfw is None:
            self.mfw = MainFeed_Window()
            self.mfw.parentWindow = self
            self.mfw.show()
            self.MFeed_infocusmode = True
            if self.isVisible():
                self.hide()
        else:    
            self.mfw.show()
            self.MFeed_infocusmode = True
            if self.isVisible():
                self.hide()

        self.pauseFeeds()
        self.mfw.OnOpen()

        pass

    def playFeeds(self):
        self.camThread1.start()
        self.camThread2.start()
        self.camThread3.start()
        

    def pauseFeeds(self):
        self.camThread1.stop()
        self.camThread2.stop()
        self.camThread3.stop()

    
    def onModuleWindowEdited(self, tempmod):
        self.module0 = tempmod
        self.updateDisplay_modData()


    def addModule(self):
        module_entry = {"Module Name": self.module0.name_identifier, "Module RFID": self.module0.rfid, "Weight": self.module0.weight, "Lint Moisture":self.module0.LM, "Seed Moisture": self.module0.SM}
        modules_dict_list.append(module_entry)
        self.saveEntryToCSV()
        self.mawp = utils.ShowDialogPopup(self, "MODULE ADDED TO DATABASE")
        self.mawp.exec()
        pass
    
    def createMainFeed(self):
        pass

    def updateDisplay_modData(self):
        #Update the module data displayed on the window
        self.weight_lbl.setText(f"Weight = {round(self.module0.weight,1)} lbs")
        self.RFID_lbl.setText(f"Module RFID: {self.module0.rfid} ")
        self.sm_lbl.setText(f"Seed Moisture = {utils.getMoistureContent_MTXV(self.curmoisturedata)[0]}%")
        self.lm_lbl.setText(f"Lint Moisture = {utils.getMoistureContent_MTXV(self.curmoisturedata)[1]}%")
        pass
    
    def initCSV(self):
        with open("ModulesSheet.csv", mode = 'w') as csvfile:
            fields = modules_dict_list[0].keys() #The module dictionary acts as a template for all the fields in the excel sheet
            w = csv.DictWriter(csvfile, delimiter=',', fieldnames=fields)
            w.writeheader()

    def saveEntryToCSV(self):
        i = 0
        with open("ModulesSheet.csv", mode = 'a') as csvfile:
            for module_entry in modules_dict_list:
                if i != 0:
                    fields = modules_dict_list[0].keys()
                    w = csv.DictWriter(csvfile, delimiter=',', fieldnames=fields)
                    w.writerow(module_entry)
                i+=1
        pass
    
    def closeEvent(self, event: QCloseEvent):
        #On window close re-enable main window and do not save changes that were made
        self.mfw = None
        self.emw = None
        if self.fTick_retrieveStrideData is not None:
            self.fTick_retrieveStrideData.stop()
            self.fTick_retrieveStrideData = None
        pass
    
    def keyPressEvent(self, event):
        """
        This method is called when a key is pressed.
        """
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier and \
           event.key() == Qt.Key.Key_Escape:
            #Close the program 
            self.close()
        

    if(not TESTMODE):
        fTick_retrieveStrideData = repeatFunction(0.5,utils.retrieveStrideDATA,stridedata0,stridedata0,c)
        fTick_retrieveStrideData.start()
        

class camThread(QThread):
    src = None
    lbl = None
    isMainFeed = False
    isTracking = False
    tracking_inclo = None

    Window = None #Will be a Qt window object
    ImageUpdate = pyqtSignal(QImage, QLabel)    
    width = 0
    height = 0
    options = {
    "CAP_PROP_FRAME_WIDTH": 1080,
    "CAP_PROP_FRAME_HEIGHT": 720,
    "CAP_PROP_FPS": 30,
    }

    def run(self):
        self.thread_isActive = True
        
        #print(cv2.getBuildInformation())
        CAP_stream = VideoGear(source= self.src, logging=True, **self.options).start()
        
        while self.thread_isActive:
            
            if self.Window is not None:
                CAP_frame = CAP_stream.read()
                
                #height, width, _ = CAP_framme.shape
                #print(f"Input Resolution: {width}x{height}")
                if CAP_frame is not None:
                    #CAP_frame_scaled = self.scaleFrame(CAP_frame, 1)
                    Image = cv2.cvtColor(CAP_frame, cv2.COLOR_BGR2RGB)
                    if self.isMainFeed:
                        if self.isTracking and self.tracking_inclo is not None:
                            if TESTMODE:
                                self.lbl.resize(1920, 1080)
                            Image = self.TrackCamFeed(Image, self.tracking_inclo)
                        pass

                    h, w, ch = Image.shape
                    if not self.isMainFeed and not self.isTracking:
                        pass
                    
                    self.width = w
                    self.height = h
                    #print(w,h)
                    if(type(self.Window) == MainFeed_Window):
                        self.Window.setSize()
                    bytes_per_line = ch * w
                    #ConvertToQtFormat = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0], QImage.Format.Format_RGB888)
                    #FlippedImage = cv2.flip(Image, 1) #flip image on vertical axis

                    ConvertToQtFormat = QImage(Image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                    Pic = None

                    try:
                        Pic = ConvertToQtFormat.scaled(self.lbl.width(), self.lbl.height(), Qt.AspectRatioMode.KeepAspectRatio)
                    except RuntimeError:
                        break
                    else:
                        self.ImageUpdate.emit(Pic, self.lbl)
                        pass
                    # try:
                    # except ValueError:
                    #     print("Value error")
                        

                if cv2.waitKey(1) & 0xFF ==ord('q'):
                    break

        CAP_stream.stop()
        cv2.destroyAllWindows()

    def scaleFrame(self,frame, scale):
        dimension = (int(frame.shape[1]*scale),int(frame.shape[0]*scale))
        return cv2.resize(frame, dimension, interpolation=cv2.INTER_AREA)

    def TrackCamFeed(self, img, inclino):
        crosshair_size = 35
        inclo_angle_X = 0
        inclo_angle_Y = 0

        if self.tracking_inclo is not None:
            inclo_angle_X = self.tracking_inclo.angleX
            inclo_angle_Y = self.tracking_inclo.angleY

        h, w, ch = img.shape
        center_x = w // 2
        center_y = h // 2

        #Draw Crosshair in center of camera feed
        img = cv2.line(img, (int(center_x - (crosshair_size/2)), center_y), (int(center_x + (crosshair_size/2)), center_y), (255,0,0), thickness=3)
        img = cv2.line(img, (center_x, int(center_y - (crosshair_size/2))), (center_x, int(center_y + (crosshair_size/2))), (255,0,0), thickness=3)

        #Add sliders and blend them in with the camera feed
        #LOAD Images
        slider1_img_rgba = cv2.imread(ROOT_DIR / "InclinometerSliderH.png", cv2.IMREAD_UNCHANGED) # Load image with alpha channel
        slider1_img_rgba = cv2.resize(slider1_img_rgba, (0, 0), fx=1, fy=0.95)
        slider1Handle_img_rgba = cv2.imread(ROOT_DIR / "SliderHandleH.png", cv2.IMREAD_UNCHANGED) # Load image with alpha channel
        slider1Handle_img_rgba = cv2.resize(slider1Handle_img_rgba, (0, 0), fx=0.75, fy=0.75)


        slider1_img = slider1_img_rgba[:, :, :3]  # RGB channels
        alpha_mask = slider1_img_rgba[:, :, 3] / 255.0  # Alpha channel, normalized to [0, 1]

        slider1_h, slider1_w, _ = slider1_img.shape
        x, y =  int(center_x-slider1_w/2),int(center_y*0.1)
        # Ensure the overlay fits within the background
        y1, y2 = max(0, y), min(img.shape[0], y + slider1_h)
        x1, x2 = max(0, x), min(img.shape[1], x + slider1_w)

        # Crop images to the overlapping region
        background_crop = img[y1:y2, x1:x2]
        overlay_crop = slider1_img[y1-y:y2-y, x1-x:x2-x]
        alpha_crop = alpha_mask[y1-y:y2-y, x1-x:x2-x, np.newaxis]

        # Blend both images together
        img[y1:y2, x1:x2] = alpha_crop * overlay_crop + (1 - alpha_crop) * background_crop

        #----------------------------Handle1--------------------------------
        slider1Handle_img = slider1Handle_img_rgba[:, :, :3]  # RGB channels
        alpha_mask = slider1Handle_img_rgba[:, :, 3] / 255.0  # Alpha channel, normalized to [0, 1]

        h,w, _ = slider1Handle_img.shape

        x, y =  int(self.IncloAngletoPos(inclo_angle_X,int(center_x-slider1_w/2),int(center_x+slider1_w/2))), int(center_y*0.05)
        x-=math.ceil(w/2) #image must be offset so that the handle appears centered

        # Ensure the overlay fits within the background
        y1, y2 = max(0, y), min(img.shape[0], y + h)
        x1, x2 = max(0, x), min(img.shape[1], x + w)

        # Crop images to the overlapping region
        background_crop = img[y1:y2, x1:x2]
        overlay_crop = slider1Handle_img[y1-y:y2-y, x1-x:x2-x]
        alpha_crop = alpha_mask[y1-y:y2-y, x1-x:x2-x, np.newaxis]

        # Blend both images together
        img[y1:y2, x1:x2] = alpha_crop * overlay_crop + (1 - alpha_crop) * background_crop



        #-------------------Slider2 and Handle 2--------------------
        #LOAD Images
        slider2_img_rgba = cv2.imread(ROOT_DIR / "InclinometerSliderV.png", cv2.IMREAD_UNCHANGED) # Load image with alpha channel
        slider2_img_rgba = cv2.resize(slider2_img_rgba, (0, 0), fx=1, fy=1)
        slider2Handle_img_rgba = cv2.imread(ROOT_DIR / "SliderHandleV1.png", cv2.IMREAD_UNCHANGED) # Load image with alpha channel
        slider2Handle_img_rgba = cv2.resize(slider2Handle_img_rgba, (0, 0), fx=0.75, fy=0.75)


        slider2_img = slider2_img_rgba[:, :, :3]  # RGB channels
        alpha_mask = slider2_img_rgba[:, :, 3] / 255.0  # Alpha channel, normalized to [0, 1]

        slider2_h, slider2_w, _ = slider2_img.shape
        x, y =  int(center_x+center_x*0.9),int(center_y-slider2_h/2)
        # Ensure the overlay fits within the background
        y1, y2 = max(0, y), min(img.shape[0], y + slider2_h)
        x1, x2 = max(0, x), min(img.shape[1], x + slider2_w)

        # Crop images to the overlapping region
        background_crop = img[y1:y2, x1:x2]
        overlay_crop = slider2_img[y1-y:y2-y, x1-x:x2-x]
        alpha_crop = alpha_mask[y1-y:y2-y, x1-x:x2-x, np.newaxis]

        # Blend both images together
        img[y1:y2, x1:x2] = alpha_crop * overlay_crop + (1 - alpha_crop) * background_crop

        #----------------------------Handle2--------------------------------
        slider2Handle_img = slider2Handle_img_rgba[:, :, :3]  # RGB channels
        alpha_mask = slider2Handle_img_rgba[:, :, 3] / 255.0  # Alpha channel, normalized to [0, 1]

        h,w, _ = slider2Handle_img.shape
        x, y =  int(center_x+center_x*0.95), int(self.IncloAngletoPos(inclo_angle_Y,int(center_y-slider2_h/2),int(center_y+slider2_h/2)))
        y-=math.ceil(h/2)
        # Ensure the overlay fits within the background
        y1, y2 = max(0, y), min(img.shape[0], y + h)
        x1, x2 = max(0, x), min(img.shape[1], x + w)

        # Crop images to the overlapping region
        background_crop = img[y1:y2, x1:x2]
        overlay_crop = slider2Handle_img[y1-y:y2-y, x1-x:x2-x]
        alpha_crop = alpha_mask[y1-y:y2-y, x1-x:x2-x, np.newaxis]

        # Blend both images together
        img[y1:y2, x1:x2] = alpha_crop * overlay_crop + (1 - alpha_crop) * background_crop

        #Display numbers which represent the horizontal and vertical angles
        img = utils.opencv_draw_label(img, f"{inclo_angle_X}",(int(center_x-slider1_w/2),int(center_y*0.1)),(0,0,255)) #Horizontally
        img = utils.opencv_draw_label(img, f"{inclo_angle_Y}",(int(center_x+center_x*0.9),int(center_y-slider2_h/2)),(0,0,255)) #Vertically

        return img



    def IncloAngletoPos(self, angle, slider_endpos0, slider_endpos1 ):
        #Range is -45 degrees to 45 degrees
        in_min = -45
        in_max = 45
        out_min = slider_endpos0 #1st end position of the slider
        out_max = slider_endpos1 #2nd end position of the slider
        return (angle - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    def stop(self):
        self.thread_isActive = False
        self.quit()

class MainFeed_Window(QWidget):
    parentWindow = None
    emw = None
    module0 = ModuleData()
    def __init__(self):
        super().__init__()
        self.showFullScreen()
        #self.showMaximized() 
        self.MFeed_infocusmode = True
        self.curinclodata = InclonometerData()
        self.curmoisturedata = (0,0) #Tuple which store the seed and lint moisture content respectively
        self.curweightdata = 0
        self.curstridedata = StrideData()
        #self.setWindowModality(Qt.ApplicationModal)
        self.layout = QVBoxLayout()
        self.btn_layout = QHBoxLayout()

        self.feed_lbl = QLabel()
        self.feed_lbl.minimumWidth = self.width()
        self.feed_lbl.minimumHeight = self.height()
        self.feed_lbl.setStyleSheet("background-color: #000000;")

        self.editModule_btn = QPushButton("EDIT MODULE")
        self.editModule_btn.clicked.connect(self.editModule)
        self.editModule_btn.setStyleSheet("""
        QPushButton {
            background-color: blue; 
            color: white;
            border-radius: 0px;
            padding: 10px 5px;
            font-size: 16px;
            font-weight: 900;                             
        }
        QPushButton:hover {
            background-color: white;
        }
        QPushButton:pressed {
            background-color: black; 
        }
    """)
        
        self.close_btn = QPushButton("RETURN")
        self.close_btn.clicked.connect(self.closeFocusedView)
        self.close_btn.setStyleSheet("""
        QPushButton {
            background-color: orange; 
            color: white;
            border-radius: 0px;
            padding: 10px 5px;
            font-size: 16px;
            font-weight: 900;                             
        }
        QPushButton:hover {
            background-color: white;
        }
        QPushButton:pressed {
            background-color: black; 
        }
    """)
        
        self.layout.addWidget(self.feed_lbl)
        self.btn_layout.addWidget(self.editModule_btn)
        self.btn_layout.addWidget(self.close_btn)
        self.layout.addLayout(self.btn_layout)
        self.setLayout(self.layout)



        self.curinclodata = utils.getIncloData(self.curstridedata.analog0/1000,self.curstridedata.analog2/1000)

        self.camThreadm = camThread()
        self.camThreadm.isMainFeed = True
        self.camThreadm.isTracking = True
        self.camThreadm.tracking_inclo = self.curinclodata
        self.camThreadm.src = M_Camera_IP
        self.camThreadm.lbl = self.feed_lbl
        self.camThreadm.Window = self
        self.camThreadm.ImageUpdate.connect(self.ImageUpdateSlot)
        self.camThreadm.start()

    def OnOpen(self):
        #When the winow is open do:
        self.camThreadm.start()
        pass

    def closeFocusedView(self):
        #Return to the normal view from before
        if self.parentWindow is not None and not self.parentWindow.isVisible():
            self.parentWindow.show()
            self.parentWindow.MFeed_infocusmode = False
            self.parentWindow.playFeeds()
            
            self.parentWindow.showFullScreen()
            self.parentWindow.initUI()

        self.close()
        self.camThreadm.stop()
        
        print("Closed main feed window")
        pass

    def editModule(self):
        self.updateModule()
        if self.emw == None:
            self.emw = EditModule_Window(parentWindow=self)
            self.emw.parentWindow = self
            self.emw.show()
            self.setDisabled(True)
        else:    
            self.emw.show()
            self.setDisabled(True)


    def onModuleWindowEdited(self, tempModule):
        utils.ShowDialogPopup(self, "MODULE ADDED TO DATABASE")
    def setSize(self):
        #self.setFixedSize(self.camThreadm.width, self.camThreadm.height)
        self.feed_lbl.minimumWidth = self.width()
        self.feed_lbl.minimumHeight = self.height()

    def ImageUpdateSlot(self, Image, lbl):
        self.updateModule()
        self.camThreadm.tracking_inclo = self.curinclodata
        #Update the module

        ##Set image of pixmap
        lbl.setPixmap(QPixmap.fromImage(Image))
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def updateModule(self):
        self.curstridedata = self.parentWindow.stridedata0  ##CHANGE THIS TO GETTING THE STRIDE DATA DYNAMICALLY
        if self.curstridedata is not None:
            #Get inclonometer data readings from stride 4mA-20mA X and Y
            self.curinclodata = utils.getIncloData(self.curstridedata.analog0/1000,self.curstridedata.analog2/1000)
            #Get Weight Data from stride 0-20mA
            self.curweightdata =  self.curstridedata.analog4
            #Get MTX-V(Moisture) data from stride 0-10V
            self.curmoisturedata = self.curstridedata.analog7
            self.module0.rfid = "000000"
            self.module0.weight = utils.get_weight_lbs(self,self.curweightdata/1000, self.curstridedata.analog0/1000, self.curstridedata.analog2/1000, 10000)
            self.module0.SM = (utils.getMoistureContent_MTXV(self.curmoisturedata))[0]
            self.module0.LM = (utils.getMoistureContent_MTXV(self.curmoisturedata))[1]

    def keyPressEvent(self, event):
        """
        This method is called when a key is pressed.
        """
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier and \
            event.key() == Qt.Key.Key_Escape:
            #Close the program 
            self.close()


class EditModule_Window(QWidget):
    parentWindow = None
    tempModule = None
    def __init__(self, parentWindow):
        super().__init__()

        self.tempModule = parentWindow.module0
        layout = QVBoxLayout()
        self.data_layout = QGridLayout()
        #ALL DATA HERE NEEDS TO BE CHANGED TO COME FROM THE MODULE CLASS INSTANCE WHICH WAS MADE
        self.modname_lbl = QLabel("Module Identifier: ")
        self.modname_ledit = QLineEdit()
        self.modname_ledit.setPlaceholderText("Enter an identifier for the module")
        self.modrfid_lbl = QLabel("Module RFID: ")
        self.modrfid_ledit = QLineEdit()
        self.modrfid_ledit.setText(f"{self.tempModule.rfid}")
        self.modweight_lbl = QLabel("Module Weight = ")
        self.modweight_ledit = QLineEdit()
        self.modweight_ledit.setText(f"{round(self.tempModule.weight,2)} lbs")
        self.modLM_lbl = QLabel("Lint Moisture = ")
        self.modLM_ledit = QLineEdit()
        self.modLM_ledit.setText(f"{self.tempModule.LM}%")
        self.modSM_lbl = QLabel("Seed Moisture = ")
        self.modSM_ledit = QLineEdit()
        self.modSM_ledit.setText(f"{self.tempModule.SM}%")
        self.setStyleSheet("background-color: #fff;")

        self.done_btn = QPushButton("ADD MODULE TO DATABASE")

        self.done_btn.clicked.connect(self.OnEditModuleCompleted)
        
        #Add widgets to grid layout
        self.data_layout.addWidget(self.modname_lbl,0,0)
        self.data_layout.addWidget(self.modname_ledit,0,1)
        self.data_layout.addWidget(self.modrfid_lbl,1,0)
        self.data_layout.addWidget(self.modrfid_ledit,1,1)
        self.data_layout.addWidget(self.modweight_lbl,2,0)
        self.data_layout.addWidget(self.modweight_ledit,2,1)
        self.data_layout.addWidget(self.modSM_lbl,3,0)
        self.data_layout.addWidget(self.modSM_ledit,3,1)
        self.data_layout.addWidget(self.modLM_lbl,4,0)
        self.data_layout.addWidget(self.modLM_ledit,4,1)

        #vertical layout widgets
        layout.addLayout(self.data_layout)
        layout.addWidget(self.done_btn)
        self.setLayout(layout)

        screen = QApplication.primaryScreen()
        screen_geometry = screen.availableGeometry() # Get available geometry excluding taskbars/docks

        window_width = int(screen_geometry.width() * 0.3)
        window_height = int(screen_geometry.height() * 0.6)

        # Set the window size
        self.resize(window_width, window_height)

        # Optionally, center the window
        center_x = screen_geometry.x() + (screen_geometry.width() - window_width) // 2
        center_y = screen_geometry.y() + (screen_geometry.height() - window_height) // 2
        self.move(center_x, center_y)


    def editCurModule(self):
        if self.tempModule is not None:
            self.tempModule.name_identifier = self.modname_ledit.text()
            self.tempModule.rfid = int(self.modrfid_ledit.text())
            self.tempModule.weight = int(re.sub(r'[^0-9]', '', self.modweight_ledit.text()))
            self.tempModule.LM = self.modLM_ledit.text()
            self.tempModule.SM = self.modSM_ledit.text()

    def closeEvent(self, event: QCloseEvent):
        #On window close re-enable main window and do not save changes that were made
        self.parentWindow.setDisabled(False)
        self.parentWindow.onModuleWindowEdited(self.tempModule)
        pass

    def OnEditModuleCompleted(self):
        #Finish Editing the module, save the class instance
        #close this window, and enable main window
        if self.parentWindow is not None:
            if self.parentWindow.module0 is not None:
                self.tempModule = self.parentWindow.module0
        else:
            pass
        self.editCurModule()
        self.parentWindow.setDisabled(False)
        self.close()
        pass
    pass

    def keyPressEvent(self, event):
        """
        This method is called when a key is pressed.
        """
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier and \
            event.key() == Qt.Key.Key_Escape:
            #Close the program 
            self.close()


def main():
    #Main will be called when the application begins

    
    print(cv2.getBuildInformation())
    
    app = QApplication(sys.argv) #This will allow the possiblility for command line functionality in the future.
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
    pass

if __name__ == "__main__":
    main()











