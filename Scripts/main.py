#This will use the OpenCV library to handle camera input, 1 from the main camera, and 2 from the auxillary safety cameras

#The GUI Library used is pyQT
#IP Cameras are streamed usign RTSP

import sys 
import time
import threading

from pathlib import Path

from scipy.interpolate import interp1d
import numpy as np
import math
import csv

from pymodbus.client.tcp import ModbusTcpClient
import asyncio
import numpy as np
import csv



from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
import cv2
from vidgear.gears import VideoGear
import imutils


M_Camera_IP = "rtsp://admin:ABEFarmData1!@192.168.0.150/media/video1/multicast"
S_Camera1_IP = "rtsp://admin:ABEFarmData1!@192.168.0.151/media/video1/multicast"
S_Camera2_IP = "rtsp://admin:ABEFarmData1!@192.168.0.152/media/video1/multicast"

STRIDE_IP = "192.168.0.131"
STRIDE_PORT = 502

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
    MFeed_infocusmode = False

    c = ModbusTcpClient(host=STRIDE_IP, port=STRIDE_PORT)

    Window_W = 1600
    Window_H = 900

    module0 = ModuleData() #Test Module
    inclodata = InclonometerData()
    stridedata0 = StrideData()
    curmoisturedata = (0,0) #Tuple which store the seed and lint moisture content respectively

    curweightdata = 0

    fTick_retrieveStrideData = None


    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("WINDOW TITLE")
        self.setGeometry(0,0,self.Window_W, self.Window_H)
        self.setFixedSize(self.Window_W, self.Window_H)
        self.initUI()
        self.initCSV()


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

        self.feed1_lbl = QLabel()
        self.feed2_lbl = QLabel(self.feed1_lbl)
        self.feed3_lbl = QLabel(self.feed1_lbl)
        self.cameraHLayout_Spacer = QLabel()
        
        self.feed1_lbl.setMinimumWidth(1280)
        self.feed1_lbl.setMinimumHeight(720)
        self.feed1_lbl.resize(self.M_Camera_W, self.M_Camera_H)

        self.feed2_lbl.setMinimumWidth(200)
        self.feed2_lbl.setMinimumHeight(200)
        self.feed3_lbl.setMinimumWidth(200)
        self.feed3_lbl.setMinimumHeight(200)
        
        self.feed1_lbl.resizeEvent = self.OnMCameraResize
        #Setup Camera MultiThreading
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

        #Define other UI Elements

        self.focusview_btn = QPushButton(text="FOCUS MAIN VIEW", parent=self)
        self.focusview_btn.clicked.connect(self.focusMCameraFeed)
        self.focusview_btn.setMinimumWidth(200)

        self.weight_lbl = QLabel(f"Weight = {self.module0.weight} lbs", self)
        self.weight_lbl.setFont(QFont("Arial", 14))
        self.weight_lbl.setStyleSheet("color: white;" "background-color: green; padding: 10px 5px;") #Use CSS properties to modify text
        self.weight_lbl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom )

        self.RFID_lbl = QLabel("Module RFID: ", self)
        self.RFID_lbl.setFont(QFont("Arial", 12))
        self.RFID_lbl.setStyleSheet("color: white;" "background-color: green; padding: 10px 5px;")
        self.RFID_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)

        self.feed1_lbl.setStyleSheet("background-color: blue;")
        self.feed2_lbl.setStyleSheet("background-color: red;")
        self.feed3_lbl.setStyleSheet("background-color: green;")

        self.feed2_lbl.setMaximumSize(320,240)
        self.feed3_lbl.setMaximumSize(320,240)

        self.pausevideo_btn = QPushButton("Pause Video Feed")
        self.pausevideo_btn.clicked.connect(self.pauseFeeds)
        self.pausevideo_btn.setStyleSheet("color: yellow;" "background-color: blue;")

        self.editModule_btn = QPushButton("EDIT MODULE")
        self.editModule_btn.clicked.connect(self.editModule)

        self.addModule_btn = QPushButton("ADD MODULE")
        self.addModule_btn.clicked.connect(self.addModule)
        

        
        #STYLING UI 

        self.focusview_btn.setStyleSheet("""
        QPushButton {
            background-color: #4CAF50; /* Green */
            color: white;
            border-radius: 0px;
            padding: 0.8rem 1.5rem;
            font-size: 16px;
        }
        QPushButton:hover {
            background-color: #45a049; /* Darker green on hover */
        }
        QPushButton:pressed {
            background-color: #3e8e41; /* Even darker green when pressed */
        }
    """)
        self.editModule_btn.setStyleSheet("""
        QPushButton {
            background-color: #E8B661; 
            color: white;
            padding: 10px 5px;
            font-size: 16px;
        }
        QPushButton:hover {
            background-color: yellow;
        }
        QPushButton:pressed {
            background-color: white; 
        }
    """)
        
        self.addModule_btn.setStyleSheet("""
        QPushButton {
            background-color: blue; /* Green */
            color: white;
            padding: 10px 5px;
            font-size: 16px;
        }
        QPushButton:hover {
            background-color: white; /* Darker green on hover */
        }
        QPushButton:pressed {
            background-color: blue; /* Even darker green when pressed */
        }
    """)
        #Create Grid Layout Manager
        grid = QGridLayout()
        grid.addWidget(self.feed1_lbl, 0,0)
        grid.addWidget(self.weight_lbl, 1,0)
        grid.addWidget(self.RFID_lbl, 1,1)
        grid.addWidget(self.editModule_btn,1,2)
        grid.addWidget(self.addModule_btn, 1,3)
        #grid.addWidget(self.pausevideo_btn, 1,2)
        central_widget.setLayout(grid)

        #Create Horizontal Layout within the main video feed that has the security camera feeds
        H_Layout = QHBoxLayout()
        
        #H_Layout.addWidget(self.cameraHLayout_Spacer)
        H_Layout.addWidget(self.feed2_lbl,25,Qt.AlignmentFlag.AlignTop)
        H_Layout.addWidget(self.feed3_lbl,25,Qt.AlignmentFlag.AlignTop)
        
        H_Layout.insertStretch(0)

        self.feed1_lbl.setLayout(H_Layout)
        
        self.updateDisplay_modData()
    def initializeCameraFeedLbl(camera_lbl, Camera_W, Camera_H):
        camera_lbl.resize(Camera_W, Camera_H)
        camera_lbl.setMinimumWidth(Camera_W)
        camera_lbl.setMinimumHeight(Camera_H)


    #On Image updated
    def ImageUpdateSlot(self, Image, lbl):
        lbl.setPixmap(QPixmap.fromImage(Image))
        self.onWindowImageUpdated()
        pass
        
    def onWindowImageUpdated(self):
        minweight = 0
        
        if(self.stridedata0 is not None):         
            #Get Weight Data from stride 0-20mA
            self.curweightdata = self.stridedata0.analog4/1000  #Convert the input voltage of the moisture content sensor to %
            #Get MTX-V(Moisture) data from stride 0-10V
            self.curmoisturedata = self.getMoistureContent_MTXV(self.stridedata0.analog7/1000) # Convert voltage from milliVolts
            #print(self.curmoisturedata)
            #Find the weight on the load cells
                
            self.inclodata = self.getIncloData(self.stridedata0.analog0/1000,self.stridedata0.analog2/1000)
            self.module0.weight = self.get_weight_lbs(self.curweightdata, self.inclodata.angleX, self.inclodata.angleY, 10000)
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
        self.retrieveStrideDATA(self.stridedata0)

    def pauseFeeds(self):
        self.camThread1.stop()
        self.camThread2.stop()
        self.camThread3.stop()

    def editModule(self):
        if self.emw == None:
            self.emw = EditModule_Window()
            self.emw.show()
            self.setDisabled(True)
        else:    
            self.emw.show()
            self.setDisabled(True)

        self.emw.parentWindow = self
        pass
    
    def onModuleWindowEdited(self, tempmod):
        self.module0 = tempmod
        self.updateDisplay_modData()


    def addModule(self):
        module_entry = {"Module Name": self.module0.name_identifier, "Module RFID": self.module0.rfid, "Weight": self.module0.weight, "Lint Moisture":self.module0.LM, "Seed Moisture": self.module0.SM}
        modules_dict_list.append(module_entry)
        self.saveEntryToCSV()
        self.mawp = Popup("MODULE ADDED TO DATABASE")
        self.mawp.exec()
        pass
    
    def createMainFeed(self):
        pass

    def retrieveStrideDATA(self, stridedata_instance,c):
        
        c.connect()
        stored_registers = []
        n = 8 #number of registers on the STRIDE I/O

        try: 
            r = c.read_holding_registers(address=34,count=15)
            w = c.write_coil(address=161, value=True)
            if r:

                print("RESISTERS FOUND")
                i = 35
                
                for register in r.registers:
                    #Registers 40041 - Analog Input #0(mA)
                    #Registers 40043 - Analog Input #2(mA)
                    #Registers 40045 - Analog Input #4(mA)
                    #Registers 40048 - Analog Input #7(V)

                    #[11997, 11997, 3898, 10628] ---Test/Starting Values
                    match i:
                        case 41:
                            stored_registers.append(register)
                        case 43:
                            stored_registers.append(register)
                        case 45:
                            stored_registers.append(register)
                        case 48:
                            stored_registers.append(register)
                        case _:
                            pass
                    i+=1
                #print(stored_registers)
                if stridedata_instance is not None:
                    if len(stored_registers) >= 4:
                        stridedata_instance.analog0 = stored_registers[0]
                        stridedata_instance.analog2 = stored_registers[1]
                        stridedata_instance.analog4 = stored_registers[2]
                        stridedata_instance.analog7 = stored_registers[3]
                    else:
                        print("Not all data was retrieved from the Stride I/O")

            else:
                print("No data received")

        except Exception as e:
            print(f"Connection or operation failed: {e}")
            self.c.connect()


    def getIncloData(self, stride_angleX_current, stride_angleY_current):
        inclodata = InclonometerData()
        min_angle_current = 4 #mA
        max_angle_current = 20 #mA
        min_angle = -45
        max_angle = 45
        inclodata.angleX = int((stride_angleX_current - min_angle_current) * (max_angle - min_angle) / (max_angle_current-min_angle_current) + min_angle)
        inclodata.angleY = int((stride_angleY_current - min_angle_current) * (max_angle - min_angle) / (max_angle_current-min_angle_current) + min_angle)

        return inclodata
    
    def getMoistureContent_MTXV(self, voltage=0):
        #This function uses known values from the Delmhorst Instrument Co on the moisture content of of Cotton based on the voltage input
        #Data points for the moisture content plot
        v = [11,10.8,10.6,10.4,10.2,10, 8.9,8.2,7.6,6.9,6.4,6,5.5,5,4.7,3.6,2.3,0.8,0]
        smc = [5.5,5.7,5.8,6,6.2,6.1,7.1,7.6,8.4,9.3,10,10.8,11.8,12.8,13.8,17,20,20.1,20.2]
        lmc = [3.7,3.8,3.9,4,4.1,4.2,4.7,5.5,6,6.5,7,7.5,8.1,8.5,8.8,10,12.3,16,16.2]

        #Create an interpreted function from the data 
        f =  interp1d(v,smc, kind = 'cubic')
        f1 = interp1d(v, lmc, kind = 'cubic')

        return math.ceil(f(voltage)*100)/100, math.ceil(f1(voltage)*100)/100

    import math

    def get_weight_lbs(self, mA, tilt_x_deg, tilt_y_deg, scale_capacity_lbs):
        """
        Calculate true weight in pounds, corrected for tilt.
        
        Parameters:
            mA (float): Output from Rice Lake SCT-10 (4.0 to 20.0 mA)
            tilt_x_deg (float): Inclinometer X-axis tilt in degrees (±45)
            tilt_y_deg (float): Inclinometer Y-axis tilt in degrees (±45)
            scale_capacity_lbs (float): Total rated capacity of your scale in pounds
                                        (e.g., 1000 lbs if full scale = 1000 lbs at 20 mA)
        
        Returns:
            float: Corrected weight in pounds
        """
        # Step 1: Convert mA to raw weight (linear 4–20 mA → 0 to full scale)
        if mA < 3.8 or mA > 20.0:
            raise ValueError("mA must be between 4.0 and 20.0")
        
        raw_weight = scale_capacity_lbs * (mA - 4.0) / 16.0

        # Step 2: Convert tilt angles to radians
        theta_x = math.radians(tilt_x_deg)
        theta_y = math.radians(tilt_y_deg)

        # Step 3: Compute gravity projection factor
        # True weight = measured_weight / (cos(x) * cos(y))
        cos_correction = math.cos(theta_x) * math.cos(theta_y)

        if cos_correction < 0.7:  # ~±45° limit
            print(f"Warning: High tilt! cos_factor = {cos_correction:.3f} (accuracy reduced)")

        if cos_correction == 0:
            raise ValueError("Tilt too extreme: cos(θx)*cos(θy) = 0")

        # Step 4: Apply tilt correction
        true_weight = raw_weight / cos_correction

        return true_weight

    def updateDisplay_modData(self):
        #Update the module data displayed on the window
        self.weight_lbl.setText(f"Weight = {self.module0.weight} lbs")
        self.RFID_lbl.setText(f"Module RFID: {self.module0.rfid} ")
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

    fTick_retrieveStrideData = repeatFunction(0.5,retrieveStrideDATA,stridedata0,stridedata0,c)
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
        
        #Attempt to set the video resolution
        '''CAP1.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
        CAP1.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)'''
        '''actual_width = int(CAP.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(CAP.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Actual resolution: {actual_width}x{actual_height}")'''
    
        stream = VideoGear(source=self.src).start() # 0 for default camera
        
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
                            Image = self.TrackCamFeed(Image, self.tracking_inclo)
                        pass
                    h, w, ch = Image.shape
                    self.width = w
                    self.height = h
                    #print(w,h)
                    if(type(self.Window) == MainFeed_Window):
                        self.Window.setSize()
                    bytes_per_line = ch * w
                    #ConvertToQtFormat = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0], QImage.Format.Format_RGB888)
                    #FlippedImage = cv2.flip(Image, 1) #flip image on vertical axis

                    ConvertToQtFormat = QImage(Image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                    Pic = ConvertToQtFormat.scaled(self.lbl.width(), self.lbl.height(), Qt.AspectRatioMode.KeepAspectRatio)

                    self.ImageUpdate.emit(Pic, self.lbl)

                if cv2.waitKey(1) & 0xFF ==ord('q'):
                    break

        CAP_stream.stop()
        cv2.destroyAllWindows()

    def scaleFrame(self,frame, scale):
        dimension = (int(frame.shape[1]*scale),int(frame.shape[0]*scale))
        return cv2.resize(frame, dimension, interpolation=cv2.INTER_AREA)

    def TrackCamFeed(self, img, inclino):
        crosshair_size = 15
        inclo_angle_X = 0
        inclo_angle_Y = 0

        if self.tracking_inclo is not None:
            inclo_angle_X = self.tracking_inclo.angleX
            inclo_angle_Y = self.tracking_inclo.angleY

        h, w, ch = img.shape
        center_x = w // 2
        center_y = h // 2

        #Draw Crosshair in center of camera feed
        img = cv2.line(img, (int(center_x - (crosshair_size/2)), center_y), (int(center_x + (crosshair_size/2)), center_y), (255,0,0), thickness=1)
        img = cv2.line(img, (center_x, int(center_y - (crosshair_size/2))), (center_x, int(center_y + (crosshair_size/2))), (255,0,0), thickness=1)

        #Add sliders and blend them in with the camera feed
        #LOAD Images
        slider1_img_rgba = cv2.imread(ROOT_DIR / "InclinometerSliderH.png", cv2.IMREAD_UNCHANGED) # Load image with alpha channel
        slider1_img_rgba = cv2.resize(slider1_img_rgba, (0, 0), fx=0.75, fy=0.75)
        slider1Handle_img_rgba = cv2.imread(ROOT_DIR / "SliderHandleH.png", cv2.IMREAD_UNCHANGED) # Load image with alpha channel
        slider1Handle_img_rgba = cv2.resize(slider1Handle_img_rgba, (0, 0), fx=0.65, fy=0.65)


        slider1_img = slider1_img_rgba[:, :, :3]  # RGB channels
        alpha_mask = slider1_img_rgba[:, :, 3] / 255.0  # Alpha channel, normalized to [0, 1]

        slider1_h, slider1_w, _ = slider1_img.shape
        x, y =  int(center_x-slider1_w/2),int(center_y*0.01)
        print(int(center_y*0.01))
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
        x, y =  int(self.IncloAngletoPos(inclo_angle_X,int(center_x-slider1_w/2),int(center_x+slider1_w/2))), int(center_y*0.01)
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
        slider2_img_rgba = cv2.resize(slider2_img_rgba, (0, 0), fx=0.75, fy=0.75)
        slider2Handle_img_rgba = cv2.imread(ROOT_DIR / "SliderHandleV.png", cv2.IMREAD_UNCHANGED) # Load image with alpha channel
        slider2Handle_img_rgba = cv2.resize(slider2Handle_img_rgba, (0, 0), fx=0.65, fy=0.65)


        slider2_img = slider2_img_rgba[:, :, :3]  # RGB channels
        alpha_mask = slider2_img_rgba[:, :, 3] / 255.0  # Alpha channel, normalized to [0, 1]

        slider2_h, slider2_w, _ = slider2_img.shape
        x, y =  int(center_x+center_x*0.95),int(center_y-slider2_h/2)
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
        x, y =  int(center_x+center_x*0.9), int(self.IncloAngletoPos(inclo_angle_Y,int(center_y-slider2_h/2),int(center_y+slider2_h/2)))
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
    
    def __init__(self):
        super().__init__()
        #self.showFullScreen()
        self.showMaximized() 
        self.MFeed_infocusmode = True
        self.curinclodata = InclonometerData()
        self.curmoisturedata = (0,0) #Tuple which store the seed and lint moisture content respectively
        self.curweightdata = 0
        self.curstridedata = StrideData()
        #self.setWindowModality(Qt.ApplicationModal)
        layout = QVBoxLayout()
        self.feed_lbl = QLabel()
        self.feed_lbl.minimumWidth = self.width()
        self.feed_lbl.minimumHeight = self.height()
        self.feed_lbl.setStyleSheet("background-color: #000000;")
        self.close_btn = QPushButton("RETURN")
        self.close_btn.clicked.connect(self.closeFocusedView)
        layout.addWidget(self.feed_lbl)
        layout.addWidget(self.close_btn)
        self.setLayout(layout)

        #self.curstridedata = self.parentWindow.stridedata0
        self.curinclodata = self.getIncloData(self.curstridedata.analog0/1000,self.curstridedata.analog2/1000)

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

        self.close()
        self.camThreadm.stop()
        
        print("Closed main feed window")
        pass
    
    def setSize(self):
        #self.setFixedSize(self.camThreadm.width, self.camThreadm.height)
        self.feed_lbl.minimumWidth = self.width()
        self.feed_lbl.minimumHeight = self.height()

    def ImageUpdateSlot(self, Image, lbl):
        self.updateStrideData()
        self.camThreadm.tracking_inclo = self.curinclodata
        lbl.setPixmap(QPixmap.fromImage(Image))
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def updateStrideData(self):
        self.curstridedata = self.parentWindow.stridedata0
        if self.curstridedata is not None:
            #Get inclonometer data readings from stride 0-20mA X and Y
            self.curinclodata = self.getIncloData(self.curstridedata.analog0/1000,self.curstridedata.analog2/1000)
            print(self.curinclodata.angleX,self.curinclodata.angleY)
            #Get Weight Data from stride 0-20mA
            self.curweightdata =  self.curstridedata.analog4
            #Get MTX-V(Moisture) data from stride 0-10V
            self.curmoisturedata = self.curstridedata.analog7


    def getIncloData(self, stride_angleX_current, stride_angleY_current):
        inclodata = InclonometerData()
        min_angle_current = 4 #mA
        max_angle_current = 20 #mA
        min_angle = -45
        max_angle = 45
        inclodata.angleX = int((stride_angleX_current - min_angle_current) * (max_angle - min_angle) / (max_angle_current-min_angle_current) + min_angle)
        inclodata.angleY = int((stride_angleY_current - min_angle_current) * (max_angle - min_angle) / (max_angle_current-min_angle_current) + min_angle)

        return inclodata

class EditModule_Window(QWidget):
    parentWindow = None
    tempModule = None
    def __init__(self):
        super().__init__()
        
        layout = QVBoxLayout()
        self.data_layout = QGridLayout()
        #ALL DATA HERE NEEDS TO BE CHANGED TO COME FROM THE MODULE CLASS INSTANCE WHICH WAS MADE
        self.modname_lbl = QLabel("Module Identifier: ")
        self.modname_ledit = QLineEdit()
        self.modname_ledit.setPlaceholderText("Enter an identifier for the module")
        self.modrfid_lbl = QLabel("Module RFID: ")
        self.modrfid_ledit = QLineEdit()
        self.modrfid_ledit.setPlaceholderText("00000")
        self.modweight_lbl = QLabel("Module Weight = ")
        self.modweight_ledit = QLineEdit()
        self.modweight_ledit.setPlaceholderText("0 lbs")
        self.modLM_lbl = QLabel("Lint Moisture = ")
        self.modLM_ledit = QLineEdit()
        self.modLM_ledit.setPlaceholderText("0")
        self.modSM_lbl = QLabel("Seed Moisture = ")
        self.modSM_ledit = QLineEdit()
        self.modSM_ledit.setPlaceholderText("0")
        self.setStyleSheet("background-color: #45a049;")

        self.done_btn = QPushButton("DONE")

        self.done_btn.clicked.connect(self.OnEditModuleCompleted)
        
        #Add widgets to grid layout
        self.data_layout.addWidget(self.modname_lbl,0,0)
        self.data_layout.addWidget(self.modname_ledit,0,1)
        self.data_layout.addWidget(self.modrfid_lbl,1,0)
        self.data_layout.addWidget(self.modrfid_ledit,1,1)
        self.data_layout.addWidget(self.modweight_lbl,2,0)
        self.data_layout.addWidget(self.modweight_ledit,2,1)
        self.data_layout.addWidget(self.modLM_lbl,3,0)
        self.data_layout.addWidget(self.modLM_ledit,3,1)
        self.data_layout.addWidget(self.modSM_lbl,4,0)
        self.data_layout.addWidget(self.modSM_ledit,4,1)

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
            self.tempModule.weight = int(self.modweight_ledit.text())
            self.tempModule.LM = self.modLM_ledit.text()
            self.tempModule.SM = self.modSM_ledit.text()

    def retrieveStrideDATA(self, stridedata_instance):
        
        c = ModbusTcpClient(host=STRIDE_IP, port=STRIDE_PORT)
        stored_registers = []
        n = 8 #number of registers on the STRIDE I/O 
        r = c.read_holding_registers(address=34,count=15)
        if r:
            print("RESISTERS FOUND")
            i = 35
            
            for register in r.registers:
                #Registers 40041 - Analog Input #0(mA)
                #Registers 40043 - Analog Input #2(mA)
                #Registers 40045 - Analog Input #4(mA)
                #Registers 40048 - Analog Input #7(V)

                #[11997, 11997, 3898, 10628] ---Test/Starting Values
                match i:
                    case 41:
                        stored_registers.append(register)
                    case 43:
                        stored_registers.append(register)
                    case 45:
                        stored_registers.append(register)
                    case 48:
                        stored_registers.append(register)
                    case _:
                        pass
                i+=1
            print(stored_registers)
            if stridedata_instance is not None:
                if len(stored_registers) >= 4:
                    stridedata_instance.analog0 = stored_registers[0]
                    stridedata_instance.analog2 = stored_registers[1]
                    stridedata_instance.analog4 = stored_registers[2]
                    stridedata_instance.analog7 = stored_registers[3]
                else:
                    print("Not all data was retrieved from the Stride I/O")
        self.c.close()
        print("Modbus connection closed.")



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
                print("MODULE FOUND")
                self.tempModule = self.parentWindow.module0
        else:
            pass
        self.editCurModule()
        self.parentWindow.setDisabled(False)
        self.close()
        pass
    pass

class Popup(QDialog):
    def __init__(self, message, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MESSAGE")
        self.layout = QVBoxLayout()
        self.label = QLabel(message)
        self.layout.addWidget(self.label)
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept) # Close dialog on button click
        self.layout.addWidget(self.close_button)
        self.setLayout(self.layout)




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











