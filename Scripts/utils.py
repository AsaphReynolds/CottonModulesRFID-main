from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *


import math 
from scipy.interpolate import interp1d

import cv2

class InclonometerData:
    angleX=0
    angleY=0

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


def getIncloData(stride_angleX_current, stride_angleY_current):
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


##WEIGHT EQUATION CORRECTION WITH ANGLE OF TILT TAKEN INTO CONSIDERATION 
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


def ShowDialogPopup(self, title = "",message="", width = 300, height = 100):
    dialog = ModalDialog(self, title, message, width, height)
    dialog.exec()



class ModalDialog(QDialog):
    def __init__(self, parent, title, message, width, height):
        super().__init__(parent)
        self.setModal(True)
        self.setWindowTitle(title)

        layout = QVBoxLayout()
        label = QLabel(message)
        label.setFont(QFont("Arial", 20))
        self.setMinimumSize(width, height)
        layout.addWidget(label)
        self.setLayout(layout)

    def keyPressEvent(self, event):
        # This method will be called when a key is pressed while the dialog has focus
        """
        This method is called when a key is pressed.
        """
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier and \
            event.key() == Qt.Key.Key_Escape:
            #Close the program 
            self.close()


        

def opencv_draw_label(img, text, pos, bg_color):
   temp_img = img
   font_face = cv2.FONT_HERSHEY_TRIPLEX
   scale = 1
   color = (0, 0, 0)
   thickness = cv2.FILLED
   margin = 2
   txt_size = cv2.getTextSize(text, font_face, scale, thickness)

   end_x = pos[0] + txt_size[0][0] + margin
   end_y = pos[1] - txt_size[0][1] - margin

   cv2.rectangle(temp_img, pos, (end_x, end_y), bg_color, thickness)
   cv2.putText(temp_img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)
   return temp_img



class hoverLayout:

    pass

class clickedLayoutWidget(QWidget):
    def __init__(self, parent = None):
        super().__init__(parent)

        self.setMouseTracking(True)

        def mousePressEvent(self, event):
            global_pos = event.globalPos()
            local_pos = self.mapFromGlobal(global_pos)

            if(self.rect().contains(local_pos)):
                print(f"Layout Area Pressed: {local_pos.x}")

            super.mousePressEvent(event)

        
    pass


class clickableLabel(QLabel):
    clicked = pyqtSignal()
    resizeOnClicked = False
    enlarged = False
    minH = 0
    minW = 0
    setsize = False
    def __init__(self, parent = None, resize = False):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.resizeOnClicked = resize
        

    def mousePressEvent(self, event):
        # global_pos = event.globalPos()
        # local_pos = self.mapFromGlobal(global_pos)

        # if(self.rect().contains(local_pos)):
        #     print(f"Layout Area Pressed: {local_pos.x}")

        if event.button() == Qt.MouseButton.LeftButton:
            if(self.resizeOnClicked==True):
                if(not self.setsize):
                    self.minH = self.minimumHeight()
                    self.minW = self.minimumWidth()
                    self.setsize = True
                if not self.enlarged:
                    self.setMinimumWidth(self.minW*3)
                    self.setMinimumHeight(self.minH*3)
                    self.enlarged = True
                else:
                    self.setMinimumWidth(self.minW)
                    self.setMinimumHeight(self.minH)   
                    self.resize(self.minW,self.minH)     
                    self.enlarged = False               
        super().mousePressEvent(event)

        
    pass


def tag_report_cb(reader,tag_reports):  
    for tag in tag_reports:
        print('tag: %r' % tag)
    try:
    # Block forever or until a disconnection of the reader
        reader.join(None)
    except (KeyboardInterrupt, SystemExit):
    # catch ctrl-C and stop inventory before disconnecting
        reader.disconnect()