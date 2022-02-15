import numpy as np
import cv2
import pytesseract
from tkinter.colorchooser import askcolor
from tkinter import filedialog as fd
import colorsys

# path to tesseract installation
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Laura\tesseract\tesseract.exe"

def rgb2hsv(*args):
	# RGB input out of (255,255,255)
	# HSV output out of (180,255,255)
	if len(args) == 1:
		r,g,b = [c/255.0 for c in args[0]]
	else:
		r,g,b = [c/255.0 for c in args]
	
	h,s,v = colorsys.rgb_to_hsv(r,g,b)
	return h*180,s*255,v*255

# color chooser for marker color
color = askcolor(title = "Choose the marker color")
chosenC = color[0]

# rgb value ton hsv values
hsvVal = rgb2hsv(chosenC)

# choose image
filename = fd.askopenfilename()
# read input image
image = cv2.imread(filename, cv2.IMREAD_COLOR)
# convert to hsv image
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# color region is chosen due to different shadings in highlights
lower_color = np.array([(hsvVal[0]-25),50,50])
upper_color = np.array([(hsvVal[0]+25),255,255])
# Threshold the HSV image to get only marked areas
mask = cv2.inRange(hsv, lower_color, upper_color)
cv2.imwrite('mask.jpg', mask)
# cut image with mask
img_maskiert = cv2.bitwise_and(image, image, mask= mask)

# choose threshold and search for contours
bw = cv2.cvtColor(img_maskiert, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(bw, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imwrite('thresh.jpg', thresh)
contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)    

# draw contours on the original image
image_copy = image.copy()
cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
cv2.imwrite('output.jpg', image_copy)

# extract text from marked ares and write to txt
with open('Output.txt', mode = 'w') as f:
    data = pytesseract.image_to_string(thresh, lang='deu',config='--psm 6')
    f.write(data)

print(f)
