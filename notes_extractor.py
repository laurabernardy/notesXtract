import numpy as np
import cv2
import pytesseract
from tkinter.colorchooser import askcolor
from tkinter import filedialog as fd
import tkinter as tk
import colorsys
from PIL import Image

# path to tesseract installation
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Laura\tesseract\tesseract.exe"
#
#
#color_explore = np.zeros((150,150,3), np.uint8)  
#color_selected = np.zeros((150,150,3), np.uint8)
#
#
##save selected color RGB in file
#def write_to_file(R,G,B):
#	f = open("saved_color.txt", "a")
#	RGB_color=str(R) + "," + str(G) + "," + str(B) + str("\n")
#	f.write(RGB_color)
#	f.close()
#
#
#
##Mouse Callback function
#def show_color(event,x,y,flags,param): 
#	
#	B=img[y,x][0]
#	G=img[y,x][1]
#	R=img[y,x][2]
#	color_explore [:] = (B,G,R)
#
#	if event == cv2.EVENT_LBUTTONDOWN:
#		color_selected [:] = (B,G,R)
#		print(R,G,B)
#		write_to_file(R,G,B)
#		cv2.destroyAllWindows()
#
##live update color with cursor
#cv2.namedWindow('color_explore')
#cv2.resizeWindow("color_explore", 50,50);
#
##Show selected color when left mouse button pressed
#cv2.namedWindow('color_selected')
#cv2.resizeWindow("color_selected", 50,50);
#
##image window for sample image
#cv2.namedWindow('image')
#
##sample image path
#img_path="D:/Projekte/MarksXtract/MarksXtract/Examples/testitest.jpg"
#
##read sample image
#image=cv2.imread(img_path)
#img = cv2.resize(image, (500, 400))    
#
##mouse call back function declaration
#cv2.setMouseCallback('image',show_color)
#
##while loop to live update
#while (1):
#	
#	cv2.imshow('image',img)
#	cv2.imshow('color_explore',color_explore)
#	cv2.imshow('color_selected',color_selected)
#	if cv2.waitKey(1) & 0xFF == 27:
#		break
#
#cv2.destroyAllWindows()
#
#
#
#def rgb2hsv(*args):
#	# RGB input out of (255,255,255)
#	# HSV output out of (180,255,255)
#	if len(args) == 1:
#		r,g,b = [c/255.0 for c in args[0]]
#	else:
#		r,g,b = [c/255.0 for c in args]
#	
#	h,s,v = colorsys.rgb_to_hsv(r,g,b)
#	return h*180,s*255,v*255


def to_hsv(image):
	# convert to hsv image
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	# color region is chosen due to different shadings in highlights
	lower_color = np.array([0,80,80])
	upper_color = np.array([179,255,255])
	return hsv, lower_color, upper_color, image

def mask(hsv, lower_color, upper_color, image):
	# Mask the HSV image to get only marked areas
	mask = cv2.inRange(hsv, lower_color, upper_color)
	#kernel  = np.ones((5,5), np.uint8)
	#mask = cv2.dilate(mask, kernel, iterations=1)
	cv2.imwrite('mask.jpg', mask)
	# cut image with mask
	img_masked = cv2.bitwise_and(image, image, mask= mask)
	return img_masked

def img_thresh(img_masked): # choose threshold and search for contours
	bw = cv2.cvtColor(img_masked, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(bw,(5,5),0)
	_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	kernel = np.ones((5,5), np.uint8)
	thresh = cv2.erode(thresh, kernel, iterations=1)
	cv2.imwrite('thresh.jpg', thresh)
	contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
	return thresh, contours, hierarchy

def draw_contours(image, contours):
# draw contours on the original image
	image_copy = image.copy()
	min_area = 5000
	for c in contours:
		area = cv2.contourArea(c)
		if area > min_area:
			cv2.drawContours(image,[c], 0, (36,255,12), 2)

	cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
	cv2.imwrite('output.jpg', image)

# color chooser for marker color
root = tk.Tk()
root.withdraw()
#color = askcolor(title = "Choose the marker color")
#chosenC = color[0]
#
## rgb value ton hsv values
#hsvVal = rgb2hsv(chosenC)

# choose image
filename = fd.askopenfilename()
# read input image
image = cv2.imread(filename, cv2.IMREAD_COLOR)

# resize image for better results 
scale_percent = 220 # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)

image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

hsv, lower_color, upper_color, image = to_hsv(image)
img_masked = mask(hsv, lower_color, upper_color, image)
thresh, contours, hierarchy = img_thresh(img_masked)
draw_contours(image, contours)

# extract text from marked areas and write to txt
with open('Output.txt', mode = 'w', encoding="utf-8") as f:
    data = pytesseract.image_to_string(thresh, lang='eng',config='-c preserve_interword_spaces=1 -c tessedit_char_whitelist="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.:;,-+_!?/\\\"[]{}()<>=*&%$#@!~ " --psm 6')
    f.write(data)

print(f)
