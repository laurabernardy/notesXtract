import numpy as np
import cv2
import pytesseract
from tkinter import filedialog as fd
import tkinter as tk
import difflib
import re
import os

# path to tesseract installation (just needed if used with an other lang package than english)
#pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Laura\tesseract\tesseract.exe"

#convert image to hsv and mark colored frequency for finding hightlighted areas
def to_hsv(image):
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	# color region is chosen due to different shadings in highlights
	lower_color = np.array([0,80,80])
	upper_color = np.array([179,255,255])
	return hsv, lower_color, upper_color, image

# resize image for better results 
def resize(image):
	scale_percent = 220
	width = int(image.shape[1] * scale_percent / 100)
	height = int(image.shape[0] * scale_percent / 100)
	dim = (width, height)
	image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
	return image

#mask input to get only marked areas
def mask(hsv, lower_color, upper_color, image):
	mask = cv2.inRange(hsv, lower_color, upper_color)
	cv2.imwrite('./temp/mask.jpg', mask)
	# cut image with mask
	img_masked = cv2.bitwise_and(image, image, mask= mask)
	return img_masked

#thresh of highlighted areas and contour searching
def img_thresh(img_masked):
	bw = cv2.cvtColor(img_masked, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(bw,(5,5),0)
	_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	cv2.imwrite('./temp/thresh.jpg', thresh)
	contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
	return thresh, contours, hierarchy

#thresh of whole document
def alltxtimg_thresh(imgalltext): # choose threshold and search for contours
	bw = cv2.cvtColor(imgalltext, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(bw,(5,5),0)
	_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_TRUNC+cv2.THRESH_OTSU)
	kernel = np.ones((5,5), np.uint8)
	thresh = cv2.erode(thresh, kernel, iterations=1)
	cv2.imwrite('./temp/threshalltext.jpg', thresh)
	return thresh

# draw contours on the original image
def draw_contours(image, contours):
	image_copy = image.copy()
	min_area = 5000
	for c in contours:
		area = cv2.contourArea(c)
		if area > min_area:
			cv2.drawContours(image,[c], 0, (36,255,12), 2)
	cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
	cv2.imwrite('./temp/output.jpg', image)

#rename new file if file with name already exists
def next(path):
    filename, extension = os.path.splitext(path)
    counter = 1
    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1
    return path

#Write OCR Output of whole document
def writealltxt(thresh):
	outputall = next('./Output/Outputall.txt')
	with open(outputall, mode = 'w', encoding="utf-8") as f:
		data = pytesseract.image_to_string(thresh, lang='eng',config='-c preserve_interword_spaces=1 -c tessedit_char_whitelist="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.:;,-+_!?/\\\"[]{}()<>=*&%$#@!~ " --psm 1')
		f.write(data)
		return outputall

#Write OCR output of highlighted areas
def writeoutput(thresh):
	output = next('./Output/Output.txt')
	with open(output, mode = 'w', encoding="utf-8") as f:
		data = pytesseract.image_to_string(thresh, lang='eng',config='-c preserve_interword_spaces=1 -c tessedit_char_whitelist="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.:;,-+_!?/\\\"[]{}()<>=*&%$#@!~ " --psm 1')
		f.write(data)
		return output

#Build ngrams
def ngrams(input, n):
    input = input.split(' ')
    output = []
    for i in range(len(input)-n+1):
        output.append(input[i:i+n])
    return output

#try to restore the original text for marked areas (and remove fragments) if not ok by using trigrams
#depends on how readable the original text is
#won't remove duplicate words, that follow each other, caused by the matching process (To do!)
#can include rarely words, that are'nt correct, because of the matching process (searches best fit autmatically)
#Information can get lost, when it's less than 80% readable
def errormanage(output, outputall):
	f1 = open(output, "r")
	f2 = open(outputall, "r") 
	f2 = f2.read().replace('\n',' ')
	f2 = re.sub(r'\s+',' ', f2)
	grams2 = ngrams(f2 , 3)
	grams2list = []
	reslist=[]
	for gram2 in grams2:
		strgr2 = ' '.join(gram2)
		grams2list.append(strgr2)
	for line1 in f1:
		line1 = re.sub(r"\s+"," ",line1).strip()
		grams = ngrams(line1, 3)
		if len(grams) == 0:
			grams = [line1.split(' ')]
		for gram in grams:
			strgr = ' '.join(gram)
			result = difflib.get_close_matches(strgr, grams2list, cutoff=0.8)
			if len(result)>0:
				reslist.append(result[0].split())
	#take first trigram
	result = []
	result.append(" ".join(reslist[0]))
	for i in range(len(reslist)-1):
	#if 1. and 2. trigram have 1 or less words in common --> take 2. trigramm
		if len(set(reslist[i])&set(reslist[i+1])) <=1:
			result.append(f"\n {' '.join(reslist[i+1])}")
		else:
	#if 1. word in 2. trigram is last word in 1. trigram --> take 2. and 3. word from 2. trigram
			if reslist[i][2] == reslist[i+1][0]:
				result.append("".join(reslist[i+1][1]),"".join(reslist[i+1][2]))
			else:
	# if 1. word in 2. trigram is 2. word in 1. trigram --> take 3. word from 2. trigram
				if reslist[i+1][0] == reslist[i][1]:
					result.append("".join(reslist[i+1][2]))
				else:
					pass
	print(" ".join(result))
	with open((next('./Output/Output_corr.txt')), mode = 'w', encoding="utf-8") as f:
		f.write(" ".join(result))
	
	f1.close()                                     

# config of tk window
root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True)

# choose image(s)
filenames = fd.askopenfilenames(parent=root, title='Choose a file')

# read input image(s)
for filename in filenames:
	image = cv2.imread(filename, cv2.IMREAD_COLOR)
	image = resize(image)
	outputall = writealltxt(alltxtimg_thresh(image))
	hsv, lower_color, upper_color, image = to_hsv(image)
	img_masked = mask(hsv, lower_color, upper_color, image)
	thresh, contours, hierarchy = img_thresh(img_masked)
	draw_contours(image, contours)
	output = writeoutput(thresh)
	errormanage(output, outputall)

