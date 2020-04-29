import cv2
import numpy as np

import os
for dirname in os.listdir("."):
	if os.path.isdir(dirname):
		for i, filename in enumerate(os.listdir(dirname)):
			#os.rename(dirname + "/" + filename, dirname + "/" + str(i) + ".png")
			img = cv2.imread(dirname + "/" + filename)
		


			height, width, channels = img.shape
			white = [255,255,255]
			gray = [222,222,222]

			for x in range(0,width):
				for y in range(0,height):
					channels_xy = img[y,x]
					if all(channels_xy == white):
						img[y,x] = gray
					elif all(channels_xy == gray):
						img[y,x] = white

			hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
			


			# # Define lower and uppper limits of what we call "brown"
			# white1=np.array([60,60,255])
			# white2=np.array([20,170,240])

			# # Mask image to only select browns
			# mask=cv2.inRange(hsv,white1,white2)
			# cv2.imshow('mask',mask)
			# cv2.waitKey(0)
			

			# # Change image to red where we found brown
			# img[mask>0]=(0,0,222)

			blur = cv2.blur(img,(5,5))

			cv2.imwrite( dirname + "/" + "A"+ str(i) + ".png",blur)