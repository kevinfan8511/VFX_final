import oil_painting.oil_painting2 as op2
import oil_painting.oil_painting as op
import cv2
import math
import numpy as np
import sys
import operator
import os
import matplotlib.pyplot as plt

def video_to_frame(video):
	if not os.path.exists("data"):
		os.mkdir("data")
	vidcap = cv2.VideoCapture(video)
	success, image = vidcap.read()
	count = 0
	success = True
	fps = vidcap.get(cv2.CAP_PROP_FPS)
	print ('fps is ',fps)
	while success:
		cv2.imwrite("./data/frame%d.jpg"%count, image)     # save frame as JPEG file
		success, image = vidcap.read()
		#print 'Read a new frame: ', success
		count += 1
	return count ,fps

def frame_to_video(total_frame, fps):

	#img1 = cv2.imread("./data/frame1.jpg")
	imgs = []
	for i in range(total_frame) :
		img = cv2.imread("./data/frame"+str(i)+'.jpg')
		print('%d / %d'%(i+1, total_frame+1) )
		#op_img = op2.oil_paint(img, [32, 8, 4, 4, 2], 'a')
		op_img = op.oil_paint(img, 5, 30)
		cv2.imwrite("./data/op_frame%d.jpg"%i, op_img)
		imgs.append(op_img)
	height, width, layers = imgs[0].shape
	print('height width layers ',height ,width ,layers)
	#video=cv2.VideoWriter('video2.avi',fourcc,fps,(width,height))
	video = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (width,height))

	for im in imgs:
		video.write(im)
	cv2.destroyAllWindows()
	video.release()
	print ('done')

#python2 test.py images\rail.jpg
def main():
	argv, argc = sys.argv, len(sys.argv)
	name, radius = argv[1], [int(i) for i in argv[2:argc]]
	
	total_frame ,fps = video_to_frame(name)
	if len(radius) == 0:
		radius = [64, 16, 8, 4]
	#show_image(oil_painting)
	frame_to_video(total_frame,fps)
	
	quit()
	
if __name__ == "__main__":
	main()