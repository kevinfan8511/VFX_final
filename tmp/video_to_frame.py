import os
import cv2
import numpy as np

def video_to_frame(video):
	if not os.path.exists("data2"):
		os.mkdir("data2")
	vidcap = cv2.VideoCapture(video)
	success,image = vidcap.read()
	count = 0
	success = True
	fps = vidcap.get(cv2.CAP_PROP_FPS)
	print ('fps is ' , fps)
	while success:
		cv2.imwrite("./data2/frame%d.jpg" % count, image)     # save frame as JPEG file
		success,image = vidcap.read()
		#print 'Read a new frame: ', success
		count += 1
	return count ,fps

def frame_to_video(total_frame,fps):
	img = []
	for i in range(total_frame) :
		img.append(cv2.imread("./data2/frame"+str(i)+'.jpg'))
	img1 = imgs[0]
	height,width,layers=img1.shape
	#fourcc = cv2.VideoWriter_fourcc(*'XVID')
	print('height width layers ' , height ,width ,layers)
	#video=cv2.VideoWriter('video2.avi',fourcc,fps,(width,height))
	video=cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc(*'XVID'),fps,(width,height))

	for j in range(total_frame):
		video.write(img[j])
	cv2.destroyAllWindows()
	video.release()
	print ('done')


#total_frame ,fps = video_to_frame('room.mp4')
#frame_to_video(total_frame,fps)
frame_to_video(446, 29.976528445288924)

