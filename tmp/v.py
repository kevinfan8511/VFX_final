import cv2
import numpy

def frame_to_video(total_frame,fps):
	img = []
	for i in range(0, total_frame) :
		img.append(cv2.imread("./data/op_frame"+str(i)+'.jpg'))
	img1 = img[0]
	height,width,layers=img1.shape
	#fourcc = cv2.VideoWriter_fourcc(*'XVID')
	print('height width layers ' , height ,width ,layers)
	#video=cv2.VideoWriter('video2.avi',fourcc,fps,(width,height))
	video = cv2.VideoWriter('output2.avi', cv2.VideoWriter_fourcc(*'XVID'), fps/2, (width,height))
	flag = True
	for im in img:
		if flag:
			flag = False
			video.write(im)
		else :
			flag = True
	cv2.destroyAllWindows()
	video.release()
	print ('done')

frame_to_video(257,30)