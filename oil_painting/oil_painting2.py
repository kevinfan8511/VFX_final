import cv2
import math
import numpy as np
import getopt
import sys
from time import time
import random
import operator
import matplotlib.pyplot as plt
from scipy.ndimage import filters
__debug_level__ = 1
#python2 oil_painting2.py Forest-300x225.jpg 8 4 2
#python2 oil_painting2.py 123.png 8 4 2
filter_name = ''


def convert_and_save(image, name):
	# Convert datatype to 8-bit and save
	image_8bit = np.clip(image*255, 0, 255).astype('uint8')
	cv2.imwrite(name, image_8bit)
	return 
def debug_messege(obj, level = 1):
	if __debug_level__ >= level :
		print(obj)
	return 
def error_messege(obj):
	print('ERR: ' + obj)
	return	
def show_image(img, name = 'image'):
	cv2.imshow(name, img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return 	
def rgb2gray(rgb):
	r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
	gray = r*0.2989 + g*0.5870 + b*0.1140
	return gray
	
def difference(img1, img2):
	shape = img1.shape
	ret = np.zeros((shape[0], shape[1]), np.uint8)
	for y in range(shape[0]):
		for x in range(shape[1]):
			ret[y, x]  = ( np.sum((img1[y, x] - img2[y, x])**2) )**0.5
	return ret

def make_spline_stroke(y0, x0, r, ref, canvas, sobely, sobelx, fc):
	K = []
	height, width= sobely.shape
	max_stroke_length, min_stroke_length = (8, 4) #tmp
	if  x0 < 0 or y0 < 0 or x0 >= width or y0 >= height :
		return K
	stroke_color = ref[y0, x0]
	(y, x) = (y0, x0)
	K.append((y0, x0))
	(lastDy, lastDx) = (0,0)
	for i in range(1, max_stroke_length):
		if i > min_stroke_length and \
		( np.sum((ref[y, x] - canvas[y, x])**2) ) < ( np.sum((ref[y, x] - stroke_color)**2) ):
			return K
		
		(gy, gx) =  sobely[y][x], sobelx[y][x]
		if gy**2 + gx**2 == 0 :
			return K
		(dy, dx) = (gx, -gy)
		if (np.dot(lastDx, dx) + np.dot(lastDy, dy) < 0):
			(dy, dx) = (-dy, -dx)
	
		(dy, dx) = (fc*(dy) + (1-fc)*(lastDy), fc*(dx) + (1-fc)*(lastDx))
		(dy, dx) = (dy, dx) / (( dy**2 + dx**2)**0.5)
		(lastDy, lastDx) = (dy, dx)
		if r <= 2:
			r = 4
		(new_y, new_x) = (y, x)
		m = 1
		while (new_y, new_x) == (y, x):
			(new_y, new_x) = (int(y + (r/2)*m*dy), int(x + (r/2)*m*dx))
			m += 1
		(y, x) = (new_y, new_x)
		if  x < 0 or y < 0 or x >= width or y >= height :
			return K
		K.append((y, x))
	
	return K

def print_layer(canvas, ref, r, threshold = 40, fc = 0.5):
	S = []
	D = difference(canvas, ref)
	img_shape = ref.shape
	gray = rgb2gray(ref)
	filter_name = ''
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = 5)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = 5)
	painted = []

	def get_filter(name, R):
		r = 8
		r2 = r**2
		if name == '':
			r0 = 6
			r02 = r0**2
			filter = np.zeros((2*r, 2*r))
			for i in range(2*r):
				y2 = (i - r)**2 
				for j in range(2*r):
					x2 = (j - r)**2
					if x2+y2 <= r2: # in the circle
						if x2+y2 < r02:
							filter[i, j] = 1	
						else :
							if r == r0:
								filter[i, j] = 1	
							else:
								filter[i, j] = 1-float((x2+y2)**0.5-r0)/(r-r0)
		else:
			filter = cv2.imread(name)
		filter = cv2.resize(filter, (2*R, 2*R), interpolation = cv2.INTER_LINEAR)
		return filter
	def get_values(t, max, radius):
		if t > radius and t < max - radius:
			ret = t - radius, t + radius, radius
		elif t <= radius :
			ret = 0, t + radius, t
		else:
			ret = t-radius, max, radius
		return ret
	for y in range(0, int(img_shape[0]/(r/2))):
			y0 = int((y+0.5)*(r/2))
			y1, y2 , oy = get_values(y0, img_shape[0], r) 
			for x in range(0, int(img_shape[1]/(r/2))):
				x0 = int((x+0.5)*(r/2))
				x1, x2 , ox = get_values(x0, img_shape[1], r) 
				bin = D[y1:y2, x1:x2]
				
				area_error = np.sum(bin) / ( (x2-x1+1)*(y2-y1+1) )
				# print area_error,
				if area_error > threshold:
					(yt, xt) = np.unravel_index(np.argmax(bin, axis=None), bin.shape)
					if (y1+yt, x1+xt) not in painted:
						painted.append((y1+yt, x1+xt))
						s = make_spline_stroke(y1+yt, x1+xt, r, ref, canvas, sobely, sobelx, fc)
						S.append(s)
	#random.shuffle(S)
	rate = 1
	'''
	for s in S:
		filter = get_filter(filter_name, r)
		if len(s) == 0:
			continue
		color = ref[s[0][0]][s[0][1]]
		for (y, x) in s:
			y1, y2, oy = get_values(y, img_shape[0], r) 
			x1, x2, ox = get_values(x, img_shape[1], r) 
			c_bin = canvas[y1:y2, x1:x2, :]
			color = rate*color + (1-rate)*ref[y][x]
			for i in range(c_bin.shape[0]):
				for j in range(c_bin.shape[1]):
					it, jt = i+r-y+y1, j+r-x+x1
					c_bin[i, j] = filter[it, jt]*color + (1-filter[it, jt])*c_bin[i, j]
	'''
	for s in S:
		if len(s) == 0:
			continue
		color = ref[s[0][0]][s[0][1]]
		for (y, x) in s:
			color = rate*color + (1-rate)*ref[y][x]
			cv2.circle(canvas, (x, y), r, (int(color[0]), int(color[1]), int(color[2])), -1)
	
	return canvas
	
def oil_paint(img, radius, style = 'default'):
	'''
	Input: 
	img: raw image (np.array)
	radius: a list of int
	'''
	sorted(radius, reverse=True)
	
	height, width, channel = img.shape
	new_height, new_width = 640, int(640*width/height)
	img = cv2.resize(img, (new_width, new_height), interpolation = cv2.INTER_LINEAR)
	canvas = np.zeros(img.shape)
	str = ''
	
	if style == 'default':	
		threshold, fc = 40, 0.5
	elif style == 'a':
		threshold, fc = 40, 0.9
	elif style == 'b':
		threshold, fc = 100, 1
	
	last_r = -1
	l = len(radius)
	for i in range(l):
		r = radius[i]
		print (r, end = " ") 
		start = time()
		if r != last_r:	
			ref = cv2.blur(img, (r, r))
		canvas = print_layer(canvas, ref, r, threshold, fc)
		last_r = r
		'''
		str += '%d_' % r
		cv2.imwrite(str + '.jpg', canvas)
		'''
		end = time()
		
		print (end - start)
		
	ret = cv2.blur(canvas, (3, 3))
	ret = cv2.resize(ret, (width, height), interpolation = cv2.INTER_LINEAR)
	ret.astype(np.uint8)
	return ret	