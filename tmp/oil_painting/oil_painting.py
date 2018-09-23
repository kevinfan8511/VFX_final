import cv2
import math
import numpy as np
import getopt
import sys
import operator
import concurrent.futures.process 
import matplotlib.pyplot as plt
from scipy.ndimage import filters
__debug_level__ = 1
#python2 oil_painting.py Forest-300x225.jpg 3 20

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
def find_value(bin, o, radius, level_of_intensity):
	'''
	Input:
		bin : a matrix (2r * 2r)
		o: (y, x)
		radius: int
		
	Output:
		the rgb value of the pixel that in the oil painting image 
	'''
	r2 = radius**2
	height, width = bin.shape[0], bin.shape[1]
	intensity_count = [ 0 for i in range(0, 255) ]
	averageRGB = np.zeros((1000, 3))
	for i in range(height):
		y2 = (i - o[0])**2 
		for j in range(width):
			x2 = (j - o[1])**2
			if x2+y2 <= r2: # in the circle
				r, g, b = int(bin[i, j, 0]), int(bin[i, j, 1]), int(bin[i, j, 2])
				curr_intensity = (int)( float(((r+g+b)/3)*level_of_intensity)/255.0 )
				intensity_count[curr_intensity] += 1
				averageRGB[curr_intensity][0] += r
				averageRGB[curr_intensity][1] += g
				averageRGB[curr_intensity][2] += b
	
	max_index, max_value = np.argmax(intensity_count), max(intensity_count)
	finalR = int(averageRGB[max_index][0]/max_value)
	finalG = int(averageRGB[max_index][1]/max_value)
	finalB = int(averageRGB[max_index][2]/max_value)
	#print finalR, finalG, finalB
	return finalR, finalG, finalB
	
def oil_paint(img, radius, level_of_intensity):
	'''
	Input :
		img: the raw image(np.array)
		radius: int
	Output:
		ret: np.array 
	'''
	def get_values(t, max, radius):
		if t > radius and t < max - radius:
			ret = t - radius, t + radius, radius
		elif t < radius :
			ret = 0, t + radius, t
		else:
			ret = t-radius, max, radius
		return ret
	
	height, width, channel = img.shape
	new_height, new_width = 640, int(640*width/height)
	img = cv2.resize(img, (new_width, new_height), interpolation = cv2.INTER_LINEAR)
	ret = np.zeros(img.shape)
	img_shape = img.shape
	for y in range(img_shape[0]):
		# range of the bin(y)
		y1, y2 , oy = get_values(y, img_shape[0], radius) 
		for x in range(img_shape[1]):
			
			x1, x2 , ox = get_values(x, img_shape[1], radius) 
			bin = img[y1:y2, x1:x2, :]
			o = oy, ox
			ret[y, x,:] = find_value(bin, o, radius, level_of_intensity)
	ret = cv2.resize(ret, (width, height), interpolation = cv2.INTER_LINEAR)
	ret.astype(np.uint8)
	return ret

def main():
	argv, argc = sys.argv, len(sys.argv)
	name, radius, level_of_intensity = argv[1], int(argv[2]), int(argv[3])
	img = cv2.imread(name)
	oil_painting = oil_paint(img, radius, level_of_intensity)
	#show_image(oil_painting)
	name_partition = name.partition('.')
	str = '_%d_%d'%(radius, level_of_intensity)
	output_name = name_partition[0]+'_oil_painting'+ str +'.'+name_partition[2]
	cv2.imwrite(output_name, oil_painting)

if __name__ == "__main__":
	main()