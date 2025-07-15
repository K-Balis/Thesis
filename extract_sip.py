import math
from PIL import Image
import itertools as it
import numpy as np
from math import log10, sqrt
from time import process_time
import cv2
import sys
import re
import matplotlib.pyplot as plt
from scipy.linalg import dft
from ellipticdisk import createEllipticDisk
import threading, queue
from encodeinteger import encodeInteger
from decodesip import decodeSip
import math
import random
import os

class ExtractPermutation:

	def openImage(self,path):
		img = Image.open(path)
		return img

	def getFFTTransform(self,image):
		dft = np.fft.fft2(image,norm='ortho')
		fftShift = np.fft.fftshift(dft)
		mag = np.abs(fftShift)
		phase = np.angle(fftShift)
		return mag,phase

	def getIFFTTransform(self,mag,phase):
		real = mag * np.cos(phase)
		imag = mag * np.sin(phase)
		complex_output = np.zeros(mag.shape, complex)

		complex_output.real = real
		complex_output.imag = imag
		back_ishift = np.fft.ifftshift(complex_output)
		img_back = np.fft.ifft2(back_ishift,norm='ortho')
		img_back = abs(img_back)
		return img_back 

	def getSipFromFrame(self,im,size,sips_list,sip_size,key,pr,pb,im_format):

		#img = self.openImage(im)
		img = Image.fromarray(im)
		
		w,h = img.size

		M,N = img.size
		channel_array = np.array(img)


		grid_size_w = math.floor((N / size))
		grid_size_h = math.floor((M / size))

		cells = []
		sips = []
		extracted = []
		ex_sips = []

		i = 0
		x = 0
		y = 0
		for r in range(0,N - grid_size_w + 1, grid_size_w):
			for c in range(0,M - grid_size_h + 1, grid_size_h):
				grid_cell = channel_array[r:r + grid_size_w,c:c + grid_size_h]

				g_cell = Image.fromarray(grid_cell)
				sip1,sip2,sip3 = self.getSipFromCell(g_cell,sip_size,pr,pb,im_format)
				sips.append((sip1,sip2,sip3))
				i += 1		
				y += 1
			x += 1
			y = 0

		#print(sips_list)
		not_ex = 0
		counter = 0
		for i in range(len(sips)):
			s1 = sips[i][0]
			s2 = sips[i][1]
			s3 = sips[i][2]
			#print(s1,s2,s3)
			ex_sips.append(("block" + str(i + 1),s1,s2,s3))
			#print(decodeSip(s1),decodeSip(s2),decodeSip(s3),decodeSip(sips_list[i]))
			if(decodeSip(s1) == decodeSip(sips_list[i])):
				counter += 1
				k1 = decodeSip(s1)
				extracted.append(("block" + str(i + 1),k1)) 
				print("Extracted key: ",k1)
			elif(decodeSip(s2) == decodeSip(sips_list[i])):
				counter += 1
				k2 = decodeSip(s2)
				extracted.append(("block" + str(i + 1),k2)) 
				print("Extracted key: ",k2)
			elif(decodeSip(s3) == decodeSip(sips_list[i])):
				counter += 1
				k3 = decodeSip(s3)
				extracted.append(("block" + str(i + 1),k3)) 
				print("Extracted key: ",k3)
			else:
				extracted.append(("block" + str(i + 1),'X'))
				#print(s1,s2,s3)
				#print(decodeSip(s1),decodeSip(s2),decodeSip(s3))
				print("Extracted key: X")
				not_ex += 1

		succ_rate = (1 - (not_ex / (sip_size**2))) * 100
		return counter
		#return extracted,ex_sips,succ_rate

	def getSipFromCell(self,img,SIZE,PR,PB,im_format):
		#img = Image.open(img)
		im1 = np.array(img)
		M,N = img.size

		if(N == M):
			if(im_format == 'jpg'):
				r,g,b = img.split()
			else:
				r = Image.fromarray(im1[:,:,0])
				g = Image.fromarray(im1[:,:,1])
				b = Image.fromarray(im1[:,:,2])

			sip1 = self.extractPermutationFromChannel(r,SIZE,PR,PB)
			sip2 = self.extractPermutationFromChannel(g,SIZE,PR,PB)
			sip3 = self.extractPermutationFromChannel(b,SIZE,PR,PB)
		else:
			if(im_format == 'jpg'):
				r,g,b = img.split()
			else:
				r = Image.fromarray(im1[:,:,0])
				g = Image.fromarray(im1[:,:,1])
				b = Image.fromarray(im1[:,:,2])

			sip1 = self.extractPermutationFromChannel(r,SIZE,PR,PB)
			sip2 = self.extractPermutationFromChannel(g,SIZE,PR,PB)
			sip3 = self.extractPermutationFromChannel(b,SIZE,PR,PB)

		return sip1,sip2,sip3

	def extractPermutationFromChannel(self,channel,SIZE,PR,PB):

		grid_cell_num = 0
		sip_cells = []
		sip = []
		avg = []
		# STEP 2: COMPUTE IMAGE SIZE

		M,N = channel.size
		channel_array = np.array(channel)
		K = int(np.abs((M - N) / 2))
		#print(N,M)

		#print("grid: ",K)
		#channel_array = cv2.resize(channel_array,(200,200))

		# GET THE SIZE OF EACH GRID SHELL

		if(N < M):
			grid_size_w = math.floor(((N) / SIZE))
			grid_size_h = math.floor(((N) / SIZE))
			#print(grid_size_w,grid_size_h)
			RED_WIDTH = PR
			BLUE_WIDTH = PB

			RED_RADIOUS_X = math.floor(((N) / (2 * SIZE)))
			RED_RADIOUS_Y = math.floor(((N) / (2 * SIZE)))

			BLUE_RADIOUS_X = (RED_RADIOUS_X - RED_WIDTH)
			BLUE_RADIOUS_Y = (RED_RADIOUS_Y - RED_WIDTH)
		elif(N > M):
			grid_size_w = math.floor(((M) / SIZE))
			grid_size_h = math.floor(((M) / SIZE))
			#print(grid_size_w,grid_size_h)
			RED_WIDTH = PR
			BLUE_WIDTH = PB

			RED_RADIOUS_X = math.floor(((M) / (2 * SIZE)))
			RED_RADIOUS_Y = math.floor(((M) / (2 * SIZE)))

			BLUE_RADIOUS_X = (RED_RADIOUS_X - RED_WIDTH)
			BLUE_RADIOUS_Y = (RED_RADIOUS_Y - RED_WIDTH)
		else:
			grid_size_w = math.floor(((M) / SIZE))
			grid_size_h = math.floor(((N) / SIZE))
			#print(grid_size_w,grid_size_h)
			RED_WIDTH = PR
			BLUE_WIDTH = PB

			RED_RADIOUS_X = math.floor(((M) / (2 * SIZE)))
			RED_RADIOUS_Y = math.floor(((N) / (2 * SIZE)))

			BLUE_RADIOUS_X = (RED_RADIOUS_X - RED_WIDTH)
			BLUE_RADIOUS_Y = (RED_RADIOUS_Y - RED_WIDTH)

		minAvg = []
		minAll = []

		x = 0
		y = 0
		i = 0
		c = 0
		mag_red_blue = []
		mag_rest = []

		if(N < M):
			for r in range(0,(N) - grid_size_h + 1, grid_size_h):
				for c in range(K,(K+N) - grid_size_h + 1, grid_size_h):
					grid_cell_num += 1
					AVG_RED = 0
					AVG_BLUE = 0
					mag_sum_red = 0
					mag_sum_blue = 0
					D = 0
					
					grid_cell = channel_array[r:r + grid_size_w,c:c + grid_size_h]

					mag,phase = self.getFFTTransform(grid_cell)
					
					#print(grid_cell.shape)
				
					cx = int(grid_cell.shape[0] / 2)
					cy = int(grid_cell.shape[1] / 2)

					red,coord_red = createEllipticDisk(mag,RED_RADIOUS_X,RED_RADIOUS_Y,RED_WIDTH,cx,cy,grid_size_w,grid_size_h)
					blue,coord_blue = createEllipticDisk(mag,BLUE_RADIOUS_X,BLUE_RADIOUS_Y,BLUE_WIDTH,cx,cy,grid_size_w,grid_size_h)
					

					AVG_RED = sum(red) / len(red)
					#print(AVG_RED)
					AVG_BLUE = sum(blue) / len(blue)
					#print(AVG_BLUE)
					avg.append(AVG_RED)

					c += 1

					extract_factor = AVG_BLUE - AVG_RED
					minAvg.append((extract_factor,y))
					#print("local min ",minAvg)

					y += 1
				minAll.append(min(minAvg))
				#print("min all ",minAll)
				x += 1
				y = 0
				del minAvg[:]
			for i in range(len(minAll)):
				sip.append((minAll[i][1] + 1))
			return sip
		
		elif(N > M):

			for r in range(K,(M+K) - grid_size_h + 1, grid_size_h):
				for c in range(0,(M) - grid_size_h + 1, grid_size_h):
					grid_cell_num += 1
					AVG_RED = 0
					AVG_BLUE = 0
					mag_sum_red = 0
					mag_sum_blue = 0
					D = 0
					
					grid_cell = channel_array[r:r + grid_size_w,c:c + grid_size_h]

					mag,phase = self.getFFTTransform(grid_cell)
					
					#print(grid_cell.shape)
				
					cx = int(grid_cell.shape[0] / 2)
					cy = int(grid_cell.shape[1] / 2)

					red,coord_red = createEllipticDisk(mag,RED_RADIOUS_X,RED_RADIOUS_Y,RED_WIDTH,cx,cy,grid_size_w,grid_size_h)
					blue,coord_blue = createEllipticDisk(mag,BLUE_RADIOUS_X,BLUE_RADIOUS_Y,BLUE_WIDTH,cx,cy,grid_size_w,grid_size_h)
					

					AVG_RED = sum(red) / len(red)
					#print(AVG_RED)
					AVG_BLUE = sum(blue) / len(blue)
					#print(AVG_BLUE)
					avg.append(AVG_RED)

					c += 1

					extract_factor = AVG_BLUE - AVG_RED
					minAvg.append((extract_factor,y))
					#print("local min ",minAvg)

					y += 1
				minAll.append(min(minAvg))
				#print("min all ",minAll)
				x += 1
				y = 0
				del minAvg[:]
			for i in range(len(minAll)):
				sip.append((minAll[i][1] + 1))
			return sip
		else:
		
			for r in range(0,N - grid_size_w + 1, grid_size_w):
				for c in range(0,M - grid_size_h + 1, grid_size_h):
					grid_cell_num += 1
					AVG_RED = 0
					AVG_BLUE = 0
					mag_sum_red = 0
					mag_sum_blue = 0
					D = 0
					
					grid_cell = channel_array[r:r + grid_size_w,c:c + grid_size_h]

					mag,phase = self.getFFTTransform(grid_cell)
					
					#print(grid_cell.shape)
				
					cx = int(grid_cell.shape[0] / 2)
					cy = int(grid_cell.shape[1] / 2)

					red,coord_red = createEllipticDisk(mag,RED_RADIOUS_X,RED_RADIOUS_Y,RED_WIDTH,cx,cy,grid_size_w,grid_size_h)
					blue,coord_blue = createEllipticDisk(mag,BLUE_RADIOUS_X,BLUE_RADIOUS_Y,BLUE_WIDTH,cx,cy,grid_size_w,grid_size_h)
					

					AVG_RED = sum(red) / len(red)
					#print(AVG_RED)
					AVG_BLUE = sum(blue) / len(blue)
					#print(AVG_BLUE)
					avg.append(AVG_RED)

					c += 1

					extract_factor = AVG_BLUE - AVG_RED
					minAvg.append((extract_factor,y))
					#print("local min ",minAvg)

					y += 1
				minAll.append(min(minAvg))
				#print("min all ",minAll)
				x += 1
				y = 0
				del minAvg[:]
			for i in range(len(minAll)):
				sip.append((minAll[i][1] + 1))
			return sip

if __name__ == '__main__':
	w = ExtractPermutation()
	path = sys.argv[1]
	block = int(sys.argv[2]) 
	bits = int(sys.argv[3])
	sip = encodeInteger(key)

	sip_cells = []
	A_matrix = []
	for i in range(0,len(sip)):
		row = []
		for j in range(0,len(sip)):
			if(j == sip[i] - 1):
				row.append("*")
				sip_cells.append((i,j))
			else:
				row.append("-")
		A_matrix.append(row)

	size = len(sip)
	#im_name = sys.argv[3]
	c = float(sys.argv[3])
	#SIZE = 8

	for file in os.listdir(os.getcwd()):
		if file.startswith("keys") and file.endswith("txt"):
			fif = open('keys.txt','r')
	keys = []
	for line in fif:
		keys.append(line.strip())
	print(keys)

	sips_list = []
	for i in range(len(keys)):
		SIP = encodeInteger(int(keys[i]))
		sips_list.append(SIP)
	SIP_SIZE = len(sips_list[0])

	img = w.getSipFromFrame(path, block, sips_list, SIP_SIZE, key, 1, 1, "jpg")