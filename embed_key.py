import cv2
import random
import sys
import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import dft
from ellipticdisk import createEllipticDisk
from encodeinteger import encodeInteger
from math import sqrt
from PIL import Image
import itertools as it
import threading, queue
import math
from functools import reduce 
from matplotlib import pyplot as plt
import time
import os

class EmbedPermutation:

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
	
	def sortCells(self,mc,nonm,sip_cells):
		cell_list = []
		k = 0
		f = 0
		for i in range(len(sip_cells)):
			for j in range(len(sip_cells)):
				if(k < len(sip_cells) and sip_cells[k][0] == i and sip_cells[k][1] == j):
					cell_list.append(mc[k])
					k += 1
				else:
					cell_list.append(nonm[f])
					f += 1
		return cell_list

	def mergeCellsToFrame(self,cells,w,h,N,im_format):
		if(im_format == 'jpg'):
			display = np.empty(((w)*N, (h)*N , 3), dtype = np.uint8)
			for i, j in it.product(range(N), range(N)):
				arr = np.array(cells[i*N+j])
				x,y = i*(w), j*(h)
				display[x : x + (w), y : y + (h)] = arr

		else:
			display = np.empty(((w)*N, (h)*N , 3), dtype = np.uint8)
			for i, j in it.product(range(N), range(N)):
				arr = np.array(cells[i*N+j])
				x,y = i*(w), j*(h)
				display[x : x + (w), y : y + (h)] = arr			
		
		return display
	
	def mergeCells(self,m,unm,w,h,sip_cells):
		sorted_cells = self.sortCells(m,unm,sip_cells)
		N = len(sip_cells)
		display = np.empty(((w)*N, (h)*N), dtype=np.uint8)

		for i, j in it.product(range(N), range(N)):
			arr = sorted_cells[i*N+j]
			
			x,y = i*(w), j*(h)
			display[x : x + (w), y : y + (h)] = arr
		
		return display
	
	def watermarkedCell(self,im,sip,SIZE,copt,PR,PB,im_format):
		#img = Image.open(im)
		im1 = np.array(im)
		M,N = im.size

		if(N == M):
			outim_w = math.floor((M / SIZE))
			outim_h = math.floor((N / SIZE))

			new_w = outim_w * SIZE
			new_h = outim_h * SIZE

			if(im_format == 'jpg'):
				r,g,b = im.split()
			else:
				r = Image.fromarray(im1[:,:,0])
				g = Image.fromarray(im1[:,:,1])
				b = Image.fromarray(im1[:,:,2])


			r_img,k = self.embedPermutationToChannel(r,sip,SIZE,copt,PR,PB)
			g_img,k = self.embedPermutationToChannel(g,sip,SIZE,copt,PR,PB)
			b_img,k = self.embedPermutationToChannel(b,sip,SIZE,copt,PR,PB)

			red = Image.fromarray(r_img)
			green = Image.fromarray(g_img)
			blue = Image.fromarray(b_img)

			rgb = Image.merge('RGB', (red,green,blue))
			rgb = cv2.resize(np.array(rgb),(N,M),interpolation = cv2.INTER_AREA)
			rgb = Image.fromarray(rgb)
			#rgb.save("dumb/watermarked_" + name + "." + im_format,quality = 100)

			return rgb

		else:
			#print(M,N)
			outim_w = math.floor((M / SIZE))
			outim_h = math.floor((N / SIZE))

			#print(outim_w,outim_h)

			new_w = outim_w * SIZE
			new_h = outim_h * SIZE

			if(im_format == 'jpg'):
				r,g,b = im.split()
			else:
				r = Image.fromarray(im1[:,:,0])
				g = Image.fromarray(im1[:,:,1])
				b = Image.fromarray(im1[:,:,2])


			r_img,k = self.embedPermutationToChannel(r,sip,SIZE,copt,PR,PB)
			g_img,k = self.embedPermutationToChannel(g,sip,SIZE,copt,PR,PB)
			b_img,k = self.embedPermutationToChannel(b,sip,SIZE,copt,PR,PB)

			
			if(r_img.shape[0] != N or r_img.shape[1] != N): r_img = cv2.resize(r_img, dsize=(N,N), interpolation=cv2.INTER_CUBIC); #print("Resized block shape: ",r_img.shape)
			if(g_img.shape[0] != N or g_img.shape[1] != N): g_img = cv2.resize(g_img, dsize=(N,N), interpolation=cv2.INTER_CUBIC); #print("Resized block shape: ",g_img.shape)
			if(b_img.shape[0] != N or b_img.shape[1] != N): b_img = cv2.resize(b_img, dsize=(N,N), interpolation=cv2.INTER_CUBIC); #print("Resized block shape: ",b_img.shape)

			im1[0:N,k:(k+N),0] = r_img
			im1[0:N,k:(k+N),1] = g_img
			im1[0:N,k:(k+N),2] = b_img


			rgb = Image.fromarray(im1)
			#rgb.save("dumb/watermarked_" + name + "." + im_format,quality = 100)

			return rgb

	def getWatermarkedFrame(self,im,size,sips,COPT,pr,pb,im_format):

		img = Image.fromarray(im)
		#img = self.openImage(im)
		
		x = 0
		y = 0

		w,h = img.size
		M,N = img.size
		channel_array = np.array(img)
		grid_size_w = math.floor((N / size))
		grid_size_h = math.floor((M / size))
		cells = []

		i = 0
		for r in range(0,N - grid_size_w + 1, grid_size_w):
			for c in range(0,M - grid_size_h + 1, grid_size_h):
				grid_cell = channel_array[r:r + grid_size_w,c:c + grid_size_h]
				SIP = encodeInteger(int(sips[i]))
				SIP_SIZE = len(SIP)
				#if(i < len(sip_cells) and sip_cells[i][0] == x and sip_cells[i][1] == y):
				g_cell = Image.fromarray(grid_cell)
				#print("Current SiP block: ",i+1," in position: ",(x+1,y+1))
				w_im = self.watermarkedCell(g_cell,SIP,SIP_SIZE,COPT,pr,pb,im_format)
				cells.append(w_im)
				i += 1
				#else:
				#	cells.append(Image.fromarray(grid_cell))

				y += 1

			x += 1
			y = 0

		img = self.mergeCellsToFrame(cells,grid_size_w,grid_size_h,size,im_format)
		img = cv2.resize(np.array(img),(w,h),interpolation = cv2.INTER_AREA)
		img = Image.fromarray(img)
		#img.save("1" + "." + "jpg",quality = 100)
		#exit()
		return img
	
	def modifyMagnitude(self,mag_marked,mag_rest,D_array,sip_cells,gw,gh,copt):
		modified_cells = []
		unmodified_cells = []
		sip_hashes = []

		for i in range(len(mag_rest)):

			MAGNITUDE = mag_rest[i][0]
			PHASE = mag_rest[i][1]
			original_cell = self.getIFFTTransform(MAGNITUDE,PHASE)
			unmodified_cells.append(original_cell)
			

		for i in range(len(mag_marked)):
			marked_cells = 0
			MAGNITUDE = mag_marked[i][0]
			PHASE = mag_marked[i][1]
			RED_AN = mag_marked[i][2]
			BLUE_AN = mag_marked[i][3]
			AVG_R = mag_marked[i][4]
			AVG_B = mag_marked[i][5]
			COORD_R = mag_marked[i][6]
			COORD_B = mag_marked[i][7]
			a = np.amax(MAGNITUDE)
			for j in range(gw):
				for k in range(gh):
					for t in range(len(COORD_R)):
						if(j == COORD_R[t][0] and k == COORD_R[t][1]):
							#print("mag ",MAGNITUDE[j,k])
							val = (AVG_B - AVG_R) + (D_array[i] + copt)
							MAGNITUDE[j,k] += val# change magnitude of red region cells
							marked_cells += 1
			original_cell = self.getIFFTTransform(MAGNITUDE,PHASE)
			#sip_hashes.append(self.hash_block(original_cell))
			modified_cells.append(original_cell)
			
		return modified_cells,unmodified_cells,marked_cells

	def embedPermutationToChannel(self,channel,sip,SIZE,copt,PR,PB):
		grid_cell_num = 0
		sip_cells = []

		# STEP 1: FIND 2D REPRESANTAION OF SIP

		A_matrix = []
		for i in range(0,len(sip)):
			row = []
			for j in range(0,SIZE):
				if(j == sip[i] - 1):
					row.append("*")
					sip_cells.append((i,j))
				else:
					row.append("-")
			A_matrix.append(row)


		# STEP 2: COMPUTE IMAGE SIZE

		M,N = channel.size
		#print(M,N)
		channel_array = np.array(channel)
		K = int(np.abs((M - N) / 2))

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

		# FOR EACH GRID CELL COMPUTE FFT MAGNITUDE AND PHASE:
		# ALSO COMPUTE IMAGINARY RED AND BLUE ANULUS RADIOUSES:


		MaxDRows = []
		MaxD = []
		avgR  = []
		avgB = []
		avg = []
		x = 0
		y = 0
		i = 0
		d = 0
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
					d +=1

					grid_cell = channel_array[r:r + grid_size_w,c:c + grid_size_h]
					#print(grid_cell.shape,d,r,c)
					mag,phase = self.getFFTTransform(grid_cell)
					
					
					cx = int(grid_cell.shape[0] / 2)
					cy = int(grid_cell.shape[1] / 2)

					
					red,coord_red = createEllipticDisk(mag,RED_RADIOUS_X,RED_RADIOUS_Y,RED_WIDTH,cx,cy,grid_size_w,grid_size_h)
					blue,coord_blue = createEllipticDisk(mag,BLUE_RADIOUS_X,BLUE_RADIOUS_Y,BLUE_WIDTH,cx,cy,grid_size_w,grid_size_h)
					

					AVG_RED = sum(red) / len(red)
					AVG_BLUE = sum(blue) / len(blue)
					avg.append(AVG_RED)

					if(AVG_BLUE <= AVG_RED):
						D = abs(AVG_BLUE - AVG_RED)
					else:
						D = 0

					MaxD.append(D)

					if(i < len(sip_cells) and sip_cells[i][0] == x and sip_cells[i][1] == y): # for every marked sip cell take the magnitude,red,blue
						mag_red_blue.append((mag,phase,red,blue,AVG_RED,AVG_BLUE,coord_red,coord_blue,x,y))

						i += 1
					else:
						mag_rest.append((mag,phase,red,blue,AVG_RED,AVG_BLUE,coord_red,coord_blue))

					y += 1
				MaxDRows.append(max(MaxD))
				del MaxD[:]
				x += 1
				y = 0
			
			m,unm,m_cells = self.modifyMagnitude(mag_red_blue,mag_rest,MaxDRows,sip_cells,grid_size_w,grid_size_h,copt)
			img = self.mergeCells(m,unm,grid_size_w,grid_size_h,sip_cells)
			#print("percentage of modified cells per channel: ",m_cells/(grid_size_w * grid_size_h))
			return img,K

		elif(N > M):

			for r in range(K,(M+K) - grid_size_h + 1, grid_size_h):
				for c in range(0,(M) - grid_size_h + 1, grid_size_h):
					grid_cell_num += 1
					AVG_RED = 0
					AVG_BLUE = 0
					mag_sum_red = 0
					mag_sum_blue = 0
					D = 0
					d +=1

					grid_cell = channel_array[r:r + grid_size_w,c:c + grid_size_h]
					#print(grid_cell.shape,d,r,c)
					mag,phase = self.getFFTTransform(grid_cell)
					
					
					cx = int(grid_cell.shape[0] / 2)
					cy = int(grid_cell.shape[1] / 2)

					
					red,coord_red = createEllipticDisk(mag,RED_RADIOUS_X,RED_RADIOUS_Y,RED_WIDTH,cx,cy,grid_size_w,grid_size_h)
					blue,coord_blue = createEllipticDisk(mag,BLUE_RADIOUS_X,BLUE_RADIOUS_Y,BLUE_WIDTH,cx,cy,grid_size_w,grid_size_h)
					

					AVG_RED = sum(red) / len(red)
					AVG_BLUE = sum(blue) / len(blue)
					avg.append(AVG_RED)

					if(AVG_BLUE <= AVG_RED):
						D = abs(AVG_BLUE - AVG_RED)
					else:
						D = 0

					MaxD.append(D)

					if(i < len(sip_cells) and sip_cells[i][0] == x and sip_cells[i][1] == y): # for every marked sip cell take the magnitude,red,blue
						mag_red_blue.append((mag,phase,red,blue,AVG_RED,AVG_BLUE,coord_red,coord_blue,x,y))

						i += 1
					else:
						mag_rest.append((mag,phase,red,blue,AVG_RED,AVG_BLUE,coord_red,coord_blue))

					y += 1
				MaxDRows.append(max(MaxD))
				del MaxD[:]
				x += 1
				y = 0
			
			m,unm,m_cells = self.modifyMagnitude(mag_red_blue,mag_rest,MaxDRows,sip_cells,grid_size_w,grid_size_h,copt)
			img = self.mergeCells(m,unm,grid_size_w,grid_size_h,sip_cells)
			#print("percentage of modified cells per channel: ",m_cells/(grid_size_w * grid_size_h))
			return img,K
		else:
			for r in range(0,N - grid_size_w + 1, grid_size_w):
				for c in range(0,M - grid_size_h + 1, grid_size_h):
					grid_cell_num += 1
					AVG_RED = 0
					AVG_BLUE = 0
					mag_sum_red = 0
					mag_sum_blue = 0
					D = 0
					c +=1

					grid_cell = channel_array[r:r + grid_size_w,c:c + grid_size_h]
					mag,phase = self.getFFTTransform(grid_cell)
					
					cx = int(grid_cell.shape[0] / 2)
					cy = int(grid_cell.shape[1] / 2)

					
					red,coord_red = createEllipticDisk(mag,RED_RADIOUS_X,RED_RADIOUS_Y,RED_WIDTH,cx,cy,grid_size_w,grid_size_h)
					blue,coord_blue = createEllipticDisk(mag,BLUE_RADIOUS_X,BLUE_RADIOUS_Y,BLUE_WIDTH,cx,cy,grid_size_w,grid_size_h)
					

					AVG_RED = sum(red) / len(red)
					AVG_BLUE = sum(blue) / len(blue)
					avg.append(AVG_RED)

					if(AVG_BLUE <= AVG_RED):
						D = abs(AVG_BLUE - AVG_RED)
					else:
						D = 0

					MaxD.append(D)

					if(i < len(sip_cells) and sip_cells[i][0] == x and sip_cells[i][1] == y): # for every marked sip cell take the magnitude,red,blue
						mag_red_blue.append((mag,phase,red,blue,AVG_RED,AVG_BLUE,coord_red,coord_blue,x,y))
						i += 1
					else:
						mag_rest.append((mag,phase,red,blue,AVG_RED,AVG_BLUE,coord_red,coord_blue))

					y += 1
				MaxDRows.append(max(MaxD))
				del MaxD[:]
				x += 1
				y = 0
			
			m,unm,m_cells = self.modifyMagnitude(mag_red_blue,mag_rest,MaxDRows,sip_cells,grid_size_w,grid_size_h,copt)
			img = self.mergeCells(m,unm,grid_size_w,grid_size_h,sip_cells)
			#print("percentage of modified cells per channel: ",m_cells/(grid_size_w * grid_size_h))

		return img,0

if __name__ == '__main__':
	w = EmbedPermutation()
	path = sys.argv[1]
	block = int(sys.argv[2]) 
	bits = int(sys.argv[3])
	c = float(sys.argv[4])
	size = block
	print(size, pow(2, bits - 1), pow(2, bits) - 1)
	keys = []
	for i in range(size*size):		# size x size grids
		random_key = random.randint(pow(2, bits - 1), pow(2, bits) - 1)
		keys.append(random_key)
	print(keys)
	
	'''file = open("keys.txt", "w")
	for i in range(len(keys)):
		file.write(str(keys[i]) + '\n')'''
	
	img = w.getWatermarkedFrame(path,size,keys,c,1,1, "jpg")
	img.save("W_" + "." + "jpg",quality = 100)