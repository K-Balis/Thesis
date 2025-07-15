import sys
import re
import cv2
import os
import shutil
from moviepy.editor import *
from embed_key import EmbedPermutation
from encodeinteger import encodeInteger
from watermark_audio import WatermarkAudio
from extract_audio_watermark import ExtractWatermark
from extract_sip import ExtractPermutation
from matplotlib.image import imread
from moviepy.audio.AudioClip import AudioArrayClip
import numpy as np
from pprint import pprint
from scipy.io.wavfile import write
import wave
from scipy.io import wavfile
import hashlib
import time
import random
from PIL import Image

def validateInput(input1, input2, input3, input4):

	# Key
	try:
		key = int(input1)
	except ValueError:
		sys.exit("KEY should be an intenger!")

	# Video
	if(not os.path.exists(input2)):
		sys.exit("Video File does not Exist!")
	else:
		video = input2
		video_name = re.split(r'\.(?!\d)', video)[0]

	# Command
	command = input3
	if(command != "embed" and command != 'extract'):
		sys.exit("Command should be either 'embed' or 'extract'!")

	# Copt
	if command == "embed":
		try:
			c = float(input4)
		except ValueError:
			sys.exit("COPT should be a float!")
	else:
		c = 0.0

	return key, video, video_name, command, c

def keysToEmbed(size):
	keys = []
	for i in range(size*size):		# size x size grids
		random_key = random.randint(8, 15)
		keys.append(random_key)
	print(keys)
	
	file = open("keys.txt", "w")
	for i in range(len(keys)):
		file.write(str(keys[i]) + '\n')
	return keys

def embedKeyToFrames(video_obj, key, c):

	w = EmbedPermutation()

	size = 8
	sips = keysToEmbed(size)
	
	video_frames = []
	hash_of_frames = hashlib.sha256()
	i = 0
	for time, video_frame in video_obj.iter_frames(with_times = True, fps=video_obj.fps, dtype=np.uint8):
		curr_frame = (video_obj.get_frame(time))
		#print(curr_frame)
		curr_frame_watermarked = w.getWatermarkedFrame(curr_frame, size, sips, c, 1, 1, "None")
		npframe = np.array(curr_frame_watermarked)
		#print(npframe)
		video_frames.append(npframe)
		hash_of_frames.update(npframe)
		print("Frame ", i, "/", int(video_obj.duration*video_obj.fps), " watermarked!")
		i += 1
	print(video_obj.duration)
	print(len(video_frames))
	print(i, hash_of_frames.hexdigest())
	clip = ImageSequenceClip(video_frames, video_obj.reader.fps)

	newclip = clip.subclip(t_start=0, t_end=video_obj.duration)
	print("Duration of subcliped video sequence : ", newclip.duration)
	video_frames = []
	hash_ = hashlib.sha256()
	i = 0
	for time, video_frame in newclip.iter_frames(with_times = True, fps=video_obj.fps, dtype=np.uint8):
		#if(i == 153):
		#	break
		video_frames.append(np.array(newclip.get_frame(time)))
		hash_.update(newclip.get_frame(time))
		i += 1
	print(len(video_frames))
	print(i, hash_.hexdigest())

	sif = open("duration.txt", "w")			# Store original duration so to subclip during the extract process
	sif.write(str(video_obj.duration))
	return newclip, hash_
	#w.getWatermarkedFrame(path, frame_path, key, c, frame_name , 2, 2, frame_format)

def embedHashToAudio(audio_obj, frames_hash):
	
	w = WatermarkAudio()

	numofchannels = audio_obj.nchannels
	audio_obj.write_audiofile("temp.wav", fps=44100)
	samplerate, audio_data = wavfile.read("temp.wav")

	watermarked_audio_data = w.getWatermarkedAudio(audio_data, frames_hash, numofchannels)

	write("WatermarkedAudio.wav", 44100, watermarked_audio_data)

def watermarkVideo(key, video, video_name, c):

	video_clip = VideoFileClip(video)
	audio = video_clip.audio

	image_sequence, frames_hash = embedKeyToFrames(video_clip, key, c)
	embedHashToAudio(audio, frames_hash)

	output_video_name = "Watermarked_" + video_name + ".avi" 

	image_sequence.write_videofile(output_video_name, fps = video_clip.fps, codec = "png", audio = "WatermarkedAudio.wav", audio_fps = 44100, audio_codec = 'raw32')

def extractHashFromFrames(new_obj, duration, key, video_name):
	
	ex = ExtractPermutation()

	SIZE = 8

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

	hash_of = hashlib.sha256()
	i = 0

	extracted = 0
	res_file = open("Results_" + video_name + ".txt", "w")

	for time, video_frame in new_obj.iter_frames(with_times = True, fps=new_obj.fps, dtype=np.uint8):
		curr_frame = new_obj.get_frame(time)
		img = Image.fromarray(curr_frame)
		#img.save(str(i) + "." + "jpg",quality = 100)
		counter = ex.getSipFromFrame(curr_frame, SIZE, sips_list, SIP_SIZE, key, 1, 1, "None")
		res_file.write("Frame " + str(i) + ", Found " + str(counter) + '\n')
		print("Frame : ", i,"Found : ", counter)
		hash_of.update(curr_frame)
		i += 1
	print(i, hash_of.hexdigest())
	print(list(bin(int(hash_of.hexdigest(), 16))[2:].zfill(len(hash_of.hexdigest()) * 4)))


	newclip = new_obj.subclip(t_start=0, t_end=duration)
	print(duration)
	print("# Frames = ", newclip.reader.nframes)
	print('Fps = ', newclip.fps)
	hash_ = hashlib.sha256()
	i = 0
	#print(i, hash_of_frames.hexdigest())
	for time, video_frame in newclip.iter_frames(with_times = True, fps=newclip.fps, dtype=np.uint8):
		hash_.update(newclip.get_frame(time))
		#print(hash_of.hexdigest())
		i += 1
	print(i, hash_.hexdigest())
	bin_hash_of_frames = list(bin(int(hash_.hexdigest(), 16))[2:].zfill(len(hash_.hexdigest()) * 4))
	print(bin_hash_of_frames)
	return bin_hash_of_frames

def extractHashFromAudio(audio_obj):

	e = ExtractWatermark()
	
	numofchannels = audio_obj.nchannels
	audio_obj.write_audiofile("temp.wav", fps=44100)
	samplerate, audio_data = wavfile.read("temp.wav")

	extracted_hash = e.getWatermarkFromAudio(audio_data, numofchannels)
	print("EXT : " ,extracted_hash)
	return extracted_hash

def extractWatermark(key, video, video_name):
	
	video_clip = VideoFileClip(video)

	aaudio = video_clip.audio
	sif = open("duration.txt", "r")
	dur = sif.readline()
	print(dur)
	audio = aaudio.subclip(t_start=0, t_end=dur)

	frames_hash = extractHashFromFrames(video_clip, dur, key, video_name)

	extracted_hash = extractHashFromAudio(audio)

	if frames_hash == extracted_hash:
		print("Hash was extracted succesfully!")
	else:
		print("Hashes do not match!")
	
def deleteTempItems():
	for audio_item in os.listdir(os.getcwd()):
		if audio_item.endswith("wav"):
			os.remove(audio_item)
	print("Temp items removed")

#------------------------------------------------------------------------------------

if __name__ == '__main__':


	if (len(sys.argv) == 5):
		key, video, video_name, command, c = validateInput(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
	elif (len(sys.argv) == 4 and sys.argv[3] == "extract"):
		c_input = 0.0
		key, video, video_name, command, c = validateInput(sys.argv[1], sys.argv[2], sys.argv[3], c_input)
	else:
		sys.exit("Wrong number of arguments!\nYour input should look like this : python Main.py [Key (Integer)] [Video file] [Command (embed or extract)] [Copt (float) (if command embed)]")
		
	
	if(command == "embed"):
		watermarkVideo(key, video, video_name, c)
	
	elif(command == "extract"):
		extractWatermark(key, video, video_name)

	deleteTempItems()