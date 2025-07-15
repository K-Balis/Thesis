import sys
import re
import cv2
import os
from moviepy.editor import *
from moviepy.audio.AudioClip import AudioArrayClip
from matplotlib.image import imread
import numpy as np
from scipy.io.wavfile import write
import wave
from scipy.io import wavfile
import hashlib
import random
import itertools as it
import math
import time
import matplotlib.pyplot as plt
import imageio
from PIL import Image

class attacks:
    def __init__(self):
        pass
    class noise:
        def __init__(self):
            pass

        def Gaussian(self, frame, option):
            std = float(option)
            w,h = frame.shape[0],frame.shape[1]
            gauss_noise = np.zeros((w,h,3),dtype=np.uint8)
            cv2.randn(gauss_noise,0,std)
            gauss_noise = (gauss_noise*0.5).astype(np.uint8)
            noisy = cv2.add(frame,gauss_noise)
            #noisy_img = Image.fromarray(noisy)
            return noisy

			# Salt and Pepper
        def SaltPepper(self, frame):
            w,h = frame.shape[0],frame.shape[1]
            for i in range(w):
                for j in range(h):
                    random_num_1 = np.random.uniform(low = 0.0, high = 1.0)
                    random_num_2 = np.random.uniform(low = 0.0, high = 1.0)
                    if(random_num_1 < 0.05):
                        frame[i,j] = 255
                    elif(random_num_2 < 0.05):
                        frame[i,j] = 0
                    else:
                        pass
            return frame

    class blur:
        def __init__(self):
            pass

        def GaussianBlur(self, frame, option):
            imblur = cv2.GaussianBlur(frame,(int(option),int(option)),0)
            #imblur = Image.fromarray(imblur)
            return imblur

        def AverageBlur(self, frame, option):
            imblur = cv2.blur(frame,(int(option),int(option)))
            #imblur = Image.fromarray(imblur)
            return imblur

        def MedianBlur(self, frame, option):
            imblur = cv2.medianBlur(frame,int(option))
            #imblur = Image.fromarray(imblur)
            return imblur

    class enhance:
        def __init__(self):
            pass

        def HistogramEQ(self, frame):
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(img)
            y_eq = cv2.equalizeHist(y)
            img_y_cr_cb_eq = cv2.merge((y_eq, cr, cb))
            img_heq = cv2.cvtColor(img_y_cr_cb_eq, cv2.COLOR_YCR_CB2BGR)
            #img_heq = Image.fromarray(img_heq)
            return img_heq

        def Gamma(self, frame, option):
            kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
            image_gamma = cv2.filter2D(frame, -1, float(option))
            #image_gamma = Image.fromarray(image_gamma)
            return image_gamma

    def Compress(self,frame,Q,subs):

        out_im = Image.fromarray(frame)
        out_name = "compressed" + ".jpg"
        out_im.save(out_name,quality = int(Q),subsampling = int(subs))
        im = np.array(Image.open(out_name))
        for item in os.listdir(os.getcwd()):
            if item.startswith(out_name):
                os.remove(item)
        return im
    
    class crop:
        def __init__(self):
            pass
        
        def crop_w(self,frame,direction,per):
            self.dir = direction
            self.per = per
            self.npimage = frame
            self.w, self.h = self.npimage.shape[1], self.npimage.shape[0]

            ''' 
            Left crops with percentages {25,50,75}
            
            '''
            if(self.dir == "left" and self.per == '25'):
                d = int(0.25 * self.w)
                self.npimage[:,0:d] = 0
                return self.npimage
            if(self.dir == "left" and self.per == '50'):
                d = int(0.5 * self.w)
                self.npimage[:,0:d] = 0
                return self.npimage
            if(self.dir == "left" and self.per == '75'):
                d = int(0.75 * self.w)
                self.npimage[:,0:d] = 0
                return self.npimage
                
            ''' 
            Right crops with percentages {25,50,75}
            
            '''
            if(self.dir == "right" and self.per == '25'):
                d = int(0.25 * self.w)
                self.npimage[:,(self.w - d):] = 0
                return self.npimage
            if(self.dir == "right" and self.per == '50'):
                d = int(0.5 * self.w)
                self.npimage[:,(self.w - d):] = 0
                return self.npimage
            if(self.dir == "right" and self.per == '75'):
                d = int(0.75 * self.w)
                self.npimage[:,(self.w - d):] = 0
                return self.npimage
                
            ''' 
            Both sides crops with percentages {25,50,75}
            
            '''
            if(self.dir == "midw" and self.per == '25'):
                d = int(0.125 * self.w)
                self.npimage[:,0:d] = 0
                self.npimage[:,(self.w - d):] = 0
                return self.npimage
            if(self.dir == "midw" and self.per == '50'):
                d = int(0.25 * self.w)
                self.npimage[:,0:d] = 0
                self.npimage[:,(self.w - d):] = 0
                return self.npimage
            if(self.dir == "midw" and self.per == '75'):
                d = int(0.375 * self.w)
                self.npimage[:,0:d] = 0
                self.npimage[:,(self.w - d):] = 0
                return self.npimage
                
        def crop_h(self,frame,direction,per):
            self.dir = direction
            self.per = per
            self.npimage = frame
            self.w, self.h = self.npimage.shape[1], self.npimage.shape[0]
            
            ''' 
            Top crops with percentages {25,50,75}
            
            '''
            if(self.dir == "top" and self.per == '25'):
                d = int(0.25 * self.h)
                self.npimage[0:d,:] = 0
                return self.npimage
            if(self.dir == "top" and self.per == '50'):
                d = int(0.5 * self.h)
                self.npimage[0:d,:] = 0
                return self.npimage
            if(self.dir == "top" and self.per == '75'):
                d = int(0.75 * self.h)
                self.npimage[0:d,:] = 0
                return self.npimage
                
            ''' 
            Bottom crops with percentages {25,50,75}
            
            '''
            if(self.dir == "bottom" and self.per == '25'):
                d = int(0.25 * self.h)
                self.npimage[(self.h - d):,:] = 0
                return self.npimage
            if(self.dir == "bottom" and self.per == '50'):
                d = int(0.5 * self.h)
                self.npimage[(self.h - d):,:] = 0
                return self.npimage
            if(self.dir == "bottom" and self.per == '75'):
                d = int(0.75 * self.h)
                self.npimage[(self.h - d):,:] = 0
                return self.npimage
                
            ''' 
            Both sides crops with percentages {25,50,75}
            
            '''
            if(self.dir == "midh" and self.per == '25'):
                d = int(0.125 * self.h)
                self.npimage[0:d,:] = 0
                self.npimage[(self.h - d):,:] = 0
                return self.npimage
            if(self.dir == "midh" and self.per == '50'):
                d = int(0.25 * self.h)
                self.npimage[0:d,:] = 0
                self.npimage[(self.h - d):,:] = 0
                return self.npimage
            if(self.dir == "midh" and self.per == '75'):
                d = int(0.375 * self.h)
                self.npimage[0:d,:] = 0
                self.npimage[(self.h - d):,:] = 0
                return self.npimage

def select_frames(nof):
    single = random.randint(0, nof)
    total_frames = list(range(0, nof))
    random.shuffle(total_frames)
    ten_percent = int(0.1 * nof)
    return single, total_frames[:ten_percent]

def deleteTempItems():
    for audio_item in os.listdir(os.getcwd()):
        if audio_item.startswith("temp") and audio_item.endswith("wav"):
            os.remove(audio_item)

def attackFrame(frame, att_type, subftype, option1, option2):
    attack = attacks()
    if(att_type == 'Noise'):
        noise_a = attack.noise()
        if(subftype == 'Gaussian'):
            im = noise_a.Gaussian(frame, option1)
            return im
        if(subftype == 'SP'):
            im = noise_a.SaltPepper(frame)
            return im
    if(att_type == 'Blur'):
        blur_a = attack.blur()
        if(subftype == 'Gaussian'):
            im = blur_a.GaussianBlur(frame, option1)
            return im
        if(subftype == 'Average'):
            im = blur_a.AverageBlur(frame, option1)
            return im
        if(subftype == 'Median'):
            im = blur_a.MedianBlur(frame, option1)
            return im
    if(att_type == 'Enhance'):
        enh_a = attack.enhance()
        if(subftype == 'HEQ'):
            im = enh_a.HistogramEQ(frame)
            return im
        if(subftype == 'Gamma'):
            im = enh_a.Gamma(frame, option1)
            return im
    if(att_type == 'Compression'):
        im = attack.Compress(frame, option1, option2)
        return im
    if(att_type == 'Crop'):
        crop_a = attack.crop()
        if(subftype == 'Width'):
            im = crop_a.crop_w(frame, option1, option2)
            return im
        if(subftype == 'Height'):
            im = crop_a.crop_h(frame, option1, option2)
            return im
# HOW TO RUN
# py filter_attacks.py arg1 arg2 arg3 arg4 arg5
# arg1: input video
# arg2: choose between {single (Single frame attack), multiple (10% frames attack)}
# arg3: choose between {Noise,Blur,Enhance,Compression}
# arg4: if you choose Noise: choose between {Gaussian,SP}
#       if you choose Blur: choose between {Gaussian,Average,Median}
#       if you choose Enhance: choose between {HEQ,Gamma}
#        if you choose Compression: set this value to None
# arg5: if you choose Noise: 
# 				if you choose Gaussian arg4 is the std value so you choose a value between 0.001 - 0.5
# 				if you choose SP arg4 must be 0
#       if you choose Blur: 
# 				if you choose Gaussian arg4 is a kernel for example (3x3) so you can a choose an odd value between 3-9 (3,5,7,9)
# 				if you choose Average arg4 is a kernel for example (3x3) so you can a choose an odd value between 3-9 (3,5,7,9)
# 				if you choose Median arg4 is a kernel for example (3x3) so you can a choose an odd value between 3-9 (3,5,7,9)
#       if you choose Enhance: 
# 				if you choose HEQ must be 0 
# 				if you choose Gamma arg4 is a corection value between 0.25-3.0
#       if you choose Compression:
#               quality a value between 0-100
#argv6: Set this value to None if you don't choose Compression. If Compression select subsampling (0,1,2)

if __name__ == '__main__':
    video, att_option, att_type, sub_type, option1, option2 = sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6]
    print("Proscessing started ...")

    sif = open("duration.txt", "r")
    dur = sif.readline()
    
    video_obj = VideoFileClip(video)
    audio = video_obj.audio
    audio_obj = audio.subclip(t_start=0, t_end=dur)
    newclip = video_obj.subclip(t_start=0, t_end=dur)

    num_of_frames = int(float(dur)*video_obj.fps)
    single_frame, ten_percent = select_frames(num_of_frames)
    print(single_frame, ten_percent)
    video_frames = []
    hashbeforecrop = hashlib.sha256()
    hashaftercrop = hashlib.sha256()

    if(att_option == "single"):
        i = 0
        for time, video_frame in newclip.iter_frames(with_times = True, fps=video_obj.fps, dtype=np.uint8):
            curr_frame = np.array(newclip.get_frame(time))
            hashbeforecrop.update(curr_frame)
            if i == single_frame:
                print(i)
                #print(curr_frame)
                tosave = Image.fromarray(curr_frame)
                tosave.save("original"+ ".jpg",quality = 100)
                newframe = attackFrame(curr_frame, att_type, sub_type, option1, option2)
                tosave = Image.fromarray(newframe)
                tosave.save("attacked"+ ".jpg",quality = 100)
                #print(newframe)
                curr_frame = np.array(newframe)
            hashaftercrop.update(curr_frame)
            video_frames.append(curr_frame)
            i += 1
        print(hashbeforecrop.hexdigest(), hashaftercrop.hexdigest())
        print(i)
        clip = ImageSequenceClip(video_frames, video_obj.reader.fps)
        newclip = clip.subclip(t_start=0, t_end=dur)
        audio_obj.write_audiofile("temp.wav", fps=44100)
        out_name = video.split(".")[0] + "_" + att_option + "_" + att_type + "_" + sub_type + "_" + option1 + "_" + option2 + ".avi"
        newclip.write_videofile(out_name, fps = video_obj.fps, codec = "png", audio = "temp.wav", audio_fps = 44100, audio_codec = 'raw32')

    elif(att_option == "multiple"):
        i = 0
        for time, video_frame in newclip.iter_frames(with_times = True, fps=video_obj.fps, dtype=np.uint8):
            curr_frame = np.array(newclip.get_frame(time))
            hashbeforecrop.update(curr_frame)
            if i in ten_percent:
                print(i)
                tosave = Image.fromarray(curr_frame)
                tosave.save("original"+ ".jpg",quality = 100)
                newframe = attackFrame(curr_frame, att_type, sub_type, option1, option2)
                tosave = Image.fromarray(newframe)
                tosave.save("attacked"+ ".jpg",quality = 100)
                curr_frame = np.array(newframe)
            hashaftercrop.update(curr_frame)
            video_frames.append(curr_frame)
            i += 1
        print(hashbeforecrop.hexdigest(), hashaftercrop.hexdigest())
        print(i)
        clip = ImageSequenceClip(video_frames, video_obj.reader.fps)
        newclip = clip.subclip(t_start=0, t_end=dur)
        audio_obj.write_audiofile("temp.wav", fps=44100)
        out_name = video.split(".")[0] + "_" + att_option + "_" + att_type + "_" + sub_type + "_" + option1 + "_" + option2 + ".avi"
        newclip.write_videofile(out_name, fps = video_obj.fps, codec = "png", audio = "temp.wav", audio_fps = 44100, audio_codec = 'raw32')

    print("Proscessing ends!!")
    deleteTempItems()