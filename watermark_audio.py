import random
import os
import hashlib
import wave
import numpy as np
from scipy.io.wavfile import write

class WatermarkAudio:
    
    def __init__(self):
        return

    def getBinaryHash(self, hexHash):
        binary = list(bin(int(hexHash, 16))[2:].zfill(len(hexHash) * 4))
        return binary
    
    def selectFrames(self, frame_size):
        if(frame_size < 199500):
            audio_frame_indexes = list(range(0, frame_size))    
        else:
            audio_frame_indexes = list(range(0, 199500))
        random.shuffle(audio_frame_indexes)
        return audio_frame_indexes[:256]            # Get the random 256 first indexes we will watermark

    def createFrameIndexFile(self, indexes):
        fif = open("frame-indexes.txt", "w")
        for i in range(len(indexes)):
            fif.write(str(indexes[i]) + '\n')
    
    def selectSamples(self, numberofchannels):
        sample_indexes = []
        max_sample_size = numberofchannels
        for i in range(0, 256):
            curr_sample = random.randint(0, max_sample_size - 1)
            sample_indexes.append(curr_sample)
        return sample_indexes

    def createSampleIndexFile(self, indexes):
        sif = open("sample-indexes.txt", "w")
        for i in range(len(indexes)):
            sif.write(str(indexes[i]) + '\n')
    
    def deleteSampleIndexFile(self):                # if mono delete posible previous instances of sample-indexes.txt file
        for file in os.listdir(os.getcwd()):
            if file.startswith("sample-indexes") and file.endswith("txt"):
                os.remove("sample-indexes.txt")
    
    def getBinary(self, number):
        B = list(bin(number))
        return B
    
    def markLSB(self, sample, hash_bit):
        binary_sample = self.getBinary(sample)
        binary_sample[-1] = hash_bit                # "Watermark" Least Significant Bit
        binary_str = ''.join(binary_sample)
        new_sample = int(binary_str, 2)
        return new_sample


    def getWatermarkedAudio(self, audio_data, frames_hash, channels):

        binary_hash = self.getBinaryHash(frames_hash.hexdigest())
        print(frames_hash.hexdigest())
        print(binary_hash)
        numofframes = len(audio_data)

        if(channels == 1):                      # Mono
            frames_to_watermark = self.selectFrames(numofframes)
            self.createFrameIndexFile(frames_to_watermark)
            print("One audio channel (Mono sound) so a frame is simply a single sample. Therefore sample-index.txt is NOT necessary!")
            self.deleteSampleIndexFile()
            for i in range(len(frames_to_watermark)):
                curr_frame = frames_to_watermark[i]
                curr_bit = binary_hash[i]
                new_sample = self.markLSB(audio_data[curr_frame], curr_bit)
                audio_data[curr_frame] = new_sample
            return audio_data
        else:                                   # Stereo
            frames_to_watermark = self.selectFrames(numofframes)
            self.createFrameIndexFile(frames_to_watermark)
            samples_to_watermark = self.selectSamples(channels)
            self.createSampleIndexFile(samples_to_watermark)
            for i in range(len(frames_to_watermark)):
                curr_frame = frames_to_watermark[i]
                curr_sample = samples_to_watermark[i]
                curr_bit = binary_hash[i]
                new_sample = self.markLSB(audio_data[curr_frame][curr_sample], curr_bit)
                audio_data[curr_frame][curr_sample] = new_sample
            return audio_data