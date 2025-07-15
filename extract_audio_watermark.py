import wave
import os
import numpy as np
import hashlib
import sys

class ExtractWatermark:

    def getIndexesFromFiles(self, noc):
        fi_flag = 0
        for file in os.listdir(os.getcwd()):
            if file.startswith("frame-indexes") and file.endswith("txt"):
                fif = open('frame-indexes.txt','r')
                fi_flag = 1    # File exists
        if(fi_flag == 0):
            sys.exit("Unable to find a file that indicates the audio frames that have been watermarked! ")
        fi = []
        for line in fif:
            fi.append(line.strip())
        si_flag = 0
        si = []
        for file in os.listdir(os.getcwd()):
            if file.startswith("sample-indexes") and file.endswith("txt"):
                sif = open('sample-indexes.txt','r')
                si_flag = 1
        if(si_flag == 0):
            if(noc != 1):
                sys.exit("Unable to find a file that indicates the audio samples that have been watermarked as audio has multiple channels! ")
            else:
                return fi, si
        for line in sif:
            si.append(line.strip())
        return fi, si

    def getBinary(self, number):
        B = list(bin(number))
        return B

    def getWatermarkedBit(self, curr_sample):
        binary_curr_sample = self.getBinary(curr_sample)
        lsb = binary_curr_sample[-1]
        return lsb

    def getWatermarkFromAudio(self, audio_data, numofchannels):

        watermarked_frames, watermarked_samples = self.getIndexesFromFiles(numofchannels)
        extracted_hash = []

        if(numofchannels == 1):   # Mono
            for i in range(len(watermarked_frames)):
                curr_frame = int(watermarked_frames[i])
                extracted_bit = self.getWatermarkedBit(audio_data[curr_frame])
                extracted_hash.append(extracted_bit)
            return extracted_hash
        else:
            for i in range(len(watermarked_frames)):
                curr_frame = int(watermarked_frames[i])
                curr_sample = int(watermarked_samples[i])
                extracted_bit = self.getWatermarkedBit(audio_data[curr_frame][curr_sample])
                extracted_hash.append(extracted_bit)
            return extracted_hash

if __name__ == '__main__':
    audio = sys.argv[1]
    a= ExtractWatermark()
    a.getWatermark(audio, "frames")