#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 14:13:21 2015

@author: davidramsay

"""

from audioSample import audioSample
from audioPlayer import audioPlayer
from itertools import izip_longest
import audioExtras as ae
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import eqFilter as eqFilter
import biquad_cookbook as bq
import sys
import copy
import pyaudio
import struct
import ipdb

"""
audioMeasure is for measuring audio devices.  This accepts multiple speakers but assumes one microphone.
It will run through many channels and make measurements of each, or do single channel measurements.

initialize by calling:
a = audioMeasure(np.array([1,1,1]),type="t",Fs=44100)


the accessible data in audioMeasure are:

audioMeasure.output = an audioSample holding the measurement audio to be played out during measurement
audioMeasure.input = an array of audioSamples recordings from the last measurement, one for each channel
audioMeasure.tf = an array of transfer function measurements, for each channel of last measurement
audioMeasure.outInfo is a dictionary with several useful pieces of stored information 
    (like number of repetitions of for test audio, test audio type, etc)
audioMeasure.fs = sampling rate


available class methods are:

audioMeasure.pinkNoise(duration, Fs) - set measurement signal to pink noise of <duration> secs, if no Fs/duration provided it overwrites the current audioMeasure.output with the same Fs and duration.
audioMeasure.pinkNoiseLoop(samples, repetitions, Fs) - set measurement signal to a <samples> long, loopable pink noise test signal that repeats <repetitions> times.  Fs is optional, and will overwrite object default.
audioMeasure.testAllChannels(channels) - give max number of channels, this will step through and test all channels with the 
    stored measurement signal.  It will place them in audioMeasure.input.  Again, this assumes multiple speakers measured at 
    one microphone.  The first speakers measured signal can be found at audioMeasure.input[0], second at input[1], etc.
audioMeasure.calcTF() - step through audio signals stored in input, and using the measurement signal from output, calculate
    and update the tf field.  audioMeasure.tf[0] is the audioSample for the TF of the first speaker
audioMeasure.plotImpulseResp()
audioMeasure.plotFreqResp()


EXPERIMENTAL

audioMeasure.differenceFromEQ(eqShape="flatter", doplot=False) -
audioMeasure.differenceFromSmoothed(doplot=False)
audioMeasure.generateEQfromTF(eqShape="flatter", doplot=False, limits=100)
audioMeasure.processEQ(eqVals, maximum=10, minimum=-10)
audioMeasure.createEQs(eq, doplot=False)
audioMeasure.compareEQFiltToEQ(filt, eq, f)


Typical use:

a = audioMeasure() #create empty object
a.pinkNoiseLoop(repetitions=30) #generate a pink noise burst that loops 30 times and store in audioMeasure.output.
a.testAllChannels() #step through and play the pink noise signal for each channel and record through channel 1 mic. Store in .input
a.calcTF() #calculate TF by dividing input/output and storing in .tf
a.plotFreqResp() #plot freq response of each channel
a.plotImpulseResp() #plot IR of each channel

firstSpeakerTFMeasurement = a.tf[0] #pull out audioSample object for the first channel TF measurement
firstTFMeasurement.toTime() #convert data to time
firstTFtimeData = firstTFMeasurement.data #pull out raw array time domain IR data
firstTFtimestamps = firstTFMeasurement.t() #get array for timestamps of time IR data

firstTFMeasurement.toFreq() #convert data to freq
firstTFfreqData = firstTFMeasurement.data #pull out raw array single-sided freq data
firstTFfreqbins = firstTFMeasurement.f() #get array for freq bins of single-sided freq data

secondSpeakerTFMeasurement = a.tf[1] #... etc

#to get raw copy of measurement signal
a.output.toTime()
a.output.data
xaxis = a.output.t()

#to get raw copy of speaker #1 measurement
a.input[0].toTime()
rawdata = a.input[0].data
xaxis = a.input[0].t()

"""


class audioMeasure(object):

    def __init__(self, dataArray = [], type = "t", Fs = 44100):
        self.output = audioSample(dataArray, type, Fs)  
        self.input = []
        self.tf = [] 
        self.outInfo = {}
        self.fs = Fs
        
        
    def pinkNoise(self, duration=None, Fs=None):
        #overwrite output data with pink noise of size and fs specified
        #defaults to size and fs that are current (0 and 44100 for empty instance)
        if Fs is None:
            Fs=self.fs
            print "Fs is assumed to be " + str(self.fs)
        if (Fs != self.fs):
            self.fs =Fs
            print "WARNING: the provided Fs is *not* a match for this object, changed object to match"
        if duration is None:
            duration = self._tLength / float(Fs)
            print "size of previous output signal used, " + str(duration) + " sec"
        
        size = int(duration * Fs)
        
        B = np.array([ 0.04992203, -0.09599354,  0.0506127 , -0.00440879])
        A = np.array([ 1.        , -2.494956  ,  2.01726588, -0.5221894 ])

        data = np.random.rand(size+5000) - 0.5
        data = signal.filtfilt(B,A,data)
        
        self.output = audioSample(data[5000:], 't', Fs)
        self.outInfo['type'] = 'pink'
        self.outInfo['reps'] = 1
        self.outInfo['lenRepeated'] = size
        

    def pinkNoiseLoop(self, samples=8192, repetitions=10, Fs=None):
        #overwrite output data with pink noise of size and fs specified,
        #repeated a certain number of times.  The pink noise is designed 
        #to be loopable without a break in between.  Make sure your loop
        #duration is long enough to capture the full impulse response.
        if Fs is None:
            Fs=self.fs
            print "Fs is assumed to be " + str(self.fs)
        if (Fs != self.fs):
            self.fs =Fs
            print "WARNING: the provided Fs is *not* a match for this object, changed object to match"
        
        B = np.array([ 0.04992203, -0.09599354,  0.0506127 , -0.00440879])
        A = np.array([ 1.        , -2.494956  ,  2.01726588, -0.5221894 ])

        data = np.random.rand(samples) - 0.5
        data = np.tile(data,10)    
        data = signal.filtfilt(B,A,data)
        data = data[8*samples:9*samples]

        data = np.tile(data, repetitions)
        
        self.output = audioSample(data, 't', Fs)
        self.outInfo['type'] = 'pink'
        self.outInfo['reps'] = repetitions
        self.outInfo['lenRepeated'] = samples
        
        
    def testAllChannels(self, channels=2):
        #using 1 mic only, channel 1 always (out[0])
        self.output.toTime()
        self.input = []
        ap = audioPlayer(self.output.data, channels)
       
        for ind in range(channels):
            out = ap.measureChannel(ind+1)
            self.input.append(audioSample(out[0],'t',self.output.fs))
        
        
    def calcTF(self):
        #calculate transfer function
        
        #output to time, trim off beginning glitches        
        self.output.toTime()
        chunk = self.outInfo['lenRepeated']
        finalInd = int(self.output._tLength/float(chunk))
        self.output.data = self.output.data[chunk*3:chunk*finalInd]
        self.output._tLength = len(self.output.data)
  
        #average in time first, turn to freq
        outs = audioSample(np.mean(self.output.data.reshape(-1, self.outInfo['lenRepeated']), axis=0), 't', self.output.fs)                
        outs.toFreq()
        
        ins = []                        
        #input loop through, trim off beginning, make correct length
        for ind in range(len(self.input)):        
            self.input[ind].toTime()
            self.input[ind].data = self.input[ind].data[chunk*3:chunk*finalInd]
            self.input[ind]._tLength = len(self.input[ind].data)
            
            #average in time first, turn to freq
            ins.append(audioSample(np.mean(self.input[ind].data.reshape(-1, self.outInfo['lenRepeated']), axis=0), 't', self.input[ind].fs))        
            ins[ind].toFreq()
                        
        self.tf = []    
        for ind in range(len(self.input)):
            self.tf.append(audioSample(ins[ind].data/outs.data, 'f', Fs = self.fs))

    
    def plotImpulseResp(self):
        for ind in range(len(self.tf)):
            self.tf[ind].toTime()
            self.tf[ind].plot()
            
            
    def plotFreqResp(self):
        for ind in range(len(self.tf)):
            self.tf[ind].toFreq()
            self.tf[ind].plot()
        
        
    def differenceFromSmoothed(self, doplot=False):
        diff = []        
        for ind in range(len(self.tf)):
            smoothed=self.tf[ind].doubleSmooth(.3,'n')           
            
            if doplot:            
                plt.semilogx(smoothed.f(), smoothed.data)
                self.tf[ind].toDb()
                plt.semilogx(self.tf[ind].f(), self.tf[ind].data)
                plt.show()

            mse = np.mean((smoothed.data.real-self.tf[ind].data.real)**2)            
            diff.append(mse)

        return diff
        
        
    def differenceFromEQ(self, eqShape="flatter", doplot=False):
        diff = []        
        for ind in range(len(self.tf)):
            smoothed=self.tf[ind].doubleSmooth(.33,'n')
            smoothed.normalize()
            eq = np.array(eqFilter.EQTargetAtFreqs(smoothed.f(), eqShape))        
            eqDiff = eq - smoothed.data.real
            eqDiff = np.array(eqDiff - max(eqDiff))            

            eqVals = []            
            freqs = [100, 200, 400, 800, 1600, 3200, 6400, 10000]
            BW = .77
            for freq in freqs:            
                bounds = ae.octaveSpacing(freq, BW)
                eqVals.append(np.mean(eqDiff[np.where((self.tf[ind].f()>=bounds[0]) & (self.tf[ind].f()<=bounds[1]))]))
            eqVals = eqVals - np.mean(eqVals)
            
            if doplot:
                plt.semilogx(smoothed.f(), smoothed.data)
                plt.semilogx(smoothed.f(), eqDiff)
                plt.semilogx(freqs, eqVals, 'ro')
                plt.show()
                
            mse = np.mean((eqVals)**2)            
            diff.append(mse)
            
        return diff
        
            
    def generateEQfromTF(self, eqShape="flatter", doplot=False, limits=100):
        diff = []        
        for ind in range(len(self.tf)):
            smoothed=self.tf[ind].doubleSmooth(.33,'n')
            smoothed.normalize()
            eq = np.array(eqFilter.EQTargetAtFreqs(smoothed.f(), eqShape))        
            eqDiff = eq - smoothed.data.real
            eqDiff = np.array(eqDiff - max(eqDiff)) 
            self.tf[ind].toDb()
            plt.semilogx(smoothed.f(), self.tf[ind].data)            
            plt.semilogx(smoothed.f(), smoothed.data.real)
            plt.semilogx(smoothed.f(), eq)
            plt.semilogx(smoothed.f(), eqDiff)
            plt.show()
            
            eqVals = []            
            freqs = [60, 100, 200, 400, 800, 1600, 3200, 6400, 10000, 16000]
            BW = .77
            for freq in freqs:            
                bounds = ae.octaveSpacing(freq, BW)
                eqVals.append(np.mean(eqDiff[np.where((self.tf[ind].f()>=bounds[0]) & (self.tf[ind].f()<=bounds[1]))]))
            eqVals = eqVals - np.mean(eqVals)
            
            #EXTRA
            eqVals = self.processEQ(eqVals, limits, -1*limits)            
            eqs = self.createEQs(eqVals, doplot=doplot)
            if doplot:
                self.compareEQFiltToEQ(eqs, eqDiff, smoothed.f())
            diff.append(eqs)
            
        return diff
    
    
    def processEQ(self, eqVals, maximum=10, minimum=-10):
        #max and min spread of EQ, normalize so 0dB is maxed
        for val in np.nditer(eqVals, op_flags=['readwrite']):
            if (val > maximum):            
                val[...] = maximum
            if (val < minimum):
                val[...] = minimum
        
        return eqVals
      
      
    def createEQs(self, eq, doplot=False):
        freqs = [60, 100, 200, 400, 800, 1600, 3200, 6400, 10000, 16000]
        BW = 1
        filters = []        
        h = None
        plotFlag = False
        for ind, freq in enumerate(freqs):
            if ind == len(freqs)-1:
                filt = bq.peaking(freq/(self.tf[0].fs/2.0), eq[ind], BW = BW)       
                plotFlag = True
            else:
                filt = bq.peaking(freq/(self.tf[0].fs/2.0), eq[ind], BW = BW)
            
            if doplot:
                h = eqFilter.plotSumFilter(filt[0],filt[1], self.tf[0].fs, h, plotFlag)
            filters.append(filt)
        eqFilter.show()
        return filters
    
    
    def compareEQFiltToEQ(self, filt, eq, f):
        fs = self.tf[0].fs
        
        sumFilt = None
        for filterBA in filt:
            w, h = signal.freqz(filterBA[0],filterBA[1])
            if sumFilt is not None:
                sumFilt = (h * sumFilt)
            else:
                sumFilt = h
                
        plt.title('EQ response with 10-band EQ vs desired')
        plt.semilogx(w*(fs/2.0/3.14), 20 * np.log10(abs(sumFilt)), 'b')
        plt.ylabel('Amplitude [dB]', color='b')
        plt.xlabel('Frequency [rad/sample]')
        plt.semilogx(f,eq)
        plt.grid()
        plt.axis('tight')
        plt.show()
        

# normalize IR, find first sample to hit threshold, rotate array so it's close to front
# envelope, 'relative energy'
# discrete reflection counter   
#phase alignment of bass in two speakers, compare one vs. the other    
   

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return izip_longest(*args, fillvalue=fillvalue)


def floatsToWavBinaryMultichannel(array, chunk, numChannels):
    max_amplitude = 32767.0
    array = array*max_amplitude
    array = array.astype(int)

    binArray = []
    xfinal = np.zeros((chunk*numChannels,),dtype=np.int)
    
    for x in grouper(array, chunk*numChannels, 0):
        binArray.append(struct.pack("%sh" % len(x), *x))

    return binArray     
# EQ difference        
        
        
        
if __name__ == "__main__":
    
    #loopable - test channels, generate tf array,calc EQ diff, calc diff around smoothed, print good/better/best
    if (str(sys.argv[1]) in ['measure', 'Measure', 'm', 'M']):
        test = audioMeasure()
        ae.disablePrint()
        test.pinkNoiseLoop()                
        test.testAllChannels()    
        test.calcTF()
        smoothMSE = test.differenceFromSmoothed()
        eqMSE = test.differenceFromEQ()
        ae.enablePrint()
        print "mse = " + str(smoothMSE) + "; eqmse = " + str(eqMSE)
        scale = np.mean(smoothMSE) + np.mean(eqMSE)        
        print "score = " + str(int(50*(np.mean(smoothMSE) + np.mean(eqMSE))/scale))
        try:
            while True:
                ae.disablePrint()
                test.pinkNoiseLoop()                
                test.testAllChannels()    
                test.calcTF()
                smoothMSE = test.differenceFromSmoothed()
                eqMSE = test.differenceFromEQ()
                ae.enablePrint()
                print "mse = " + str(smoothMSE) + "; eqmse = " + str(eqMSE)
                print "score = " + str(int(50*(np.mean(smoothMSE) + np.mean(eqMSE))/scale))

        except KeyboardInterrupt:
            exit
            
    #measure with long measurement, plot, play audio, play audio EQed
    elif (str(sys.argv[1]) in ['eq', 'EQ', 'e', 'E']):     
        
        test = audioMeasure()
        test.pinkNoiseLoop(repetitions=100)
        test.testAllChannels()    
        test.calcTF()
        test.plotFreqResp()
        test.plotImpulseResp()
        eq = test.generateEQfromTF(limits=100, doplot=True, eqShape="bass")  #eq[channel][band][0] = b, eq[channel][band][1] = a
        
        fs, wav = wavfile.read("./trim2test.wav") #wav[sample][channel]
        left = wav[:, 0]
        left = left / float(np.max(np.abs(left)))
        originalleft = copy.deepcopy(left)
        right = wav[:, 1]
        right = right / float(np.max(np.abs(right)))
        originalright = copy.deepcopy(right)
        for band in range(len(eq[0])):        
            left = signal.filtfilt(eq[0][band][0], eq[0][band][1], left)
            right = signal.filtfilt(eq[1][band][0], eq[1][band][1], right)
        left = left / float(np.max(np.abs(left)))
        right = right / float(np.max(np.abs(right)))
        
        zipped = np.zeros(len(left)+len(right))
        zipped[0::2] = left        
        zipped[1::2] = right

        zippedoriginal = np.zeros(len(originalleft)+len(originalright))
        zippedoriginal[0::2] = originalleft        
        zippedoriginal[1::2] = originalright
        
        half = np.concatenate([zippedoriginal[0:len(zippedoriginal)/2], zipped[len(zipped)/2:]])

        inputVals = floatsToWavBinaryMultichannel(half, 1024, 2)
             
        pa = pyaudio.PyAudio()
        stream = pa.open(format = pyaudio.paInt16,
                 channels = 2,
                 rate = 44100,
                 output = True,
                 frames_per_buffer = 1024)
        
        currentIndex = 0
        
        while currentIndex<len(inputVals):
            stream.write(inputVals[currentIndex])
            currentIndex = currentIndex+1
        
        stream.stop_stream()
        stream.close()
    
    #walk through measurement once with plots
    elif (str(sys.argv[1]) in ['demo', 'Demo', 'd', 'D']):     
       
        test = audioMeasure()
        test.pinkNoiseLoop(repetitions=30)
        test.testAllChannels()    
        test.calcTF()
        test.plotFreqResp()
        test.plotImpulseResp()
        print "mse = " + str(test.differenceFromSmoothed(doplot=True))
        print "eqmse = " + str(test.differenceFromEQ(doplot=True))
