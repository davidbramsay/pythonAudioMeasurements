# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 16:58:59 2015

@author: davidramsay
"""

import pyaudio
import numpy as np
from itertools import izip_longest
#import ipdb
import struct
import time

"""
audioPlayer plays and plays/records simultaneously with default audio card, on any channel.

user always interacts with a float np array between [-1,1]


initialize by calling:
a = audioPlayer(audio=np.array([0,0,0]), channels=1, chunk=1024, Fs=44100)
-this sets up the audio interface (channel number, chunk size, and sample rate)
-this also sets the default audio to be played


available class methods are:

audioPlayer.setAudio(audio) - sets the default audio to be played to the passed np array
audioPlayer.normalize() - normalizes the default audio to [1, -1]

[The following two methods:
  -play either default audio (audioToPlay=None) or the passed audioToPlay.
  -passed audioToPlay is set as the default audio
  -normalize to full scale [-1,1]  (normalizeTestSignal=1), or doesn't alter the audio if set to 0.
  -plays on all channels at once (if channel=0), or an individual specified channel (channel=1,2,3...)
  -pass/return all data as [-1,1] floats]

audioPlayer.playAudio(channel=0, audioToPlay=None, normalizeTestSignal=1) - plays audio.
audioPlayer.measureChannel(channel=0, audioToPlay=None, normalizeTestSignal=1) - measures an audio channel.
  -returns an array of arrays (as long as the # of channels).  Each array has the microphone signal
  associated with that input channel. output[0] is the first microphone, output[1] is the second, etc.


Potential to add/change:

-fix normalizeTestSignal to be a True/False flag
-play audio on combinations of channels
-right now, audioToPlay replaces the default audio.  perhaps (probably) it shouldn't.
-methods to handle checking audio cards and printing available, as well as selecting new ones
-more support for other audio formats potentially (everything in paInt16 right now)

"""

class audioPlayer(object):

    def __init__(self, audio=None, channels=1, chunk=1024, Fs=44100):
        self._pa = pyaudio.PyAudio()
        self._channels = channels
        self._chunk = chunk
        self._fs = Fs
        self._format = pyaudio.paInt16
        self._audio = audio


    def setAudio(self, audio):
        self._audio = audio


    def normalize(self):
        #normalize the input audio to max output
        self._audio = self._audio/max(abs(self._audio))


    def measureChannel(self, channel=0, audioToPlay=None, normalizeTestSignal=0):
        #return float arrays from microphone channels, channel 0 will play on all channels
        if audioToPlay is not None:
            self._audio = audioToPlay

        if normalizeTestSignal:
            self.normalize()

        global input
        global output
        global currentIndex
        global exit_flag

        if (channel==0):
            input = self.floatsToWavBinaryAllChannels(self._audio, self._chunk, self._channels)
        else:
            input = self.floatsToWavBinary(self._audio, self._chunk, channel, self._channels)

        output = []
        currentIndex = 0
        exit_flag = False

        def callback(in_data, frame_count, time_info, flag):
            global input
            global output
            global currentIndex
            global exit_flag

            if flag:
                print("Playback Error: %i" % flag)

            output.append(in_data)
            currentIndex = currentIndex+1

            if (currentIndex==len(input)):
                return input[currentIndex-1], pyaudio.paComplete

            return input[currentIndex-1], pyaudio.paContinue


        stream = self._pa.open(format = self._format,
                 channels = self._channels,
                 rate = self._fs,
                 input = True,
                 output = True,
                 frames_per_buffer = self._chunk,
                 stream_callback = callback)

        while stream.is_active():
            time.sleep(0.1)

        stream.close()

        outputFloat = np.zeros(self._channels*self._chunk*len(output))
        counter = 0

        for in_data in output:
            in_data_len = (len(in_data) // 2)
            fmt = "<%dh" % in_data_len
            outputFloat[counter:counter+in_data_len] = struct.unpack(fmt, in_data)
            counter=counter+in_data_len


        outputFloat = self.int16ToFloat(outputFloat)

        finalOut = []
        for ind in range(self._channels):
            finalOut.append(outputFloat[ind::self._channels])

        return finalOut


    def playAudio(self, channel=0, audioToPlay=None, normalizeTestSignal=0):
        #channel 0 will play audio on all channels.
        if audioToPlay is not None:
            self._audio = audioToPlay

        if normalizeTestSignal:
            self.normalize()

        if (channel==0):
            input = self.floatsToWavBinaryAllChannels(self._audio, self._chunk, self._channels)
        else:
            input = self.floatsToWavBinary(self._audio, self._chunk, channel, self._channels)

        stream = self._pa.open(format = self._format,
                 channels = self._channels,
                 rate = self._fs,
                 output = True,
                 frames_per_buffer = self._chunk)

        currentIndex = 0

        while currentIndex<len(input):
            stream.write(input[currentIndex])
            currentIndex = currentIndex+1

        stream.stop_stream()
        stream.close()


    @staticmethod
    def int16ToFloat(array):
        return array/32767.0


    @staticmethod
    def grouper(iterable, n, fillvalue=None):
        args = [iter(iterable)] * n
        return izip_longest(*args, fillvalue=fillvalue)


    @classmethod
    def floatsToWavBinary(cls, array, chunk, curChannel, numChannels):
        #change float vals into a chunk for pyaudio.  Since we're doing measurement,
        #only the current channel will get audio and the rest will get silence.
        max_amplitude = 32767.0
        array = array*max_amplitude
        array = array.astype(int)

        binArray = []
        xfinal = np.zeros((chunk*numChannels,),dtype=np.int)

        for x in cls.grouper(array, chunk, 0):
            xfinal[curChannel-1::numChannels] = x
            binArray.append(struct.pack("%sh" % len(xfinal), *xfinal))

        return binArray


    @classmethod
    def floatsToWavBinaryAllChannels(cls, array, chunk, numChannels):
        #play mono audio back on all channels
        max_amplitude = 32767.0
        array = array*max_amplitude
        array = array.astype(int)

        binArray = []
        xfinal = np.zeros((chunk*numChannels,),dtype=np.int)

        for x in cls.grouper(array, chunk, 0):
            for ind in range(numChannels):
                xfinal[ind::numChannels] = x
            binArray.append(struct.pack("%sh" % len(xfinal), *xfinal))

        return binArray

