# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 16:58:59 2015

@author: davidramsay
"""

from __future__ import print_function
import pyaudio
import numpy as np
from itertools import izip_longest
import struct
import time
import math

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


Volume is now a concept-- we store rawaudio and playaudio at the right volume.  Normalize sets it to
full scale, setVolume sets it to a certain volume relative to full scale, denormalize sets it back to
it's original raw audio volume.  New audio can be passed and either easily be set to simply play/replace
the audio (1) at its raw provided volume, (2) normalized to full scale, or (3) inheriting the volume/scaling
of the default audio already in the object.

"""


class audioPlayer(object):

    def __init__(self, audio=None, channels=1, chunk=1024, Fs=44100):
        self._pa = pyaudio.PyAudio()
        self._channels = channels
        self._chunk = chunk
        self._fs = Fs
        self._format = pyaudio.paInt16
        self._rawaudio = None
        self._volume = 0.0
        self._normalized = True

        if audio is not None and audio != []:
            self._rawaudio = audio - np.mean(audio)
            self._playaudio = audio - np.mean(audio)
            self._normalized = False
            self._rawvolume = 20.0 * math.log10(np.amax(np.abs(self._rawaudio)))
            self._volume = 20.0 * math.log10(np.amax(np.abs(self._playaudio)))


    def setAudio(self, audio, keepPreviousVol=False):
        self._rawaudio = audio - np.mean(audio)
        self._rawvolume = 20.0 * math.log10(np.amax(np.abs(self._rawaudio)))

        if keepPreviousVol:
            linear_gain = 10**(self._volume/20.0)
            self._playaudio = self._rawaudio * linear_gain / float(np.amax(np.abs(self._rawaudio)))
        else:
            self._playaudio = audio - np.mean(audio)
            self._volume = 20.0 * math.log10(np.amax(np.abs(self._playaudio)))
            if self._volume == 0.0: self._normalized = True
            else: self._normalized = False


    def normalize(self):
        #normalize the input audio to max output
        assert (self._rawaudio is not None), 'must have audio to normalize!'
        self._playaudio = self._rawaudio/max(abs(self._rawaudio))
        self._normalized = True
        self._volume = 0.0
        print('set to full volume (0 dBFS).')


    def denormalize(self):
        assert (self._rawaudio is not None), 'must have audio to denormalize!'
        #replace normalized play output with raw data
        self._playaudio = self._rawaudio
        self._volume = self._rawvolume
        self._normalized = False
        print('set to original volume ' + str(self._volume) + ' dBFS.')


    def setVolume(self, volume=0):
        #set volume of playback
        assert (volume <= 0), 'volume must be in relative dBFS to full-scale (<=0)'

        if volume != self._volume:
            self._volume = volume
            if self._rawaudio is not None:
                linear_gain = 10**(volume/20.0)
                self._playaudio = self._rawaudio * linear_gain / float(np.amax(np.abs(self._rawaudio)))
                if self._volume == 0.0: self._normalized = True
                else: self._normalized = False

        print('set to volume ' + str(self._volume) + ' dBFS.')


    def _getInput(self, channel, audioToPlay, useVolume, normalizeTestSignal):
        #get the interleaved signal to playback given configuration options
        assert (audioToPlay is not None or self._rawaudio is not None), 'either audio must be passed or default audio must be initiated.'
        if (useVolume and normalizeTestSignal): print('WARNING: cannot use volume and normalize at the same time; assuming normalization is desired.')

        if audioToPlay is not None:

            audioToPlay = audioToPlay - np.mean(audioToPlay)

            if normalizeTestSignal:
                print('playing passed audio at full scale.')
                out_audio = audioToPlay / np.amax(np.abs(audioToPlay))
            elif useVolume:
                print('playing passed audio to match default audio volume of ' + str(self._volume) + ' dBFS.')
                linear_gain = 10**(self._volume/20.0)
                out_audio = audioToPlay * linear_gain / float(np.amax(np.abs(audioToPlay)))
            else:
                try:
                    print('playing passed audio as given, at ' + str(20.0 * math.log10(np.amax(np.abs(audioToPlay)))) + ' dBFS.')
                except:
                    print('playing passed audio as given.')

                out_audio = audioToPlay
        else:
            if normalizeTestSignal and not self._normalized:
                print('playing default audio normalized to full scale, which is not how it\'s currently configured (this change is not saved for future measurements.)')
                out_audio = self._rawaudio/max(abs(self._rawaudio))
            elif not useVolume and self._volume != self._rawvolume:
                print('playing default audio at raw audio volume of ' + str(self._rawvolume) + ' dBFS, which is not how it\'s currently configured (this change is not saved for future measurements.)')
                out_audio = self._rawaudio
            else:
                print('playing default saved audio at configured volume of ' + str(self._volume) + ' dBFS.')
                out_audio = self._playaudio

        if (channel==0):
            input = self.floatsToWavBinaryAllChannels(out_audio, self._chunk, self._channels)
        else:
            input = self.floatsToWavBinary(out_audio, self._chunk, channel, self._channels)

        return input


    def measureChannel(self, channel=0, audioToPlay=None, useVolume=True, normalizeTestSignal=False):
        #return float arrays from microphone channels, channel 0 will play on all channels
        global input
        global output
        global currentIndex
        global exit_flag

        input = self._getInput(channel, audioToPlay, useVolume, normalizeTestSignal)
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


    def playAudio(self, channel=0, audioToPlay=None, useVolume=True, normalizeTestSignal=False):
        #channel 0 will play audio on all channels.
        input = self._getInput(channel, audioToPlay, useVolume, normalizeTestSignal)

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

