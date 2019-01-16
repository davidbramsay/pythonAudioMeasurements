import audioSample
import numpy as np
from audioPlayer import audioPlayer
from audioMeasure import audioMeasure

a = audioMeasure(channels=1)
a.pinkNoiseLoop(samples=8192, repetitions=20)
a.setVolLinearity()
a.setRepeatsSNR(repeats=[10, 15, 20, 25, 30, 35, 40, 45, 50])
a.testAllChannels()
a.calcTF()
print '----'
a.plotImpulseResp()
a.plotFreqResp()

'''
a = audioPlayer(Fs=44100)
a.setVolume(-20)
print a.playAudio(audioToPlay=np.random.random(44100))
print a._volume
print a._rawaudio
a.setAudio(np.random.random(44100))
print a._volume
a.setAudio(np.random.random(44100))
print a._volume
a.setAudio(np.random.random(44100), keepPreviousVol=True)
print a._volume
a.normalize()
print a._volume
print a.measureChannel()
print '-'*20
print a.measureChannel(audioToPlay=np.random.random(44100), useVolume=False)
a.setVolume(-6)
print '-'*20
print a.measureChannel(audioToPlay=np.random.random(44100), normalizeTestSignal=True)
a.setVolume(-12)
print '-'*20
print a.measureChannel(audioToPlay=np.random.random(44100), useVolume=False)
a.setVolume(-18)
print '-'*20
print a.measureChannel(audioToPlay=np.random.random(44100), normalizeTestSignal=True)
print '-'*20
print a.measureChannel(normalizeTestSignal=True)
print a.measureChannel(useVolume=False)
print a.measureChannel()
'''


'''
sig =np.random.random(44100)
a = audioSample.audioSample(sig)
a.removeDCOffset()
a.plot()
a.setVolume(-6)
a.plot()
a.PDF()
a.toFreq()
a.zeroPadEnd()
a.toTime()
a.setVolume(-12)
a.plot()
a.setVolume(-18)
a.plot()
'''


'''
a= audioSample.audioSample([1,2,1,0,-1,-2,1], 't',Fs=44100)

print a.data
print a.fs
print a.type

print '-'
a.fs = 32000
print '-'
a.type = 't'
a.type = 'db'
a.type = 'f'
print '-'
a.PDF()
a.data = [1,2]
'''
