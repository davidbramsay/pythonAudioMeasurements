# pythonAudioMeasurements
*an audio measurement and manipulation library written in Python.*

This is a library designed to make audio measurment and audio data manipulation easier, faster, and less error-prone.

##AudioMeasure

audioMeasure is for measuring audio devices.  This accepts multiple speakers but assumes one microphone.

It will run through many channels and make measurements of each, or do single channel measurements.

initialize by calling:
```
a = audioMeasure(np.array([1,1,1]),type="t",Fs=44100)
```

###the accessible data in audioMeasure are:

+ `audioMeasure.output` = an audioSample holding the measurement audio to be played out during measurement
+ `audioMeasure.input` = an array of audioSamples recordings from the last measurement, one for each channel
+ `audioMeasure.tf` = an array of transfer function measurements, for each channel of last measurement
+ `audioMeasure.outInfo` is a dictionary with several useful pieces of stored information 
    (like number of repetitions of for test audio, test audio type, etc)
+ `audioMeasure.fs` = sampling rate

###available class methods are:

+ `audioMeasure.pinkNoise(duration, Fs)` - set measurement signal to pink noise of <duration> secs, if no Fs/duration provided it overwrites the current audioMeasure.output with the same Fs and duration.
+ `audioMeasure.pinkNoiseLoop(samples, repetitions, Fs)` - set measurement signal to a <samples> long, loopable pink noise test signal that repeats <repetitions> times.  Fs is optional, and will overwrite object default.
+ `audioMeasure.testAllChannels(channels)` - give max number of channels, this will step through and test all channels with the 
    stored measurement signal.  It will place them in audioMeasure.input.  Again, this assumes multiple speakers measured at 
    one microphone.  The first speakers measured signal can be found at audioMeasure.input[0], second at input[1], etc.
+ `audioMeasure.calcTF()` - step through audio signals stored in input, and using the measurement signal from output, calculate
    and update the tf field.  audioMeasure.tf[0] is the audioSample for the TF of the first speaker
+ `audioMeasure.plotImpulseResp()`
+ `audioMeasure.plotFreqResp()`

*EXPERIMENTAL*

+ `audioMeasure.differenceFromEQ(eqShape="flatter", doplot=False)`
+ `audioMeasure.differenceFromSmoothed(doplot=False)`
+ `audioMeasure.generateEQfromTF(eqShape="flatter", doplot=False, limits=100)`
+ `audioMeasure.processEQ(eqVals, maximum=10, minimum=-10)`
+ `audioMeasure.createEQs(eq, doplot=False)`
+ `audioMeasure.compareEQFiltToEQ(filt, eq, f)`

### Typical use:

```
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
```
