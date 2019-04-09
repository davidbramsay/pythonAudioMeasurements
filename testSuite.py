from audioSample import audioSample 
from polarData import polarData
import numpy as np
from math import pi, atan
import matplotlib.pyplot as plt

# rad/samp = (sec/samp)*(cycles/sce)*(rad/cycle) = 2*pi*f/fs


"""
Things to fix:

"""

def plotCompReal(audioSamp):
    audioSamp.toFreq()
    reals = [a.real for a in audioSamp.data]
    imags = [a.imag for a in audioSamp.data]

    plt.subplot(2,1,1)
    plt.plot(reals)

    plt.subplot(2,1,2)
    plt.plot(imags)

    plt.show()

def plotMagPhase(audioSamp):
    audioSamp.toFreq()
    mag = [abs(a) for a in audioSamp.data]
    phase = [atan(a.imag/a.real) for a in audioSamp.data]

    plt.subplot(2,1,1)
    plt.plot(mag)

    plt.subplot(2,1,2)
    plt.plot(phase)

    plt.show()




################################
# audioSample tests
################################


sec = 1
fs = 44100
N = sec*fs

samples = np.arange(N)



freqs = [440, 880]
sig = np.zeros(len(samples))

for f in freqs:
    omega = 2*pi*f/fs
    sig += np.sin(samples*omega)



s = audioSample(dataArray=sig)


def test_plot():
    s.plot()
    s.plot(both=True)

def test_conversion():
    s.toDb()
    s.plot()
    s.toTime()
    s.plot()
    s.toFreq()
    print s.type

def test_remove_1():
    """
    remove a single frequency
    this works, but shoudl really not be used that much because
    """
    s.toDb()
    s.plot()
    s.removeFreqs(freqs[0])
    s.plot()

def test_remove_2():
    """
    remove a range
    """
    s.toDb()
    s.plot()
    s.removeFreqs(freqRange=[100,1000])
    s.plot()

def test_db_amp_only():
    s.toDb()
    plotMagPhase(s)
    s.toDb()
    s.changeFreqs(0,freqRange=[400, 1000], dbOnly=True)
    plotMagPhase(s)
    s.plot()

def test_change_freqs():
    """
    change in complex amp domain
    """
    s.toFreq()
    s.plot()
    s.changeFreqs(complex(0.5, 0.5),freqRange=[400, 500])
    s.plot()

def test_freq_0_reject():
    """
    cannot set complex amplitude to 0
    """
    s.toFreq()
    s.changeFreqs(0,freqRange=[400, 500])

def test_db_reject():
    """
    should reject and suggest changeing to db
    """
    s.toTime()
    s.changeFreqs(0, dbOnly=True)

def test_copy():
    """
    removing freqs from s should not affect s_2
    """
    s.toDb()
    s_2 = s.copy()
    s.removeFreqs(440)
    s.plot(both=True)
    s_2.plot(both=True)

def test_iter():
    """
    test that it can be iterated twice
    """
    new = []
    for val in s:
        new.append(val)

    assert all([n==o for n,o in zip(new, s._data)])
    print "success"

    new = []
    for val in s:
        new.append(val)

    assert all([n==o for n,o in zip(new, s._data)])
    print "success"


################################
# polarData
################################

"""
Things to fix:
    > deleting of the old attributes/handling that whole situation generally
    > can you put a title on just a figure?
"""


pd = polarData("reference_mic_rokit8_3ft_fortest.pkl")



def plotAngle():
    pd.test_plot_angle(0, both=True)

def test_plot_freq():
    pd.plotFreqs([400, 1000, 1500])


def change_freqs_theta():
    pd.changeFreqs(0.5, thetaRange=[30, 50], freqRange=[380, 500])
    pd.plotFreqs([400, 1000, 1500])



change_freqs_theta()