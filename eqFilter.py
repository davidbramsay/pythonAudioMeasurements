# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 01:38:13 2015

@author: davidramsay
"""
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import biquad_cookbook as bq
import ipdb
import operator


def plotSumFilter(b, a, Fs, prev=None, plotFlag=False):
    w, h = signal.freqz(b,a)
    
    if prev is not None:
        h = (h * prev)

    if plotFlag:
        fig = plt.figure()
        plt.title('Digital filter frequency response')
        ax1 = fig.add_subplot(111)
        plt.semilogx(w*(Fs/2.0/3.14), 20 * np.log10(abs(h)), 'b')
        plt.ylabel('Amplitude [dB]', color='b')
        plt.xlabel('Frequency [rad/sample]')
        ax2 = ax1.twinx()
        angles = np.unwrap(np.angle(h))
        plt.semilogx(w*(Fs/2.0/3.14), angles, 'g')
        plt.ylabel('Angle (radians)', color='g')
        plt.grid()
        plt.axis('tight')
    
    return h    
   
   
def plotFilter(b, a, Fs, fig=None):
    w, h = signal.freqz(b,a)
    
    if (fig is None):
        fig = plt.figure()
    
    plt.title('Digital filter frequency response')
    ax1 = fig.add_subplot(111)
    plt.semilogx(w*(Fs/2.0/3.14), 20 * np.log10(abs(h)), 'b')
    plt.ylabel('Amplitude [dB]', color='b')
    plt.xlabel('Frequency [rad/sample]')
    ax2 = ax1.twinx()
    angles = np.unwrap(np.angle(h))
    plt.semilogx(w*(Fs/2.0/3.14), angles, 'g')
    plt.ylabel('Angle (radians)', color='g')
    plt.grid()
    plt.axis('tight')
    return fig


def show():
    plt.show()
   
   
def interpolate(x, x_values, y_values):
    def _basis(j):
        p = [(x - x_values[m])/(x_values[j] - x_values[m]) for m in xrange(k) if m != j]
        return reduce(operator.mul, p)
    assert len(x_values) != 0 and (len(x_values) == len(y_values)), 'x and y cannot be empty and must have the same length'
    k = len(x_values)
    return sum(_basis(j)*y_values[j] for j in xrange(k))  
  
  
def generateRoughEQTarget(Fs):
    dbGain = [2.5, 1.5, 0.5, 0.0, -1.0, -2.0, -2.5, -2.5]    
    freqs = [50, 100, 180, 445, 1085, 2660, 6525, 10000]
    BW = 2
    targ = 0
    for ind, freq in enumerate(freqs):
        if ind == 0:
            filt = bq.shelf(freq/(Fs/2.0), dbGain[ind], BW, 'low')
        if ind == len(freqs)-1:
            filt = bq.shelf(freq/(Fs/2.0), dbGain[ind], BW, 'high')
            plotFlag = True
        else:
            filt = bq.peaking(freq/(Fs/2.0), dbGain[ind], BW = BW)    
    
        w, h = signal.freqz(filt[0],filt[1])
                
        targ = targ + h 
    
    return targ, w*(Fs/2.0/3.14)


def EQTargetAtFreqs(freqs, style="normal"):  
    
    f=np.array([15.625,20,31.25,40,62.5,125,250,500,1000,2000,4000,8000,10000,16000,20000,32000])
    linf = np.log(f/15.625)/np.log(2.0) + 1    

    if style=="normal":
        db= [3, 3, 3, 3, 2.5, 1, 0.5, 0, 0, -0.5, -1, -2.5, -3, -3, -3, -3]
        cutLow = 31.25
        lowVal = 3
        cutHigh = 10000
        highVal = -3

    if style=="flat":
        db= [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        cutLow = 31.25
        lowVal = 0
        cutHigh = 10000
        highVal = 0
    
    if style=="flatter":
        db= [3, 3, 3, 3, 2.5, 1, 0, 0, 0, 0, -0.5, -2, -3, -3, -3, -3]
        cutLow = 31.25
        lowVal = 3
        cutHigh = 10000
        highVal = -3
        
    if style=="bass":
        db= [10, 10, 10, 12, 10, 8, 6, 4, 0, -2, -4, -5, -9, -6, -6, -6]
        cutLow = 31.25
        lowVal = 6
        cutHigh = 15000
        highVal = -6
        
    db2=[]
    for fdex in freqs:
        if (fdex<cutLow):
            db2.append(lowVal)
        elif (fdex>cutHigh):
            db2.append(highVal)
        else:
            db2.append(interpolate(np.log(fdex/15.625)/np.log(2.0) + 1,linf,db))
    
    return db2







if __name__ == "__main__":
    """
    Fs = 44100    
    low = bq.peaking(75/(Fs/2.0), 5, BW = 2)
    fig = plotFilter(low[0], low[1], Fs)

    mid = bq.peaking(180/(Fs/2.0), 5, BW = 2)
    plotFilter(mid[0], mid[1], Fs, fig)
    """ 
    
    
    
    Fs = 44100    
    #dbGain = [2.5, 1.5, 0.5, 0.0, -1.0, -2.0, -2.5, -2.5, -2.5]    
    #freqs = [75, 180, 445, 1085, 2660, 6525]
    freqs = [60, 100, 200, 400, 800, 1600, 3200, 6400, 10000, 16000]
    dbGain = np.ones(len(freqs))*5   
    
    BW = 1.77
    fig = None
    for ind, freq in enumerate(freqs):
        filt = bq.peaking(freq/(Fs/2.0), dbGain[ind], BW = BW)
        fig = plotFilter(filt[0],filt[1], Fs, fig)
    show()
    
    h = None
    plotFlag = False
    for ind, freq in enumerate(freqs):
        #if ind == 0:
        #    filt = bq.shelf(freq/(Fs/2.0), dbGain[ind], BW, 'low')
        if ind == len(freqs)-1:
        #    filt = bq.shelf(freq/(Fs/2.0), dbGain[ind], BW, 'high')
             filt = bq.peaking(freq/(Fs/2.0), dbGain[ind], BW = BW)       
             plotFlag = True
        else:
            filt = bq.peaking(freq/(Fs/2.0), dbGain[ind], BW = BW)
            
        h = plotSumFilter(filt[0],filt[1], Fs, h, plotFlag)
    show()
    
    
    
    """
    temp, f = generateRoughEQTarget(44100)
    plt.semilogx(f, 20*np.log10(abs(temp)))
    temp, f = handEQTarget()
    plt.semilogx(f, temp)
    plt.show()
    """
    
    """
    f = xrange(20,20000,10)
    db = EQTargetAtFreqs(f,"normal")
    db2 = EQTargetAtFreqs(f,"flatter")
    db3 = EQTargetAtFreqs(f,"flat")
    plt.semilogx(f, db)
    plt.semilogx(f, db2)
    plt.semilogx(f, db3)
    plt.show()    
    """