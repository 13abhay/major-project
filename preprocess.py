# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 16:59:28 2019

@author: pc
"""

import os
import numpy as np
import pandas as pd
import soundfile as sf
import scipy.signal as signal
import matplotlib.pyplot as plt
import librosa
import librosa.display
import scipy


def calculate_mask(norm_Sxx,t):
    col_median = np.median(norm_Sxx,axis = 0).reshape((1,norm_Sxx.shape[1]))
    row_median = np.median(norm_Sxx,axis = 1).reshape((norm_Sxx.shape[0],1))
    bin_img = np.bitwise_and(norm_Sxx>=t*row_median,norm_Sxx>t*col_median)
    bin_img = np.asarray(bin_img,dtype = np.bool)
                
    bin_img_eroded=scipy.ndimage.morphology.binary_erosion(bin_img,structure=np.ones((4,4)))
    bin_img_dilated=scipy.ndimage.morphology.binary_dilation(bin_img_eroded,structure=np.ones((4,4)))

    signal_mask=np.zeros_like(norm_Sxx).astype(np.bool)
    
    for j in range(norm_Sxx.shape[1]):
        if any(bin_img_dilated[:,j]==1):
            signal_mask[:,j]=np.ones((norm_Sxx.shape[0]))
    
    return signal_mask    

metadata=pd.read_csv('birdsong_metadata.csv')
files=os.listdir('songs')

data=[]
c=0
for i in files:
    if i.endswith('.flac'):
        data, samplerate=sf.read('songs/'+i)
        for j in range(len(metadata)):
            if i=='xc'+str(metadata['file_id'][j])+'.flac':
                y=metadata['species'][j]
        data_resampled=librosa.core.resample(data,samplerate,22050)
        print('resampled')
        
        freq, time, Sxx = signal.spectrogram(data, samplerate,mode='magnitude')
        plt.pcolormesh(time, freq, 10*np.log10(Sxx))
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')

        #norm_Sxx=(Sxx-Sxx.mean())/Sxx.std()
        norm_Sxx=(Sxx-Sxx.min())/(Sxx.max()-Sxx.min())
        print('normalized')

        signal_mask=calculate_mask(norm_Sxx,3)
        signal_mask=scipy.ndimage.morphology.binary_dilation(signal_mask,structure=np.ones((4,4)))
        signal_mask=scipy.ndimage.morphology.binary_dilation(signal_mask,structure=np.ones((4,4)))
        print('signal_mask ready')
        
        noise_mask=calculate_mask(norm_Sxx,2.5)
        noise_mask=np.invert(noise_mask)
        print('noise_mask ready')        
        
        signal=[]
        for j in range(signal_mask.shape[1]):
            if signal_mask[0,j]==1:
                signal.append(norm_Sxx[:,j])
        signal=np.array(signal)
        signal=signal.T
        
        noise=[]
        for j in range(noise_mask.shape[1]):
            if noise_mask[0,j]==1:
                noise.append(norm_Sxx[:,j])
        noise=np.array(noise)
        noise=noise.T
        
        plt.imshow(10*np.log10(norm_Sxx),aspect='auto')
        
        
        """
        D = np.abs(librosa.stft(data))**2
        S = librosa.feature.melspectrogram(S=D)
        librosa.display.specshow(librosa.power_to_db(S,ref=np.max),y_axis='mel', fmax=8000,x_axis='time')
        mfcc=librosa.feature.mfcc(data)
        #data.append([mfcc,y])"""
        break
    c+=1
    print(c,len(files))