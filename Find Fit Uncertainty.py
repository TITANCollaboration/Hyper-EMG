#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Nov 22 13:00:23 2019

@author: msmith
"""


# coding: utf-8

# questions

# =============================================================================
# Author Matthew Smith
# =============================================================================

# =============================================================================
# Update Notes:
#
#March 19th, 2020:   
#   Added the ability to print the individual uncertainties
#   that quadraticaly add up to the total peak shape unsertainty 
#   change the 'QuadErrorProp' Function to take a dictionary instead
#   added 'printDictionary' Function which neatly prints a dictionary's vals and keys.
#
#April 4th, 2020 : 
#   CalibParams dictionary relocation to resolve large error problem
# =============================================================================

# # Import Needed Packages
import matplotlib.pyplot as plt
import numpy
import math
from scipy.optimize import curve_fit
import scipy.special
import csv

# =============================================================================
# Function Definitions
# =============================================================================

#---------------------------H-EMG for 3 peaks----------------------------------
def triplefunc(x, calibdict, cI, cII, cIII):
    negtail1I = 0
    negtail2I = 0
    negtail3I = 0
    negtail1II = 0
    negtail2II = 0
    negtail3II = 0
    negtail1III = 0
    negtail2III = 0
    negtail3III = 0
    postail1I = 0
    postail2I = 0
    postail3I = 0
    postail1II = 0
    postail2II = 0
    postail3II = 0
    postail1III = 0
    postail2III = 0
    postail3III = 0
    s = calibdict['stdev'][0]
    w = calibdict['weight'][0]
    aI= calibdict['aITrip'][0]
    aII= calibdict['aIITrip'][0]
    aIII= calibdict['aIIITrip'][0]
    if leftTailNumber > 0:
        nm1 = calibdict['nm1'][0]
        tm1 = calibdict['tm1'][0]
        negtail1I = (nm1/(2*tm1))*((numpy.exp(((s/(math.sqrt(2)*tm1))**2+((x-cI)/tm1))))*scipy.special.erfc((s/(math.sqrt(2)*tm1))+((x-cI)/(math.sqrt(2)*s))))
        negtail1I = numpy.nan_to_num(negtail1I)
        negtail1II = (nm1/(2*tm1))*((numpy.exp(((s/(math.sqrt(2)*tm1))**2+((x-cII)/tm1))))*scipy.special.erfc((s/(math.sqrt(2)*tm1))+((x-cII)/(math.sqrt(2)*s))))
        negtail1II = numpy.nan_to_num(negtail1II)
        negtail1III = (nm1/(2*tm1))*((numpy.exp(((s/(math.sqrt(2)*tm1))**2+((x-cIII)/tm1))))*scipy.special.erfc((s/(math.sqrt(2)*tm1))+((x-cIII)/(math.sqrt(2)*s))))
        negtail1III = numpy.nan_to_num(negtail1III)
    if leftTailNumber > 1:
        nm2 = calibdict['nm2'][0]
        tm2 = calibdict['tm2'][0]
        negtail2I = (nm2/(2*tm2))*((numpy.exp(((s/(math.sqrt(2)*tm2))**2+((x-cI)/tm2))))*scipy.special.erfc((s/(math.sqrt(2)*tm2))+((x-cI)/(math.sqrt(2)*s))))
        negtail2I = numpy.nan_to_num(negtail2I)
        negtail2II = (nm2/(2*tm2))*((numpy.exp(((s/(math.sqrt(2)*tm2))**2+((x-cII)/tm2))))*scipy.special.erfc((s/(math.sqrt(2)*tm2))+((x-cII)/(math.sqrt(2)*s))))
        negtail2II = numpy.nan_to_num(negtail2II)
        negtail2III = (nm2/(2*tm2))*((numpy.exp(((s/(math.sqrt(2)*tm2))**2+((x-cIII)/tm2))))*scipy.special.erfc((s/(math.sqrt(2)*tm2))+((x-cIII)/(math.sqrt(2)*s))))
        negtail2III = numpy.nan_to_num(negtail2III)
    if leftTailNumber >2:
        nm3 = calibdict['nm3'][0]
        tm3 = calibdict['tm3'][0]
        negtail3I = (nm3/(2*tm3))*((numpy.exp(((s/(math.sqrt(2)*tm3))**2+((x-cI)/tm3))))*scipy.special.erfc((s/(math.sqrt(2)*tm3))+((x-cI)/(math.sqrt(2)*s))))
        negtail3I = numpy.nan_to_num(negtail3I)
        negtail3II = (nm3/(2*tm3))*((numpy.exp(((s/(math.sqrt(2)*tm3))**2+((x-cII)/tm3))))*scipy.special.erfc((s/(math.sqrt(2)*tm3))+((x-cII)/(math.sqrt(2)*s))))
        negtail3II = numpy.nan_to_num(negtail3II)
        negtail3III = (nm3/(2*tm3))*((numpy.exp(((s/(math.sqrt(2)*tm3))**2+((x-cIII)/tm3))))*scipy.special.erfc((s/(math.sqrt(2)*tm3))+((x-cIII)/(math.sqrt(2)*s))))
        negtail3III = numpy.nan_to_num(negtail3III)
    if rightTailNumber > 0:
        np1 = calibdict['np1'][0]
        tp1 = calibdict['tp1'][0]
        postail1I = (np1/(2*tp1))*((numpy.exp(((s/(math.sqrt(2)*tp1))**2-((x-cI)/tp1))))*scipy.special.erfc((s/(math.sqrt(2)*tp1))-((x-cI)/(math.sqrt(2)*s))))
        postail1I = numpy.nan_to_num(postail1I)
        postail1II = (np1/(2*tp1))*((numpy.exp(((s/(math.sqrt(2)*tp1))**2-((x-cII)/tp1))))*scipy.special.erfc((s/(math.sqrt(2)*tp1))-((x-cII)/(math.sqrt(2)*s))))
        postail1II = numpy.nan_to_num(postail1II)
        postail1III = (np1/(2*tp1))*((numpy.exp(((s/(math.sqrt(2)*tp1))**2-((x-cIII)/tp1))))*scipy.special.erfc((s/(math.sqrt(2)*tp1))-((x-cIII)/(math.sqrt(2)*s))))
        postail1III = numpy.nan_to_num(postail1III)
    if rightTailNumber > 1:
        np2 = calibdict['np2'][0]
        tp2 = calibdict['tp2'][0]
        postail2I = (np2/(2*tp2))*((numpy.exp(((s/(math.sqrt(2)*tp2))**2-((x-cI)/tp2))))*scipy.special.erfc((s/(math.sqrt(2)*tp2))-((x-cI)/(math.sqrt(2)*s))))
        postail2I = numpy.nan_to_num(postail2I)
        postail2II = (np2/(2*tp2))*((numpy.exp(((s/(math.sqrt(2)*tp2))**2-((x-cII)/tp2))))*scipy.special.erfc((s/(math.sqrt(2)*tp2))-((x-cII)/(math.sqrt(2)*s))))
        postail2II = numpy.nan_to_num(postail2II)
        postail2III = (np2/(2*tp2))*((numpy.exp(((s/(math.sqrt(2)*tp2))**2-((x-cIII)/tp2))))*scipy.special.erfc((s/(math.sqrt(2)*tp2))-((x-cIII)/(math.sqrt(2)*s))))
        postail2III = numpy.nan_to_num(postail2III)
    if rightTailNumber > 2:
        np3 = calibdict['np3'][0]
        tp3 = calibdict['tp3'][0]
        postail3I = (np3/(2*tp3))*((numpy.exp(((s/(math.sqrt(2)*tp3))**2-((x-cI)/tp3))))*scipy.special.erfc((s/(math.sqrt(2)*tp3))-((x-cI)/(math.sqrt(2)*s))))
        postail3I = numpy.nan_to_num(postail3I)
        postail3II = (np3/(2*tp3))*((numpy.exp(((s/(math.sqrt(2)*tp3))**2-((x-cII)/tp3))))*scipy.special.erfc((s/(math.sqrt(2)*tp3))-((x-cII)/(math.sqrt(2)*s))))
        postail3II = numpy.nan_to_num(postail3II)
        postail3III = (np3/(2*tp3))*((numpy.exp(((s/(math.sqrt(2)*tp3))**2-((x-cIII)/tp3))))*scipy.special.erfc((s/(math.sqrt(2)*tp3))-((x-cIII)/(math.sqrt(2)*s))))
        postail3III = numpy.nan_to_num(postail3III)
    
    hminI = negtail1I + negtail2I + negtail3I
    hminII = negtail1II + negtail2II + negtail3II
    hminIII = negtail1III + negtail2III + negtail3III
    hpluI = postail1I + postail2I + postail3I
    hpluII = postail1II + postail2II + postail3II
    hpluIII = postail1III + postail2III + postail3III
    return w*(aI*hminI + aII*hminII + aIII*hminIII) + (1-w)*(aI*hpluI + aII*hpluII + aIII*hpluIII)

#---------------------------H-EMG for 2 peaks----------------------------------
def doublefunc(x, calibdict, cI, cII):
    negtail1I = 0
    negtail2I = 0
    negtail3I = 0
    negtail1II = 0
    negtail2II = 0
    negtail3II = 0
    postail1I = 0
    postail2I = 0
    postail3I = 0
    postail1II = 0
    postail2II = 0
    postail3II = 0
    s = calibdict['stdev'][0]
    w = calibdict['weight'][0]
    aI = calibdict['aIDouble'][0]
    aII = calibdict['aIIDouble'][0]
    if leftTailNumber > 0:
        nm1 = calibdict['nm1'][0]
        tm1 = calibdict['tm1'][0]
        negtail1I = (nm1/(2*tm1))*((numpy.exp(((s/(math.sqrt(2)*tm1))**2+((x-cI)/tm1))))*scipy.special.erfc((s/(math.sqrt(2)*tm1))+((x-cI)/(math.sqrt(2)*s))))
        negtail1I = numpy.nan_to_num(negtail1I)
        negtail1II = (nm1/(2*tm1))*((numpy.exp(((s/(math.sqrt(2)*tm1))**2+((x-cII)/tm1))))*scipy.special.erfc((s/(math.sqrt(2)*tm1))+((x-cII)/(math.sqrt(2)*s))))
        negtail1II = numpy.nan_to_num(negtail1II)
    if leftTailNumber > 1:
        nm2 = calibdict['nm2'][0]
        tm2 = calibdict['tm2'][0]
        negtail2I = (nm2/(2*tm2))*((numpy.exp(((s/(math.sqrt(2)*tm2))**2+((x-cI)/tm2))))*scipy.special.erfc((s/(math.sqrt(2)*tm2))+((x-cI)/(math.sqrt(2)*s))))
        negtail2I = numpy.nan_to_num(negtail2I)
        negtail2II = (nm2/(2*tm2))*((numpy.exp(((s/(math.sqrt(2)*tm2))**2+((x-cII)/tm2))))*scipy.special.erfc((s/(math.sqrt(2)*tm2))+((x-cII)/(math.sqrt(2)*s))))
        negtail2II = numpy.nan_to_num(negtail2II)
    if leftTailNumber >2:
        nm3 = calibdict['nm3'][0]
        tm3 = calibdict['tm3'][0]
        negtail3I = (nm3/(2*tm3))*((numpy.exp(((s/(math.sqrt(2)*tm3))**2+((x-cI)/tm3))))*scipy.special.erfc((s/(math.sqrt(2)*tm3))+((x-cI)/(math.sqrt(2)*s))))
        negtail3I = numpy.nan_to_num(negtail3I)
        negtail3II = (nm3/(2*tm3))*((numpy.exp(((s/(math.sqrt(2)*tm3))**2+((x-cII)/tm3))))*scipy.special.erfc((s/(math.sqrt(2)*tm3))+((x-cII)/(math.sqrt(2)*s))))
        negtail3II = numpy.nan_to_num(negtail3II)
    if rightTailNumber > 0:
        np1 = calibdict['np1'][0]
        tp1 = calibdict['tp1'][0]
        postail1I = (np1/(2*tp1))*((numpy.exp(((s/(math.sqrt(2)*tp1))**2-((x-cI)/tp1))))*scipy.special.erfc((s/(math.sqrt(2)*tp1))-((x-cI)/(math.sqrt(2)*s))))
        postail1I = numpy.nan_to_num(postail1I)
        postail1II = (np1/(2*tp1))*((numpy.exp(((s/(math.sqrt(2)*tp1))**2-((x-cII)/tp1))))*scipy.special.erfc((s/(math.sqrt(2)*tp1))-((x-cII)/(math.sqrt(2)*s))))
        postail1II = numpy.nan_to_num(postail1II)
    if rightTailNumber > 1:
        np2 = calibdict['np2'][0]
        tp2 = calibdict['tp2'][0]
        postail2I = (np2/(2*tp2))*((numpy.exp(((s/(math.sqrt(2)*tp2))**2-((x-cI)/tp2))))*scipy.special.erfc((s/(math.sqrt(2)*tp2))-((x-cI)/(math.sqrt(2)*s))))
        postail2I = numpy.nan_to_num(postail2I)
        postail2II = (np2/(2*tp2))*((numpy.exp(((s/(math.sqrt(2)*tp2))**2-((x-cII)/tp2))))*scipy.special.erfc((s/(math.sqrt(2)*tp2))-((x-cII)/(math.sqrt(2)*s))))
        postail2II = numpy.nan_to_num(postail2II)
    if rightTailNumber > 2:
        np3 = calibdict['np3'][0]
        tp3 = calibdict['tp3'][0]
        postail3I = (np3/(2*tp3))*((numpy.exp(((s/(math.sqrt(2)*tp3))**2-((x-cI)/tp3))))*scipy.special.erfc((s/(math.sqrt(2)*tp3))-((x-cI)/(math.sqrt(2)*s))))
        postail3I = numpy.nan_to_num(postail3I)
        postail3II = (np3/(2*tp3))*((numpy.exp(((s/(math.sqrt(2)*tp3))**2-((x-cII)/tp3))))*scipy.special.erfc((s/(math.sqrt(2)*tp3))-((x-cII)/(math.sqrt(2)*s))))
        postail3II = numpy.nan_to_num(postail3II)
    
    hminI = negtail1I + negtail2I + negtail3I
    hminII = negtail1II + negtail2II + negtail3II
    hpluI = postail1I + postail2I + postail3I
    hpluII = postail1II + postail2II + postail3II
    return w*(aI*hminI + aII*hminII) + (1-w)*(aI*hpluI + aII*hpluII)
    
#---------------------------H-EMG for 1 peak-----------------------------------
def singlefunc(x, calibdict, cI):
    negtail1I = 0
    negtail2I = 0
    negtail3I = 0
    postail1I = 0
    postail2I = 0
    postail3I = 0
    s = calibdict['stdev'][0]
    w = calibdict['weight'][0]
    
    if leftTailNumber > 0:
        nm1 = calibdict['nm1'][0]
        tm1 = calibdict['tm1'][0]
        negtail1I = (nm1/(2*tm1))*((numpy.exp(((s/(math.sqrt(2)*tm1))**2+((x-cI)/tm1))))*scipy.special.erfc((s/(math.sqrt(2)*tm1))+((x-cI)/(math.sqrt(2)*s))))
        negtail1I = numpy.nan_to_num(negtail1I)
    if leftTailNumber > 1:
        nm2 = calibdict['nm2'][0]
        tm2 = calibdict['tm2'][0]
        negtail2I = (nm2/(2*tm2))*((numpy.exp(((s/(math.sqrt(2)*tm2))**2+((x-cI)/tm2))))*scipy.special.erfc((s/(math.sqrt(2)*tm2))+((x-cI)/(math.sqrt(2)*s))))
        negtail2I = numpy.nan_to_num(negtail2I)
    if leftTailNumber >2:
        nm3 = calibdict['nm3'][0]
        tm3 = calibdict['tm3'][0]
        negtail3I = (nm3/(2*tm3))*((numpy.exp(((s/(math.sqrt(2)*tm3))**2+((x-cI)/tm3))))*scipy.special.erfc((s/(math.sqrt(2)*tm3))+((x-cI)/(math.sqrt(2)*s))))
        negtail3I = numpy.nan_to_num(negtail3I)
    if rightTailNumber > 0:
        np1 = calibdict['np1'][0]
        tp1 = calibdict['tp1'][0]
        postail1I = (np1/(2*tp1))*((numpy.exp(((s/(math.sqrt(2)*tp1))**2-((x-cI)/tp1))))*scipy.special.erfc((s/(math.sqrt(2)*tp1))-((x-cI)/(math.sqrt(2)*s))))
        postail1I = numpy.nan_to_num(postail1I)
    if rightTailNumber > 1:
        np2 = calibdict['np2'][0]
        tp2 = calibdict['tp2'][0]
        postail2I = (np2/(2*tp2))*((numpy.exp(((s/(math.sqrt(2)*tp2))**2-((x-cI)/tp2))))*scipy.special.erfc((s/(math.sqrt(2)*tp2))-((x-cI)/(math.sqrt(2)*s))))
        postail2I = numpy.nan_to_num(postail2I)
    if rightTailNumber > 2:
        np3 = calibdict['np3'][0]
        tp3 = calibdict['tp3'][0]
        postail3I = (np3/(2*tp3))*((numpy.exp(((s/(math.sqrt(2)*tp3))**2-((x-cI)/tp3))))*scipy.special.erfc((s/(math.sqrt(2)*tp3))-((x-cI)/(math.sqrt(2)*s))))
        postail3I = numpy.nan_to_num(postail3I)
    
    hminI = negtail1I + negtail2I + negtail3I
    hpluI = postail1I + postail2I + postail3I
    return w*(hminI) + (1-w)*(hpluI)

#-------------------------Chi & reduced chi squared----------------------------
def chi2(y_measure,y_predict,errors):
    """Calculate the chi squared value given a measurement with errors and prediction"""
    return numpy.sum( (y_measure - y_predict)**2 / errors**2 )

def chi2reduced(y_measure, y_predict, errors, number_of_parameters):
    """Calculate the reduced chi squared value given a measurement with errors and prediction,
    and knowing the number of parameters in the model."""
    return chi2(y_measure, y_predict, errors)/(numpy.asarray(y_measure).size - number_of_parameters)

#--------------------------Print Dictionary Values-----------------------------
#Neatly Prints the values out of a dictionaty
def printDictionary(Dict):
    KevPerAmu=931494.0954 
    keylist=list(Dict.keys())
    keylist.sort()
    print('\nTotal Uncertainty Breakdown for the Desired Peak in Kev: \n')
    for parameter in keylist:
        err= Dict[parameter]*KevPerAmu

        print('{:<15}= {:<30}'.format(parameter, err))
        
#--------------------------Trapezoidal integration-----------------------------
# integrating using trap sum
def TrapezoidalArea(xData,yData):
    area=0
    for count in range(0,len(xData)-2):
        area+=((xData[count+1] - xData[count]) * 0.5 * ((yData[count+1] + yData[count])))
        count+=1
    return area

#---------------------------import data----------------------------------
# imports data from a txt file froma  given location
def getData(fileLocation):
    with open(fileLocation, newline = '') as file:
        data_reader = csv.reader(file, delimiter='\t')
        for j in range(18):
            next(data_reader,None)
        data = [line for line in data_reader]
        xdata = [i[0] for i in data]
        ydata = [i[1] for i in data]
        xdata = [float(i) for i in xdata]
        ydata = [float(i) for i in ydata]
    return xdata,ydata   

#-----------------------------make extremas------------------------------------
# given an nx2 list, first index val, 2nd uncertainty, return the extrema values
def MakeExtrema(parameter):
    maximum=parameter[0]+parameter[1]
    minimum=parameter[0]-parameter[1]
    return [maximum,parameter[1]],[minimum,parameter[1]]

#--------------------------Get dictionaty key----------------------------------
# return the key of a given value in a dict. If key not in dict, retun False
def getKey(val,dictionary): 
    for key, value in dictionary.items(): 
         if val == value: 
             return key
    return False
 
#--------------------Quadratic error propagation-------------------------------
 #error propagation for multiplications. Input is a list of all uncertainties
def QuadErrorProp(uncertainties):
    error=0
    for uncertainty in uncertainties:
        error+=uncertainty**2
    return (error**0.5)

def QuadErrorPropDict(Dict):
    keylist=list(Dict.keys())
    keylist.sort()
    error=0
    for parameter in keylist:
        err= Dict[parameter]
        error+=err**2
    return (error**0.5)
#-------------------------------get mean---------------------------------------
# gets mean of a list
def GetMean(listForCalculation):
    mean=sum(listForCalculation)/len(listForCalculation)
    return mean

#------------------------------get stDev---------------------------------------
#gets stDev of a list 
def GetDeviation(listForCalculation,originalVal):
    dev=0
    for x in listForCalculation:
        dev+=abs(x-originalVal)
    return((dev/(len(listForCalculation))))

#--------------------------MakeP0andBounds-------------------------------------
#  making the lists that are input as initial guess, upper and lower bounds used 
    #by th curve_Fit function
def MakeP0andBounds(guess1,guess2,guess3,stdev):  

    pnaught=[]
    upperLim=[]
    lowerLim=[]
    modifier=(2*stdev[0])
    
    if guess1!=None and guess2==None and guess3==None:
        pnaught.extend([guess1[0]])
        upperLim.extend([guess1[0]+(modifier)])
        lowerLim.extend([guess1[0]-(modifier)])
        
    if guess1!=None and guess2!=None and guess3==None:
        pnaught.extend([guess1[0],guess2[0]])
        upperLim.extend([guess1[0]+modifier,guess2[0]+modifier])
        lowerLim.extend([guess1[0]-modifier,guess2[0]-modifier])
        
    if guess1!=None and guess2!=None and guess3!=None:
        pnaught.extend([guess1[0],guess2[0],guess3[0]])
        upperLim.extend([guess1[0]+modifier,guess2[0]+modifier,guess3[0]+modifier])
        lowerLim.extend([guess1[0]-modifier,guess2[0]-modifier,guess3[0]-modifier])   
    return pnaught,lowerLim,upperLim 

#-------------------------getCountsinPeaks-------------------------------------
# fir a single, doublet or triplet peak, determin the fraction of counts that exist within
# each individual peak in the spectrum
def getCountsInPeak(A1,A2,A3,spectraTotalCount):
    
    if A2==None and A3==None:
        return spectraTotalCount,None,None
    
    elif A2!=None and A3==None:
        peak1Counts=round(((A1[0])/(A1[0]+A2[0]))*spectraTotalCount)
        peak2Counts=round(((A2[0])/(A1[0]+A2[0]))*spectraTotalCount)
        return peak1Counts,peak2Counts,None
    
    elif A3!=None:
        peak1Counts=round(((A1[0])/(A1[0]+A2[0]+A3[0]))*spectraTotalCount)
        peak2Counts=round(((A2[0])/(A1[0]+A2[0]+A3[0]))*spectraTotalCount)
        peak3Counts=round(((A3[0])/(A1[0]+A2[0]+A3[0]))*spectraTotalCount)
        return peak1Counts,peak2Counts,peak3Counts
    
#-------------------------getStatUncertainty-----------------------------------   
 # this is A*FWHM/sqrt(N) uncertainty for each individual peak
def getStatUncertainty(peak1Counts,peak2Counts,peak3Counts,stdev):
    constant=0.53*2.355*stdev[0]
    peak1StatUncertainty=constant/(peak1Counts)**0.5
    
    if peak2Counts==None:
        return peak1StatUncertainty
    if peak3Counts==None:
        peak2StatUncertainty=constant/(peak2Counts)**0.5
        return peak1StatUncertainty,peak2StatUncertainty
    
    peak2StatUncertainty=constant/(peak2Counts)**0.5
    peak3StatUncertainty=constant/(peak3Counts)**0.5
    
    return peak1StatUncertainty,peak2StatUncertainty,peak3StatUncertainty

#------------------------------Plotcalib---------------------------------------  
# used to visualy see the centroid variation as each individual perameter is varied
def Plotcalib(calibxdata,calibydata,calibparams,popt):
        #plot to calib 
        plt.plot(calibxdata, calibydata, 'k-', label='data')
        ploterCalib=singlefunc(calibxdata, calibparams, popt[0])
        logGraphingCopy=[]
        for yValue in ploterCalib:
            if yValue<(0.5):
                logGraphingCopy.append(0.0)
            else:
                logGraphingCopy.append(yValue)
        plt.plot(calibxdata, logGraphingCopy, 'b--', label='func10')
        plt.yscale('log')
        plt.show()
        
#------------------------------PlotDoublet-------------------------------------  
# used to visualy see the centroid variation as each individual perameter is varied
def PlotDoublet(ioixdata2,ioiydata2,calibparams,popt2Peak):#popt2Peak1
#        #plot to IOI data (2 peak)
        plt.plot(ioixdata2,ioiydata2, 'k-', label='data')
        ploterDouble=doublefunc(ioixdata2, calibparams, *popt2Peak)
        logGraphingCopy=[]
        for yValue in ploterDouble:
            if yValue<(0.5):
                logGraphingCopy.append(0.0)
            else:
                logGraphingCopy.append(yValue)
        plt.plot(ioixdata2, logGraphingCopy, 'b--', label='HypEMG')
        plt.yscale('log')
        plt.show() 

#-------------------------------PlotTriplet------------------------------------  
# used to visualy see the centroid variation as each individual perameter is varied
def PlotTriplet(ioixdata,ioiydata,calibparams,popt3Peak):#popt3Peak1
        # plot to IOI data (3 peak)
        plt.plot(ioixdata,ioiydata, 'k-', label='data')
        ploterTrip=triplefunc(ioixdata, calibparams, *popt3Peak)
        logGraphingCopy=[]
        for yValue in ploterTrip:
            if yValue<(0.5):
                logGraphingCopy.append(0.0)
            else:
                logGraphingCopy.append(yValue)
        plt.plot(ioixdata, logGraphingCopy, 'b--', label='HypEMG')
        plt.yscale('log')
        plt.show()
# =============================================================================
# Main 
# =============================================================================

leftTailNumber = 2
rightTailNumber = 2
    
def Main(): 
    nm1=[None,None]#initialize a bunch of variables so that the users 
    nm2=[None,None]#can simply copy paste, and not have to worrrie about 
    nm3=[None,None]#which were not printed in the H-EMG fit
    tm1=[None,None]
    tm2=[None,None]
    tm3=[None,None]# users can ignore these please
    np1=[None,None]
    np2=[None,None]
    np3=[None,None]
    np1=[None,None]
    tp1=[None,None]
    tp2=[None,None]
    tp3=[None,None]
    
    # file paths to data
    calibfile = '/Applications/coding_stuff/Triumf_Code/Matthew Smith Co-op 2019 Files/Mass 96 files/Final H-EMG Fit txt Files/Final_Calibrant_Cutoff_Test_Files/95.95100-95.9560.txt'
    triplePeakFile = '/Applications/coding_stuff/Triumf_Code/Matthew Smith Co-op 2019 Files/Mass 96 files/Final H-EMG Fit txt Files/Fit Used In Publication/Kr Peak Data.txt'
    doublePeakFile ='/Applications/coding_stuff/Triumf_Code/Matthew Smith - Archive/Kr Data/Mo Calibration for H-EMG/Txt Files/TRC 3 Block/Mo Peak TRC.txt'

#-----------------------Parameters from calibration----------------------------   

    center         =[95.95269820934844             ,1.59809001501288e-05]
    stdev          =[0.00013014977846647843        ,1.3719159661340796e-05]
    weight         =[0.14756874970889566           ,0.04477469947293635 ]
    scaling        =[1                             ,0                   ]
    tm1            =[0.003096478428660216          ,0.0029064830859289814]
    nm1            =[0.21111926658371635           ,0.09732316119300333 ]
    tm2            =[0.00017836330699497318        ,4.290803221946973e-05]
    nm2            =[0.7888807334162836            ,0.3636634781027106  ]
    np1            =[0.17570284090864577           ,0.09568397969027687 ]
    tp1            =[0.000611221853257956          ,0.00016032954270512784]
    tp2            =[0.00020856713772129663        ,2.9717723097950295e-05]
    np2            =[0.8242971590913543            ,0.44889446420652057 ]
       
#--------------------Perameters for double peak error analysis-----------------  

    AI2            =[7.5043199833352645            ,0.14016946478846604 ]
    AII2           =[0.6455061202110107            ,0.04226976609920313 ]
    CI2            =[95.90399691410488             ,4.933377213054878e-06]
    CII2           =[95.90761936242832             ,1.6383969794799112e-05]
    
#--------------------Perameters for Triple peak error analysis-----------------   
    
    AI3            =[0.33022621688675824           ,0.0064203309356999725]
    AII3           =[0.8680816323864224            ,0.010277560357001137]
    AIII3          =[0.006777046131809915          ,0.0010875037645273291]
    CI3            =[95.93411988118544             ,5.0801611718788195e-06]
    CII3           =[95.93910515319644             ,3.003965504009841e-06]
    CIII3          =[95.9423176336484              ,4.203102656840622e-05]
    
#-------------------------Individual Deviations--------------------------------
    # print the deviations that makeup the total uncertainty for a given peak
    ShowAllCalib=False
    ShowAllDoubletPeak1=False
    ShowAllDoubletPeak2=False
    ShowAllTripletPeak1=False
    ShowAllTripletPeak2=False      
    ShowAllTripletPeak3=True
    
    #import calibrant data, gets normalized
    calibxdata,calibydata=getData(calibfile)
    errorbarsCalib = numpy.asarray([math.sqrt(i+1) for i in calibydata])
    NormalizationFactor = TrapezoidalArea(calibxdata,calibydata)
    
    calibStatErr=getStatUncertainty(sum(calibydata),None,None,stdev)
    
    calibydata=list(numpy.asarray(calibydata)/NormalizationFactor)    
    errorbarsCalib=numpy.asarray(errorbarsCalib)/NormalizationFactor
    
    
    # import data for  3 peak set
    ioixdata,ioiydata = getData(triplePeakFile)
    errorbarsIOI =numpy.asarray([math.sqrt(i+1) for i in ioiydata])
    Tripletpeak1Counts,Tripletpeak2Counts,Tripletpeak3Counts = getCountsInPeak(AI3,AII3,AIII3,sum(ioiydata))
    TripPeak1StatErr,TripPeak2StatErr,TripPeak3StatErr=getStatUncertainty(Tripletpeak1Counts,Tripletpeak2Counts,Tripletpeak3Counts,stdev)
    
      
    # import data with 2 peaks, not currently used
    ioixdata2,ioiydata2 = getData(doublePeakFile)
    errorbarsIOI2 =numpy.asarray([math.sqrt(i+1) for i in ioiydata2])
    Doubletpeak1Counts,Doubletpeak2Counts,placeholder = getCountsInPeak(AI2,AII2,None,sum(ioiydata2))
    DblPeak1StatErr,DblPeak2StatErr=getStatUncertainty(Doubletpeak1Counts,Doubletpeak2Counts,None,stdev)
    
    
    # set perameters to be tested. IF a new perameter is added, please append it to the list below
    parameters=[stdev,weight,tm1,nm1,tm2,nm2,tm3,nm3,np1,tp1,np2,tp2,np3,tp3,AI3,AII3,AIII3,AI2,AII2] 
    calibErrs={}
    TripErrsI={}
    TripErrsII={}
    TripErrsIII={}
    DoubleErrsI={}
    DoubleErrsII={}
    parametersforTriplePeak=['aITrip','aIITrip','aIIITrip']# key names for 'calibparams' of all perameters that only pertain to the Triplet
    parametersforDoublePeak=['aIDouble', 'aIIDouble']# key names for 'calibparams' of all perameters that only pertain to the Doublet
         
 
    # make p0, upper limits and lower limits
    pnaught1,lowerLim1,upperLim1=MakeP0andBounds(center,None,None,stdev)
    pnaught2,lowerLim2,upperLim2=MakeP0andBounds(CI2,CII2,None,stdev,)
    pnaught3,lowerLim3,upperLim3=MakeP0andBounds(CI3,CII3,CIII3,stdev)
    

    #skip perameters tat are not used/ are fixed
    for parameter in parameters:
        if parameter[0]==None or parameter[0]==1:
            continue
        
        calibparams = { # containts all parameters This is used to pass thevalues into the curve_Fit function without being optimized
           'stdev':stdev,'weight':weight,        
           'tm1':tm1,'nm1':nm1,
           'tm2':tm2,'nm2':nm2,
           'tm3':tm3,'nm3':nm3,
           'np1':np1,'tp1':tp1,
           'np2':np2,'tp2':tp2,
           'np3':np3,'tp3':tp3,
           'aIDouble':AI2, 'aIIDouble':AII2,
           'aITrip':AI3,'aIITrip':AII3,
           'aIIITrip':AIII3}
    
        key=getKey(parameter,calibparams)
        paramUpper , paramLower = MakeExtrema(parameter)
        
        calibparams[key]=paramUpper
        
        #fit to calib data, skip parameters that do not apply
        if key not in parametersforTriplePeak and key not in parametersforDoublePeak:
            poptCal1, pcovCal1 = curve_fit(lambda x, c: singlefunc(x, calibparams, c), calibxdata, calibydata, p0=pnaught1, sigma=errorbarsCalib, bounds=(lowerLim1,upperLim1))
            #Plotcalib(calibxdata,calibydata,calibparams,poptCal1) #This plots the fits as they change if you would like to see it
       
        #IOI data (2 peak),skip parameters that do not apply   
        if key not in parametersforTriplePeak:
            popt2Peak1, pcov2Peak1 = curve_fit(lambda x, c1, c2: doublefunc(x, calibparams, c1, c2), ioixdata2, ioiydata2, sigma=errorbarsIOI2, p0=pnaught2, bounds=(lowerLim2,upperLim2))
            #PlotDoublet(ioixdata2,ioiydata2,calibparams,popt2Peak1) #This plots the fits as they change if you would like to see it
        
        #IOI data (3 peak),skip parameters that do not apply   
        if key not in parametersforDoublePeak:
            popt3Peak1, pcov3Peak1 = curve_fit(lambda x, c1, c2, c3: triplefunc(x, calibparams, c1, c2, c3), ioixdata, ioiydata, sigma=errorbarsIOI, p0=pnaught3, bounds=(lowerLim3,upperLim3))
            #PlotTriplet(ioixdata,ioiydata,calibparams,popt3Peak1) #This plots the fits as they change if you would like to see it
            
    #Do it again        
        calibparams[key]=paramLower 
        if paramLower[0]<0:
            paramLower[0]=0.00000000001# cant have negative time constants, (some fits error bars make them negative)
            
            
        if key not in parametersforTriplePeak and key not in parametersforDoublePeak:  
            poptCal2, pcovCal2 = curve_fit(lambda x, c: singlefunc(x, calibparams, c), calibxdata, calibydata, p0=pnaught1,sigma=errorbarsCalib, bounds=(lowerLim1,upperLim1))
                # these are here so they ignore the unwanted parameter changes
            calibErrs[key]=GetDeviation([poptCal1[0],poptCal2[0]],center[0])
            
        if key not in parametersforTriplePeak:
            popt2Peak2, pcov2Peak2 = curve_fit(lambda x, c1, c2: doublefunc(x, calibparams, c1, c2), ioixdata2, ioiydata2, sigma=errorbarsIOI2, p0=pnaught2, bounds=(lowerLim2,upperLim2))

            DoubleErrsI[key]=GetDeviation([popt2Peak1[0],popt2Peak2[0]],CI2[0])
            DoubleErrsII[key]=GetDeviation([popt2Peak1[1],popt2Peak2[1]],CII2[0])
            
        if key not in parametersforDoublePeak:       
            popt3Peak2, pcov3Peak2 = curve_fit(lambda x, c1, c2, c3: triplefunc(x, calibparams, c1, c2, c3), ioixdata, ioiydata, sigma=errorbarsIOI, p0=pnaught3, bounds=(lowerLim3,upperLim3))

            TripErrsI[key]=GetDeviation([popt3Peak1[0],popt3Peak2[0]],CI3[0])
            TripErrsII[key]=GetDeviation([popt3Peak1[1],popt3Peak2[1]],CII3[0])
            TripErrsIII[key]=GetDeviation([popt3Peak1[2],popt3Peak2[2]],CIII3[0])
        
    #Print statements    
    KevPerAmu=931494.0954  
    print('\n---------------------------Calibrant------------------------------')
    print('Calibrant Peak Shape Uncertainty (Kev) : ' + str( QuadErrorPropDict(calibErrs)*KevPerAmu)) 
    print('Statistical Uncertainty in Calib Peak (Kev) : ' + str(calibStatErr*KevPerAmu))
    print('\n--------------------------Triplet Peak----------------------------')
    print('Triple Peak 1 Shape Uncertainty (Kev) : ' + str(QuadErrorPropDict(TripErrsI)*KevPerAmu))
    print('Statistical Uncertainty in Triple Peak 1 (Kev) : ' + str(TripPeak1StatErr*KevPerAmu))
    print('Triple Peak 2 Shape Uncertainty (Kev) : ' + str(QuadErrorPropDict(TripErrsII)*KevPerAmu))
    print('Statistical Uncertainty in Triple Peak 2 (Kev) : ' + str(TripPeak2StatErr*KevPerAmu))
    print('Triple Peak 3 Shape Uncertainty (Kev) : ' + str(QuadErrorPropDict(TripErrsIII)*KevPerAmu))
    print('Statistical Uncertainty in Triple Peak 3 (Kev) : ' + str(TripPeak3StatErr*KevPerAmu))
    print('\n--------------------------Doublet Peak----------------------------')
    print('Double Peak 1 Shape Uncertainty (Kev) : ' + str(QuadErrorPropDict(DoubleErrsI)*KevPerAmu))
    print('Statistical Uncertainty in Double Peak 1 (Kev) : ' + str(DblPeak1StatErr*KevPerAmu))
    print('Double Peak 2 Shape Uncertainty (Kev) : ' + str(QuadErrorPropDict(DoubleErrsII)*KevPerAmu))
    print('Statistical Uncertainty in Double Peak 1 (Kev) : ' + str(DblPeak2StatErr*KevPerAmu))
    print('------------------------------------------------------------------')


    if ShowAllCalib:
        printDictionary(calibErrs)  
    if ShowAllDoubletPeak1:
        printDictionary(DoubleErrsI) 
    if ShowAllDoubletPeak2:
        printDictionary(DoubleErrsII) 
    if ShowAllTripletPeak1:
        printDictionary(TripErrsI) 
    if ShowAllTripletPeak2: 
        printDictionary(TripErrsII) 
    if ShowAllTripletPeak3:
        printDictionary(TripErrsIII)
        
Main()           
