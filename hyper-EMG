
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 18:14:54 2019

@author: matthewsmith
"""

# coding: utf-8

# questions

# =============================================================================
# Author: Chris Izzo , Matthew Smith
# Updated September 2019 
# =============================================================================



# =============================================================================
# Update Notes:
# =============================================================================



# # Import Needed Packages
import matplotlib.pyplot as plt
import numpy
import math
from scipy.optimize import curve_fit
import scipy.special
import csv

# =============================================================================
#User Inputs
# =============================================================================


# =============================================================================
# # # Definitions for Fit Functions
# =============================================================================

#-----------------------------------logGraphing--------------------------------
# graphing values of the order E-6 can cause log graphs to become distorted
#Remove all values below 0.5 for graphing purposes, original lists are NOT mutated
def logGraphing(xdata,functionValues,colour,labl):
    logGraphingCopy=[]
    for yValue in  functionValues:
        if yValue<(0.5):
            logGraphingCopy.append(0.0)
        else:
            logGraphingCopy.append(yValue)
    plt.plot(xdata,logGraphingCopy, colour, label=labl)

#-------------------------------getCountsInPeak--------------------------------
#Dreturn the number of counts within each HEMG peak making up a given spectrum
def getCountsInPeak(EMGValues,spectraTotalCount,numberOfPeaks):
    
    if numberOfPeaks==1:
        return spectraTotalCount,None,None
    
    elif numberOfPeaks==2:
        AI=EMGValues['AI'+str(numberOfPeaks)][0]
        AII=EMGValues['AII'+str(numberOfPeaks)][0]
        peak1Counts=round(((AI)/(AI+AII))*spectraTotalCount)
        peak2Counts=round(((AII)/(AI+AII))*spectraTotalCount)
        
        return peak1Counts,peak2Counts,None
    
    elif numberOfPeaks==3:
        AI=EMGValues['AI'+str(numberOfPeaks)][0]
        AII=EMGValues['AII'+str(numberOfPeaks)][0]
        AIII=EMGValues['AIII'+str(numberOfPeaks)][0]
        peak1Counts=round(((AI)/(AI+AII+AIII))*spectraTotalCount)
        peak2Counts=round(((AII)/(AI+AII+AIII))*spectraTotalCount)
        peak3Counts=round(((AIII)/(AI+AII+AIII))*spectraTotalCount)
        
        return peak1Counts,peak2Counts,peak3Counts

#-------------------------------Printing Blocks--------------------------------
# these are for organization, they pring out information pertaining to the given fit    
# block 1 is for H-EMG calibration fitting
# Block 2 is for fitting that peak shape calibrant to other peaks
def PrintingBlock1(calibparams,xdata,ydata,redchi2,leftTailNumber,rightTailNumber):
    print('----------------------------------------------------------------------')
    print('Hyper-EMG ('+str(leftTailNumber)+','+str(rightTailNumber)+') Fit:\n')
    print('{:<15} {:<30} {:<20}'.format('Parameter','Fit Value','Uncertainty'))
    for par, v in calibparams.items():
        val, err = v
        print('{:<15}=[{:<30},{:<20}]'.format(par, val, err))
    print('\nintegral of fitted peak for A=1: %s'% str(TrapezoidalArea(xdata,ydata)))  
    print('\nCorresponding Reduced Chi Squared: %s' % str(redchi2))


def printBlock2(EMGValues,spectraTotalCount,numberOfPeaks,redchi2):
    print('----------------------------------------------------------------------')
    print('Hyper-EMG ('+str(numberOfPeaks)+' Peak Fit):\n')
    peak1Counts,peak2Counts,peak3Counts = getCountsInPeak(EMGValues,spectraTotalCount,numberOfPeaks) 
    keylist=list(EMGValues.keys())
    keylist.sort()
    for parameter in keylist:
        val= EMGValues[parameter][0]
        err= EMGValues[parameter][1]
        print('{:<15}=[{:<30},{:<20}]'.format(parameter, val, err))
    
    print('\nTotal counts in spectra =%s' % str(round(spectraTotalCount)))    
    print('Counts in first peak =%s' % str(peak1Counts))  
    
    if peak2Counts!=None:
        print('Counts in second peak =%s' % str(peak2Counts))
        
    if peak3Counts!=None:
        print('Counts in third peak =%s' % str(peak3Counts))
    print('\nCorresponding Reduced Chi Squared: %s' % str(redchi2))     
 
#-----------------------------------Chi squared--------------------------------
def chi2(y_measure,y_predict,errors):
    """Calculate the chi squared value given a measurement with errors and prediction"""
    return numpy.sum( (y_measure - y_predict)**2 / errors**2 )

def chi2reduced(y_measure, y_predict, errors, number_of_parameters):
    """Calculate the reduced chi squared value given a measurement with errors and prediction,
    and knowing the number of parameters in the model."""
    return chi2(y_measure, y_predict, errors)/(numpy.asarray(y_measure).size - number_of_parameters)

#--------------------------------TrapezoidalArea-------------------------------
# for integration
def TrapezoidalArea(xData,yData):
    area=0
    for count in range(0,len(xData)-2):
        area+=((xData[count+1] - xData[count]) * 0.5 * ((yData[count+1] + yData[count])))
        count+=1
    return area

#-------------------------------------LoadFile---------------------------------
#Takes path to txt file as a string and outputs the xdata and y data in the txt file
def LoadFile(filePath):
    with open(filePath, newline = '') as file:
        data_reader = csv.reader(file, delimiter='\t')
        for j in range(18):
            next(data_reader,None)
        data = [line for line in data_reader]
        xdata = [i[0] for i in data]
        ydata = [i[1] for i in data]
        xdata = [float(i) for i in xdata]
        ydata = [float(i) for i in ydata]
    return xdata,ydata

#-----------------------SubtractFit--SubtractFWHM------------------------------
# subtracts a fit from the originl y data, leaving only the residules
def SubtractFit(yData,fitYData):
    subtractedData=numpy.asarray(yData)-numpy.asarray(fitYData)
    return subtractedData

#use in combination with SubtractFit. S
#somwtimes the peak max values are non gaussian, making it dificult to look
#at the ploted results of SubtractFit. SO we subtract the FWHM to focus
#on the tails
def SubtractFWHM(xData,yData,gaussianSigma,yMaxXVal):
    FWHM=gaussianSigma*2.355
    newyData=[]
    for counter,x in enumerate(xData):
        if abs(x-yMaxXVal)>(FWHM/2):
            newyData.append(yData[counter])
        else:
            newyData.append(0)
    return newyData

#-------------------------------RemoveNegatives--------------------------------
# removes the negative values froma list
def RemoveNegatives(yData):
    copy=[]
    for y in yData:
        if y<0:
            copy.append(0)
        else:
            copy.append(y)
    return copy

#--------------------------text file printing----------------------------------
# print xdata    ydata, if y is normalized an d you wish to un normalize,
#enter normalization factor, if not, enter None
def printoutFunctionforTxt(xdata,fitdata,normalization):
    if normalization==None:
        normalization=1  
    for x in range(0,len(fitdata)):
        print(str(xdata[x])+'\t'+str(fitdata[x]*normalization))
#------------------------------Make P0 and Bounds------------------------------
# this wil make the intital guess, upper lim and lower lim lists that curve_Fit uses, based off 
#your inital inputs for the gusses. Note that for the positive tails, the fuction
#definition was made wonky, so i had to adjust to keep up with the ordering
    
# Version 1 is for the calibration peak, versino 2 is for fitting the peak shape to
# a multi peak spectrum
def MakeP0andBounds1(inputDict,leftTailNumber,rightTailNumber):
    pnaught=[]
    upperLim=[]
    lowerLim=[]
    
    pnaught.append(inputDict['center'])
    lowerLim.append(inputDict['center']-inputDict['stdev'])
    upperLim.append(inputDict['center']+inputDict['stdev'])
    
    pnaught.append(inputDict['stdev'])
    lowerLim.append(inputDict['stdev']/2)
    upperLim.append(inputDict['stdev']*2)

    pnaught.append(inputDict['weight'])
    lowerLim.append(0)
    upperLim.append(1)
    
    if leftTailNumber>0:
        pnaught.append(inputDict['tm1'])
        lowerLim.append(0)
        upperLim.append(1000)
        
    if leftTailNumber>1:
        pnaught.append(inputDict['nm1'])
        lowerLim.append(0)
        upperLim.append(1)

        pnaught.append(inputDict['tm2'])
        lowerLim.append(0)
        upperLim.append(1000) 
        
    if leftTailNumber>2:
        pnaught.append(inputDict['nm2'])
        lowerLim.append(0)
        upperLim.append(1)
        
        pnaught.append(inputDict['tm3'])  
        lowerLim.append(0)
        upperLim.append(1000)
        
    if rightTailNumber>1:
        pnaught.append(inputDict['np1'])
        lowerLim.append(0)
        upperLim.append(1)
        
    if rightTailNumber>0:
        pnaught.append(inputDict['tp1'])
        lowerLim.append(0)
        upperLim.append(1000)   
        
    if rightTailNumber>2:
        pnaught.append(inputDict['np2']) 
        lowerLim.append(0)
        upperLim.append(1)
        
    if rightTailNumber>1:
        pnaught.append(inputDict['tp2'])
        lowerLim.append(0)
        upperLim.append(1000)
        
        
        
    if rightTailNumber>2:
        pnaught.append(inputDict['tp3'])  
        lowerLim.append(0)
        upperLim.append(1000) 
        
    return pnaught,lowerLim,upperLim
# version 2
def MakeP0andBounds2(inputDict,guess1,guess2,guess3):  
    pnaught=[]
    upperLim=[]
    lowerLim=[]
    modifier=(2*inputDict['stdev'][0])
    pnaught.append(1)
    upperLim.append(500)
    lowerLim.append(0)
    # i made the initil guess 100 for amplitudes 100, with bounds 0,500
    # you can change those if you want
    
    if guess2==None and guess3==None:
        pnaught.extend(guess1)
        upperLim.extend(guess1+(modifier))
        lowerLim.extend(guess1-(modifier))
        
    if guess2!=None and guess3==None:
        pnaught.extend([1,guess1,guess2])
        upperLim.extend([500,guess1+modifier,guess2+modifier])
        lowerLim.extend([0,guess1-modifier,guess2-modifier])
        
    if guess3!=None:
        pnaught.extend([1,1,guess1,guess2,guess3])
        upperLim.extend([500,500,guess1+modifier,guess2+modifier,guess3+modifier])
        lowerLim.extend([0,0,guess1-modifier,guess2-modifier,guess3-modifier])

        
    return pnaught,lowerLim,upperLim
#-------------------------------CurveFitAndAnalyse-----------------------------
# this wil fit your H-EMG Fn to the calibration data. It is generalized for all
#combinatino of tails up to 3,3
def CurveFitAndAnalyse(function,xdata,ydata,errorbars,pnaught,lowerLim,upperLim):
    popt, pcov = curve_fit(function, xdata, ydata, sigma=errorbars, p0=pnaught,  bounds=(lowerLim,upperLim))
    errs = numpy.sqrt(numpy.diag(pcov))
    fit = function(xdata, *popt)
    nParams = len(pnaught)
    funcredchi2 = chi2reduced(ydata, fit, errorbars, nParams)
    return fit,funcredchi2,popt,errs

#--------------------------------createCalibparams-----------------------------
#this will create the 'calibration parameters' dictionary that will
#be used to save the optimized parameters found from th peak shape calibrant
# is generalized for any combinatino of tails up to 3,3
def createCalibparams(popt,errs,leftTailNumber,rightTailNumber):
    # note the indecies of perameters within each popt and errs is different 
    #for every version of Hyper EMG. The indecies of eah parameter are dependent on
    #the order of initialization of that parameter withing the H-EMG functino definitions
    calibparams = {'center':[popt[0],errs[0]],
               'stdev':[popt[1],errs[1]],
               'weight':[popt[2],errs[2]],
               'scaling':[1,0]}

    if leftTailNumber>0:
        calibparams['tm1']=[popt[3],errs[3]]
        if leftTailNumber==1:
            calibparams['nm1']=[1,0]
            
    if leftTailNumber>1:
        calibparams['nm1']=[popt[4],errs[4]]
        calibparams['tm2']=[popt[5],errs[5]]
        
        if leftTailNumber==2:
            nm2=1-(popt[4])
            nm2error=errs[4]/popt[4]*nm2
            calibparams['nm2']=[nm2,nm2error]

    if leftTailNumber>2:
        calibparams['nm2']=[popt[6],errs[6]]
        calibparams['tm3']=[popt[7],errs[7]]
        nm3=1-(popt[6])-(popt[4])
        nm3error=math.sqrt(((errs[6]/popt[6])**2)+((errs[4]/popt[4])**2))*nm3
        calibparams['nm3']=[nm3,nm3error]
    
    if leftTailNumber==0:
        placeholder=3 
    else:
        placeholder=2+(2*leftTailNumber) 
        
    if rightTailNumber==1:
        calibparams['np1']=[1,0]
        calibparams['tp1']=[popt[placeholder],errs[placeholder]] 
            
    elif rightTailNumber==2:
        calibparams['np1']=[popt[placeholder],errs[placeholder]]
        calibparams['tp1']=[popt[placeholder+1],errs[placeholder+1]] 
        calibparams['tp2']=[popt[placeholder+2],errs[placeholder+2]] 
        np2=1-(popt[placeholder])
        np2error=errs[placeholder]/popt[placeholder]*np2  
        calibparams['np2']=[np2,np2error]
        
    elif rightTailNumber==3:
        calibparams['np1']=[popt[placeholder],errs[placeholder]]
        calibparams['tp1']=[popt[placeholder+1],errs[placeholder+1]] 
        calibparams['np2']=[popt[placeholder+2],errs[placeholder+2]] 
        calibparams['tp2']=[popt[placeholder+3],errs[placeholder+3]] 
        calibparams['tp3']=[popt[placeholder+4],errs[placeholder+4]]
        np3=1-(popt[placeholder+2])-(popt[placeholder])
        np3error=math.sqrt(((errs[placeholder+2]/popt[placeholder+2])**2)+((errs[placeholder]/popt[placeholder])**2))*np3
        calibparams['np3']=[np3,np3error]
        
    return calibparams

#---------------------------------createEMGValues------------------------------

#this creates the dictionary that stores the optimized values from the triple, double and
# single peak fitting functions
def createEMGValues(popt,errs,numberOfPeaks):
    EMGValues={}
    EMGValues['AI'+str(numberOfPeaks)]=[popt[0],errs[0]]
    EMGValues['CI'+str(numberOfPeaks)]=[popt[numberOfPeaks],errs[numberOfPeaks]]
    
    if numberOfPeaks>1:
        EMGValues['AII'+str(numberOfPeaks)]=[popt[1],errs[1]]
        EMGValues['CII'+str(numberOfPeaks)]=[popt[numberOfPeaks+1],errs[numberOfPeaks+1]]
        
        seperation21=popt[numberOfPeaks+1]-popt[numberOfPeaks]
        seperationError21=(errs[numberOfPeaks+1]**2+errs[numberOfPeaks]**2)**0.5

        EMGValues['Peak seperation (2-1)']=[seperation21,seperationError21]
    
    if numberOfPeaks>2:
        EMGValues['AIII'+str(numberOfPeaks)]=[popt[2],errs[2]]
        EMGValues['CIII'+str(numberOfPeaks)]=[popt[numberOfPeaks+2],errs[numberOfPeaks+2]]
  
        seperation31=popt[numberOfPeaks+2]-popt[numberOfPeaks]
        seperationError31=(errs[numberOfPeaks+2]**2+errs[numberOfPeaks]**2)**0.5
        seperation32=popt[numberOfPeaks+2]-popt[numberOfPeaks+1]
        seperationError32=(errs[numberOfPeaks+2]**2+errs[numberOfPeaks+1]**2)**0.5
        
        EMGValues['Peak seperation (3-1)']=[seperation31,seperationError31]
        EMGValues['Peak seperation (3-2)']=[seperation32,seperationError32]
    
    return EMGValues
#-------------------------------findIOIMaxLocation-----------------------------
#finds the x value that corsponds to the maximum y value in 2 list of the same length
def findIOIMaxLocation(xdata,ydata):
    ymax = max(ydata)
    ymaxindex = ydata.index(ymax)
    maxlocation = xdata[ymaxindex]
    return maxlocation

#-------------------------Get the H-emg Function-------------------------------
# just sifts through the possible func and returns the one the user wants
def GethemgFunction(leftTailNumber,rightTailNumber): 
    if leftTailNumber==1 and rightTailNumber==0:
        function=func10
    if leftTailNumber==2 and rightTailNumber==0:
        function=func20
    if leftTailNumber==3 and rightTailNumber==0:
        function=func30
    if leftTailNumber==0 and rightTailNumber==1:
        function=func01
    if leftTailNumber==0 and rightTailNumber==2:
        function=func02
    if leftTailNumber==0 and rightTailNumber==3:
        function=func03
    if leftTailNumber==1 and rightTailNumber==1:
        function=func11
    if leftTailNumber==1 and rightTailNumber==2:
        function=func12
    if leftTailNumber==1 and rightTailNumber==3:
        function=func13
    if leftTailNumber==2 and rightTailNumber==1:
        function=func21
    if leftTailNumber==2 and rightTailNumber==2:
        function=func22
    if leftTailNumber==2 and rightTailNumber==3:
        function=func23
    if leftTailNumber==3 and rightTailNumber==1:
        function=func31
    if leftTailNumber==3 and rightTailNumber==2:
        function=func32
    if leftTailNumber==3 and rightTailNumber==3:
        function=func33
    return function

#---------------------------Gaussian Function----------------------------------
def gaussfunc(x,m,w,a):
    #return ((a/numpy.pi)*(w/((x-m)**2+w**2)))
    #return a*numpy.exp(-(x-m)**2/(w**2))
    return (a/(w*math.sqrt(numpy.pi*2)))*numpy.exp(-0.5*((x-m)/w)**2)
#--------------------H-EMG Function Definitions--------------------------------
def func01(x,c,s,w,tp1):
    penelization=0
    np1=1
    if (s/tp1)>10:
        penelization=5000
    postail1 = (np1/(2*tp1))*((numpy.exp(((s/(math.sqrt(2)*tp1))**2-((x-c)/tp1))))*scipy.special.erfc((s/(math.sqrt(2)*tp1))-((x-c)/(math.sqrt(2)*s))))
    postail1 = numpy.nan_to_num(postail1)
    hmin = 0
    hplu = postail1
    return (w*hmin +(1-w)*hplu)+penelization

def func02(x,c,s,w,np1,tp1,tp2):
    penelization=0
    np2=1-np1
    if  (s/tp1)>10 or (s/tp2)>10:
        penelization=5000
    if (np1+np2)>1.05:
        penelization=5000
    if  np1<0 or np2<0:
        penelization=5000
    postail1 = (np1/(2*tp1))*((numpy.exp(((s/(math.sqrt(2)*tp1))**2-((x-c)/tp1))))*scipy.special.erfc((s/(math.sqrt(2)*tp1))-((x-c)/(math.sqrt(2)*s))))
    postail1 = numpy.nan_to_num(postail1)
    postail2 = (np2/(2*tp2))*((numpy.exp(((s/(math.sqrt(2)*tp2))**2-((x-c)/tp2))))*scipy.special.erfc((s/(math.sqrt(2)*tp2))-((x-c)/(math.sqrt(2)*s))))
    postail2 = numpy.nan_to_num(postail2)
    hmin = 0
    hplu = postail1 + postail2
    return (w*hmin + (1-w)*hplu)+penelization


def func03(x,c,s,w,np1,tp1,np2,tp2,tp3):
    penelization=0
    np3=1-np2-np1
    if (s/tp1)>10 or (s/tp2)>10 or (s/tp3)>10:
        penelization=5000
    if (np1+np2)>1.05 or (np1+np2)<0.95 :
        penelization=5000
    if np1<0  or np2<0 or np3<0:
        penelization=5000
    postail1 = (np1/(2*tp1))*((numpy.exp(((s/(math.sqrt(2)*tp1))**2-((x-c)/tp1))))*scipy.special.erfc((s/(math.sqrt(2)*tp1))-((x-c)/(math.sqrt(2)*s))))
    postail1 = numpy.nan_to_num(postail1)
    postail2 = (np2/(2*tp2))*((numpy.exp(((s/(math.sqrt(2)*tp2))**2-((x-c)/tp2))))*scipy.special.erfc((s/(math.sqrt(2)*tp2))-((x-c)/(math.sqrt(2)*s))))
    postail2 = numpy.nan_to_num(postail2)
    postail3 = (np3/(2*tp3))*((numpy.exp(((s/(math.sqrt(2)*tp3))**2-((x-c)/tp3))))*scipy.special.erfc((s/(math.sqrt(2)*tp3))-((x-c)/(math.sqrt(2)*s))))
    postail3 = numpy.nan_to_num(postail3)
    hmin = 0
    hplu = postail1 + postail2 + postail3
    return (w*hmin + (1-w)*hplu)+penelization



def func10(x,c,s,w,tm1):
    penelization=0
    nm1=1
    if (s/tm1)>10:
        penelization=5000
    negtail1 = negtail1 = (nm1/(2*tm1))*((numpy.exp(((s/(math.sqrt(2)*tm1))**2+((x-c)/tm1))))*scipy.special.erfc((s/(math.sqrt(2)*tm1))+((x-c)/(math.sqrt(2)*s))))
    negtail1 = numpy.nan_to_num(negtail1)
    hmin = negtail1
    hplu = 0
    return (w*hmin + (1-w)*hplu)+penelization

def func20(x,c,s,w,tm1,nm1,tm2):
    penelization=0
    nm2=1-nm1
    if (s/tm1)>10 or (s/tm2)>10:
        penelization=5000
    if (nm1+nm2)>1.05 or (nm1+nm2)<0.95:
        penelization=5000
    if nm1<0 or nm2<0:
        penelization=5000
    negtail1 = (nm1/(2*tm1))*((numpy.exp(((s/(math.sqrt(2)*tm1))**2+((x-c)/tm1))))*scipy.special.erfc((s/(math.sqrt(2)*tm1))+((x-c)/(math.sqrt(2)*s))))
    negtail1 = numpy.nan_to_num(negtail1)
    negtail2 = (nm2/(2*tm2))*((numpy.exp(((s/(math.sqrt(2)*tm2))**2+((x-c)/tm2))))*scipy.special.erfc((s/(math.sqrt(2)*tm2))+((x-c)/(math.sqrt(2)*s))))
    negtail2 = numpy.nan_to_num(negtail2)
    hmin = negtail1 + negtail2
    hplu = 0
    return (w*hmin + (1-w)*hplu)+penelization

def func30(x,c,s,w,tm1,nm1,tm2,nm2,tm3):
    penelization=0
    nm3=1-nm2-nm1
    if (nm1+nm2+nm3)>1.05 or (nm1+nm2+nm3)<0.95:
        penelization=5000
    if (s/tm1)>10 or (s/tm2)>10 or (s/tm3)>10:
        penelization=5000
    if nm1<0  or nm2<0 or nm3<0:
        penelization=5000
    negtail1 = (nm1/(2*tm1))*((numpy.exp(((s/(math.sqrt(2)*tm1))**2+((x-c)/tm1))))*scipy.special.erfc((s/(math.sqrt(2)*tm1))+((x-c)/(math.sqrt(2)*s))))
    negtail1 = numpy.nan_to_num(negtail1)
    negtail2 = (nm2/(2*tm2))*((numpy.exp(((s/(math.sqrt(2)*tm2))**2+((x-c)/tm2))))*scipy.special.erfc((s/(math.sqrt(2)*tm2))+((x-c)/(math.sqrt(2)*s))))
    negtail2 = numpy.nan_to_num(negtail2)
    negtail3 = (nm3/(2*tm3))*((numpy.exp(((s/(math.sqrt(2)*tm3))**2+((x-c)/tm3))))*scipy.special.erfc((s/(math.sqrt(2)*tm3))+((x-c)/(math.sqrt(2)*s))))
    negtail3 = numpy.nan_to_num(negtail3)
    hmin = negtail1 + negtail2 + negtail3
    hplu = 0
    return (w*hmin + (1-w)*hplu)+penelization

def func11(x,c,s,w,tm1,tp1):
    penelization=0
    nm1=1
    np1=1
    if (s/tm1)>10  or (s/tp1)>10:
        penelization=5000
    if nm1<0 or np1<0:
        penelization=5000
    negtail1 = (nm1/(2*tm1))*((numpy.exp(((s/(math.sqrt(2)*tm1))**2+((x-c)/tm1))))*scipy.special.erfc((s/(math.sqrt(2)*tm1))+((x-c)/(math.sqrt(2)*s))))
    negtail1 = numpy.nan_to_num(negtail1)
    postail1 = (np1/(2*tp1))*((numpy.exp(((s/(math.sqrt(2)*tp1))**2-((x-c)/tp1))))*scipy.special.erfc((s/(math.sqrt(2)*tp1))-((x-c)/(math.sqrt(2)*s))))
    postail1 = numpy.nan_to_num(postail1)
    hmin = negtail1
    hplu = postail1
    return (w*hmin + (1-w)*hplu)+penelization

def func21(x,c,s,w,tm1,nm1,tm2,tp1):
    penelization=0
    np1=1
    nm2=1-nm1
    if (s/tm1)>10 or (s/tm2)>10 or (s/tp1)>10:
        penelization=5000
    if (nm1+nm2)>1.05 or (nm1+nm2)<0.95 :
        penelization=5000
    if nm1<0 or np1<0 or nm2<0:
        penelization=5000
    negtail1 = (nm1/(2*tm1))*((numpy.exp(((s/(math.sqrt(2)*tm1))**2+((x-c)/tm1))))*scipy.special.erfc((s/(math.sqrt(2)*tm1))+((x-c)/(math.sqrt(2)*s))))
    negtail1 = numpy.nan_to_num(negtail1)
    negtail2 = (nm2/(2*tm2))*((numpy.exp(((s/(math.sqrt(2)*tm2))**2+((x-c)/tm2))))*scipy.special.erfc((s/(math.sqrt(2)*tm2))+((x-c)/(math.sqrt(2)*s))))
    negtail2 = numpy.nan_to_num(negtail2)
    postail1 = (np1/(2*tp1))*((numpy.exp(((s/(math.sqrt(2)*tp1))**2-((x-c)/tp1))))*scipy.special.erfc((s/(math.sqrt(2)*tp1))-((x-c)/(math.sqrt(2)*s))))
    postail1 = numpy.nan_to_num(postail1)
    hmin = negtail1 + negtail2
    hplu = postail1
    return (w*hmin + (1-w)*hplu)+penelization

def func31(x,c,s,w,tm1,nm1,tm2,nm2,tm3,tp1):
    penelization=0
    nm3=1-nm1-nm2
    np1=1
    if (s/tm1)>10 or (s/tm2)>10 or (s/tm3)>10 or (s/tp1)>10:
        penelization=5000
    if (nm1+nm2+nm3)>1.05 or (nm1+nm2+nm3)<0.95 :
        penelization=5000
    if nm1<0 or np1<0 or nm2<0 or nm3<0:
        penelization=5000
    negtail1 = (nm1/(2*tm1))*((numpy.exp(((s/(math.sqrt(2)*tm1))**2+((x-c)/tm1))))*scipy.special.erfc((s/(math.sqrt(2)*tm1))+((x-c)/(math.sqrt(2)*s))))
    negtail1 = numpy.nan_to_num(negtail1)
    negtail2 = (nm2/(2*tm2))*((numpy.exp(((s/(math.sqrt(2)*tm2))**2+((x-c)/tm2))))*scipy.special.erfc((s/(math.sqrt(2)*tm2))+((x-c)/(math.sqrt(2)*s))))
    negtail2 = numpy.nan_to_num(negtail2)
    negtail3 = (nm3/(2*tm3))*((numpy.exp(((s/(math.sqrt(2)*tm3))**2+((x-c)/tm3))))*scipy.special.erfc((s/(math.sqrt(2)*tm3))+((x-c)/(math.sqrt(2)*s))))
    negtail3 = numpy.nan_to_num(negtail3)
    postail1 = (np1/(2*tp1))*((numpy.exp(((s/(math.sqrt(2)*tp1))**2-((x-c)/tp1))))*scipy.special.erfc((s/(math.sqrt(2)*tp1))-((x-c)/(math.sqrt(2)*s))))
    postail1 = numpy.nan_to_num(postail1)
    hmin = negtail1 + negtail2 + negtail3
    hplu = postail1
    return (w*hmin + (1-w)*hplu)+penelization

def func32(x,c,s,w,tm1,nm1,tm2,nm2,tm3,np1,tp1,tp2):
    penelization=0
    nm3=1-nm2-nm1
    np2=1-np1
    if (s/tm1)>10 or (s/tm2)>10 or (s/tm3)>10 or (s/tp1)>10 or (s/tp2)>10:
        penelization=5000
    if (nm1+nm2+nm3)>1.05 or (nm1+nm2+nm3)<0.95 or (np1+np2)>1.05 or (np1+np2)<0.95:
        penelization=5000
    if nm1<0 or np1<0 or nm2<0 or np2<0 or nm3<0:
        penelization=5000
    negtail1 = (nm1/(2*tm1))*((numpy.exp(((s/(math.sqrt(2)*tm1))**2+((x-c)/tm1))))*scipy.special.erfc((s/(math.sqrt(2)*tm1))+((x-c)/(math.sqrt(2)*s))))
    negtail1 = numpy.nan_to_num(negtail1)
    negtail2 = (nm2/(2*tm2))*((numpy.exp(((s/(math.sqrt(2)*tm2))**2+((x-c)/tm2))))*scipy.special.erfc((s/(math.sqrt(2)*tm2))+((x-c)/(math.sqrt(2)*s))))
    negtail2 = numpy.nan_to_num(negtail2)
    negtail3 = (nm3/(2*tm3))*((numpy.exp(((s/(math.sqrt(2)*tm3))**2+((x-c)/tm3))))*scipy.special.erfc((s/(math.sqrt(2)*tm3))+((x-c)/(math.sqrt(2)*s))))
    negtail3 = numpy.nan_to_num(negtail3)
    postail1 = (np1/(2*tp1))*((numpy.exp(((s/(math.sqrt(2)*tp1))**2-((x-c)/tp1))))*scipy.special.erfc((s/(math.sqrt(2)*tp1))-((x-c)/(math.sqrt(2)*s))))
    postail1 = numpy.nan_to_num(postail1)
    postail2 = (np2/(2*tp2))*((numpy.exp(((s/(math.sqrt(2)*tp2))**2-((x-c)/tp2))))*scipy.special.erfc((s/(math.sqrt(2)*tp2))-((x-c)/(math.sqrt(2)*s))))
    postail2 = numpy.nan_to_num(postail2)
    hmin = negtail1 + negtail2 + negtail3
    hplu = postail1 + postail2
    return (w*hmin + (1-w)*hplu)+penelization

def func12(x,c,s,w,tm1,np1,tp1,tp2):
    penelization=0
    nm1=1
    np2=1-np1
    if (s/tm1)>10 or (s/tp1)>10 or (s/tp2)>10:
        penelization=5000
    if (np1+np2)>1.05 or (np1+np2)<0.95:
        penelization=5000
    if nm1<0 or np1<0 or np2<0:
        penelization=5000
    negtail1 = (nm1/(2*tm1))*((numpy.exp(((s/(math.sqrt(2)*tm1))**2+((x-c)/tm1))))*scipy.special.erfc((s/(math.sqrt(2)*tm1))+((x-c)/(math.sqrt(2)*s))))
    negtail1 = numpy.nan_to_num(negtail1)
    postail1 = (np1/(2*tp1))*((numpy.exp(((s/(math.sqrt(2)*tp1))**2-((x-c)/tp1))))*scipy.special.erfc((s/(math.sqrt(2)*tp1))-((x-c)/(math.sqrt(2)*s))))
    postail1 = numpy.nan_to_num(postail1)
    postail2 = (np2/(2*tp2))*((numpy.exp(((s/(math.sqrt(2)*tp2))**2-((x-c)/tp2))))*scipy.special.erfc((s/(math.sqrt(2)*tp2))-((x-c)/(math.sqrt(2)*s))))
    postail2 = numpy.nan_to_num(postail2)
    hmin = negtail1
    hplu = postail1 + postail2
    return (w*hmin + (1-w)*hplu)+penelization

def func22(x,c,s,w,tm1,nm1,tm2,np1,tp1,tp2):
    penelization=0
    np2=1-np1
    nm2=1-nm1
    if (s/tm1)>10 or (s/tm2)>10 or (s/tp1)>10 or (s/tp2)>10 :
        penelization=5000
    if (nm1+nm2)>1.05 or (nm1+nm2)<0.95 or (np1+np2)>1.05 or (np1+np2)<0.95:
        penelization=5000
    if nm1<0 or np1<0 or nm2<0 or np2<0:
        penelization=5000
    negtail1 = (nm1/(2*tm1))*((numpy.exp(((s/(math.sqrt(2)*tm1))**2+((x-c)/tm1))))*scipy.special.erfc((s/(math.sqrt(2)*tm1))+((x-c)/(math.sqrt(2)*s))))
    negtail1 = numpy.nan_to_num(negtail1)
    negtail2 = (nm2/(2*tm2))*((numpy.exp(((s/(math.sqrt(2)*tm2))**2+((x-c)/tm2))))*scipy.special.erfc((s/(math.sqrt(2)*tm2))+((x-c)/(math.sqrt(2)*s))))
    negtail2 = numpy.nan_to_num(negtail2)
    postail1 = (np1/(2*tp1))*((numpy.exp(((s/(math.sqrt(2)*tp1))**2-((x-c)/tp1))))*scipy.special.erfc((s/(math.sqrt(2)*tp1))-((x-c)/(math.sqrt(2)*s))))
    postail1 = numpy.nan_to_num(postail1)
    postail2 = (np2/(2*tp2))*((numpy.exp(((s/(math.sqrt(2)*tp2))**2-((x-c)/tp2))))*scipy.special.erfc((s/(math.sqrt(2)*tp2))-((x-c)/(math.sqrt(2)*s))))
    postail2 = numpy.nan_to_num(postail2)
    hmin = negtail1 + negtail2
    hplu = postail1 + postail2
    return (w*hmin + (1-w)*hplu)+penelization

def func13(x,c,s,w,tm1,np1,tp1,np2,tp2,tp3):
    penelization=0
    nm1=1
    np3=1-np2-np1
    if (s/tm1)>10 or (s/tp1)>10 or (s/tp2)>10 or (s/tp3)>10:
        penelization=5000
    if  (np1+np2+np3)>1.05 or (np1+np2+np3)<0.95:
        penelization=5000
    if nm1<0 or np1<0 or np2<0 or np3<0:
        penelization=5000
    negtail1 = (nm1/(2*tm1))*((numpy.exp(((s/(math.sqrt(2)*tm1))**2+((x-c)/tm1))))*scipy.special.erfc((s/(math.sqrt(2)*tm1))+((x-c)/(math.sqrt(2)*s))))
    negtail1 = numpy.nan_to_num(negtail1)
    postail1 = (np1/(2*tp1))*((numpy.exp(((s/(math.sqrt(2)*tp1))**2-((x-c)/tp1))))*scipy.special.erfc((s/(math.sqrt(2)*tp1))-((x-c)/(math.sqrt(2)*s))))
    postail1 = numpy.nan_to_num(postail1)
    postail2 = (np2/(2*tp2))*((numpy.exp(((s/(math.sqrt(2)*tp2))**2-((x-c)/tp2))))*scipy.special.erfc((s/(math.sqrt(2)*tp2))-((x-c)/(math.sqrt(2)*s))))
    postail2 = numpy.nan_to_num(postail2)
    postail3 = (np3/(2*tp3))*((numpy.exp(((s/(math.sqrt(2)*tp3))**2-((x-c)/tp3))))*scipy.special.erfc((s/(math.sqrt(2)*tp3))-((x-c)/(math.sqrt(2)*s))))
    postail3 = numpy.nan_to_num(postail3)
    hmin = negtail1
    hplu = postail1 + postail2 + postail3
    return (w*hmin + (1-w)*hplu)+penelization

def func23(x,c,s,w,tm1,nm1,tm2,np1,tp1,np2,tp2,tp3):
    penelization=0
    nm2=1-nm1
    np3=1-np2-np1
    if (s/tm1)>10 or (s/tm2)>10 or (s/tp1)>10 or (s/tp2)>10 or (s/tp3)>10:
        penelization=5000
    if (nm1+nm2)>1.05 or (nm1+nm2)<0.95 or (np1+np2+np3)>1.05 or (np1+np2+np3)<0.95:
        penelization=5000
    if nm1<0 or np1<0 or nm2<0 or np2<0 or np3<0:
        penelization=5000
    negtail1 = (nm1/(2*tm1))*((numpy.exp(((s/(math.sqrt(2)*tm1))**2+((x-c)/tm1))))*scipy.special.erfc((s/(math.sqrt(2)*tm1))+((x-c)/(math.sqrt(2)*s))))
    negtail1 = numpy.nan_to_num(negtail1)
    negtail2 = (nm2/(2*tm2))*((numpy.exp(((s/(math.sqrt(2)*tm2))**2+((x-c)/tm2))))*scipy.special.erfc((s/(math.sqrt(2)*tm2))+((x-c)/(math.sqrt(2)*s))))
    negtail2 = numpy.nan_to_num(negtail2)
    postail1 = (np1/(2*tp1))*((numpy.exp(((s/(math.sqrt(2)*tp1))**2-((x-c)/tp1))))*scipy.special.erfc((s/(math.sqrt(2)*tp1))-((x-c)/(math.sqrt(2)*s))))
    postail1 = numpy.nan_to_num(postail1)
    postail2 = (np2/(2*tp2))*((numpy.exp(((s/(math.sqrt(2)*tp2))**2-((x-c)/tp2))))*scipy.special.erfc((s/(math.sqrt(2)*tp2))-((x-c)/(math.sqrt(2)*s))))
    postail2 = numpy.nan_to_num(postail2)
    postail3 = (np3/(2*tp3))*((numpy.exp(((s/(math.sqrt(2)*tp3))**2-((x-c)/tp3))))*scipy.special.erfc((s/(math.sqrt(2)*tp3))-((x-c)/(math.sqrt(2)*s))))
    postail3 = numpy.nan_to_num(postail3)
    hmin = negtail1 + negtail2
    hplu = postail1 + postail2 + postail3
    return (w*hmin + (1-w)*hplu)+penelization

def func33(x,c,s,w,tm1,nm1,tm2,nm2,tm3,np1,tp1,np2,tp2,tp3):
    penelization=0
    nm3=1-nm2-nm1
    np3=1-np2-np1
    if (s/tm1)>10 or (s/tm2)>10 or (s/tm3)>10 or (s/tp1)>10 or (s/tp2)>10 or (s/tp3)>10:
        penelization=5000
    if (nm1+nm2+nm3)>1.05 or (nm1+nm2+nm3)<0.95 or (np1+np2+np3)>1.05 or (np1+np2+np3)<0.95:
        penelization=5000
    if nm1<0 or np1<0 or nm2<0 or np2<0 or nm3<0 or np3<0:
        penelization=5000
    negtail1 = (nm1/(2*tm1))*((numpy.exp(((s/(math.sqrt(2)*tm1))**2+((x-c)/tm1))))*scipy.special.erfc((s/(math.sqrt(2)*tm1))+((x-c)/(math.sqrt(2)*s))))
    negtail1 = numpy.nan_to_num(negtail1)
    negtail2 = (nm2/(2*tm2))*((numpy.exp(((s/(math.sqrt(2)*tm2))**2+((x-c)/tm2))))*scipy.special.erfc((s/(math.sqrt(2)*tm2))+((x-c)/(math.sqrt(2)*s))))
    negtail2 = numpy.nan_to_num(negtail2)
    negtail3 = (nm3/(2*tm3))*((numpy.exp(((s/(math.sqrt(2)*tm3))**2+((x-c)/tm3))))*scipy.special.erfc((s/(math.sqrt(2)*tm3))+((x-c)/(math.sqrt(2)*s))))
    negtail3 = numpy.nan_to_num(negtail3)
    postail1 = (np1/(2*tp1))*((numpy.exp(((s/(math.sqrt(2)*tp1))**2-((x-c)/tp1))))*scipy.special.erfc((s/(math.sqrt(2)*tp1))-((x-c)/(math.sqrt(2)*s))))
    postail1 = numpy.nan_to_num(postail1)
    postail2 = (np2/(2*tp2))*((numpy.exp(((s/(math.sqrt(2)*tp2))**2-((x-c)/tp2))))*scipy.special.erfc((s/(math.sqrt(2)*tp2))-((x-c)/(math.sqrt(2)*s))))
    postail2 = numpy.nan_to_num(postail2)
    postail3 = (np3/(2*tp3))*((numpy.exp(((s/(math.sqrt(2)*tp3))**2-((x-c)/tp3))))*scipy.special.erfc((s/(math.sqrt(2)*tp3))-((x-c)/(math.sqrt(2)*s))))
    postail3 = numpy.nan_to_num(postail3)
    hmin = negtail1 + negtail2 + negtail3
    hplu = postail1 + postail2 + postail3
    return (w*hmin + (1-w)*hplu)+penelization


#---------------------------------Multiple H-EMG Fit functions--------------------------------
# these functions are for fitting your peak shape calibrant to the spectra of interest
# for a file with 2 peaks
def doublefunc(x, calibdict, aI, aII, cI, cII):
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
    
    if leftTailNumber> 0:
        nm1 = calibdict['nm1'][0]
        tm1 = calibdict['tm1'][0]
        negtail1I = (nm1/(2*tm1))*((numpy.exp(((s/(math.sqrt(2)*tm1))**2+((x-cI)/tm1))))*scipy.special.erfc((s/(math.sqrt(2)*tm1))+((x-cI)/(math.sqrt(2)*s))))
        negtail1I = numpy.nan_to_num(negtail1I)
        negtail1II = (nm1/(2*tm1))*((numpy.exp(((s/(math.sqrt(2)*tm1))**2+((x-cII)/tm1))))*scipy.special.erfc((s/(math.sqrt(2)*tm1))+((x-cII)/(math.sqrt(2)*s))))
        negtail1II = numpy.nan_to_num(negtail1II)
    if leftTailNumber> 1:
        nm2 = calibdict['nm2'][0]
        tm2 = calibdict['tm2'][0]
        negtail2I = (nm2/(2*tm2))*((numpy.exp(((s/(math.sqrt(2)*tm2))**2+((x-cI)/tm2))))*scipy.special.erfc((s/(math.sqrt(2)*tm2))+((x-cI)/(math.sqrt(2)*s))))
        negtail2I = numpy.nan_to_num(negtail2I)
        negtail2II = (nm2/(2*tm2))*((numpy.exp(((s/(math.sqrt(2)*tm2))**2+((x-cII)/tm2))))*scipy.special.erfc((s/(math.sqrt(2)*tm2))+((x-cII)/(math.sqrt(2)*s))))
        negtail2II = numpy.nan_to_num(negtail2II)
    if leftTailNumber>2:
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
    
# for a file with 3 peaks
def triplefunc(x, calibdict, aI, aII, aIII, cI, cII, cIII):
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
    if leftTailNumber> 0:
        nm1 = calibdict['nm1'][0]
        tm1 = calibdict['tm1'][0]
        negtail1I = (nm1/(2*tm1))*((numpy.exp(((s/(math.sqrt(2)*tm1))**2+((x-cI)/tm1))))*scipy.special.erfc((s/(math.sqrt(2)*tm1))+((x-cI)/(math.sqrt(2)*s))))
        negtail1I = numpy.nan_to_num(negtail1I)
        negtail1II = (nm1/(2*tm1))*((numpy.exp(((s/(math.sqrt(2)*tm1))**2+((x-cII)/tm1))))*scipy.special.erfc((s/(math.sqrt(2)*tm1))+((x-cII)/(math.sqrt(2)*s))))
        negtail1II = numpy.nan_to_num(negtail1II)
        negtail1III = (nm1/(2*tm1))*((numpy.exp(((s/(math.sqrt(2)*tm1))**2+((x-cIII)/tm1))))*scipy.special.erfc((s/(math.sqrt(2)*tm1))+((x-cIII)/(math.sqrt(2)*s))))
        negtail1III = numpy.nan_to_num(negtail1III)
    if leftTailNumber> 1:
        nm2 = calibdict['nm2'][0]
        tm2 = calibdict['tm2'][0]
        negtail2I = (nm2/(2*tm2))*((numpy.exp(((s/(math.sqrt(2)*tm2))**2+((x-cI)/tm2))))*scipy.special.erfc((s/(math.sqrt(2)*tm2))+((x-cI)/(math.sqrt(2)*s))))
        negtail2I = numpy.nan_to_num(negtail2I)
        negtail2II = (nm2/(2*tm2))*((numpy.exp(((s/(math.sqrt(2)*tm2))**2+((x-cII)/tm2))))*scipy.special.erfc((s/(math.sqrt(2)*tm2))+((x-cII)/(math.sqrt(2)*s))))
        negtail2II = numpy.nan_to_num(negtail2II)
        negtail2III = (nm2/(2*tm2))*((numpy.exp(((s/(math.sqrt(2)*tm2))**2+((x-cIII)/tm2))))*scipy.special.erfc((s/(math.sqrt(2)*tm2))+((x-cIII)/(math.sqrt(2)*s))))
        negtail2III = numpy.nan_to_num(negtail2III)
    if leftTailNumber>2:
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
      

# =============================================================================
# BEGIN CODE BODY
# =============================================================================


# =============================================================================
# Fit calibrant datawith a Hyper EMG
# =============================================================================
def FitCalibrantData(calibfile,function,printfit):
    # get function from specified left and right tail numbers      
    calibxdata,calibydata=LoadFile(calibfile)# load calibrant file
        
    NormalizationFactor=TrapezoidalArea(calibxdata,calibydata)#get normalization factor
    errorbars = numpy.asarray([math.sqrt(i+1) for i in calibydata]) # get error bars
    # normalize data and error bars  
    calibydata=list(numpy.asarray(calibydata)/NormalizationFactor)
    errorbars=numpy.asarray(errorbars)/NormalizationFactor
     
    #fit and plot a gaussian fit to the data
    calibmaxlocation=findIOIMaxLocation(calibxdata,calibydata)
    gaussianFit,gaussredchi2,popt,errs=CurveFitAndAnalyse(gaussfunc,calibxdata,calibydata,errorbars,[calibmaxlocation, 1, 1],None,None)
    plt.plot(calibxdata, calibydata, 'k-', label='data')
    logGraphing(calibxdata,gaussianFit, 'g--', 'gaussfit')
    initialGuesses['center']=popt[0]
    initialGuesses['stdev']=popt[1]
    
    #Fit Calibration Peak with HyperEMG Fit
    pnaught,lowerLim,upperLim=MakeP0andBounds1(initialGuesses,leftTailNumber,rightTailNumber)
    fitdata,funcredchi2,poptfunc,errsfunc=CurveFitAndAnalyse(function,calibxdata,calibydata,errorbars,pnaught,lowerLim,upperLim)
    calibparams = createCalibparams(poptfunc,errsfunc,leftTailNumber,rightTailNumber)
    logGraphing(calibxdata,fitdata,'b--','HypEMG('+str(leftTailNumber)+','+str(rightTailNumber)+')')
    placeholder=[]
    placeholder.append([calibparams,leftTailNumber,rightTailNumber,funcredchi2])


    #print the info
    print('\nCounts in calibration peak =%5.3d' % round(sum(calibydata) ))
    print('integral of all data: %s'% str(TrapezoidalArea(calibxdata,calibydata))) 
    print('Unmodified Gussian Centroid and StDev : ' + str([popt[0],popt[1]]))
    PrintingBlock1(calibparams,calibxdata,fitdata,funcredchi2,leftTailNumber,rightTailNumber)
    
    # if the user wants to export the fits, print it out
    if printfit[0]:
        print('\n---------Calibration Peak Fit---------\n')
        printoutFunctionforTxt(calibxdata,fitdata,NormalizationFactor)
        
    #plotting details
    fig_size = plt.rcParams['figure.figsize']
    fig_size[0] = 8.0
    fig_size[1] = 6.0
    plt.xlabel('Mass (u)')
    plt.ylabel('Counts')
    plt.legend()
    plt.yscale('log')
    plt.show()
    
    subtractedFityIOI=SubtractFit(calibydata,fitdata)
    subtractedyIOI=SubtractFWHM(calibxdata,subtractedFityIOI,calibparams['stdev'][0],poptfunc[3])
    fig_size = plt.rcParams['figure.figsize']
    fig_size[0] = 9.0
    fig_size[1] = 6.0
    plt.xlabel('Mass (u)')
    plt.ylabel('Counts')
    plt.plot(calibxdata,(subtractedyIOI),label='Residual')
    plt.legend()
    plt.show()
    
    
    print('\n----------------------------------------------------------------------')
    return calibparams,poptfunc,errsfunc


# =============================================================================
# Fit a triplet peak with a Hyper EMG
# =============================================================================
def FitTriplePeak(filepath,function,calibparams,popt,errs,peak1guess,peak2guess,peak3guess,printfit):
    
    #load and plot data from txt file
    ioixdata,ioiydata=LoadFile(filepath)# load file data
    errorbarsioi = numpy.asarray([math.sqrt(i+1) for i in ioiydata])# get error bars
    ioipeakcount=sum(ioiydata)
    plt.plot(ioixdata, ioiydata, 'k-', label='data')# plot the data
    
    #fit the tripple peak with the hyper EMG, using pre-determined peak shape
    pnaught,lowerLim,upperLim=MakeP0andBounds2(calibparams,peak1guess,peak2guess,peak3guess)
    
    poptEMG, pcovEMG = curve_fit(lambda x, a1, a2, a3, c1, c2, c3: triplefunc(x, calibparams, a1, a2, a3, c1, c2, c3), ioixdata, ioiydata, sigma=errorbarsioi, p0=pnaught, bounds=(lowerLim,upperLim))
    fitErrsEMG = numpy.sqrt(numpy.diag(pcovEMG))
    fitparams = 6   
    funcIoIredchi2 = chi2reduced(ioiydata, triplefunc(ioixdata, calibparams, *poptEMG), errorbarsioi, fitparams)      
    finalFit=triplefunc(ioixdata, calibparams, *poptEMG)

    # get data from the fit and display it
    EMGValues=createEMGValues(poptEMG,fitErrsEMG,3)
    printBlock2(EMGValues,ioipeakcount,3,funcIoIredchi2) 

    #plot each peaks fit individualy
    plt.plot(ioixdata, finalFit, 'b--', label='HypEMG')
    ploter1=poptEMG[0]*function(ioixdata,poptEMG[3],*popt[1:])
    ploter2=poptEMG[1]*function(ioixdata,poptEMG[4],*popt[1:])
    ploter3=poptEMG[2]*function(ioixdata,poptEMG[5],*popt[1:])
    plt.plot(ioixdata,ploter1,'r--')
    plt.plot(ioixdata,ploter2,'r--')
    plt.plot(ioixdata,ploter3,'r--')
    
    # if the user wants to export the fits, print it out
    if printfit[0]:
        if printfit[1]==0:
            print('\n--------Triplet Peak Sum Fit--------\n')
            printoutFunctionforTxt(ioixdata,finalFit,None)
        if printfit[1]==1:
            print('\n---------Peak 1 Triplet Fit---------\n')
            printoutFunctionforTxt(ioixdata,ploter1,None)
        if printfit[1]==2:
            print('\n---------Peak 2 Triplet Fit---------\n')
            printoutFunctionforTxt(ioixdata,ploter2,None)
        if printfit[1]==3:
            print('\n---------Peak 3 Triplet Fit---------\n')
            printoutFunctionforTxt(ioixdata,ploter3,None)
        
        
    # plot formatting  
    fig_size = plt.rcParams['figure.figsize']
    fig_size[0] = 8.0
    fig_size[1] = 6.0
    plt.xlabel('Mass (u)')
    plt.ylabel('Counts')
    plt.legend()
    plt.yscale('log')
    plt.ylim(1,max(ioiydata)*1.1)
    plt.show()
    
    # plot how much the fit missed (residuals plot)
    subtractedFityIOI=SubtractFit(ioiydata,finalFit)
    subtractedyIOI=SubtractFWHM(ioixdata,subtractedFityIOI,calibparams['stdev'][0],poptEMG[0])
    fig_size = plt.rcParams['figure.figsize']
    fig_size[0] = 9.0
    fig_size[1] = 6.0
    plt.xlabel('Mass (u)')
    plt.ylabel('Counts')
    plt.plot(ioixdata,subtractedyIOI,label='Residual')
    plt.legend()
    plt.show()
    
#---------------------------------------------------------------------------------------------------------------------------------
# =============================================================================
# Fit a doublet peak with a Hyper EMG
# =============================================================================
def FitDoublePeak(filepath,function,calibparams,popt,errs,peak1guess,peak2guess,printfit):
    #load and plot data from txt file
    ioixdata,ioiydata=LoadFile(filepath)  # load file data  
    errorbars = numpy.asarray([math.sqrt(i+1) for i in ioiydata]) # get error bars
    ioipeakcount=sum(ioiydata)
    plt.plot(ioixdata, ioiydata, 'k-', label='data')# plot the data
    
    #fit the doublet peak with the hyper EMG, using pre-determined peak shape
    pnaught,lowerLim,upperLim=MakeP0andBounds2(calibparams,peak1guess,peak2guess,None)
    poptEMG, pcovEMG = curve_fit(lambda x, a1, a2, c1, c2: doublefunc(x, calibparams, a1, a2, c1, c2), ioixdata, ioiydata,sigma=errorbars, p0=pnaught, bounds=(lowerLim,upperLim))
    fitErrsEMG = numpy.sqrt(numpy.diag(pcovEMG))
    fitparams = 4
    funcIoIredchi2 = chi2reduced(ioiydata, doublefunc(ioixdata, calibparams, *poptEMG), errorbars, fitparams)  
    finalFit=doublefunc(ioixdata, calibparams, *poptEMG)

    # get data from the fit and display it
    EMGValues=createEMGValues(poptEMG,fitErrsEMG,2)
    printBlock2(EMGValues,ioipeakcount,2,funcIoIredchi2) 
    
    #plot each peaks fit individualy
    plt.plot(ioixdata, finalFit, 'b--', label='HypEMG')
    ploter1=poptEMG[0]*function(ioixdata,poptEMG[2],*popt[1:])
    ploter2=poptEMG[1]*function(ioixdata,poptEMG[3],*popt[1:])
    plt.plot(ioixdata,ploter1,'r--')
    plt.plot(ioixdata,ploter2,'r--')

    # if the user wants to export the fits, print it out
    if printfit[0]:
        if printfit[1]==0:
            print('\n--------Doublet Peak Sum Fit--------\n')
            printoutFunctionforTxt(ioixdata,finalFit,None)
        if printfit[1]==1:
            print('\n---------Peak 1 Doublet Fit---------\n')
            printoutFunctionforTxt(ioixdata,ploter1,None)
        if printfit[1]==2:
            print('\n---------Peak 2 Doublet Fit---------\n')
            printoutFunctionforTxt(ioixdata,ploter2,None)
    # plot formatting  
    fig_size = plt.rcParams['figure.figsize']
    fig_size[0] = 8.0
    fig_size[1] = 6.0
    plt.xlabel('Mass (u)')
    plt.ylabel('Counts')
    plt.legend()
    plt.yscale('log')
    plt.ylim(1,max(ioiydata)*1.1)
    plt.show()
    
    # plot how much the fit missed (residuals plot)
    subtractedFityIOI=SubtractFit(ioiydata,finalFit)
    subtractedyIOI=SubtractFWHM(ioixdata,subtractedFityIOI,calibparams['stdev'][0],poptEMG[3])
    subtractedyIOI=RemoveNegatives(subtractedyIOI)
    fig_size = plt.rcParams['figure.figsize']
    fig_size[0] = 9.0
    fig_size[1] = 6.0
    plt.xlabel('Mass (u)')
    plt.ylabel('Counts')
    plt.plot(ioixdata,(subtractedyIOI),label='Residual')
    plt.legend()
    plt.show()
    
    print('\n----------------------------------------------------------------------')
  
      
  
# =============================================================================
# User Inputs
# =============================================================================

calibfile = 'H:/Files to put in public folder/Mass 96 files/Final H-EMG Fit txt Files/Calibrant Data (95.95150-95.9565).txt'
triplePeakFile = 'H:/Files to put in public folder/Mass 96 files/Final H-EMG Fit txt Files/Kr Peak Data.txt'
doublePeakFile ='H:/Kr Data/Mo Calibration for H-EMG/Txt Files/TRC 3 Block/Mo Peak TRC.txt'

triplepeak1guess = 95.93420
triplepeak2guess = 95.93919
triplepeak3guess = 95.94246

doublepeak1guess = 95.90412
doublepeak2guess = 95.90772

leftTailNumber = 1
rightTailNumber = 2

initialGuesses={'weight':0.2,
               'tm1':0.05,
               'nm1':0.45,
               'tm2':0.15,
               'nm2':0.3,
               'tm3':0.1,
               'np1':0.31,
               'tp1':0.1,
               'np2':0.15,
               'tp2':0.05,
               'tp3':0.05} 

# to display the fit data for exporting to another program
#1st index is if you want to see the fit, enter True or False

#2nd index is which peak you wish to see;
#0 = the sum peak
#1 = farthest left individual peak
#2 = second peak from the left
#3 = third peak from the left
# if a invalid numberis enteres, it will not print.

displayCalibFitData=[False,0]
displayTripletFitData=[True,0]
displayDoubletFitData=[False,0]
# call functions
function=GethemgFunction(leftTailNumber,rightTailNumber)
calibparams,popt,errs=FitCalibrantData(calibfile,function,displayCalibFitData)
FitTriplePeak(triplePeakFile,function,calibparams,popt,errs,triplepeak1guess,triplepeak2guess,triplepeak3guess,displayTripletFitData)
FitDoublePeak(doublePeakFile,function,calibparams,popt,errs,doublepeak1guess,doublepeak2guess,displayDoubletFitData)



