import numpy as np
import scipy.interpolate as sp
import math
import csv
import cv2
import pdb
#from programs.construct_model import f_x_to_model_evaluation
#from programs.construct_model_instrinsic_parameters import f_x_to_model_evaluation

import numpy.matlib
from sklearn.linear_model import LinearRegression

#########################################################################################################################
# Interparc
# Credit: Robert Yi (https://github.com/rsyi)
def diffCOL(matrix):
    newMAT = []
    newROW = []
    for i in range(len(matrix)-1):
        for j in range(len(matrix[i])):
            diff = matrix[i+1][j]-matrix[i][j]
            newROW.append(diff) 
        newMAT.append(newROW)
        newROW = []
    #Stack the matrix to get xyz in columns
    newMAT = np.vstack(newMAT)
    return newMAT

def squareELEM(matrix):            
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            matrix[i][j] = matrix[i][j]*matrix[i][j]
    return matrix
        
def sumROW(matrix):      
    newMAT = []
    for j in range(len(matrix)):
        rowSUM = 0
        for k in range(len(matrix[j])):
            rowSUM = rowSUM + matrix[j][k]
        newMAT.append(rowSUM)  
    return newMAT
            
def sqrtELEM(matrix):
    for i in range(len(matrix)):
        matrix[i] = math.sqrt(matrix[i])
    return matrix      

def sumELEM(matrix):
    sum = 0
    for i in range(len(matrix)):
        sum = sum + matrix[i]
    return sum

def diffMAT(matrix,denom):
    newMAT = []
    for i in range(len(matrix)):
        newMAT.append(matrix[i]/denom)
    return newMAT
    
def cumsumMAT(matrix):
    first = 0
    newmat = []
    newmat.append(first)
    #newmat.append(matrix)
    for i in range(len(matrix)):
        newmat.append(matrix[i])
    cum = 0
    for i in range(len(newmat)):
        cum = cum + newmat[i] 
        newmat[i] = cum
    return newmat
        
def divMAT(A,B):
    newMAT = []
    for i in range(len(A)):
        newMAT.append(A[i] / B[i])
    return newMAT

def minusVector(t,cumarc):
    newMAT = []
    for i in range(len(t)):
        newMAT.append(t[i] - cumarc[i])
    return newMAT
    
def replaceIndex(A,B):
    newMAT = []
    for i in range(len(B)):
        index = B[i]
        newMAT.append(A[index])
    return newMAT        
   
def matSUB(first,second):
    newMAT = []
    newCOL = []
    for i in range(len(first)):
        for j in range(len(first[i])):
            newMAT.append(first[i][j] - second[i][j])
        #newMAT.append(newCOL)
    return newMAT
    
def matADD(first,second):
    newMAT = []
    newCOL = []
    for i in range(len(first)):
        for j in range(len(first[i])):
            newMAT.append(first[i][j]+second[i][j])
        #newMAT.append(newCOL)
    return newMAT
    
def matMULTI(first,second):
    """
    Take in two matrix
    multiply each element against the other at the same index
    return a new matrix
    """
    newMAT = []
    newCOL = []
    for i in range(len(first)):
        for j in range(len(first[i])):
            newMAT.append(first[i][j]*second[i][j])
        #newMAT.append(newCOL)
    return newMAT 
    
def matDIV(first,second):
    """
    Take in two matrix
    multiply each element against the other at the same index
    return a new matrix
    """
    newMAT = []
    newCOL = []
    for i in range(len(first)):
        for j in range(len(first[i])):
            newMAT.append(first[i][j]/second[i][j])
        #newMAT.append(newCOL)
    return newMAT

def vecDIV(first,second):
    """
    Take in two arrays
    multiply each element against the other at the same index
    return a new array
    """
    newMAT = []
    for i in range(len(first)):
        newMAT.append(first[i]/second[i])
    return newMAT

def replaceROW(matrix,replacer,adder):
    newMAT = []
    if adder != 0:
        for i in range(len(replacer)):
            newMAT.append(matrix[replacer[i]+adder])
    else:
        for i in range(len(replacer)):
            newMAT.append(matrix[replacer[i]])
    return np.vstack(newMAT)
            
            
def interparc(t,px,py,*args):
    inputs = [t,px,py,args]
    #If we dont get at least a t, x, and y value we error
    if len(inputs)<3:
        print("ERROR: NOT ENOUGH ARGUMENTS")
        
    #Should check to make sure t is a single integer greater than 1
    t = t
    #if (t>1) and (t%1==0):
    #    t = np.linspace(0,1,t)
    #elif t<0 or t>1:
    #    print("Error: STEP SIZE t IS NOT ALLOWED")
       
    nt = len(t)
    
    px = px
    py = py
    n = len(px)
    
    if len(px) != len(py):
        print("ERROR: MUST BE SAME LENGTH")
    elif n < 2:
        print("ERROR: MUST BE OF LENGTH 2")
        
    pxy = [px,py]
    #pxy = np.transpose(pxy)
    ndim = 2
    
    method = 'linear'
    
    if len(args) > 1:
        if isinstance(args[len(args)-1], basestring) == True:
            method = args[len(args)-1]
            if method != 'linear' and method != 'pchip' and method != 'spline':
                print("ERROR: INVALID METHOD")
    elif len(args)==1:
        method = args[0]
    method = 'linear'
    # Try to append all the arguments together
    for i in range(len(args)):
        if isinstance(args[i], basestring) != True:
            pz = args[i]
            if len(pz) != n:
                print("ERROR: LENGTH MUST BE SAME AS OTHER INPUT")
            pxy.append(pz)
    ndim = len(pxy)
    
    pt = np.zeros((nt,ndim))
#Check for rounding errors here
    # Transpose the matrix to align with matlab codes method
    pxy = np.transpose(pxy)
    chordlen = sqrtELEM(sumROW(squareELEM(diffCOL(pxy))))
    chordlen = diffMAT(chordlen,sumELEM(chordlen))
    cumarc = cumsumMAT(chordlen)
    if method == 'linear':
        inter = np.histogram(bins=t,a=cumarc)
        tbins = inter[1]
        hist= inter[0]
        tbinset=[]
        index=0
        tbinset.append(index)
        
        for i in range(len(hist)):
            if hist[i]>0:
                index=index+hist[i]
                tbinset.append(index)
            else:
                tbinset.append(index)

        for i in range(len(tbinset)):
            if tbinset[i]<= 0 or t[i]<=0:
                tbinset[i] = 1
            elif tbinset[i]>=n or t[i]>=1:
                tbinset[i] = n-1
        #Take off one value to match the way matlab does indexing
        for i in range(len(tbinset)):
            tbinset[i]=tbinset[i]-1

        s = divMAT(minusVector(t,replaceIndex(cumarc,tbinset)),replaceIndex(chordlen,tbinset) )

        #Breakup the parts of pt
        repmat = np.transpose(np.reshape(np.vstack(np.tile(s,(1,ndim))[0]),(ndim,-1)))
        sub = np.reshape( np.vstack( matSUB( replaceROW(pxy,tbinset,1) , replaceROW(pxy,tbinset,0) ) ) , (-1,ndim) )
        multi = np.reshape( np.vstack(matMULTI( sub , repmat )) ,(-1,ndim) )
        pt = np.reshape( np.vstack( matADD( replaceROW(pxy,tbinset,0) , multi ) ) ,(-1,ndim) )
        return pt



#########################################################################################################################
# Form a binary mask around the real image
def mask_real_image(im_real, im_gray):
    ret, bw = cv2.threshold(np.uint8(im_gray), 1, 255, cv2.THRESH_BINARY)  
    kernel = np.ones((7,7), dtype = np.uint8)
    mask = cv2.dilate(bw, kernel, iterations = 1)
    return mask



# Normalize array to 0 - 1
def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def interpolatePts(pt):
    vec = np.diff(pt, n = 1, axis = 1)
    #theta_0 = np.arctan2(vec[1, 0], vec[0, 0])
    #theta_i = np.diff(np.arctan2(vec[1, 0:9], vec[0,0:9]))
    t = normalize_data(np.concatenate((np.array([0, 0.8]),  0.8 + np.cumsum(np.ones((1, 9), dtype = np.float64)))))
    pt_for_interpolation = np.concatenate((np.mean(pt[:, 10 : 12, None], axis = 1), pt[:, 0 : 10]), axis = 1)
    pt = interparc(t, pt_for_interpolation[0, :], pt_for_interpolation[1, :]).T
    return pt

def regressionSmoothing(interpFish, windowSize = 5, fps = 50):
    firstFishInterp = np.copy(interpFish)

    firstFishXInterp = firstFishInterp[:, 0]
    firstFishYInterp = firstFishInterp[:, 1]

    # Linear Regression Smoothing

    # windowSize = 5
    offsetFromCenter = (windowSize - 1) // 2

    choppedfirstFishXInterp = firstFishXInterp[offsetFromCenter:-offsetFromCenter]
    choppedfirstFishYInterp = firstFishYInterp[offsetFromCenter:-offsetFromCenter]

    amountOfDataPoints = len(firstFishXInterp)
    amountOfPointsToCompute = amountOfDataPoints - (2 * offsetFromCenter)

    DT = (1 / fps)
    X = np.arange(0, windowSize)

    # Getting the ys and xs, also getting the speed from that

    x_ = np.matlib.repmat(X, amountOfPointsToCompute, 1)
    # x_ = np.squeeze(x_)
    # x_ = x_.reshape(-1, 3)
    offset = np.arange(x_.shape[0])
    # offset = offset.reshape(1, -1)
    offset = np.stack([offset for _ in range(windowSize)], axis=1)
    indices = x_ + offset
    # print(indices[0])
    # print(indices[1])
    # reshaping them to be able to get all of the values at once
    indices = indices.reshape(-1, 1)
    indices = np.squeeze(indices)
    xYs = firstFishXInterp[indices]
    yYs = firstFishYInterp[indices]
    xYs = xYs.reshape(-1, windowSize)
    yYs = yYs.reshape(-1, windowSize)

    xs = (X + 1) * DT
    xs = xs.reshape(-1, 1)  # Got to reshape it into the certain format
    xSpeeds = []
    ySpeeds = []
    for xy, yy in zip(xYs, yYs):
        if np.any(np.isnan(xy)) or np.any(np.isnan(yy)):
            xSpeeds.append(np.nan)
            ySpeeds.append(np.nan)
            continue

        reg = LinearRegression().fit(xs, xy)
        xSpeed = (reg.coef_)[0]
        xSpeeds.append(xSpeed)

        reg = LinearRegression().fit(xs, yy)
        ySpeed = (reg.coef_)[0]
        ySpeeds.append(ySpeed)

    pads = []
    for _ in range(offsetFromCenter): pads.append(0)
    xSpeeds2 = pads + xSpeeds + pads
    ySpeeds2 = pads + ySpeeds + pads

    xSpeeds2 = np.array(xSpeeds2)
    ySpeeds2 = np.array(ySpeeds2)

    return xSpeeds2, ySpeeds2

def regressionSmoothingWithCC(interpFish, fishIdx, ccData, windowSize = 5, fps = 50, threshold = .7):
    firstFishInterp = np.copy(interpFish)

    firstFishXInterp = firstFishInterp[:, 0]
    firstFishYInterp = firstFishInterp[:, 1]

    # Linear Regression Smoothing

    # windowSize = 5
    offsetFromCenter = (windowSize - 1) // 2

    choppedfirstFishXInterp = firstFishXInterp[offsetFromCenter:-offsetFromCenter]
    choppedfirstFishYInterp = firstFishYInterp[offsetFromCenter:-offsetFromCenter]

    amountOfDataPoints = len(firstFishXInterp)
    amountOfPointsToCompute = amountOfDataPoints - (2 * offsetFromCenter)

    DT = (1 / fps)
    X = np.arange(0, windowSize)

    # Getting the ys and xs, also getting the speed from that

    x_ = np.matlib.repmat(X, amountOfPointsToCompute, 1)
    # x_ = np.squeeze(x_)
    # x_ = x_.reshape(-1, 3)
    offset = np.arange(x_.shape[0])
    # offset = offset.reshape(1, -1)
    offset = np.stack([offset for _ in range(windowSize)], axis=1)
    indices = x_ + offset
    # print(indices[0])
    # print(indices[1])
    # reshaping them to be able to get all of the values at once
    indices = indices.reshape(-1, 1)
    indices = np.squeeze(indices)
    xYs = firstFishXInterp[indices]
    yYs = firstFishYInterp[indices]
    xYs = xYs.reshape(-1, windowSize)
    yYs = yYs.reshape(-1, windowSize)

    xs = (X + 1) * DT
    xs = xs.reshape(-1, 1)  # Got to reshape it into the certain format
    xSpeeds = []
    ySpeeds = []

    for frameIdx, (xy, yy) in enumerate(zip(xYs, yYs)):
        frameIdx += offsetFromCenter

        ccs = ccData[frameIdx - offsetFromCenter: frameIdx + offsetFromCenter + 1, fishIdx]
        if np.any(ccs < threshold) or np.any(np.isnan(xy)) or np.any(np.isnan(yy)):
            xSpeeds.append(np.nan)
            ySpeeds.append(np.nan)
            continue

        reg = LinearRegression().fit(xs, xy)
        xSpeed = (reg.coef_)[0]
        xSpeeds.append(xSpeed)

        reg = LinearRegression().fit(xs, yy)
        ySpeed = (reg.coef_)[0]
        ySpeeds.append(ySpeed)

    pads = []
    for _ in range(offsetFromCenter): pads.append(0)
    xSpeeds2 = pads + xSpeeds + pads
    ySpeeds2 = pads + ySpeeds + pads

    xSpeeds2 = np.array(xSpeeds2)
    ySpeeds2 = np.array(ySpeeds2)

    return xSpeeds2, ySpeeds2



