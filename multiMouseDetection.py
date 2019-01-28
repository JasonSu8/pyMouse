# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 09:41:00 2018

@author: Administrator
"""

import os
import numpy as np
import cv2
import mouseLocalization
from skimage.morphology import skeletonize as skn
from matplotlib import pyplot as plt


def npBackAdd(front,back):
    frontLogic=np.int32(front!=0)
    backLogic=np.int32(back!=0)
    addLogic=np.int32(np.logical_or(frontLogic,backLogic))
    back0to1=back.copy()
    backAddLogic=np.int32(back0to1==0)
    back0to1[backAddLogic==1]=1
    backAdd=addLogic*back0to1
    frontTo1=front.copy()
    frontTo1[backAddLogic==0]=1
    result=frontTo1*backAdd
    return(result)

def npMaxCross(front,back):
    backLogic=np.int32(back>1)
    result=backLogic.copy()
    cross=front*backLogic
    for numi in range(2,back.max()+1):
        backMask=np.int32(back==numi)
        maskCross=cross*backMask
        count=[]
        maskCrossList=list(set(maskCross.flat))
        try:
            maskCrossList.remove(-1)
        except:
            pass
        for numj in maskCrossList:
            if numj!=0 and numj!=1:
                count.append((np.sum(maskCross==numj),numj))
        count=sorted(count)[::-1]
        for numk in range(len(count)):
            maskCross[maskCross==count[numk][1]]=count[0][1]
        maskCross0to1=maskCross.copy()
        maskCross0to1[maskCross0to1==0]=1
        result=result*maskCross0to1
    return(result)

def npCompelSplit_bodyDst(front,back):
    backLogic=np.int32(back==255)
    cross=front*backLogic
    cross1position=np.argwhere(cross==1)
    for num in range(0,len(cross1position),1):
        square9=cross[cross1position[num][0]-1:cross1position[num][0]+2,cross1position[num][1]-1:cross1position[num][1]+2]
        square9Element=list(set(square9.flat))
        try:
            square9Element.remove(0)
        except:
            pass
        try:
            square9Element.remove(1)
        except:
            pass
        if len(square9Element)==1:
            cross[cross1position[num][0],cross1position[num][1]]=square9Element[0]
    cross1position=np.argwhere(cross==1)
    for num in range(len(cross1position)-1,-1,-1):
        square9=cross[cross1position[num][0]-1:cross1position[num][0]+2,cross1position[num][1]-1:cross1position[num][1]+2]
        square9Element=list(set(square9.flat))
        try:
            square9Element.remove(0)
        except:
            pass
        try:
            square9Element.remove(1)
        except:
            pass
        if len(square9Element)==1:
            cross[cross1position[num][0],cross1position[num][1]]=square9Element[0]
    split=cross.copy()
    split1position=np.argwhere(split==1)
    if len(split1position)==0:
        crossLogic=np.int32(cross>1)
        crossLogicPosition=np.argwhere(crossLogic==1)
        for num in range(0,len(crossLogicPosition),1):
            square4=split[crossLogicPosition[num][0]:crossLogicPosition[num][0]+2,crossLogicPosition[num][1]:crossLogicPosition[num][1]+2]
            square4Element=list(set(square4.flat))
            try:
                square4Element.remove(0)
            except:
                pass
            if len(square4Element)>1:
                split[crossLogicPosition[num][0],crossLogicPosition[num][1]]=1
                break
        for num in range(len(crossLogicPosition)-1,-1,-1):
            square4=split[crossLogicPosition[num][0]-1:crossLogicPosition[num][0]+1,crossLogicPosition[num][1]-1:crossLogicPosition[num][1]+1]
            square4Element=list(set(square4.flat))
            try:
                square4Element.remove(0)
            except:
                pass
            if len(square4Element)>1:
                split[crossLogicPosition[num][0],crossLogicPosition[num][1]]=1
                break
        split1position=np.argwhere(split==1)
    result=split.copy()
    split1positionAccumulation=np.zeros([2],np.int64)
    while len(split1position)>0:
        split1positionAccumulation=np.vstack([split1positionAccumulation,split1position])
        for num in range(len(split1position)):
            left=split[split1position[num][0],split1position[num][1]-1]
            right=split[split1position[num][0],split1position[num][1]+1]
            up=split[split1position[num][0]-1,split1position[num][1]]
            down=split[split1position[num][0]+1,split1position[num][1]]
            leftUp=split[split1position[num][0]-1,split1position[num][1]-1]
            leftDown=split[split1position[num][0]+1,split1position[num][1]-1]
            rightUp=split[split1position[num][0]-1,split1position[num][1]+1]
            rightDown=split[split1position[num][0]+1,split1position[num][1]+1]
            if left!=0 and left!=1 and up!=0 and up!=1 and leftUp!=0 and left!=up:
                split[split1position[num][0]-1,split1position[num][1]-1]=1
            if left!=0 and left!=1 and down!=0 and down!=1 and leftDown!=0 and left!=down:
                split[split1position[num][0]+1,split1position[num][1]-1]=1
            if right!=0 and right!=1 and up!=0 and up!=1 and rightUp!=0 and right!=up:
                split[split1position[num][0]-1,split1position[num][1]+1]=1
            if right!=0 and right!=1 and down!=0 and down!=1 and rightDown!=0 and right!=down:
                split[split1position[num][0]+1,split1position[num][1]+1]=1
            split[split1position[num][0],split1position[num][1]]=0
        split1position=np.argwhere(split==1)
    splitRetVal=split1positionAccumulation.any()
    if splitRetVal:
        split1positionAccumulation=split1positionAccumulation[1:,:]
        exp1=4
        for num in range(len(split1positionAccumulation)):
            result[split1positionAccumulation[num][0]-exp1:split1positionAccumulation[num][0]+exp1+1,split1positionAccumulation[num][1]-exp1:split1positionAccumulation[num][1]+exp1+1]=0
    result=255*np.int32(result!=0)
    return(splitRetVal,split1positionAccumulation,result)

def npCompelSplit_markerCross(splitPosition,front,back):
    backLogic=np.int32(back>1)
    cross=front*backLogic
    cross1position=np.argwhere(cross==1)
    for num in range(0,len(cross1position),1):
        square9=cross[cross1position[num][0]-2:cross1position[num][0]+3,cross1position[num][1]-2:cross1position[num][1]+3]
        square9Element=list(set(square9.flat))
        try:
            square9Element.remove(0)
        except:
            pass
        try:
            square9Element.remove(1)
        except:
            pass
        if len(square9Element)==1:
            cross[cross1position[num][0],cross1position[num][1]]=square9Element[0]
    cross1position=np.argwhere(cross==1)
    for num in range(len(cross1position)-1,-1,-1):
        square9=cross[cross1position[num][0]-1:cross1position[num][0]+2,cross1position[num][1]-1:cross1position[num][1]+2]
        square9Element=list(set(square9.flat))
        try:
            square9Element.remove(0)
        except:
            pass
        try:
            square9Element.remove(1)
        except:
            pass
        if len(square9Element)==1:
            cross[cross1position[num][0],cross1position[num][1]]=square9Element[0]
    result=cross.copy()
    exp2=1
    for num in range(len(splitPosition)):
        cross[splitPosition[num][0],splitPosition[num][1]]=1
        result[splitPosition[num][0]-exp2:splitPosition[num][0]+exp2+1,splitPosition[num][1]-exp2:splitPosition[num][1]+exp2+1]=-1
    split=cross.copy()
    split1position=np.argwhere(split==1)
    exp3=4
    while len(split1position)>0:
        for num in range(len(split1position)):
            left=split[split1position[num][0],split1position[num][1]-1]
            right=split[split1position[num][0],split1position[num][1]+1]
            up=split[split1position[num][0]-1,split1position[num][1]]
            down=split[split1position[num][0]+1,split1position[num][1]]
            leftUp=split[split1position[num][0]-1,split1position[num][1]-1]
            leftDown=split[split1position[num][0]+1,split1position[num][1]-1]
            rightUp=split[split1position[num][0]-1,split1position[num][1]+1]
            rightDown=split[split1position[num][0]+1,split1position[num][1]+1]
            if left!=0 and left!=1 and up!=0 and up!=1 and leftUp!=0 and left!=up:
                split[split1position[num][0]-1,split1position[num][1]-1]=1
                result[split1position[num][0]-exp3-1:split1position[num][0]+exp3,split1position[num][1]-exp3-1:split1position[num][1]+exp3]=-1
            if left!=0 and left!=1 and down!=0 and down!=1 and leftDown!=0 and left!=down:
                split[split1position[num][0]+1,split1position[num][1]-1]=1
                result[split1position[num][0]-exp3+1:split1position[num][0]+exp3+2,split1position[num][1]-exp3-1:split1position[num][1]+exp3]=-1
            if right!=0 and right!=1 and up!=0 and up!=1 and rightUp!=0 and right!=up:
                split[split1position[num][0]-1,split1position[num][1]+1]=1
                result[split1position[num][0]-exp3-1:split1position[num][0]+exp3,split1position[num][1]-exp3+1:split1position[num][1]+exp3+2]=-1
            if right!=0 and right!=1 and down!=0 and down!=1 and rightDown!=0 and right!=down:
                split[split1position[num][0]+1,split1position[num][1]+1]=1
                result[split1position[num][0]-exp3+1:split1position[num][0]+exp3+2,split1position[num][1]-exp3+1:split1position[num][1]+exp3+2]=-1
            split[split1position[num][0],split1position[num][1]]=0
        split1position=np.argwhere(split==1)
    return(result)

def npRecovery(front,back,iloop,startFrame,DstRate,watershedMat):
    global countAccumulation
    if iloop==startFrame:
        countAccumulation=np.array([],np.int64)
    identity=list(set(back.flat))
    try:
        identity.remove(1)
    except:
        pass
    try:
        identity.remove(0)
    except:
        pass
    try:
        identity.remove(-1)
    except:
        pass
    count=int(len(identity))
    countAccumulation=np.append(countAccumulation,count)
    countAccumulationNum=list(set(countAccumulation.flat))
    countAccumulationList=[]
    for num in range(len(countAccumulationNum)):
        countAccumulationList.append((np.sum(countAccumulation==countAccumulationNum[num]),countAccumulationNum[num]))
    countAccumulationList=sorted(countAccumulationList)[::-1]
    countAccumulationMost=countAccumulationList[0][1]
    
    result=back.copy()
    recoveryRetVal=False
    global identityMost
    if iloop==startFrame:
        identityMost=[]
    if count==countAccumulationMost:
        identityMost=identity
    if count<countAccumulationMost:
        backLogic=255*np.uint8(back>1)
        backDst=cv2.distanceTransform(backLogic,cv2.DIST_L2,5)
        retVal,backDstThresh=cv2.threshold(backDst,DstRate*backDst.max(),255,0)
        backDstThresh=np.uint8(backDstThresh)
        resultMarkers=np.int32(backDstThresh/255)
        retVal,backMarkers=cv2.connectedComponents(backDstThresh)
        backIdentity=list(set(backMarkers.flat))
        try:
            backIdentity.remove(0)
        except:
            pass
        front=front+1
        recoveryRetVal=(len(backIdentity)==len(identityMost))
        if recoveryRetVal:
            for numi in backIdentity:
                backMask=np.uint8(backMarkers==numi)
                cross=front*backMask
                crossIdentity=list(set(cross.flat))
                try:
                    crossIdentity.remove(0)
                except:
                    pass
                try:
                    crossIdentity.remove(-1)
                except:
                    pass
                crossIdentityList=[]
                for numj in range(len(crossIdentity)):
                    crossIdentityList.append((np.sum(cross==crossIdentity[numj]),crossIdentity[numj]))
                crossIdentityList=sorted(crossIdentityList)[::-1]
                crossIdentityMost=crossIdentityList[0][1]
                if crossIdentityMost!=1:
                    resultMarkers[backMask==1]=crossIdentityMost
                    
            resultMarkersIdentity=list(set(resultMarkers.flat))    
            try:
                resultMarkersIdentity.remove(0)
            except:
                pass
            try:
                resultMarkersIdentity.remove(1)
            except:
                pass
            if resultMarkersIdentity!=identityMost:
                difference=[]
                for num in identityMost:
                    if num not in resultMarkersIdentity:
                        difference.append(num)
                resultMarkers[backMask==1]=difference[0]
            result=np.int32(backLogic/255*resultMarkers)
            result[result==0]=-1
            result[result==1]=0
            result[result==-1]=1
            result=cv2.watershed(watershedMat,result)
    return(result,count,countAccumulationMost,recoveryRetVal)


folderPath='E:\\videos for analysis\\videos for analysis\\MultiMouse\\01'
os.chdir(folderPath)
vidPath=folderPath+'\\multiColor.mp4'

grayBack,roi = mouseLocalization.backgroundCalculation(vidPath)

vid = cv2.VideoCapture(vidPath)
nFrames = np.int0(vid.get(cv2.CAP_PROP_FRAME_COUNT))
width = np.int0(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
height = np.int0(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = vid.get(cv2.CAP_PROP_FPS)
term_crit = (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,10,1)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fourcc=cv2.VideoWriter_fourcc(*'XVID')
#out1=cv2.VideoWriter('multiColorCut.mp4',fourcc,fps,(width,height))
#out2=cv2.VideoWriter('frameMatCut.mp4',fourcc,fps,(width,height))
splitRetVal = False
MouseThreshold = 8
DstRate = 0.45
referenceNum = 5
markerAddReference=np.zeros([height,width,referenceNum],np.int32)
markerCrossReference=np.zeros([height,width],np.int32)
markerRecoveryReference=np.zeros([height,width],np.int32)
markerID=np.zeros([height,width],np.int32)
startFrame = 1400
endFrame = nFrames

for iloop in range(nFrames):
    ret,frameMat=vid.read()
    if ret==True:
        grayFrame=cv2.cvtColor(frameMat,cv2.COLOR_BGR2GRAY)
        grayDiff=np.int0(grayBack)-np.int0(grayFrame)
        grayDiff[grayDiff<0]=0
        mouseDark=255*np.uint8((grayFrame<MouseThreshold)&(grayDiff>MouseThreshold))
        mouseDark=mouseDark&roi
        
        mouseOpening=cv2.morphologyEx(mouseDark,cv2.MORPH_OPEN,kernel,iterations=3)
        mouseMask=cv2.dilate(mouseOpening,kernel,iterations=3)
        mouseDst=cv2.distanceTransform(mouseOpening,cv2.DIST_L2,5)
        retVal,bodyDst=cv2.threshold(mouseDst,DstRate*mouseDst.max(),255,0)
        
        if iloop>=startFrame:
            markerBackAdd=np.zeros([height,width],np.int32)
            for num in range(referenceNum):
                markerBackAdd=npBackAdd(markerBackAdd,markerAddReference[:,:,referenceNum-num-1])
            markerBackAddList=list(set(markerBackAdd.flat))
            print("markerBackAddList =",markerBackAddList)
            
            markerBackAddAdd=np.uint8(255*markerBackAdd/markerBackAdd.max())
            cv2.imshow('MarkerBackAddAdd',markerBackAddAdd)
            markerBackAdd=markerBackAdd+1
            splitRetVal,splitPosition,bodyDst=npCompelSplit_bodyDst(markerBackAdd,bodyDst)
            
            bodyDstList=list(set(bodyDst.flat))
            print("bodyDstList =",bodyDstList)
    
        bodyDst=np.uint8(bodyDst)
        cv2.imshow('bodyDst',bodyDst)
        mouseUnknown=cv2.subtract(mouseMask,bodyDst)
        retVal,mouseMarkers=cv2.connectedComponents(bodyDst)
        mouseMarkers=mouseMarkers+1
        mouseMarkers[mouseUnknown==255]=0
        watershedMat=frameMat.copy()
        mouseMarkers=cv2.watershed(watershedMat,mouseMarkers)
        mouseMarkers[mouseMarkers==-1]=1
        
        mouseMarkersList=list(set(mouseMarkers.flat))
        print("mouseMarkersList =",mouseMarkersList)
        
        if iloop>=startFrame:
            markerCross=npMaxCross(markerBackAdd,mouseMarkers)
            markerCross[markerCross==0]=-1
            markerCross[markerCross==1]=0
            markerCross[markerCross==-1]=1
            markerCross=cv2.watershed(watershedMat,markerCross)
            
            markerCrossList=list(set(markerCross.flat))
            print("markerCrossList =",markerCrossList)
            
            markerID=markerCross.copy()
            
            if splitRetVal:
                markerSplit=npCompelSplit_markerCross(splitPosition,markerCrossReference,markerCross)
                markerSplit[markerSplit==0]=1
                
                markerSplitList=list(set(markerSplit.flat))
                print("markerSplitList =",markerSplitList)
                
                markerID=markerSplit.copy()
            
            print("splitRetVal =",splitRetVal)
            
            markerID,count,countAccumulationMost,recoveryRetVal=npRecovery(markerRecoveryReference,markerID,iloop,startFrame,DstRate,watershedMat)
            
            print("count =",count)
            print("countAccumulationMost =",countAccumulationMost)
            print("recoveryRetVal =",recoveryRetVal)
            
            markerIDMask=np.int32(markerID>0)
            markerID=markerID-markerIDMask
            
            markerIDList=list(set(markerID.flat))
            print("markerIDList =",markerIDList)
            
            print(" ")
        
        markerAddReference=np.roll(markerAddReference,1,axis=2)
        if iloop>=startFrame:
            markerCrossReference=markerCross.copy()
            markerRecoveryReference=markerID.copy()
            markerNormalized=markerID.copy()
        if iloop<startFrame:
            markerNormalized=mouseMarkers.copy()
            markerNormalized=markerNormalized-1
        if markerNormalized.any():
            markerAddReference[:,:,0]=markerNormalized
        markerIDcolor=np.zeros([height,width,3],np.uint8)
        for num in range(3):
            markerIDcolor[:,:,num]=255*np.uint8(markerID==num+1)
        
        cv2.imshow('MarkerIDcolor',markerIDcolor)
        
#        if iloop>=startFrame and iloop<=endFrame:
#            out1.write(markerIDcolor)
        
        blue=255*np.uint8(markerID==1)
        if blue.max()!=0:
            img,blueContours,hierarchy=cv2.findContours(blue,1,2)
            blueContours=np.vstack(blueContours)
            blueRect=cv2.minAreaRect(blueContours)
            bluePts=np.int0(cv2.boxPoints(blueRect))
            blueBox=cv2.polylines(frameMat,[bluePts],True,[255,0,0],2)
        green=255*np.uint8(markerID==2)
        if green.max()!=0:
            img,greenContours,hierarchy=cv2.findContours(green,1,2)
            greenContours=np.vstack(greenContours)
            greenRect=cv2.minAreaRect(greenContours)
            greenPts=np.int0(cv2.boxPoints(greenRect))
            greenBox=cv2.polylines(frameMat,[greenPts],True,[0,255,0],2)
        red=255*np.uint8(markerID==3)
        if red.max()!=0:
            img,redContours,hierarchy=cv2.findContours(red,1,2)
            redContours=np.vstack(redContours)
            redRect=cv2.minAreaRect(redContours)
            redPts=np.int0(cv2.boxPoints(redRect))
            redBox=cv2.polylines(frameMat,[redPts],True,[0,0,255],2)
        
        cv2.imshow("FrameMat",frameMat)
        
#        if iloop>=startFrame and iloop<=endFrame:
#            out2.write(frameMat)

        if cv2.waitKey(1) & 0xFF == 27:
            break
        
    else:
        break

cv2.destroyAllWindows()
vid.release()
#out1.release()
#out2.release()
