#! /usr/bin/python

#
# Description:
# ================================================================
# Time-stamp: "2021-10-06 05:58:34 trottar"
# ================================================================
#
# Author:  Richard L. Trotta III <trotta@cua.edu>
#
# Copyright (c) trottar
#

import uproot as up
import numpy as np
import pandas as pd
import scipy
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import sys, math, os, subprocess

RunType = sys.argv[1]
ROOTPrefix = sys.argv[2]
runNum = sys.argv[3]
MaxEvent=sys.argv[4]

# Add this to all files for more dynamic pathing
USER = subprocess.getstatusoutput("whoami") # Grab user info for file finding
HOST = subprocess.getstatusoutput("hostname")

if ("farm" in HOST[1]):
    REPLAYPATH="/group/c-pionlt/online_analysis/hallc_replay_lt"
elif ("lark" in HOST[1]):
    REPLAYPATH = "/home/%s/work/JLab/hallc_replay_lt" % USER[1]
elif ("cdaq" in HOST[1]):
    REPLAYPATH = "/home/cdaq/hallc-online/hallc_replay_lt"
elif ("trottar" in HOST[1]):
    REPLAYPATH = "/home/trottar/Analysis/hallc_replay_lt"

sys.path.insert(0, '%s/UTIL_PION/bin/python/' % REPLAYPATH)
import kaonlt as klt

rootName = "%s/UTIL_PION/ROOTfiles/Analysis/%s/%s_%s_%s.root" % (REPLAYPATH,RunType,ROOTPrefix,runNum,MaxEvent)     # Input file location and variables taking
report = "%s/UTIL_PION/REPORT_OUTPUT/Analysis/%s/%s_%s_%s.report" % (REPLAYPATH,RunType,ROOTPrefix,runNum,MaxEvent)

f = open(report)
    
psList = ['SW_Ps1_factor','SW_Ps2_factor','SW_Ps3_factor','SW_Ps4_factor','SW_Ps5_factor','SW_Ps6_factor']
    
psActual = [-1,1,2,3,5,9,17,33,65,129,257,513,1025,2049,4097,8193,16385,32769]
psValue = [-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

for line in f:
    data = line.split(':')
    track_data = line.split(':')
    if ('SW_SHMS_Electron_Singles_TRACK_EFF' in track_data[0]):
        SHMS_track_info = track_data[1].split("+-")
    if ('SW_HMS_Electron_Singles_TRACK_EFF' in track_data[0]):
        HMS_track_info = track_data[1].split("+-")
    for i, obj in enumerate(psList) :
        if (psList[i] in data[0]) : 
            if (i == 0) :  
                ps1_tmp = data[1].strip()
            if (i == 1) : 
                ps2_tmp = data[1].strip()
            if (i == 2) :
                ps3_tmp = data[1].strip()
            if (i == 3) :
                ps4_tmp = data[1].strip()
            if (i == 4) :
                ps5_tmp = data[1].strip()
            if (i == 5) :
                ps6_tmp = data[1].strip()
ps1=int(ps1_tmp)
ps2=int(ps2_tmp)
ps3=int(ps3_tmp)
ps4=int(ps4_tmp)
ps5=int(ps5_tmp)
ps6=int(ps6_tmp)
SHMS_track_eff = float(SHMS_track_info[0])
SHMS_track_uncern = float(SHMS_track_info[1])
HMS_track_eff = float(HMS_track_info[0])
HMS_track_uncern = float(HMS_track_info[1])

for i,index in enumerate(psActual):
    #psValue
    if (index == ps1) :
        if(index == -1):
            PS1 = 0
        else:
            PS1 = psActual[i]
    if (index == ps2) :
        if(index == -1):
            PS2 = 0
        else:
            PS2 = psActual[i]            
    if (index == ps3) :
        if(index == -1):
            PS3 = 0
        else:
            PS3 = psActual[i]
    if (index == ps4) :
        if(index == -1):
            PS4 = 0
        else:
            PS4 = psActual[i]            
    if (index == ps5) :
        if(index == -1):
            PS5 = 0
        else:
            PS5 = psActual[i]
    if (index == ps6) :
        if(index == -1):
            PS6 = 0
        else:
            PS6 = psActual[i]            
f.close()

#print("\nPre-scale values...\nPS1:{0}, PS2:{1}, PS3:{2}, PS4:{3}, PS5:{4}, PS6:{5}\n".format(PS1,PS2,PS3,PS4,PS5,PS6))

PS_list = [["PS1",PS1],["PS2",PS2],["PS3",PS3],["PS4",PS4],["PS5",PS5],["PS6",PS6]]
PS_used = []

for val in PS_list:
    if val[1] != 0:
        PS_used.append(val)

if len(PS_used) > 2:
    PS_names = [PS_used[0][0],PS_used[1][0],PS_used[2][0]]
    SHMS_PS = PS_used[0][1]
    HMS_PS = PS_used[1][1]
    COIN_PS = PS_used[2][1]
else:
    PS_names = [PS_used[0][0],PS_used[1][0]]
    SHMS_PS = PS_used[0][1]
    HMS_PS = PS_used[1][1]

'''
ANALYSIS TREE, T
'''

tree = up.open(rootName)["T"]
branch = klt.pyBranch(tree)

H_bcm_bcm4a_AvgCurrent = tree.array("H.bcm.bcm4a.AvgCurrent")

if PS_names[0] is "PS1":
    T_coin_pTRIG_SHMS_ROC1_tdcTimeRaw = tree.array("T.coin.pTRIG1_ROC1_tdcTimeRaw")
    T_coin_pTRIG_SHMS_ROC2_tdcTimeRaw = tree.array("T.coin.pTRIG1_ROC2_tdcTimeRaw")
    T_coin_pTRIG_SHMS_ROC1_tdcTime = tree.array("T.coin.pTRIG1_ROC1_tdcTime")
    T_coin_pTRIG_SHMS_ROC2_tdcTime = tree.array("T.coin.pTRIG1_ROC2_tdcTime")

if PS_names[0] is "PS2":
    T_coin_pTRIG_SHMS_ROC1_tdcTimeRaw = tree.array("T.coin.pTRIG2_ROC1_tdcTimeRaw")
    T_coin_pTRIG_SHMS_ROC2_tdcTimeRaw = tree.array("T.coin.pTRIG2_ROC2_tdcTimeRaw")
    T_coin_pTRIG_SHMS_ROC1_tdcTime = tree.array("T.coin.pTRIG2_ROC1_tdcTime")
    T_coin_pTRIG_SHMS_ROC2_tdcTime = tree.array("T.coin.pTRIG2_ROC2_tdcTime")

if PS_names[1] is "PS3":
    T_coin_pTRIG_HMS_ROC1_tdcTimeRaw = tree.array("T.coin.pTRIG3_ROC1_tdcTimeRaw")
    T_coin_pTRIG_HMS_ROC2_tdcTimeRaw = tree.array("T.coin.pTRIG3_ROC2_tdcTimeRaw")
    T_coin_pTRIG_HMS_ROC1_tdcTime = tree.array("T.coin.pTRIG3_ROC1_tdcTime")
    T_coin_pTRIG_HMS_ROC2_tdcTime = tree.array("T.coin.pTRIG3_ROC2_tdcTime")

if PS_names[1] is "PS4":
    T_coin_pTRIG_HMS_ROC1_tdcTimeRaw = tree.array("T.coin.pTRIG4_ROC1_tdcTimeRaw")
    T_coin_pTRIG_HMS_ROC2_tdcTimeRaw = tree.array("T.coin.pTRIG4_ROC2_tdcTimeRaw")
    T_coin_pTRIG_HMS_ROC1_tdcTime = tree.array("T.coin.pTRIG4_ROC1_tdcTime")
    T_coin_pTRIG_HMS_ROC2_tdcTime = tree.array("T.coin.pTRIG4_ROC2_tdcTime")

if len(PS_used) > 2:
    if PS_names[2] is "PS5":
        T_coin_pTRIG_COIN_ROC1_tdcTimeRaw = tree.array("T.coin.pTRIG5_ROC1_tdcTimeRaw")
        T_coin_pTRIG_COIN_ROC2_tdcTimeRaw = tree.array("T.coin.pTRIG5_ROC2_tdcTimeRaw")
        T_coin_pTRIG_COIN_ROC1_tdcTime = tree.array("T.coin.pTRIG5_ROC1_tdcTime")
        T_coin_pTRIG_COIN_ROC2_tdcTime = tree.array("T.coin.pTRIG5_ROC2_tdcTime")
        
    if PS_names[2] is "PS6":
        T_coin_pTRIG_COIN_ROC1_tdcTimeRaw = tree.array("T.coin.pTRIG6_ROC1_tdcTimeRaw")
        T_coin_pTRIG_COIN_ROC2_tdcTimeRaw = tree.array("T.coin.pTRIG6_ROC2_tdcTimeRaw")
        T_coin_pTRIG_COIN_ROC1_tdcTime = tree.array("T.coin.pTRIG6_ROC1_tdcTime")
        T_coin_pTRIG_COIN_ROC2_tdcTime = tree.array("T.coin.pTRIG6_ROC2_tdcTime")

T_coin_pEDTM_tdcTimeRaw = tree.array("T.coin.pEDTM_tdcTimeRaw")
T_coin_pEDTM_tdcTime = tree.array("T.coin.pEDTM_tdcTime")

fout = REPLAYPATH+'/UTIL_PION/DB/CUTS/run_type/lumi.cuts'

# read in cuts file and make dictionary
c = klt.pyPlot(REPLAYPATH)
readDict = c.read_dict(fout,runNum)

def make_cutDict(cut,inputDict=None):
    '''
    This method calls several methods in kaonlt package. It is required to create properly formated
    dictionaries. The evaluation must be in the analysis script because the analysis variables (i.e. the
    leaves of interest) are not defined in the kaonlt package. This makes the system more flexible
    overall, but a bit more cumbersome in the analysis script. Perhaps one day a better solution will be
    implimented.
    '''

    global c

    c = klt.pyPlot(REPLAYPATH,readDict)
    x = c.w_dict(cut)
    print("\n%s" % cut)
    print(x, "\n")

    # Grab current cuts
    if cut == "c_curr":
        global report_current
        # e.g. Grabbing threshold current (ie 2.5) from something like this [' {"H_bcm_bcm4a_AvgCurrent" : (abs(H_bcm_bcm4a_AvgCurrent-55) < 2.5)}']
        report_current = x[0]
    
    if inputDict == None:
        inputDict = {}
        
    for key,val in readDict.items():
        if key == cut:
            inputDict.update({key : {}})

    for i,val in enumerate(x):
        tmp = x[i]
        if tmp == "":
            continue
        else:
            inputDict[cut].update(eval(tmp))
        
    return inputDict

cutDict = make_cutDict("c_nozero")
cutDict = make_cutDict("c_noedtm",cutDict)
cutDict = make_cutDict("c_edtm",cutDict)
cutDict = make_cutDict("c_ptrigHMS",cutDict)
cutDict = make_cutDict("c_ptrigSHMS",cutDict)
if len(PS_used) > 2:
    cutDict = make_cutDict("c_ptrigCOIN",cutDict)
cutDict = make_cutDict("c_curr",cutDict)
c = klt.pyPlot(REPLAYPATH,cutDict)

def trig_Plots():

    f = plt.figure(figsize=(11.69,8.27))

    if len(PS_used) > 2:

        ax = f.add_subplot(241)
        ax.hist(c.add_cut(T_coin_pTRIG_HMS_ROC1_tdcTimeRaw,"c_nozero"),bins=200,label='no cut',histtype='step', alpha=0.5, stacked=True, fill=True)
        ax.hist(c.add_cut(T_coin_pTRIG_HMS_ROC1_tdcTimeRaw,"c_ptrigHMS"),bins=200,label='cut',histtype='step', alpha=0.5, stacked=True, fill=True)
        plt.yscale('log')
        plt.xlabel('T_coin_pTRIG_HMS_ROC1_tdcTimeRaw')
        plt.ylabel('Count')
        
        ax = f.add_subplot(242)
        ax.hist(c.add_cut(T_coin_pTRIG_SHMS_ROC2_tdcTimeRaw,"c_nozero"),bins=200,label='no cut',histtype='step', alpha=0.5, stacked=True, fill=True)
        ax.hist(c.add_cut(T_coin_pTRIG_SHMS_ROC2_tdcTimeRaw,"c_ptrigSHMS"),bins=200,label='cut',histtype='step', alpha=0.5, stacked=True, fill=True)
        plt.yscale('log')
        plt.xlabel('T_coin_pTRIG_SHMS_ROC2_tdcTimeRaw')
        plt.ylabel('Count')

        plt.title("Run %s" % runNum)

        ax = f.add_subplot(243)
        ax.hist(c.add_cut(T_coin_pTRIG_COIN_ROC1_tdcTimeRaw,"c_nozero"),bins=200,label='no cut',histtype='step', alpha=0.5, stacked=True, fill=True)
        ax.hist(c.add_cut(T_coin_pTRIG_COIN_ROC1_tdcTimeRaw,"c_ptrigCOIN"),bins=200,label='cut',histtype='step', alpha=0.5, stacked=True, fill=True)
        plt.yscale('log')
        plt.xlabel('T_coin_pTRIG_COIN_ROC1_tdcTimeRaw')
        plt.ylabel('Count')

        ax = f.add_subplot(244)
        ax.hist(c.add_cut(T_coin_pEDTM_tdcTimeRaw,"c_nozero"),bins=200,label='no cut',histtype='step', alpha=0.5, stacked=True, fill=True)
        ax.hist(c.add_cut(T_coin_pEDTM_tdcTimeRaw,"c_edtm"),bins=200,label='cut',histtype='step', alpha=0.5, stacked=True, fill=True)
        plt.yscale('log')
        plt.xlabel('T_coin_pEDTM_tdcTimeRaw')
        plt.ylabel('Count')
        
        ax = f.add_subplot(245)
        ax.hist(c.add_cut(T_coin_pTRIG_HMS_ROC1_tdcTime,"c_nozero"),bins=200,label='no cut',histtype='step', alpha=0.5, stacked=True, fill=True)
        ax.hist(c.add_cut(T_coin_pTRIG_HMS_ROC1_tdcTime,"c_ptrigHMS"),bins=200,label='cut',histtype='step', alpha=0.5, stacked=True, fill=True)
        plt.yscale('log')
        plt.xlabel('T_coin_pTRIG_HMS_ROC1_tdcTime')
        plt.ylabel('Count')

        ax = f.add_subplot(246)
        ax.hist(c.add_cut(T_coin_pTRIG_SHMS_ROC2_tdcTime,"c_nozero"),bins=200,label='no cut',histtype='step', alpha=0.5, stacked=True, fill=True)
        ax.hist(c.add_cut(T_coin_pTRIG_SHMS_ROC2_tdcTime,"c_ptrigSHMS"),bins=200,label='cut',histtype='step', alpha=0.5, stacked=True, fill=True)
        plt.yscale('log')
        plt.xlabel('T_coin_pTRIG_SHMS_ROC2_tdcTime')
        plt.ylabel('Count')

        ax = f.add_subplot(247)
        ax.hist(c.add_cut(T_coin_pTRIG_COIN_ROC1_tdcTimeRaw,"c_nozero"),bins=200,label='no cut',histtype='step', alpha=0.5, stacked=True, fill=True)
        ax.hist(c.add_cut(T_coin_pTRIG_COIN_ROC1_tdcTimeRaw,"c_ptrigCOIN"),bins=200,label='cut',histtype='step', alpha=0.5, stacked=True, fill=True)
        plt.yscale('log')
        plt.xlabel('T_coin_pTRIG_COIN_ROC1_tdcTimeRaw')
        plt.ylabel('Count')

        ax = f.add_subplot(248)
        ax.hist(c.add_cut(T_coin_pEDTM_tdcTime,"c_nozero"),bins=200,label='no cut',histtype='step', alpha=0.5, stacked=True, fill=True)
        ax.hist(c.add_cut(T_coin_pEDTM_tdcTime,"c_edtm"),bins=200,label='cut',histtype='step', alpha=0.5, stacked=True, fill=True)
        plt.yscale('log')
        plt.xlabel('T_coin_pEDTM_tdcTime')
        plt.ylabel('Count')

    else:

        ax = f.add_subplot(231)
        ax.hist(c.add_cut(T_coin_pTRIG_HMS_ROC1_tdcTimeRaw,"c_nozero"),bins=200,label='no cut',histtype='step', alpha=0.5, stacked=True, fill=True)
        ax.hist(c.add_cut(T_coin_pTRIG_HMS_ROC1_tdcTimeRaw,"c_ptrigHMS"),bins=200,label='cut',histtype='step', alpha=0.5, stacked=True, fill=True)
        plt.yscale('log')
        plt.xlabel('T_coin_pTRIG_HMS_ROC1_tdcTimeRaw')
        plt.ylabel('Count')
        
        ax = f.add_subplot(232)
        ax.hist(c.add_cut(T_coin_pTRIG_SHMS_ROC2_tdcTimeRaw,"c_nozero"),bins=200,label='no cut',histtype='step', alpha=0.5, stacked=True, fill=True)
        ax.hist(c.add_cut(T_coin_pTRIG_SHMS_ROC2_tdcTimeRaw,"c_ptrigSHMS"),bins=200,label='cut',histtype='step', alpha=0.5, stacked=True, fill=True)
        plt.yscale('log')
        plt.xlabel('T_coin_pTRIG_SHMS_ROC2_tdcTimeRaw')
        plt.ylabel('Count')

        plt.title("Run %s" % runNum)

        ax = f.add_subplot(233)
        ax.hist(c.add_cut(T_coin_pEDTM_tdcTimeRaw,"c_nozero"),bins=200,label='no cut',histtype='step', alpha=0.5, stacked=True, fill=True)
        ax.hist(c.add_cut(T_coin_pEDTM_tdcTimeRaw,"c_edtm"),bins=200,label='cut',histtype='step', alpha=0.5, stacked=True, fill=True)
        plt.yscale('log')
        plt.xlabel('T_coin_pEDTM_tdcTimeRaw')
        plt.ylabel('Count')
        
        ax = f.add_subplot(234)
        ax.hist(c.add_cut(T_coin_pTRIG_HMS_ROC1_tdcTime,"c_nozero"),bins=200,label='no cut',histtype='step', alpha=0.5, stacked=True, fill=True)
        ax.hist(c.add_cut(T_coin_pTRIG_HMS_ROC1_tdcTime,"c_ptrigHMS"),bins=200,label='no cut',histtype='step', alpha=0.5, stacked=True, fill=True)
        plt.yscale('log')
        plt.xlabel('T_coin_pTRIG_HMS_ROC1_tdcTime')
        plt.ylabel('Count')

        ax = f.add_subplot(235)
        ax.hist(c.add_cut(T_coin_pTRIG_SHMS_ROC2_tdcTime,"c_nozero"),bins=200,label='no cut',histtype='step', alpha=0.5, stacked=True, fill=True)
        ax.hist(c.add_cut(T_coin_pTRIG_SHMS_ROC2_tdcTime,"c_ptrigSHMS"),bins=200,label='no cut',histtype='step', alpha=0.5, stacked=True, fill=True)
        plt.yscale('log')
        plt.xlabel('T_coin_pTRIG_SHMS_ROC2_tdcTime')
        plt.ylabel('Count')

        ax = f.add_subplot(236)
        ax.hist(c.add_cut(T_coin_pEDTM_tdcTime,"c_nozero"),bins=200,label='no cut',histtype='step', alpha=0.5, stacked=True, fill=True)
        ax.hist(c.add_cut(T_coin_pEDTM_tdcTime,"c_edtm"),bins=200,label='no cut',histtype='step', alpha=0.5, stacked=True, fill=True)
        plt.yscale('log')
        plt.xlabel('T_coin_pEDTM_tdcTime')
        plt.ylabel('Count')
        
    plt.tight_layout()      
    plt.savefig('%s/UTIL_PION/scripts/trig_windows/OUTPUTS/trig_%s_%s.png' % (REPLAYPATH,ROOTPrefix,runNum))     # Input file location and variables taking)

def currentPlots():

    f = plt.figure(figsize=(11.69,8.27))

    ax = f.add_subplot(111)
    ax.hist(H_bcm_bcm4a_AvgCurrent,bins=100,label='no cut',histtype='step', alpha=0.5, stacked=True, fill=True)
    ax.hist(c.add_cut(H_bcm_bcm4a_AvgCurrent,"c_curr"),bins=100,label='cut',histtype='step', alpha=0.5, stacked=True, fill=True)
    plt.yscale('log')
    plt.xlabel('H_bcm_bcm4a_AvgCurrent')
    plt.ylabel('Count')
    plt.title("Run %s, %s" % (runNum,report_current))

    plt.savefig('%s/UTIL_PION/scripts/trig_windows/OUTPUTS/curr_%s_%s.png' % (REPLAYPATH,ROOTPrefix,runNum))     # Input file location and variables taking)


def main():

    trig_Plots()
    currentPlots()
    #plt.show()

if __name__ == '__main__': main()