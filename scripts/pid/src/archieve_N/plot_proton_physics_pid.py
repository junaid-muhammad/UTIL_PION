#! /usr/bin/python
#
# Description:
# ================================================================
# Time-stamp: "2024-03-15 01:29:19 junaid"
# ================================================================
#
# Author:  Muhammad Junaid III <mjo147@uregina.ca>
#
# Copyright (c) junaid
#
###################################################################################################################################################

# Import relevant packages
import uproot
import uproot as up
import numpy as np

np.bool = bool
np.float = float

import root_numpy as rnp
import pandas as pd
import root_pandas as rpd
import ROOT
import scipy
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import sys, math, os, subprocess
import array
import csv
from ROOT import TCanvas, TList, TPaveLabel, TColor, TGaxis, TH1F, TH2F, TPad, TStyle, gStyle, gPad, TLegend, TGaxis, TLine, TMath, TLatex, TPaveText, TArc, TGraphPolar, TText, TString
from ROOT import kBlack, kCyan, kRed, kGreen, kMagenta, kBlue
from functools import reduce
import math as ma
import re # Regexp package - for string manipulation

##################################################################################################################################################

# Check the number of arguments provided to the script
if len(sys.argv)-1!=7:
    print("!!!!! ERROR !!!!!\n Expected 8 arguments\n Usage is with - ROOTfileSuffixs Beam Energy MaxEvents RunList CVSFile\n!!!!! ERROR !!!!!")
    sys.exit(1)

##################################################################################################################################################

# Defining some constants here
minbin = 0.70 # minimum bin for selecting neutrons events in missing mass distribution
maxbin = 0.85 # maximum bin for selecting neutrons events in missing mass distribution

##################################################################################################################################################

# Input params - run number and max number of events
BEAM_ENERGY = sys.argv[1]
Q2 = sys.argv[2]
W = sys.argv[3]
ptheta = sys.argv[4]
DATA_Suffix = sys.argv[5]
MaxEvent = sys.argv[6]
runNum = sys.argv[7]
################################################################################################################################################
'''
ltsep package import and pathing definitions
'''

# Import package for cuts
from ltsep import Root

lt=Root(os.path.realpath(__file__), "Plot_Prod")

# Add this to all files for more dynamic pathing
USER=lt.USER # Grab user info for file finding
HOST=lt.HOST
REPLAYPATH=lt.REPLAYPATH
UTILPATH=lt.UTILPATH
ANATYPE=lt.ANATYPE
OUTPATH=lt.OUTPATH

#################################################################################################################################################

# Output PDF File Name
print("Running as %s on %s, hallc_replay_lt path assumed as %s" % (USER, HOST, REPLAYPATH))
Pion_Analysis_Distributions = "%s/%s_%s_%s_%s_%s_%s_Proton_PID_Analysis_Distributions.pdf" % (OUTPATH, BEAM_ENERGY, Q2, W, ptheta, DATA_Suffix, MaxEvent)

# Input file location and variables taking
rootFile_DATA = "%s/%s_%s_%s_%s_%s_%s.root" % (OUTPATH, BEAM_ENERGY, Q2, W, ptheta, DATA_Suffix, MaxEvent)

###############################################################################################################################################

# Section for grabing Prompt/Random selection parameters from PARAM file
PARAMPATH = "%s/DB/PARAM" % UTILPATH
print("Running as %s on %s, hallc_replay_lt path assumed as %s" % (USER, HOST, REPLAYPATH))
TimingCutFile = "%s/Timing_Parameters.csv" % PARAMPATH # This should match the param file actually being used!
TimingCutf = open(TimingCutFile)
try:
    TimingCutFile
except NameError:
    print("!!!!! ERRROR !!!!!\n One (or more) of the cut files not found!\n!!!!! ERRORR !!!!!")
    sys.exit(2)
print("Reading timing cuts from %s" % TimingCutFile)
PromptWindow = [0, 0]
RandomWindows = [0, 0, 0, 0]
linenum = 0 # Count line number we're on
TempPar = -1 # To check later
for line in TimingCutf: # Read all lines in the cut file
    linenum += 1 # Add one to line number at start of loop
    if(linenum > 1): # Skip first line
        line = line.partition('#')[0] # Treat anything after a # as a comment and ignore it
        line = line.rstrip()
        array = line.split(",") # Convert line into an array, anything after a comma is a new entry 
        if(int(runNum) in range (int(array[0]), int(array[1])+1)): # Check if run number for file is within any of the ranges specified in the cut file
            TempPar += 2 # If run number is in range, set to non -1 value
            BunchSpacing = float(array[2])
            CoinOffset = float(array[3]) # Coin offset value
            nSkip = float(array[4]) # Number of random windows skipped 
            nWindows = float(array[5]) # Total number of random windows
            PromptPeak = float(array[6]) # Pion CT prompt peak positon 
TimingCutf.close() # After scanning all lines in file, close file

if(TempPar == -1): # If value is still -1, run number provided din't match any ranges specified so exit 
    print("!!!!! ERROR !!!!!\n Run number specified does not fall within a set of runs for which cuts are defined in %s\n!!!!! ERROR !!!!!" % TimingCutFile)
    sys.exit(3)
elif(TempPar > 1):
    print("!!! WARNING!!! Run number was found within the range of two (or more) line entries of %s !!! WARNING !!!" % TimingCutFile)
    print("The last matching entry will be treated as the input, you should ensure this is what you want")

# From our values from the file, reconstruct our windows 
PromptWindow[0] = PromptPeak - (BunchSpacing/2) - CoinOffset
PromptWindow[1] = PromptPeak + (BunchSpacing/2) + CoinOffset
RandomWindows[0] = PromptPeak - (BunchSpacing/2) - CoinOffset - (nSkip*BunchSpacing) - ((nWindows/2)*BunchSpacing)
RandomWindows[1] = PromptPeak - (BunchSpacing/2) - CoinOffset - (nSkip*BunchSpacing)
RandomWindows[2] = PromptPeak + (BunchSpacing/2) + CoinOffset + (nSkip*BunchSpacing)
RandomWindows[3] = PromptPeak + (BunchSpacing/2) + CoinOffset + (nSkip*BunchSpacing) + ((nWindows/2)*BunchSpacing)

##############################################################################################################################################
ROOT.gROOT.SetBatch(ROOT.kTRUE) # Set ROOT to batch mode explicitly, does not splash anything to screen
###############################################################################################################################################

# Read stuff from the main event tree

infile_DATA = ROOT.TFile.Open(rootFile_DATA, "READ")

Uncut_Pion_Events_Data_tree = infile_DATA.Get("Uncut_Pion_Events")
Cut_Pion_Events_Accpt_Data_tree = infile_DATA.Get("Cut_Pion_Events_Accpt")
Cut_Pion_Events_Prompt_Data_tree = infile_DATA.Get("Cut_Pion_Events_Prompt")
Cut_Pion_Events_Random_Data_tree = infile_DATA.Get("Cut_Pion_Events_Random")
nEntries_TBRANCH_DATA  = Cut_Pion_Events_Accpt_Data_tree.GetEntries()

###################################################################################################################################################
nbins = 200

# Defining Histograms for Pions
# Uncut Data Histograms
H_gtr_beta_protons_data_uncut = ROOT.TH1D("H_gtr_beta_protons_data_uncut", "HMS #beta; HMS_gtr_#beta; Counts", nbins, 0.8, 1.2)
H_cal_etottracknorm_protons_data_uncut = ROOT.TH1D("H_cal_etottracknorm_protons_data_uncut", "HMS cal etottracknorm (uncut); HMS_cal_etottracknorm; Counts", nbins, 0.0, 1.8)
H_cer_npeSum_protons_data_uncut = ROOT.TH1D("H_cer_npeSum_protons_data_uncut", "HMS cer npeSum (uncut); HMS_cer_npeSum; Counts", nbins, 0, 50)
H_RFTime_Dist_protons_data_uncut = ROOT.TH1D("H_RFTime_Dist_protons_data_uncut", "HMS RFTime (uncut); HMS_RFTime; Counts", nbins, 0, 4)
P_gtr_beta_protons_data_uncut = ROOT.TH1D("P_gtr_beta_protons_data_uncut", "SHMS #beta (uncut); SHMS_gtr_#beta; Counts", nbins, 0.0, 1.4)
P_gtr_dp_protons_data_uncut = ROOT.TH1D("P_gtr_dp_protons_protons_data_uncut", "SHMS #delta (uncut); SHMS_gtr_dp; Counts", nbins, -30, 30)
P_cal_etottracknorm_protons_data_uncut = ROOT.TH1D("P_cal_etottracknorm_protons_data_uncut", "SHMS cal etottracknorm (uncut); SHMS_cal_etottracknorm; Counts", nbins, 0.0, 1.8)
P_hgcer_npeSum_protons_data_uncut = ROOT.TH1D("P_hgcer_npeSum_protons_data_uncut", "SHMS HGC npeSum (uncut); SHMS_hgcer_npeSum; Counts", nbins, 0, 50)
P_hgcer_xAtCer_protons_data_uncut = ROOT.TH1D("P_hgcer_xAtCer_protons_data_uncut", "SHMS HGC xAtCer (uncut); SHMS_hgcer_xAtCer; Counts", nbins, -60, 60)
P_hgcer_yAtCer_protons_data_uncut = ROOT.TH1D("P_hgcer_yAtCer_protons_data_uncut", "SHMS HGC yAtCer (uncut); SHMS_hgcer_yAtCer; Counts", nbins, -60, 60)
P_ngcer_npeSum_protons_data_uncut = ROOT.TH1D("P_ngcer_npeSum_protons_data_uncut", "SHMS NGC npeSum (uncut); SHMS_ngcer_npeSum; Counts", nbins, 0, 50)
P_ngcer_xAtCer_protons_data_uncut = ROOT.TH1D("P_ngcer_xAtCer_protons_data_uncut", "SHMS NGC xAtCer (uncut); SHMS_ngcer_xAtCer; Counts", nbins, -60, 60)
P_ngcer_yAtCer_protons_data_uncut = ROOT.TH1D("P_ngcer_yAtCer_protons_data_uncut", "SHMS NGC yAtCer (uncut); SHMS_ngcer_yAtCer; Counts", nbins, -60, 60)
P_aero_npeSum_protons_data_uncut = ROOT.TH1D("P_aero_npeSum_protons_data_uncut", "SHMS aero npeSum (uncut); SHMS_aero_npeSum; Counts", nbins, 0, 50)
P_aero_xAtAero_protons_data_uncut = ROOT.TH1D("P_acero_xAtAero_protons_data_uncut", "SHMS aero xAtAero (uncut); SHMS_aero_xAtAero; Counts", nbins, -60, 60)
P_aero_yAtAero_protons_data_uncut = ROOT.TH1D("P_aero_yAtAero_protons_data_uncut", "SHMS aero yAtAero (uncut); SHMS_aero_yAtAero; Counts", nbins, -60, 60)
P_kin_MMp_protons_data_uncut = ROOT.TH1D("P_kin_MMp_protons_data_uncut", "MIssing Mass data (uncut); MM_{\pi}; Counts", nbins, 0, 2.0)
P_RFTime_Dist_protons_data_uncut = ROOT.TH1D("P_RFTime_Dist_protons_data_uncut", "SHMS RFTime (uncut); SHMS_RFTime; Counts", nbins, 0, 4)
CTime_epCoinTime_ROC1_protons_data_uncut = ROOT.TH1D("CTime_epCoinTime_ROC1_protons_data_uncut", "Electron-Pion CTime (uncut); e pi Coin_Time; Counts", nbins, -50, 50)

# Acceptance Cut Data Histograms
H_gtr_beta_protons_data_accpt_cut_all = ROOT.TH1D("H_gtr_beta_protons_data_accpt_cut_all", "HMS #beta (accpt_cut); HMS_gtr_#beta; Counts", nbins, 0.0, 1.2)
H_cal_etottracknorm_protons_data_accpt_cut_all = ROOT.TH1D("H_cal_etottracknorm_protons_data_accpt_cut_all", "HMS cal etottracknorm (accpt_cut); HMS_cal_etottracknorm; Counts", nbins, 0.0, 1.8)
H_cer_npeSum_protons_data_accpt_cut_all = ROOT.TH1D("H_cer_npeSum_protons_data_accpt_cut_all", "HMS cer npeSum (accpt_cut); HMS_cer_npeSum; Counts", nbins, 0, 50)
H_RFTime_Dist_protons_data_accpt_cut_all = ROOT.TH1D("H_RFTime_Dist_protons_data_accpt_cut_all", "HMS RFTime (accpt_cut); HMS_RFTime; Counts", nbins, 0, 4)
P_gtr_beta_protons_data_accpt_cut_all = ROOT.TH1D("P_gtr_beta_protons_data_accpt_cut_all", "SHMS #beta (accpt_cut); SHMS_gtr_#beta; Counts", nbins, 0.0, 1.4)
P_gtr_dp_protons_data_accpt_cut_all = ROOT.TH1D("P_gtr_dp_protons_protons_data_accpt_cut_all", "SHMS #delta (accpt_cut); SHMS_gtr_dp; Counts", nbins, -30, 30)
P_cal_etottracknorm_protons_data_accpt_cut_all = ROOT.TH1D("P_cal_etottracknorm_protons_data_accpt_cut_all", "SHMS cal etottracknorm (accpt_cut_all); SHMS_cal_etottracknorm; Counts", nbins, 0, 1.8)
P_hgcer_npeSum_protons_data_accpt_cut_all = ROOT.TH1D("P_hgcer_npeSum_protons_data_accpt_cut_all", "SHMS HGC npeSum (accpt_cut); SHMS_hgcer_npeSum; Counts", nbins, 0, 50)
P_hgcer_xAtCer_protons_data_accpt_cut_all = ROOT.TH1D("P_hgcer_xAtCer_protons_data_accpt_cut_all", "SHMS HGC xAtCer (accpt_cut); SHMS_hgcer_xAtCer; Counts", nbins, -60, 60)
P_hgcer_yAtCer_protons_data_accpt_cut_all = ROOT.TH1D("P_hgcer_yAtCer_protons_data_accpt_cut_all", "SHMS HGC yAtCer (accpt_cut); SHMS_hgcer_yAtCer; Counts", nbins, -60, 60)
P_ngcer_npeSum_protons_data_accpt_cut_all = ROOT.TH1D("P_ngcer_npeSum_protons_data_accpt_cut_all", "SHMS NGC npeSum (accpt_cut); SHMS_ngcer_npeSum; Counts", nbins, 0, 50)
P_ngcer_xAtCer_protons_data_accpt_cut_all = ROOT.TH1D("P_ngcer_xAtCer_protons_data_accpt_cut_all", "SHMS NGC xAtCer (accpt_cut); SHMS_ngcer_xAtCer; Counts", nbins, -60, 60)
P_ngcer_yAtCer_protons_data_accpt_cut_all = ROOT.TH1D("P_ngcer_yAtCer_protons_data_accpt_cut_all", "SHMS NGC yAtCer (accpt_cut); SHMS_ngcer_yAtCer; Counts", nbins, -60, 60)
P_aero_npeSum_protons_data_accpt_cut_all = ROOT.TH1D("P_aero_npeSum_protons_data_accpt_cut_all", "SHMS aero npeSum (accpt_cut); SHMS_aero_npeSum; Counts", nbins, 0, 50)
P_aero_xAtAero_protons_data_accpt_cut_all = ROOT.TH1D("P_acero_xAtAero_protons_data_accpt_cut_all", "SHMS aero xAtAero (accpt_cut); SHMS_aero_xAtAero; Counts", nbins, -60, 60)
P_aero_yAtAero_protons_data_accpt_cut_all = ROOT.TH1D("P_aero_yAtAero_protons_data_accpt_cut_all", "SHMS aero yAtAero (accpt_cut); SHMS_aero_yAtAero; Counts", nbins, -60, 60)
P_kin_MMp_protons_data_accpt_cut_all = ROOT.TH1D("P_kin_MMp_protons_data_accpt_cut_all", "MIssing Mass data (accpt_cut); MM_{\pi}; Counts", nbins, 0, 2.0)
P_RFTime_Dist_protons_data_accpt_cut_all = ROOT.TH1D("P_RFTime_Dist_protons_data_accpt_cut_all", "SHMS RFTime (accpt_cut); SHMS_RFTime; Counts", nbins, 0, 4)
CTime_epCoinTime_ROC1_protons_data_accpt_cut_all = ROOT.TH1D("CTime_epCoinTime_ROC1_protons_data_accpt_cut_all", "Electron-Pion CTime (accpt_cut); e pi Coin_Time; Counts", nbins, -50, 50)

# Prompt Cut Data Histograms
H_gtr_beta_protons_data_prompt_cut_all = ROOT.TH1D("H_gtr_beta_protons_data_prompt_cut_all", "HMS #beta (prompt+accpt_cut); HMS_gtr_#beta; Counts", nbins, 0.0, 1.2)
H_cal_etottracknorm_protons_data_prompt_cut_all = ROOT.TH1D("H_cal_etottracknorm_protons_data_prompt_cut_all", "HMS cal etottracknorm (prompt+accpt_cut); HMS_cal_etottracknorm; Counts", nbins, 0.0, 1.8)
H_cer_npeSum_protons_data_prompt_cut_all = ROOT.TH1D("H_cer_npeSum_protons_data_prompt_cut_all", "HMS cer npeSum (prompt+accpt_cut); HMS_cer_npeSum; Counts", nbins, 0, 50)
H_RFTime_Dist_protons_data_prompt_cut_all = ROOT.TH1D("H_RFTime_Dist_protons_data_prompt_cut_all", "HMS RFTime (prompt+accpt_cut); HMS_RFTime; Counts", nbins, 0, 4)
P_gtr_beta_protons_data_prompt_cut_all = ROOT.TH1D("P_gtr_beta_protons_data_prompt_cut_all", "SHMS #beta (prompt+accpt_cut); SHMS_gtr_#beta; Counts", nbins, 0.0, 1.4)
P_gtr_dp_protons_data_prompt_cut_all = ROOT.TH1D("P_gtr_dp_protons_protons_data_prompt_cut_all", "SHMS #delta (prompt+accpt_cut); SHMS_gtr_dp; Counts", nbins, -30, 30)
P_cal_etottracknorm_protons_data_prompt_cut_all = ROOT.TH1D("P_cal_etottracknorm_protons_data_prompt_cut_all", "SHMS cal etottracknorm (prompt+accpt_cut); SHMS_cal_etottracknorm; Counts", nbins, 0, 1.8)
P_hgcer_npeSum_protons_data_prompt_cut_all = ROOT.TH1D("P_hgcer_npeSum_protons_data_prompt_cut_all", "SHMS HGC npeSum (prompt+accpt_cut); SHMS_hgcer_npeSum; Counts", nbins, 0, 50)
P_hgcer_xAtCer_protons_data_prompt_cut_all = ROOT.TH1D("P_hgcer_xAtCer_protons_data_prompt_cut_all", "SHMS HGC xAtCer (prompt+accpt_cut); SHMS_hgcer_xAtCer; Counts", nbins, -60, 60)
P_hgcer_yAtCer_protons_data_prompt_cut_all = ROOT.TH1D("P_hgcer_yAtCer_protons_data_prompt_cut_all", "SHMS HGC yAtCer (prompt+accpt_cut); SHMS_hgcer_yAtCer; Counts", nbins, -60, 60)
P_ngcer_npeSum_protons_data_prompt_cut_all = ROOT.TH1D("P_ngcer_npeSum_protons_data_prompt_cut_all", "SHMS NGC npeSum (prompt+accpt_cut); SHMS_ngcer_npeSum; Counts", nbins, 0, 50)
P_ngcer_xAtCer_protons_data_prompt_cut_all = ROOT.TH1D("P_ngcer_xAtCer_protons_data_prompt_cut_all", "SHMS NGC xAtCer (prompt+accpt_cut); SHMS_ngcer_xAtCer; Counts", nbins, -60, 60)
P_ngcer_yAtCer_protons_data_prompt_cut_all = ROOT.TH1D("P_ngcer_yAtCer_protons_data_prompt_cut_all", "SHMS NGC yAtCer (prompt+accpt_cut); SHMS_ngcer_yAtCer; Counts", nbins, -60, 60)
P_aero_npeSum_protons_data_prompt_cut_all = ROOT.TH1D("P_aero_npeSum_protons_data_prompt_cut_all", "SHMS aero npeSum (prompt+accpt_cut); SHMS_aero_npeSum; Counts", nbins, 0, 50)
P_aero_xAtAero_protons_data_prompt_cut_all = ROOT.TH1D("P_acero_xAtAero_protons_data_prompt_cut_all", "SHMS aero xAtAero (prompt+accpt_cut); SHMS_aero_xAtAero; Counts", nbins, -60, 60)
P_aero_yAtAero_protons_data_prompt_cut_all = ROOT.TH1D("P_aero_yAtAero_protons_data_prompt_cut_all", "SHMS aero yAtAero (prompt+accpt_cut); SHMS_aero_yAtAero; Counts", nbins, -60, 60)
P_kin_MMp_protons_data_prompt_cut_all = ROOT.TH1D("P_kin_MMp_protons_data_prompt_cut_all", "MIssing Mass data (prompt+accpt_cut); MM_{\pi}; Counts", nbins, 0.0, 2.0)
P_RFTime_Dist_protons_data_prompt_cut_all = ROOT.TH1D("P_RFTime_Dist_protons_data_prompt_cut_all", "SHMS RFTime (prompt+accpt_cut); SHMS_RFTime; Counts", nbins, 0, 4)
CTime_epCoinTime_ROC1_protons_data_prompt_cut_all = ROOT.TH1D("CTime_epCoinTime_ROC1_protons_data_prompt_cut_all", "Electron-Pion CTime (prompt+accpt_cut); e pi Coin_Time; Counts", nbins, -50, 50)

# Random Cut Data Histograms
H_gtr_beta_protons_data_random_cut_all = ROOT.TH1D("H_gtr_beta_protons_data_random_cut_all", "HMS #beta (random+accpt_cut); HMS_gtr_#beta; Counts", nbins, 0.0, 1.2)
H_cal_etottracknorm_protons_data_random_cut_all = ROOT.TH1D("H_cal_etottracknorm_protons_data_random_cut_all", "HMS cal etottracknorm (random+accpt_cut); HMS_cal_etottracknorm; Counts", nbins, 0.0, 1.8)
H_cer_npeSum_protons_data_random_cut_all = ROOT.TH1D("H_cer_npeSum_protons_data_random_cut_all", "HMS cer npeSum (random+accpt_cut); HMS_cer_npeSum; Counts", nbins, 0, 50)
H_RFTime_Dist_protons_data_random_cut_all = ROOT.TH1D("H_RFTime_Dist_protons_data_random_cut_all", "HMS RFTime (random+accpt_cut); HMS_RFTime; Counts", nbins, 0, 4)
P_gtr_beta_protons_data_random_cut_all = ROOT.TH1D("P_gtr_beta_protons_data_random_cut_all", "SHMS #beta (random+accpt_cut); SHMS_gtr_#beta; Counts", nbins, 0.0, 1.4)
P_gtr_dp_protons_data_random_cut_all = ROOT.TH1D("P_gtr_dp_protons_protons_data_random_cut_all", "SHMS #delta (random+accpt_cut); SHMS_gtr_dp; Counts", nbins, -30, 30)
P_cal_etottracknorm_protons_data_random_cut_all = ROOT.TH1D("P_cal_etottracknorm_protons_data_random_cut_all", "SHMS cal etottracknorm (random+accpt_cut); SHMS_cal_etottracknorm; Counts", nbins, 0, 1.8)
P_hgcer_npeSum_protons_data_random_cut_all = ROOT.TH1D("P_hgcer_npeSum_protons_data_random_cut_all", "SHMS HGC npeSum (random+accpt_cut); SHMS_hgcer_npeSum; Counts", nbins, 0, 50)
P_hgcer_xAtCer_protons_data_random_cut_all = ROOT.TH1D("P_hgcer_xAtCer_protons_data_random_cut_all", "SHMS HGC xAtCer (random+accpt_cut); SHMS_hgcer_xAtCer; Counts", nbins, -60, 60)
P_hgcer_yAtCer_protons_data_random_cut_all = ROOT.TH1D("P_hgcer_yAtCer_protons_data_random_cut_all", "SHMS HGC yAtCer (random+accpt_cut); SHMS_hgcer_yAtCer; Counts", nbins, -60, 60)
P_ngcer_npeSum_protons_data_random_cut_all = ROOT.TH1D("P_ngcer_npeSum_protons_data_random_cut_all", "SHMS NGC npeSum (random+accpt_cut); SHMS_ngcer_npeSum; Counts", nbins, 0, 50)
P_ngcer_xAtCer_protons_data_random_cut_all = ROOT.TH1D("P_ngcer_xAtCer_protons_data_random_cut_all", "SHMS NGC xAtCer (random+accpt_cut); SHMS_ngcer_xAtCer; Counts", nbins, -60, 60)
P_ngcer_yAtCer_protons_data_random_cut_all = ROOT.TH1D("P_ngcer_yAtCer_protons_data_random_cut_all", "SHMS NGC yAtCer (random+accpt_cut); SHMS_ngcer_yAtCer; Counts", nbins, -60, 60)
P_aero_npeSum_protons_data_random_cut_all = ROOT.TH1D("P_aero_npeSum_protons_data_random_cut_all", "SHMS aero npeSum (random+accpt_cut); SHMS_aero_npeSum; Counts", nbins, 0, 50)
P_aero_xAtAero_protons_data_random_cut_all = ROOT.TH1D("P_acero_xAtAero_protons_data_random_cut_all", "SHMS aero xAtAero (random+accpt_cut); SHMS_aero_xAtAero; Counts", nbins, -60, 60)
P_aero_yAtAero_protons_data_random_cut_all = ROOT.TH1D("P_aero_yAtAero_protons_data_random_cut_all", "SHMS aero yAtAero (random+accpt_cut); SHMS_aero_yAtAero; Counts", nbins, -60, 60)
P_kin_MMp_protons_data_random_cut_all = ROOT.TH1D("P_kin_MMp_protons_data_random_cut_all", "MIssing Mass data (random+accpt_cut); MM_{\pi}; Counts", nbins, 0, 2.0)
P_RFTime_Dist_protons_data_random_cut_all = ROOT.TH1D("P_RFTime_Dist_protons_data_random_cut_all", "SHMS RFTime (random+accpt_cut); SHMS_RFTime; Counts", nbins, 0, 4)
CTime_epCoinTime_ROC1_protons_data_random_cut_all = ROOT.TH1D("CTime_epCoinTime_ROC1_protons_data_random_cut_all", "Electron-Pion CTime (random+accpt_cut); e pi Coin_Time; Counts", nbins, -50, 50)
CTime_epCoinTime_ROC1_protons_data_random_unsub_cut_all = ROOT.TH1D("CTime_epCoinTime_ROC1_protons_data_random_unsub_cut_all", "Electron-Pion CTime (random+accpt_cut); e pi Coin_Time; Counts", nbins, -50, 50)

# Cut All Data Histograms
H_gtr_beta_protons_data_cut_all = ROOT.TH1D("H_gtr_beta_protons_data_cut_all", "HMS #beta (all_cut); HMS_gtr_#beta; Counts", nbins, 0.0, 1.2)
H_cal_etottracknorm_protons_data_cut_all = ROOT.TH1D("H_cal_etottracknorm_protons_data_cut_all", "HMS cal etottracknorm (all_cut); HMS_cal_etottracknorm; Counts", nbins, 0.0, 1.8)
H_cer_npeSum_protons_data_cut_all = ROOT.TH1D("H_cer_npeSum_protons_data_cut_all", "HMS cer npeSum (all_cut); HMS_cer_npeSum; Counts", nbins, 0, 50)
H_RFTime_Dist_protons_data_cut_all = ROOT.TH1D("H_RFTime_Dist_protons_data_cut_all", "HMS RFTime (all_cut); HMS_RFTime; Counts", nbins, 0, 4)
P_gtr_beta_protons_data_cut_all = ROOT.TH1D("P_gtr_beta_protons_data_cut_all", "SHMS #beta (all_cut); SHMS_gtr_#beta; Counts", nbins, 0.0, 1.4)
P_gtr_dp_protons_data_cut_all = ROOT.TH1D("P_gtr_dp_protons_protons_data_cut_all", "SHMS #delta (cut_all); SHMS_gtr_dp; Counts", nbins, -30, 30)
P_cal_etottracknorm_protons_data_cut_all = ROOT.TH1D("P_cal_etottracknorm_protons_data_cut_all", "SHMS cal etottracknorm (cut_all); SHMS_cal_etottracknorm; Counts", nbins, 0, 1.8)
P_hgcer_npeSum_protons_data_cut_all = ROOT.TH1D("P_hgcer_npeSum_protons_data_cut_all", "SHMS HGC npeSum (all_cut); SHMS_hgcer_npeSum; Counts", nbins, 0, 50)
P_hgcer_xAtCer_protons_data_cut_all = ROOT.TH1D("P_hgcer_xAtCer_protons_data_cut_all", "SHMS HGC xAtCer (all_cut); SHMS_hgcer_xAtCer; Counts", nbins, -60, 60)
P_hgcer_yAtCer_protons_data_cut_all = ROOT.TH1D("P_hgcer_yAtCer_protons_data_cut_all", "SHMS HGC yAtCer (all_cut); SHMS_hgcer_yAtCer; Counts", nbins, -60, 60)
P_ngcer_npeSum_protons_data_cut_all = ROOT.TH1D("P_ngcer_npeSum_protons_data_cut_all", "SHMS NGC npeSum (all_cut); SHMS_ngcer_npeSum; Counts", nbins, 0, 50)
P_ngcer_xAtCer_protons_data_cut_all = ROOT.TH1D("P_ngcer_xAtCer_protons_data_cut_all", "SHMS NGC xAtCer (all_cut); SHMS_ngcer_xAtCer; Counts", nbins, -60, 60)
P_ngcer_yAtCer_protons_data_cut_all = ROOT.TH1D("P_ngcer_yAtCer_protons_data_cut_all", "SHMS NGC yAtCer (all_cut); SHMS_ngcer_yAtCer; Counts", nbins, -60, 60)
P_aero_npeSum_protons_data_cut_all = ROOT.TH1D("P_aero_npeSum_protons_data_cut_all", "SHMS aero npeSum (all_cut); SHMS_aero_npeSum; Counts", nbins, 0, 50)
P_aero_xAtAero_protons_data_cut_all = ROOT.TH1D("P_acero_xAtAero_protons_data_cut_all", "SHMS aero xAtAero (all_cut); SHMS_aero_xAtAero; Counts", nbins, -60, 60)
P_aero_yAtAero_protons_data_cut_all = ROOT.TH1D("P_aero_yAtAero_protons_data_cut_all", "SHMS aero yAtAero (all_cut); SHMS_aero_yAtAero; Counts", nbins, -60, 60)
P_kin_MMp_protons_data_cut_all = ROOT.TH1D("P_kin_MMp_protons_data_cut_all", "MIssing Mass data (all_cut); MM_{\pi}; Counts", nbins, 0, 2.0)
P_RFTime_Dist_protons_data_cut_all = ROOT.TH1D("P_RFTime_Dist_protons_data_cut_all", "SHMS RFTime (all_cut); SHMS_RFTime; Counts", nbins, 0, 4)
CTime_epCoinTime_ROC1_protons_data_cut_all = ROOT.TH1D("CTime_epCoinTime_ROC1_protons_data_cut_all", "Electron-Pion CTime (all_cut); e pi Coin_Time; Counts", nbins, -50, 50)

# 2D Histograms
# Uncut Histograms
P_RFTime_vs_gtr_dp_protons_uncut = ROOT.TH2D("P_RFTime_vs_gtr_dp_protons_uncut","P_RFTime_Dist vs P_gtr_dp (uncut); P_RFTime_Dist; P_gtr_dp", 200, 0, 4, 200, -30, 30)
H_cal_etottracknorm_vs_cer_npeSum_protons_uncut = ROOT.TH2D("H_cal_etottracknorm_vs_cer_npeSum_protons_uncut","HMS cal etottracknorm vs HMS cer npeSum (uncut); H_cal_etottracknorm; H_cer_npeSum",100, 0, 2, 100, 0, 50)
P_hgcer_vs_aero_npe_protons_uncut = ROOT.TH2D("P_hgcer_vs_aero_npe_protons_uncut", "SHMS HGC npeSum vs SHMS Aero npeSum (uncut); SHMS_hgcer_npeSum; SHMS_aero_npeSum", 100, 0, 50, 100, 0, 50)
P_ngcer_vs_hgcer_npe_protons_uncut = ROOT.TH2D("P_ngcer_vs_hgcer_npe_protons_uncut", "SHMS NGC npeSum vs SHMS HGC npeSum (uncut); SHMS_ngcer_npeSum; SHMS_hgcer_npeSum", 100, 0, 50, 100, 0, 50)
P_ngcer_vs_aero_npe_protons_uncut = ROOT.TH2D("P_ngcer_vs_aero_npe_protons_uncut", "SHMS NGC npeSum vs SHMS aero npeSum (uncut); SHMS_ngcer_npeSum; SHMS_aero_npeSum", 100, 0, 50, 100, 0, 50)
P_hgcer_yAtCer_vs_hgcer_xAtCer_protons_uncut = ROOT.TH2D("P_hgcer_yAtCer_vs_hgcer_xAtCer_protons_uncut", "SHMS HGC yAtCer vs SHMS HGC xAtCer (uncut); SHMS_hgcer_yAtCer; SHMS_hgcer_xAtCer", 100, -50, 50, 100, -50, 50)
P_aero_yAtAero_vs_aero_xAtAero_protons_uncut = ROOT.TH2D("P_aero_yAtAero_vs_aero_xAtAero_protons_uncut", "SHMS aero yAtAero vs SHMS aero xAtAero (uncut); SHMS_aero_yAtAero; SHMS_aero_xAtAero", 100, -50, 50, 100, -50, 50)
P_ngcer_yAtCer_vs_ngcer_xAtCer_protons_uncut = ROOT.TH2D("P_ngcer_yAtCer_vs_ngcer_xAtCer_protons_uncut", "SHMS NGC yAtCer vs SHMS NGC xAtCer (uncut); SHMS_ngcer_yAtCer; SHMS_ngcer_xAtCer", 100, -50, 50, 100, -50, 50)
P_cal_etottracknorm_vs_ngcer_npe_protons_uncut = ROOT.TH2D("P_cal_etottracknorm_vs_ngcer_npe_protons_uncut", "SHMS cal etottracknorm vs SHMS NGC xAtCer (uncut); SHMS_cal_etottracknorm; SHMS_ngcer_xAtCer", 100, -10, 10, 100, -10, 10)
CTime_epCoinTime_vs_MMp_protons_uncut = ROOT.TH2D("CTime_epCoinTime_vs_MMp_protons_uncut","Electron-Pion CTime vs Missing Mass (uncut); e #pi Coin_Time; MM_{#pi}", 300, -30, 30, 100, 0, 2)
CTime_epCoinTime_vs_beta_protons_uncut = ROOT.TH2D("CTime_epCoinTime_vs_beta_protons_uncut", "Electron-Pion CTime vs SHMS #beta (uncut); e #pi Coin_Time; SHMS_#beta", 120, -30, 30, 200, 0, 2)
P_RFTime_vs_MMp_protons_uncut = ROOT.TH2D("P_RFTime_vs_MMp_protons_uncut", "SHMS RFTime vs Missing Mass (uncut); SHMS_RFTime_Dist; MM_{#pi}", 100, 0, 4, 100, 0, 2)
CTime_epCoinTime_vs_RFTime_protons_uncut = ROOT.TH2D("CTime_epCoinTime_vs_RFTime_protons_uncut", "Electron-Pion CTime vs SHMS RFTime (uncut); e #pi Coin_Time; SHMS_RFTime_Dist", 300, -30, 30, 200, 0, 4)

# Acceptance Cut Histograms
P_RFTime_vs_gtr_dp_protons_accpt_cut_all = ROOT.TH2D("P_RFTime_vs_gtr_dp_protons_accpt_cut_all","P_RFTime_Dist vs P_gtr_dp (accpt_cut); P_RFTime_Dist; P_gtr_dp", 200, 0, 4, 200, -30, 30)
H_cal_etottracknorm_vs_cer_npeSum_protons_accpt_cut_all = ROOT.TH2D("H_cal_etottracknorm_vs_cer_npeSum_protons_accpt_cut_all","HMS cal etottracknorm vs HMS cer npeSum (accpt_cut); H_cal_etottracknorm; H_cer_npeSum",100, 0, 2, 100, 0, 40)
P_hgcer_vs_aero_npe_protons_accpt_cut_all = ROOT.TH2D("P_hgcer_vs_aero_npe_protons_accpt_cut_all", "SHMS HGC npeSum vs SHMS Aero npeSum (accpt_cut); SHMS_hgcer_npeSum; SHMS_aero_npeSum", 100, 0, 50, 100, 0, 50)
P_ngcer_vs_hgcer_npe_protons_accpt_cut_all = ROOT.TH2D("P_ngcer_vs_hgcer_npe_protons_accpt_cut_all", "SHMS NGC npeSum vs SHMS HGC npeSum (accpt_cut); SHMS_ngcer_npeSum; SHMS_hgcer_npeSum", 100, 0, 50, 100, 0, 50)
P_ngcer_vs_aero_npe_protons_accpt_cut_all = ROOT.TH2D("P_ngcer_vs_aero_npe_protons_accpt_cut_all", "SHMS NGC npeSum vs SHMS aero npeSum (accpt_cut); SHMS_ngcer_npeSum; SHMS_aero_npeSum", 100, 0, 50, 100, 0, 50)
P_hgcer_yAtCer_vs_hgcer_xAtCer_protons_accpt_cut_all = ROOT.TH2D("P_hgcer_yAtCer_vs_hgcer_xAtCer_protons_accpt_cut_all", "SHMS HGC yAtCer vs SHMS HGC xAtCer (accpt_cut); SHMS_hgcer_yAtCer; SHMS_hgcer_xAtCer", 100, -50, 50, 100, -50, 50)
P_aero_yAtAero_vs_aero_xAtAero_protons_accpt_cut_all = ROOT.TH2D("P_aero_yAtAero_vs_aero_xAtAero_protons_accpt_cut_all", "SHMS aero yAtAero vs SHMS aero xAtAero (accpt_cut); SHMS_aero_yAtAero; SHMS_aero_xAtAero", 100, -50, 50, 100, -50, 50)
P_ngcer_yAtCer_vs_ngcer_xAtCer_protons_accpt_cut_all = ROOT.TH2D("P_ngcer_yAtCer_vs_ngcer_xAtCer_protons_accpt_cut_all", "SHMS NGC yAtCer vs SHMS NGC xAtCer (accpt_cut); SHMS_ngcer_yAtCer; SHMS_ngcer_xAtCer", 100, -50, 50, 100, -50, 50)
P_cal_etottracknorm_vs_ngcer_npe_protons_accpt_cut_all = ROOT.TH2D("P_cal_etottracknorm_vs_ngcer_npe_protons_accpt_cut_all", "SHMS cal etottracknorm vs SHMS NGC xAtCer (accpt_cut); SHMS_cal_etottracknorm; SHMS_ngcer_xAtCer", 100, -10, 10, 100, -10, 10)
CTime_epCoinTime_vs_MMp_protons_accpt_cut_all = ROOT.TH2D("CTime_epCoinTime_vs_MMp_protons_accpt_cut_all","Electron-Pion CTime vs Missing Mass (accpt_cut); e #pi Coin_Time; MM_{#pi}", 300, -30, 30, 100, 0, 2)
CTime_epCoinTime_vs_beta_protons_accpt_cut_all = ROOT.TH2D("CTime_epCoinTime_vs_beta_protons_accpt_cut_all", "Electron-Pion CTime vs SHMS #beta (accpt_cut); e #pi Coin_Time; SHMS_#beta", 120, -30, 30, 200, 0, 2)
P_RFTime_vs_MMp_protons_accpt_cut_all = ROOT.TH2D("P_RFTime_vs_MMp_protons_accpt_cut_all", "SHMS RFTime vs Missing Mass (accpt_cut); SHMS_RFTime_Dist; MM_{#pi}", 100, 0, 4, 100, 0, 2)
CTime_epCoinTime_vs_RFTime_protons_accpt_cut_all = ROOT.TH2D("CTime_epCoinTime_vs_RFTime_protons_accpt_cut_all", "Electron-Pion CTime vs SHMS RFTime (accpt_cut); e #pi Coin_Time; SHMS_RFTime_Dist", 300, -30, 30, 200, 0, 4)

# Prompt + Acceptance Cut Histograms
P_RFTime_vs_gtr_dp_protons_prompt_cut_all = ROOT.TH2D("P_RFTime_vs_gtr_dp_protons_prompt_cut_all","P_RFTime_Dist vs P_gtr_dp (prompt+accpt_cut); P_RFTime_Dist; P_gtr_dp", 200, 0, 4, 200, -30, 30)
H_cal_etottracknorm_vs_cer_npeSum_protons_prompt_cut_all = ROOT.TH2D("H_cal_etottracknorm_vs_cer_npeSum_protons_prompt_cut_all","HMS cal etottracknorm vs HMS cer npeSum (prompt+accpt_cut); H_cal_etottracknorm; H_cer_npeSum",100, 0, 2, 100, 0, 40)
P_hgcer_vs_aero_npe_protons_prompt_cut_all = ROOT.TH2D("P_hgcer_vs_aero_npe_protons_prompt_cut_all", "SHMS HGC npeSum vs SHMS Aero npeSum (prompt+accpt_cut); SHMS_hgcer_npeSum; SHMS_aero_npeSum", 100, 0, 50, 100, 0, 50)
P_ngcer_vs_hgcer_npe_protons_prompt_cut_all = ROOT.TH2D("P_ngcer_vs_hgcer_npe_protons_prompt_cut_all", "SHMS NGC npeSum vs SHMS HGC npeSum (prompt+accpt_cut); SHMS_ngcer_npeSum; SHMS_hgcer_npeSum", 100, 0, 50, 100, 0, 50)
P_ngcer_vs_aero_npe_protons_prompt_cut_all = ROOT.TH2D("P_ngcer_vs_aero_npe_protons_prompt_cut_all", "SHMS NGC npeSum vs SHMS aero npeSum (prompt+accpt_cut); SHMS_ngcer_npeSum; SHMS_aero_npeSum", 100, 0, 50, 100, 0, 50)
P_hgcer_yAtCer_vs_hgcer_xAtCer_protons_prompt_cut_all = ROOT.TH2D("P_hgcer_yAtCer_vs_hgcer_xAtCer_protons_prompt_cut_all", "SHMS HGC yAtCer vs SHMS HGC xAtCer (prompt+accpt_cut); SHMS_hgcer_yAtCer; SHMS_hgcer_xAtCer", 100, -50, 50, 100, -50, 50)
P_aero_yAtAero_vs_aero_xAtAero_protons_prompt_cut_all = ROOT.TH2D("P_aero_yAtAero_vs_aero_xAtAero_protons_prompt_cut_all", "SHMS aero yAtAero vs SHMS aero xAtAero (prompt+accpt_cut); SHMS_aero_yAtAero; SHMS_aero_xAtAero", 100, -50, 50, 100, -50, 50)
P_ngcer_yAtCer_vs_ngcer_xAtCer_protons_prompt_cut_all = ROOT.TH2D("P_ngcer_yAtCer_vs_ngcer_xAtCer_protons_prompt_cut_all", "SHMS NGC yAtCer vs SHMS NGC xAtCer (prompt+accpt_cut); SHMS_ngcer_yAtCer; SHMS_ngcer_xAtCer", 100, -50, 50, 100, -50, 50)
P_cal_etottracknorm_vs_ngcer_npe_protons_prompt_cut_all = ROOT.TH2D("P_cal_etottracknorm_vs_ngcer_npe_protons_prompt_cut_all", "SHMS cal etottracknorm vs SHMS NGC xAtCer (prompt+accpt_cut); SHMS_cal_etottracknorm; SHMS_ngcer_xAtCer", 100, -10, 10, 100, -10, 10)
CTime_epCoinTime_vs_MMp_protons_prompt_cut_all = ROOT.TH2D("CTime_epCoinTime_vs_MMp_protons_prompt_cut_all","Electron-Pion CTime vs Missing Mass (prompt+accpt_cut); e #pi Coin_Time; MM_{#pi}", 300, -30, 30, 200, 0, 2)
CTime_epCoinTime_vs_beta_protons_prompt_cut_all = ROOT.TH2D("CTime_epCoinTime_vs_beta_protons_prompt_cut_all", "Electron-Pion CTime vs SHMS #beta (prompt+accpt_cut); e #pi Coin_Time; SHMS_#beta", 120, -30, 30, 200, 0, 2)
P_RFTime_vs_MMp_protons_prompt_cut_all = ROOT.TH2D("P_RFTime_vs_MMp_protons_prompt_cut_all", "SHMS RFTime vs Missing Mass (prompt+accpt_cut); SHMS_RFTime_Dist; MM_{#pi}", 100, 0, 4, 100, 0, 2)
CTime_epCoinTime_vs_RFTime_protons_prompt_cut_all = ROOT.TH2D("CTime_epCoinTime_vs_RFTime_protons_prompt_cut_all", "Electron-Pion CTime vs SHMS RFTime (prompt+accpt_cut); e #pi Coin_Time; SHMS_RFTime_Dist", 300, -30, 30, 100, 0, 4)

# Acceptance + Random Cut Histograms
P_RFTime_vs_gtr_dp_protons_random_cut_all = ROOT.TH2D("P_RFTime_vs_gtr_dp_protons_random_cut_all","P_RFTime_Dist vs P_gtr_dp (random+accpt_cut); P_RFTime_Dist; P_gtr_dp", 200, 0, 4, 200, -30, 30)
H_cal_etottracknorm_vs_cer_npeSum_protons_random_cut_all = ROOT.TH2D("H_cal_etottracknorm_vs_cer_npeSum_protons_random_cut_all","HMS cal etottracknorm vs HMS cer npeSum (random+accpt_cut); H_cal_etottracknorm; H_cer_npeSum",100, 0, 2, 100, 0, 40)
P_hgcer_vs_aero_npe_protons_random_cut_all = ROOT.TH2D("P_hgcer_vs_aero_npe_protons_random_cut_all", "SHMS HGC npeSum vs SHMS Aero npeSum (random+accpt_cut); SHMS_hgcer_npeSum; SHMS_aero_npeSum", 100, 0, 50, 100, 0, 50)
P_ngcer_vs_hgcer_npe_protons_random_cut_all = ROOT.TH2D("P_ngcer_vs_hgcer_npe_protons_random_cut_all", "SHMS NGC npeSum vs SHMS HGC npeSum (random+accpt_cut); SHMS_ngcer_npeSum; SHMS_hgcer_npeSum", 100, 0, 50, 100, 0, 50)
P_ngcer_vs_aero_npe_protons_random_cut_all = ROOT.TH2D("P_ngcer_vs_aero_npe_protons_random_cut_all", "SHMS NGC npeSum vs SHMS aero npeSum (random+accpt_cut); SHMS_ngcer_npeSum; SHMS_aero_npeSum", 100, 0, 50, 100, 0, 50)
P_hgcer_yAtCer_vs_hgcer_xAtCer_protons_random_cut_all = ROOT.TH2D("P_hgcer_yAtCer_vs_hgcer_xAtCer_protons_random_cut_all", "SHMS HGC yAtCer vs SHMS HGC xAtCer (random+accpt_cut); SHMS_hgcer_yAtCer; SHMS_hgcer_xAtCer", 100, -50, 50, 100, -50, 50)
P_aero_yAtAero_vs_aero_xAtAero_protons_random_cut_all = ROOT.TH2D("P_aero_yAtAero_vs_aero_xAtAero_protons_random_cut_all", "SHMS aero yAtAero vs SHMS aero xAtAero (random+accpt_cut); SHMS_aero_yAtAero; SHMS_aero_xAtAero", 100, -50, 50, 100, -50, 50)
P_ngcer_yAtCer_vs_ngcer_xAtCer_protons_random_cut_all = ROOT.TH2D("P_ngcer_yAtCer_vs_ngcer_xAtCer_protons_random_cut_all", "SHMS NGC yAtCer vs SHMS NGC xAtCer (random+accpt_cut); SHMS_ngcer_yAtCer; SHMS_ngcer_xAtCer", 100, -50, 50, 100, -50, 50)
P_cal_etottracknorm_vs_ngcer_npe_protons_random_cut_all = ROOT.TH2D("P_cal_etottracknorm_vs_ngcer_npe_protons_random_cut_all", "SHMS cal etottracknorm vs SHMS NGC xAtCer (random+accpt_cut); SHMS_cal_etottracknorm; SHMS_ngcer_xAtCer", 100, -10, 10, 100, -10, 10)
CTime_epCoinTime_vs_MMp_protons_random_cut_all = ROOT.TH2D("CTime_epCoinTime_vs_MMp_protons_random_cut_all","Electron-Pion CTime vs Missing Mass (random+accpt_cut); e #pi Coin_Time; MM_{#pi}", 300, -30, 30, 200, 0, 2)
CTime_epCoinTime_vs_beta_protons_random_cut_all = ROOT.TH2D("CTime_epCoinTime_vs_beta_protons_random_cut_all", "Electron-Pion CTime vs SHMS #beta (random+accpt_cut); e #pi Coin_Time; SHMS_#beta", 120, -30, 30, 200, 0, 2)
P_RFTime_vs_MMp_protons_random_cut_all = ROOT.TH2D("P_RFTime_vs_MMp_protons_random_cut_all", "SHMS RFTime vs Missing Mass (random+accpt_cut); SHMS_RFTime_Dist; MM_{#pi}", 100, 0, 4, 100, 0, 2)
CTime_epCoinTime_vs_RFTime_protons_random_cut_all = ROOT.TH2D("CTime_epCoinTime_vs_RFTime_protons_random_cut_all", "Electron-Pion CTime vs SHMS RFTime (random+accpt_cut); e #pi Coin_Time; SHMS_RFTime_Dist", 300, -30, 30, 100, 0, 4)

# All Cuts Histograms
P_RFTime_vs_gtr_dp_protons_cut_all = ROOT.TH2D("P_RFTime_vs_gtr_dp_protons_cut_all","P_RFTime_Dist vs P_gtr_dp (all_cut); P_RFTime_Dist; P_gtr_dp", 200, 0, 4, 200, -30, 30)
H_cal_etottracknorm_vs_cer_npeSum_protons_cut_all = ROOT.TH2D("H_cal_etottracknorm_vs_cer_npeSum_protons_cut_all","HMS cal etottracknorm vs HMS cer npeSum (all_cut); H_cal_etottracknorm; H_cer_npeSum",100, 0, 2, 100, 0, 40)
P_hgcer_vs_aero_npe_protons_cut_all = ROOT.TH2D("P_hgcer_vs_aero_npe_protons_cut_all", "SHMS HGC npeSum vs SHMS Aero npeSum (all_cut); SHMS_hgcer_npeSum; SHMS_aero_npeSum", 100, 0, 50, 100, 0, 50)
P_ngcer_vs_hgcer_npe_protons_cut_all = ROOT.TH2D("P_ngcer_vs_hgcer_npe_protons_cut_all", "SHMS NGC npeSum vs SHMS HGC npeSum (all_cut); SHMS_ngcer_npeSum; SHMS_hgcer_npeSum", 100, 0, 50, 100, 0, 50)
P_ngcer_vs_aero_npe_protons_cut_all = ROOT.TH2D("P_ngcer_vs_aero_npe_protons_cut_all", "SHMS NGC npeSum vs SHMS aero npeSum (all_cut); SHMS_ngcer_npeSum; SHMS_aero_npeSum", 100, 0, 50, 100, 0, 50)
P_hgcer_yAtCer_vs_hgcer_xAtCer_protons_cut_all = ROOT.TH2D("P_hgcer_yAtCer_vs_hgcer_xAtCer_protons_cut_all", "SHMS HGC yAtCer vs SHMS HGC xAtCer (all_cut); SHMS_hgcer_yAtCer; SHMS_hgcer_xAtCer", 100, -50, 50, 100, -50, 50)
P_aero_yAtAero_vs_aero_xAtAero_protons_cut_all = ROOT.TH2D("P_aero_yAtAero_vs_aero_xAtAero_protons_cut_all", "SHMS aero yAtAero vs SHMS aero xAtAero (all_cut); SHMS_aero_yAtAero; SHMS_aero_xAtAero", 100, -50, 50, 100, -50, 50)
P_ngcer_yAtCer_vs_ngcer_xAtCer_protons_cut_all = ROOT.TH2D("P_ngcer_yAtCer_vs_ngcer_xAtCer_protons_cut_all", "SHMS NGC yAtCer vs SHMS NGC xAtCer (all_cut); SHMS_ngcer_yAtCer; SHMS_ngcer_xAtCer", 100, -50, 50, 100, -50, 50)
P_cal_etottracknorm_vs_ngcer_npe_protons_cut_all = ROOT.TH2D("P_cal_etottracknorm_vs_ngcer_npe_protons_cut_all", "SHMS cal etottracknorm vs SHMS NGC xAtCer (all_cut); SHMS_cal_etottracknorm; SHMS_ngcer_xAtCer", 100, -10, 10, 100, -10, 10)
CTime_epCoinTime_vs_MMp_protons_cut_all = ROOT.TH2D("CTime_epCoinTime_vs_MMp_protons_cut_all","Electron-Pion CTime vs Missing Mass (all_cut); e #pi Coin_Time; MM_{#pi}", 300, -30, 30, 200, 0.0, 2.0)
CTime_epCoinTime_vs_beta_protons_cut_all = ROOT.TH2D("CTime_epCoinTime_vs_beta_protons_cut_all", "Electron-Pion CTime vs SHMS #beta (all_cut); e #pi Coin_Time; SHMS_#beta", 120, -30, 30, 200, 0, 2)
P_RFTime_vs_MMp_protons_cut_all = ROOT.TH2D("P_RFTime_vs_MMp_protons_cut_all", "SHMS RFTime vs Missing Mass (all_cut); SHMS_RFTime_Dist; MM_{#pi}", 100, 0, 4, 100, 0, 2)
CTime_epCoinTime_vs_RFTime_protons_cut_all = ROOT.TH2D("CTime_epCoinTime_vs_RFTime_protons_cut_all", "Electron-Pion CTime vs SHMS RFTime (all_cut); e #pi Coin_Time; SHMS_RFTime_Dist", 300, -30, 30, 100, 0, 4)

# 3D Histograms
P_HGC_xy_npe_protons_uncut = ROOT.TH3D("P_HGC_xy_npe_protons_uncut", "SHMS HGC NPE as fn of yAtCer vs SHMS HGC xAtCer (no cuts); HGC_yAtCer(cm); HGC_xAtCer(cm); NPE", 100, -50, 50, 100, -50, 50, 100, 0.1 , 50)
P_Aero_xy_npe_protons_uncut = ROOT.TH3D("P_Aero_xy_npe_protons_uncut", "SHMS Aerogel NPE as fn of yAtCer vs xAtCer (no cuts); Aero_yAtCer(cm); Aero_xAtCer(cm); NPE", 100, -50, 50, 100, -50, 50, 100, 0.1 , 50)
P_NGC_xy_npe_protons_uncut = ROOT.TH3D("P_NGC_xy_npe_protons_uncut", "SHMS NGC NPE as fn of yAtCer vs xAtCer (no cuts); NGC_yAtCer(cm); NGC_xAtCer(cm); NPE", 100, -50, 50, 100, -50, 50, 100, 0.1 , 50)
P_HGC_xy_npe_protons_accpt_cut_all = ROOT.TH3D("P_HGC_xy_npe_protons_accpt_cut_all", "SHMS HGC NPE as fn of yAtCer vs SHMS HGC xAtCer (accpt_cut); HGC_yAtCer(cm); HGC_xAtCer(cm); NPE", 100, -50, 50, 100, -50, 50, 100, 0.1 , 50)
P_Aero_xy_npe_protons_accpt_cut_all = ROOT.TH3D("P_Aero_xy_npe_protons_accpt_cut_all", "SHMS Aerogel NPE as fn of yAtCer vs xAtCer (accpt_cut); Aero_yAtCer(cm); Aero_xAtCer(cm); NPE", 100, -50, 50, 100, -50, 50, 100, 0.1 , 50)
P_NGC_xy_npe_protons_accpt_cut_all = ROOT.TH3D("P_NGC_xy_npe_protons_accpt_cut_all", "SHMS NGC NPE as fn of yAtCer vs xAtCer (accpt_cut); NGC_yAtCer(cm); NGC_xAtCer(cm); NPE", 100, -50, 50, 100, -50, 50, 100, 0.1 , 50)
P_HGC_xy_npe_protons_prompt_cut_all = ROOT.TH3D("P_HGC_xy_npe_protons_prompt_cut_all", "SHMS HGC NPE as fn of yAtCer vs SHMS HGC xAtCer (prompt+accpt_cut); HGC_yAtCer(cm); HGC_xAtCer(cm); NPE", 100, -50, 50, 100, -50, 50, 100, 0.1 , 50)
P_Aero_xy_npe_protons_prompt_cut_all = ROOT.TH3D("P_Aero_xy_npe_protons_prompt_cut_all", "SHMS Aerogel NPE as fn of yAtCer vs xAtCer (prompt+accpt_cut); Aero_yAtCer(cm); Aero_xAtCer(cm); NPE", 100, -50, 50, 100, -50, 50, 100, 0.1 , 50)
P_NGC_xy_npe_protons_prompt_cut_all = ROOT.TH3D("P_NGC_xy_npe_protons_prompt_cut_all", "SHMS NGC NPE as fn of yAtCer vs xAtCer (prompt+accpt_cut); NGC_yAtCer(cm); NGC_xAtCer(cm); NPE", 100, -50, 50, 100, -50, 50, 100, 0.1 , 50)
P_HGC_xy_npe_protons_random_cut_all = ROOT.TH3D("P_HGC_xy_npe_protons_random_cut_all", "SHMS HGC NPE as fn of yAtCer vs SHMS HGC xAtCer (random+accpt_cut); HGC_yAtCer(cm); HGC_xAtCer(cm); NPE", 100, -50, 50, 100, -50, 50, 100, 0.1 , 50)
P_Aero_xy_npe_protons_random_cut_all = ROOT.TH3D("P_Aero_xy_npe_protons_random_cut_all", "SHMS Aerogel NPE as fn of yAtCer vs xAtCer (random+accpt_cut); Aero_yAtCer(cm); Aero_xAtCer(cm); NPE", 100, -50, 50, 100, -50, 50, 100, 0.1 , 50)
P_NGC_xy_npe_protons_random_cut_all = ROOT.TH3D("P_NGC_xy_npe_protons_random_cut_all", "SHMS NGC NPE as fn of yAtCer vs xAtCer (random+accpt_cut); NGC_yAtCer(cm); NGC_xAtCer(cm); NPE", 100, -50, 50, 100, -50, 50, 100, 0.1 , 50)
P_HGC_xy_npe_protons_cut_all = ROOT.TH3D("P_HGC_xy_npe_protons_cut_all", "SHMS HGC NPE as fn of yAtCer vs SHMS HGC xAtCer (with cuts); HGC_yAtCer(cm); HGC_xAtCer(cm); NPE", 100, -50, 50, 100, -50, 50, 100, 0.1 , 50)
P_Aero_xy_npe_protons_cut_all = ROOT.TH3D("P_Aero_xy_npe_protons_cut_all", "SHMS Aerogel NPE as fn of yAtCer vs xAtCer (with cuts); Aero_yAtCer(cm); Aero_xAtCer(cm); NPE", 100, -50, 50, 100, -50, 50, 100, 0.1 , 50)
P_NGC_xy_npe_protons_cut_all = ROOT.TH3D("P_NGC_xy_npe_protons_cut_all", "SHMS NGC NPE as fn of yAtCer vs xAtCer (with cuts); NGC_yAtCer(cm); NGC_xAtCer(cm); NPE", 100, -50, 50, 100, -50, 50, 100, 0.1 , 50)

#################################################################################################################################################

# PID Cut Values
P_aero_npeSum_cut_value = 3.0
P_hgcer_npeSum_cut_value = 1.5
#P_ngcer_npeSum_cut_value = 0.5
#P_RF_Dist_low_cut_value = 1.2
#P_RF_Dist_high_cut_value = 3.4

# Fill Uncut Hitograms
for event in Uncut_Pion_Events_Data_tree:
    H_gtr_beta_protons_data_uncut.Fill(event.H_gtr_beta)
    H_cal_etottracknorm_protons_data_uncut.Fill(event.H_cal_etottracknorm)
    H_cer_npeSum_protons_data_uncut.Fill(event.H_cer_npeSum)
    H_RFTime_Dist_protons_data_uncut.Fill(event.H_RF_Dist)
    P_gtr_beta_protons_data_uncut.Fill(event.P_gtr_beta)
    P_gtr_dp_protons_data_uncut.Fill(event.P_gtr_dp)
    P_cal_etottracknorm_protons_data_uncut.Fill(event.P_cal_etottracknorm)
    P_hgcer_npeSum_protons_data_uncut.Fill(event.P_hgcer_npeSum)
    P_hgcer_xAtCer_protons_data_uncut.Fill(event.P_hgcer_xAtCer)
    P_hgcer_yAtCer_protons_data_uncut.Fill(event.P_hgcer_yAtCer)
    P_ngcer_npeSum_protons_data_uncut.Fill(event.P_ngcer_npeSum)
    P_ngcer_xAtCer_protons_data_uncut.Fill(event.P_ngcer_xAtCer)
    P_ngcer_yAtCer_protons_data_uncut.Fill(event.P_ngcer_yAtCer)
    P_aero_npeSum_protons_data_uncut.Fill(event.P_aero_npeSum)
    P_aero_xAtAero_protons_data_uncut.Fill(event.P_aero_xAtAero)
    P_aero_yAtAero_protons_data_uncut.Fill(event.P_aero_yAtAero)
    P_kin_MMp_protons_data_uncut.Fill(event.MMp)
    P_RFTime_Dist_protons_data_uncut.Fill(event.P_RF_Dist)
    CTime_epCoinTime_ROC1_protons_data_uncut.Fill(event.CTime_epCoinTime_ROC1)
    H_cal_etottracknorm_vs_cer_npeSum_protons_uncut.Fill(event.H_cal_etottracknorm, event.H_cer_npeSum)
    P_hgcer_vs_aero_npe_protons_uncut.Fill(event.P_hgcer_npeSum, event.P_aero_npeSum)
    P_ngcer_vs_hgcer_npe_protons_uncut.Fill(event.P_ngcer_npeSum, event.P_hgcer_npeSum)
    P_ngcer_vs_aero_npe_protons_uncut.Fill(event.P_ngcer_npeSum, event.P_aero_npeSum)
    P_hgcer_yAtCer_vs_hgcer_xAtCer_protons_uncut.Fill(event.P_hgcer_yAtCer, event.P_hgcer_xAtCer)
    P_aero_yAtAero_vs_aero_xAtAero_protons_uncut.Fill(event.P_aero_yAtAero, event.P_aero_xAtAero)
    P_ngcer_yAtCer_vs_ngcer_xAtCer_protons_uncut.Fill(event.P_ngcer_yAtCer, event.P_ngcer_xAtCer)
    P_cal_etottracknorm_vs_ngcer_npe_protons_uncut.Fill(event.P_cal_etottracknorm, event.P_ngcer_npeSum)
    CTime_epCoinTime_vs_MMp_protons_uncut.Fill(event.CTime_epCoinTime_ROC1, event.MMp)
    CTime_epCoinTime_vs_beta_protons_uncut.Fill(event.CTime_epCoinTime_ROC1, event.P_gtr_beta)
    P_RFTime_vs_MMp_protons_uncut.Fill(event.P_RF_Dist, event.MMp)
    CTime_epCoinTime_vs_RFTime_protons_uncut.Fill(event.CTime_epCoinTime_ROC1, event.P_RF_Dist)
    P_HGC_xy_npe_protons_uncut.Fill(event.P_hgcer_yAtCer,event.P_hgcer_xAtCer,event.P_hgcer_npeSum)
    P_Aero_xy_npe_protons_uncut.Fill(event.P_aero_yAtAero,event.P_aero_xAtAero,event.P_aero_npeSum)
    P_NGC_xy_npe_protons_uncut.Fill(event.P_ngcer_yAtCer,event.P_ngcer_xAtCer,event.P_ngcer_npeSum)
    P_RFTime_vs_gtr_dp_protons_uncut.Fill(event.P_RF_Dist, event.P_gtr_dp)

# Fill Accpt Cut Hitograms
for event in Cut_Pion_Events_Accpt_Data_tree:
    H_gtr_beta_protons_data_accpt_cut_all.Fill(event.H_gtr_beta)
    H_cal_etottracknorm_protons_data_accpt_cut_all.Fill(event.H_cal_etottracknorm)
    H_cer_npeSum_protons_data_accpt_cut_all.Fill(event.H_cer_npeSum)
    H_RFTime_Dist_protons_data_accpt_cut_all.Fill(event.H_RF_Dist)
    P_gtr_beta_protons_data_accpt_cut_all.Fill(event.P_gtr_beta)
    P_gtr_dp_protons_data_accpt_cut_all.Fill(event.P_gtr_dp)
    P_cal_etottracknorm_protons_data_accpt_cut_all.Fill(event.P_cal_etottracknorm)
    P_hgcer_npeSum_protons_data_accpt_cut_all.Fill(event.P_hgcer_npeSum)
    P_hgcer_xAtCer_protons_data_accpt_cut_all.Fill(event.P_hgcer_xAtCer)
    P_hgcer_yAtCer_protons_data_accpt_cut_all.Fill(event.P_hgcer_yAtCer)
    P_ngcer_npeSum_protons_data_accpt_cut_all.Fill(event.P_ngcer_npeSum)
    P_ngcer_xAtCer_protons_data_accpt_cut_all.Fill(event.P_ngcer_xAtCer)
    P_ngcer_yAtCer_protons_data_accpt_cut_all.Fill(event.P_ngcer_yAtCer)
    P_aero_npeSum_protons_data_accpt_cut_all.Fill(event.P_aero_npeSum)
    P_aero_xAtAero_protons_data_accpt_cut_all.Fill(event.P_aero_xAtAero)
    P_aero_yAtAero_protons_data_accpt_cut_all.Fill(event.P_aero_yAtAero)
    P_kin_MMp_protons_data_accpt_cut_all.Fill(event.MMp)
    P_RFTime_Dist_protons_data_accpt_cut_all.Fill(event.P_RF_Dist)
    CTime_epCoinTime_ROC1_protons_data_accpt_cut_all.Fill(event.CTime_epCoinTime_ROC1)
    H_cal_etottracknorm_vs_cer_npeSum_protons_accpt_cut_all.Fill(event.H_cal_etottracknorm, event.H_cer_npeSum)
    P_hgcer_vs_aero_npe_protons_accpt_cut_all.Fill(event.P_hgcer_npeSum, event.P_aero_npeSum)
    P_ngcer_vs_hgcer_npe_protons_accpt_cut_all.Fill(event.P_ngcer_npeSum, event.P_hgcer_npeSum)
    P_ngcer_vs_aero_npe_protons_accpt_cut_all.Fill(event.P_ngcer_npeSum, event.P_aero_npeSum)
    P_hgcer_yAtCer_vs_hgcer_xAtCer_protons_accpt_cut_all.Fill(event.P_hgcer_yAtCer, event.P_hgcer_xAtCer)
    P_aero_yAtAero_vs_aero_xAtAero_protons_accpt_cut_all.Fill(event.P_aero_yAtAero, event.P_aero_xAtAero)
    P_ngcer_yAtCer_vs_ngcer_xAtCer_protons_accpt_cut_all.Fill(event.P_ngcer_yAtCer, event.P_ngcer_xAtCer)
    P_cal_etottracknorm_vs_ngcer_npe_protons_accpt_cut_all.Fill(event.P_cal_etottracknorm, event.P_ngcer_npeSum)
    CTime_epCoinTime_vs_MMp_protons_accpt_cut_all.Fill(event.CTime_epCoinTime_ROC1, event.MMp)
    CTime_epCoinTime_vs_beta_protons_accpt_cut_all.Fill(event.CTime_epCoinTime_ROC1, event.P_gtr_beta)
    P_RFTime_vs_MMp_protons_accpt_cut_all.Fill(event.P_RF_Dist, event.MMp)
    CTime_epCoinTime_vs_RFTime_protons_accpt_cut_all.Fill(event.CTime_epCoinTime_ROC1, event.P_RF_Dist)
    P_HGC_xy_npe_protons_accpt_cut_all.Fill(event.P_hgcer_yAtCer,event.P_hgcer_xAtCer,event.P_hgcer_npeSum)
    P_Aero_xy_npe_protons_accpt_cut_all.Fill(event.P_aero_yAtAero,event.P_aero_xAtAero,event.P_aero_npeSum)
    P_NGC_xy_npe_protons_accpt_cut_all.Fill(event.P_ngcer_yAtCer,event.P_ngcer_xAtCer,event.P_ngcer_npeSum)
    P_RFTime_vs_gtr_dp_protons_accpt_cut_all.Fill(event.P_RF_Dist, event.P_gtr_dp)

# Fill Accpt + Prompt Cut Hitograms
for event in Cut_Pion_Events_Prompt_Data_tree:
    SHMS_PID_Cut = (event.P_hgcer_npeSum < P_hgcer_npeSum_cut_value) & (event.P_aero_npeSum < P_aero_npeSum_cut_value)
#    SHMS_PID_Cut = (event.P_aero_npeSum > P_aero_npeSum_cut_value)
#    SHMS_PID_Cut = (event.P_RF_Dist > P_RF_Dist_low_cut_value) & (event.P_RF_Dist < P_RF_Dist_high_cut_value) & (event.P_aero_npeSum > P_aero_npeSum_cut_value)# & (event.P_hgcer_npeSum > P_hgcer_npeSum_cut_value)
    if (SHMS_PID_Cut):
        H_gtr_beta_protons_data_prompt_cut_all.Fill(event.H_gtr_beta)
        H_cal_etottracknorm_protons_data_prompt_cut_all.Fill(event.H_cal_etottracknorm)
        H_cer_npeSum_protons_data_prompt_cut_all.Fill(event.H_cer_npeSum)
        H_RFTime_Dist_protons_data_prompt_cut_all.Fill(event.H_RF_Dist)
        P_gtr_beta_protons_data_prompt_cut_all.Fill(event.P_gtr_beta)
        P_gtr_dp_protons_data_prompt_cut_all.Fill(event.P_gtr_dp)
        P_cal_etottracknorm_protons_data_prompt_cut_all.Fill(event.P_cal_etottracknorm)
        P_hgcer_npeSum_protons_data_prompt_cut_all.Fill(event.P_hgcer_npeSum)
        P_hgcer_xAtCer_protons_data_prompt_cut_all.Fill(event.P_hgcer_xAtCer)
        P_hgcer_yAtCer_protons_data_prompt_cut_all.Fill(event.P_hgcer_yAtCer)
        P_ngcer_npeSum_protons_data_prompt_cut_all.Fill(event.P_ngcer_npeSum)
        P_ngcer_xAtCer_protons_data_prompt_cut_all.Fill(event.P_ngcer_xAtCer)
        P_ngcer_yAtCer_protons_data_prompt_cut_all.Fill(event.P_ngcer_yAtCer)
        P_aero_npeSum_protons_data_prompt_cut_all.Fill(event.P_aero_npeSum)
        P_aero_xAtAero_protons_data_prompt_cut_all.Fill(event.P_aero_xAtAero)
        P_aero_yAtAero_protons_data_prompt_cut_all.Fill(event.P_aero_yAtAero)
        P_kin_MMp_protons_data_prompt_cut_all.Fill(event.MMp)
        P_RFTime_Dist_protons_data_prompt_cut_all.Fill(event.P_RF_Dist)
        CTime_epCoinTime_ROC1_protons_data_prompt_cut_all.Fill(event.CTime_epCoinTime_ROC1)
        H_cal_etottracknorm_vs_cer_npeSum_protons_prompt_cut_all.Fill(event.H_cal_etottracknorm, event.H_cer_npeSum)
        P_hgcer_vs_aero_npe_protons_prompt_cut_all.Fill(event.P_hgcer_npeSum, event.P_aero_npeSum)
        P_ngcer_vs_hgcer_npe_protons_prompt_cut_all.Fill(event.P_ngcer_npeSum, event.P_hgcer_npeSum)
        P_ngcer_vs_aero_npe_protons_prompt_cut_all.Fill(event.P_ngcer_npeSum, event.P_aero_npeSum)
        P_hgcer_yAtCer_vs_hgcer_xAtCer_protons_prompt_cut_all.Fill(event.P_hgcer_yAtCer, event.P_hgcer_xAtCer)
        P_aero_yAtAero_vs_aero_xAtAero_protons_prompt_cut_all.Fill(event.P_aero_yAtAero, event.P_aero_xAtAero)
        P_ngcer_yAtCer_vs_ngcer_xAtCer_protons_prompt_cut_all.Fill(event.P_ngcer_yAtCer, event.P_ngcer_xAtCer)
        P_cal_etottracknorm_vs_ngcer_npe_protons_prompt_cut_all.Fill(event.P_cal_etottracknorm, event.P_ngcer_npeSum)
        CTime_epCoinTime_vs_MMp_protons_prompt_cut_all.Fill(event.CTime_epCoinTime_ROC1, event.MMp)
        CTime_epCoinTime_vs_beta_protons_prompt_cut_all.Fill(event.CTime_epCoinTime_ROC1, event.P_gtr_beta)
        P_RFTime_vs_MMp_protons_prompt_cut_all.Fill(event.P_RF_Dist, event.MMp)
        CTime_epCoinTime_vs_RFTime_protons_prompt_cut_all.Fill(event.CTime_epCoinTime_ROC1, event.P_RF_Dist)
        P_HGC_xy_npe_protons_prompt_cut_all.Fill(event.P_hgcer_yAtCer,event.P_hgcer_xAtCer,event.P_hgcer_npeSum)
        P_Aero_xy_npe_protons_prompt_cut_all.Fill(event.P_aero_yAtAero,event.P_aero_xAtAero,event.P_aero_npeSum)
        P_NGC_xy_npe_protons_prompt_cut_all.Fill(event.P_ngcer_yAtCer,event.P_ngcer_xAtCer,event.P_ngcer_npeSum)
        P_RFTime_vs_gtr_dp_protons_prompt_cut_all.Fill(event.P_RF_Dist, event.P_gtr_dp)


# Fill Accpt + Random Cut Hitograms
for event in Cut_Pion_Events_Random_Data_tree:
    SHMS_PID_Cut = (event.P_hgcer_npeSum < P_hgcer_npeSum_cut_value) & (event.P_aero_npeSum < P_aero_npeSum_cut_value)
#    SHMS_PID_Cut = (event.P_aero_npeSum > P_aero_npeSum_cut_value)
#    SHMS_PID_Cut = (event.P_RF_Dist > P_RF_Dist_low_cut_value) & (event.P_RF_Dist < P_RF_Dist_high_cut_value) & (event.P_aero_npeSum > P_aero_npeSum_cut_value)# & (event.P_hgcer_npeSum > P_hgcer_npeSum_cut_value)
    if (SHMS_PID_Cut):
        H_gtr_beta_protons_data_random_cut_all.Fill(event.H_gtr_beta)
        H_cal_etottracknorm_protons_data_random_cut_all.Fill(event.H_cal_etottracknorm)
        H_cer_npeSum_protons_data_random_cut_all.Fill(event.H_cer_npeSum)
        H_RFTime_Dist_protons_data_random_cut_all.Fill(event.H_RF_Dist)
        P_gtr_beta_protons_data_random_cut_all.Fill(event.P_gtr_beta)
        P_gtr_dp_protons_data_random_cut_all.Fill(event.P_gtr_dp)
        P_cal_etottracknorm_protons_data_random_cut_all.Fill(event.P_cal_etottracknorm)
        P_hgcer_npeSum_protons_data_random_cut_all.Fill(event.P_hgcer_npeSum)
        P_hgcer_xAtCer_protons_data_random_cut_all.Fill(event.P_hgcer_xAtCer)
        P_hgcer_yAtCer_protons_data_random_cut_all.Fill(event.P_hgcer_yAtCer)
        P_ngcer_npeSum_protons_data_random_cut_all.Fill(event.P_ngcer_npeSum)
        P_ngcer_xAtCer_protons_data_random_cut_all.Fill(event.P_ngcer_xAtCer)
        P_ngcer_yAtCer_protons_data_random_cut_all.Fill(event.P_ngcer_yAtCer)
        P_aero_npeSum_protons_data_random_cut_all.Fill(event.P_aero_npeSum)
        P_aero_xAtAero_protons_data_random_cut_all.Fill(event.P_aero_xAtAero)
        P_aero_yAtAero_protons_data_random_cut_all.Fill(event.P_aero_yAtAero)
        P_kin_MMp_protons_data_random_cut_all.Fill(event.MMp)
        P_RFTime_Dist_protons_data_random_cut_all.Fill(event.P_RF_Dist)
        CTime_epCoinTime_ROC1_protons_data_random_cut_all.Fill(event.CTime_epCoinTime_ROC1)
        CTime_epCoinTime_ROC1_protons_data_random_unsub_cut_all.Fill(event.CTime_epCoinTime_ROC1)
        H_cal_etottracknorm_vs_cer_npeSum_protons_random_cut_all.Fill(event.H_cal_etottracknorm, event.H_cer_npeSum)
        P_hgcer_vs_aero_npe_protons_random_cut_all.Fill(event.P_hgcer_npeSum, event.P_aero_npeSum)
        P_ngcer_vs_hgcer_npe_protons_random_cut_all.Fill(event.P_ngcer_npeSum, event.P_hgcer_npeSum)
        P_ngcer_vs_aero_npe_protons_random_cut_all.Fill(event.P_ngcer_npeSum, event.P_aero_npeSum)
        P_hgcer_yAtCer_vs_hgcer_xAtCer_protons_random_cut_all.Fill(event.P_hgcer_yAtCer, event.P_hgcer_xAtCer)
        P_aero_yAtAero_vs_aero_xAtAero_protons_random_cut_all.Fill(event.P_aero_yAtAero, event.P_aero_xAtAero)
        P_ngcer_yAtCer_vs_ngcer_xAtCer_protons_random_cut_all.Fill(event.P_ngcer_yAtCer, event.P_ngcer_xAtCer)
        P_cal_etottracknorm_vs_ngcer_npe_protons_random_cut_all.Fill(event.P_cal_etottracknorm, event.P_ngcer_npeSum)
        CTime_epCoinTime_vs_MMp_protons_random_cut_all.Fill(event.CTime_epCoinTime_ROC1, event.MMp)
        CTime_epCoinTime_vs_beta_protons_random_cut_all.Fill(event.CTime_epCoinTime_ROC1, event.P_gtr_beta)
        P_RFTime_vs_MMp_protons_random_cut_all.Fill(event.P_RF_Dist, event.MMp)
        CTime_epCoinTime_vs_RFTime_protons_random_cut_all.Fill(event.CTime_epCoinTime_ROC1, event.P_RF_Dist)
        P_HGC_xy_npe_protons_random_cut_all.Fill(event.P_hgcer_yAtCer,event.P_hgcer_xAtCer,event.P_hgcer_npeSum)
        P_Aero_xy_npe_protons_random_cut_all.Fill(event.P_aero_yAtAero,event.P_aero_xAtAero,event.P_aero_npeSum)
        P_NGC_xy_npe_protons_random_cut_all.Fill(event.P_ngcer_yAtCer,event.P_ngcer_xAtCer,event.P_ngcer_npeSum)
        P_RFTime_vs_gtr_dp_protons_random_cut_all.Fill(event.P_RF_Dist, event.P_gtr_dp)

print("Histograms filled")

#################################################################################################################################################

# Random subtraction from missing mass
#for event in Cut_Pion_Events_Random_tree:
#    P_kin_MMp_protons_cut_random_scaled.Fill(event.MMp)
#    P_kin_MMp_protons_cut_random_scaled.Scale(1.0/nWindows)
#P_kin_MMp_protons_cut_random_sub.Add(P_kin_MMp_protons_cut_prompt, P_kin_MMp_protons_cut_random_scaled, 1, -1)

H_gtr_beta_protons_data_random_cut_all.Scale(1.0/nWindows)
H_cal_etottracknorm_protons_data_random_cut_all.Scale(1.0/nWindows)
H_cer_npeSum_protons_data_random_cut_all.Scale(1.0/nWindows)
H_RFTime_Dist_protons_data_random_cut_all.Scale(1.0/nWindows)
P_gtr_beta_protons_data_random_cut_all.Scale(1.0/nWindows)
P_gtr_dp_protons_data_random_cut_all.Scale(1.0/nWindows)
P_cal_etottracknorm_protons_data_random_cut_all.Scale(1.0/nWindows)
P_hgcer_npeSum_protons_data_random_cut_all.Scale(1.0/nWindows)
P_hgcer_xAtCer_protons_data_random_cut_all.Scale(1.0/nWindows)
P_hgcer_yAtCer_protons_data_random_cut_all.Scale(1.0/nWindows)
P_ngcer_npeSum_protons_data_random_cut_all.Scale(1.0/nWindows)
P_ngcer_xAtCer_protons_data_random_cut_all.Scale(1.0/nWindows)
P_ngcer_yAtCer_protons_data_random_cut_all.Scale(1.0/nWindows)
P_aero_npeSum_protons_data_random_cut_all.Scale(1.0/nWindows)
P_aero_xAtAero_protons_data_random_cut_all.Scale(1.0/nWindows)
P_aero_yAtAero_protons_data_random_cut_all.Scale(1.0/nWindows)
P_RFTime_Dist_protons_data_random_cut_all.Scale(1.0/nWindows)
CTime_epCoinTime_ROC1_protons_data_random_cut_all.Scale(1.0/nWindows)
H_cal_etottracknorm_vs_cer_npeSum_protons_random_cut_all.Scale(1.0/nWindows)
P_hgcer_vs_aero_npe_protons_random_cut_all.Scale(1.0/nWindows)
P_ngcer_vs_hgcer_npe_protons_random_cut_all.Scale(1.0/nWindows)
P_ngcer_vs_aero_npe_protons_random_cut_all.Scale(1.0/nWindows)
P_hgcer_yAtCer_vs_hgcer_xAtCer_protons_random_cut_all.Scale(1.0/nWindows)
P_aero_yAtAero_vs_aero_xAtAero_protons_random_cut_all.Scale(1.0/nWindows)
P_ngcer_yAtCer_vs_ngcer_xAtCer_protons_random_cut_all.Scale(1.0/nWindows)
P_cal_etottracknorm_vs_ngcer_npe_protons_random_cut_all.Scale(1.0/nWindows)
CTime_epCoinTime_vs_MMp_protons_random_cut_all.Scale(1.0/nWindows)
CTime_epCoinTime_vs_beta_protons_random_cut_all.Scale(1.0/nWindows)
P_RFTime_vs_MMp_protons_random_cut_all.Scale(1.0/nWindows)
CTime_epCoinTime_vs_RFTime_protons_random_cut_all.Scale(1.0/nWindows)
P_HGC_xy_npe_protons_random_cut_all.Scale(1.0/nWindows)
P_Aero_xy_npe_protons_random_cut_all.Scale(1.0/nWindows)
P_NGC_xy_npe_protons_random_cut_all.Scale(1.0/nWindows)
P_RFTime_vs_gtr_dp_protons_random_cut_all.Scale(1.0/nWindows)
P_kin_MMp_protons_data_random_cut_all.Scale(1.0/nWindows)


H_gtr_beta_protons_data_cut_all.Add(H_gtr_beta_protons_data_prompt_cut_all, H_gtr_beta_protons_data_random_cut_all, 1, -1)
H_cal_etottracknorm_protons_data_cut_all.Add(H_cal_etottracknorm_protons_data_prompt_cut_all, H_cal_etottracknorm_protons_data_random_cut_all, 1, -1)
H_cer_npeSum_protons_data_cut_all.Add(H_cer_npeSum_protons_data_prompt_cut_all, H_cer_npeSum_protons_data_random_cut_all, 1, -1)
H_RFTime_Dist_protons_data_cut_all.Add(H_RFTime_Dist_protons_data_prompt_cut_all, H_RFTime_Dist_protons_data_random_cut_all, 1, -1)
P_gtr_beta_protons_data_cut_all.Add(P_gtr_beta_protons_data_prompt_cut_all, P_gtr_beta_protons_data_random_cut_all, 1, -1)
P_gtr_dp_protons_data_cut_all.Add(P_gtr_dp_protons_data_prompt_cut_all, P_gtr_dp_protons_data_random_cut_all, 1, -1)
P_cal_etottracknorm_protons_data_cut_all.Add(P_cal_etottracknorm_protons_data_prompt_cut_all, P_cal_etottracknorm_protons_data_random_cut_all, 1, -1)
P_hgcer_npeSum_protons_data_cut_all.Add(P_hgcer_npeSum_protons_data_prompt_cut_all, P_hgcer_npeSum_protons_data_random_cut_all, 1, -1)
P_hgcer_xAtCer_protons_data_cut_all.Add(P_hgcer_xAtCer_protons_data_prompt_cut_all, P_hgcer_xAtCer_protons_data_random_cut_all, 1, -1)
P_hgcer_yAtCer_protons_data_cut_all.Add(P_hgcer_yAtCer_protons_data_prompt_cut_all, P_hgcer_yAtCer_protons_data_random_cut_all, 1, -1)
P_ngcer_npeSum_protons_data_cut_all.Add(P_ngcer_npeSum_protons_data_prompt_cut_all, P_ngcer_npeSum_protons_data_random_cut_all, 1, -1)
P_ngcer_xAtCer_protons_data_cut_all.Add(P_ngcer_xAtCer_protons_data_prompt_cut_all, P_ngcer_xAtCer_protons_data_random_cut_all, 1, -1)
P_ngcer_yAtCer_protons_data_cut_all.Add(P_ngcer_yAtCer_protons_data_prompt_cut_all, P_ngcer_yAtCer_protons_data_random_cut_all, 1, -1)
P_aero_npeSum_protons_data_cut_all.Add(P_aero_npeSum_protons_data_prompt_cut_all, P_aero_npeSum_protons_data_random_cut_all, 1, -1)
P_aero_xAtAero_protons_data_cut_all.Add(P_aero_xAtAero_protons_data_prompt_cut_all, P_aero_xAtAero_protons_data_random_cut_all, 1, -1)
P_aero_yAtAero_protons_data_cut_all.Add(P_aero_yAtAero_protons_data_prompt_cut_all, P_aero_yAtAero_protons_data_random_cut_all, 1, -1)
P_RFTime_Dist_protons_data_cut_all.Add(P_RFTime_Dist_protons_data_prompt_cut_all, P_RFTime_Dist_protons_data_random_cut_all, 1, -1)
CTime_epCoinTime_ROC1_protons_data_cut_all.Add(CTime_epCoinTime_ROC1_protons_data_prompt_cut_all, CTime_epCoinTime_ROC1_protons_data_random_cut_all, 1, -1)
H_cal_etottracknorm_vs_cer_npeSum_protons_cut_all.Add(H_cal_etottracknorm_vs_cer_npeSum_protons_prompt_cut_all, H_cal_etottracknorm_vs_cer_npeSum_protons_random_cut_all, 1, -1)
P_hgcer_vs_aero_npe_protons_cut_all.Add(P_hgcer_vs_aero_npe_protons_prompt_cut_all, P_hgcer_vs_aero_npe_protons_random_cut_all, 1, -1)
P_ngcer_vs_hgcer_npe_protons_cut_all.Add(P_ngcer_vs_hgcer_npe_protons_prompt_cut_all, P_ngcer_vs_hgcer_npe_protons_random_cut_all, 1, -1)
P_ngcer_vs_aero_npe_protons_cut_all.Add(P_ngcer_vs_aero_npe_protons_prompt_cut_all, P_ngcer_vs_aero_npe_protons_random_cut_all, 1, -1)
P_hgcer_yAtCer_vs_hgcer_xAtCer_protons_cut_all.Add(P_hgcer_yAtCer_vs_hgcer_xAtCer_protons_prompt_cut_all, P_hgcer_yAtCer_vs_hgcer_xAtCer_protons_random_cut_all, 1, -1)
P_aero_yAtAero_vs_aero_xAtAero_protons_cut_all.Add(P_aero_yAtAero_vs_aero_xAtAero_protons_prompt_cut_all, P_aero_yAtAero_vs_aero_xAtAero_protons_random_cut_all, 1, -1)
P_ngcer_yAtCer_vs_ngcer_xAtCer_protons_cut_all.Add(P_ngcer_yAtCer_vs_ngcer_xAtCer_protons_prompt_cut_all, P_ngcer_yAtCer_vs_ngcer_xAtCer_protons_random_cut_all, 1, -1)
P_cal_etottracknorm_vs_ngcer_npe_protons_cut_all.Add(P_cal_etottracknorm_vs_ngcer_npe_protons_prompt_cut_all, P_cal_etottracknorm_vs_ngcer_npe_protons_random_cut_all, 1, -1)
CTime_epCoinTime_vs_MMp_protons_cut_all.Add(CTime_epCoinTime_vs_MMp_protons_prompt_cut_all, CTime_epCoinTime_vs_MMp_protons_random_cut_all, 1, -1)
CTime_epCoinTime_vs_beta_protons_cut_all.Add(CTime_epCoinTime_vs_beta_protons_prompt_cut_all, CTime_epCoinTime_vs_beta_protons_random_cut_all, 1, -1)
P_RFTime_vs_MMp_protons_cut_all.Add(P_RFTime_vs_MMp_protons_prompt_cut_all, P_RFTime_vs_MMp_protons_random_cut_all, 1, -1)
CTime_epCoinTime_vs_RFTime_protons_cut_all.Add(CTime_epCoinTime_vs_RFTime_protons_prompt_cut_all, CTime_epCoinTime_vs_RFTime_protons_random_cut_all, 1, -1)
P_HGC_xy_npe_protons_cut_all.Add(P_HGC_xy_npe_protons_prompt_cut_all, P_HGC_xy_npe_protons_random_cut_all, 1, -1)
P_Aero_xy_npe_protons_cut_all.Add(P_Aero_xy_npe_protons_prompt_cut_all, P_Aero_xy_npe_protons_random_cut_all, 1, -1)
P_NGC_xy_npe_protons_cut_all.Add(P_NGC_xy_npe_protons_prompt_cut_all, P_NGC_xy_npe_protons_random_cut_all, 1, -1)
P_RFTime_vs_gtr_dp_protons_cut_all.Add(P_RFTime_vs_gtr_dp_protons_prompt_cut_all, P_RFTime_vs_gtr_dp_protons_random_cut_all, 1, -1)
P_kin_MMp_protons_data_cut_all.Add(P_kin_MMp_protons_data_prompt_cut_all, P_kin_MMp_protons_data_random_cut_all, 1, -1)

'''
H_gtr_beta_protons_data_cut_all = H_gtr_beta_protons_data_prompt_cut_all.Clone()
H_cal_etottracknorm_protons_data_cut_all = H_cal_etottracknorm_protons_data_prompt_cut_all.Clone()
H_cer_npeSum_protons_data_cut_all = H_cer_npeSum_protons_data_prompt_cut_all.Clone()
H_RFTime_Dist_protons_data_cut_all = H_RFTime_Dist_protons_data_prompt_cut_all.Clone()
P_gtr_beta_protons_data_cut_all = P_gtr_beta_protons_data_prompt_cut_all.Clone()
P_cal_etottracknorm_protons_data_cut_all = P_cal_etottracknorm_protons_data_prompt_cut_all.Clone()
P_hgcer_npeSum_protons_data_cut_all = P_hgcer_npeSum_protons_data_prompt_cut_all.Clone()
P_hgcer_xAtCer_protons_data_cut_all = P_hgcer_xAtCer_protons_data_prompt_cut_all.Clone()
P_hgcer_yAtCer_protons_data_cut_all = P_hgcer_yAtCer_protons_data_prompt_cut_all.Clone()
P_ngcer_npeSum_protons_data_cut_all = P_ngcer_npeSum_protons_data_prompt_cut_all.Clone()
P_ngcer_xAtCer_protons_data_cut_all = P_ngcer_xAtCer_protons_data_prompt_cut_all.Clone()
P_ngcer_yAtCer_protons_data_cut_all = P_ngcer_yAtCer_protons_data_prompt_cut_all.Clone()
P_aero_npeSum_protons_data_cut_all = P_aero_npeSum_protons_data_prompt_cut_all.Clone()
P_aero_xAtAero_protons_data_cut_all = P_aero_xAtAero_protons_data_prompt_cut_all.Clone()
P_aero_yAtAero_protons_data_cut_all = P_aero_yAtAero_protons_data_prompt_cut_all.Clone()
P_RFTime_Dist_protons_data_cut_all = P_RFTime_Dist_protons_data_prompt_cut_all.Clone()
CTime_epCoinTime_ROC1_protons_data_cut_all = CTime_epCoinTime_ROC1_protons_data_prompt_cut_all.Clone()
H_cal_etottracknorm_vs_cer_npeSum_protons_cut_all = H_cal_etottracknorm_vs_cer_npeSum_protons_prompt_cut_all.Clone()
P_hgcer_vs_aero_npe_protons_cut_all = P_hgcer_vs_aero_npe_protons_prompt_cut_all.Clone()
P_ngcer_vs_hgcer_npe_protons_cut_all = P_ngcer_vs_hgcer_npe_protons_prompt_cut_all.Clone()
P_ngcer_vs_aero_npe_protons_cut_all = P_ngcer_vs_aero_npe_protons_prompt_cut_all.Clone()
P_hgcer_yAtCer_vs_hgcer_xAtCer_protons_cut_all = P_hgcer_yAtCer_vs_hgcer_xAtCer_protons_prompt_cut_all.Clone()
P_aero_yAtAero_vs_aero_xAtAero_protons_cut_all = P_aero_yAtAero_vs_aero_xAtAero_protons_prompt_cut_all.Clone()
P_ngcer_yAtCer_vs_ngcer_xAtCer_protons_cut_all = P_ngcer_yAtCer_vs_ngcer_xAtCer_protons_prompt_cut_all.Clone()
P_cal_etottracknorm_vs_ngcer_npe_protons_cut_all = P_cal_etottracknorm_vs_ngcer_npe_protons_prompt_cut_all.Clone()
CTime_epCoinTime_vs_MMp_protons_cut_all = CTime_epCoinTime_vs_MMp_protons_prompt_cut_all.Clone()
CTime_epCoinTime_vs_beta_protons_cut_all = CTime_epCoinTime_vs_beta_protons_prompt_cut_all.Clone()
P_RFTime_vs_MMp_protons_cut_all = P_RFTime_vs_MMp_protons_prompt_cut_all.Clone()
CTime_epCoinTime_vs_RFTime_protons_cut_all = CTime_epCoinTime_vs_RFTime_protons_prompt_cut_all.Clone()
P_HGC_xy_npe_protons_cut_all = P_HGC_xy_npe_protons_prompt_cut_all.Clone()
P_Aero_xy_npe_protons_cut_all = P_Aero_xy_npe_protons_prompt_cut_all.Clone()
P_NGC_xy_npe_protons_cut_all = P_NGC_xy_npe_protons_prompt_cut_all.Clone()
'''
############################################################################################################################################

# HGC/NGC/Aero XY Projection vs npe for protons.
HGC_proj_yx_protons_uncut = ROOT.TProfile2D(P_HGC_xy_npe_protons_uncut.Project3DProfile("yx"))
NGC_proj_yx_protons_uncut = ROOT.TProfile2D(P_NGC_xy_npe_protons_uncut.Project3DProfile("yx"))
Aero_proj_yx_protons_uncut = ROOT.TProfile2D(P_Aero_xy_npe_protons_uncut.Project3DProfile("yx"))
HGC_proj_yx_protons_accpt_cut_all = ROOT.TProfile2D(P_HGC_xy_npe_protons_accpt_cut_all.Project3DProfile("yx"))
NGC_proj_yx_protons_accpt_cut_all = ROOT.TProfile2D(P_NGC_xy_npe_protons_accpt_cut_all.Project3DProfile("yx"))
Aero_proj_yx_protons_accpt_cut_all = ROOT.TProfile2D(P_Aero_xy_npe_protons_accpt_cut_all.Project3DProfile("yx"))
HGC_proj_yx_protons_prompt_cut_all = ROOT.TProfile2D(P_HGC_xy_npe_protons_prompt_cut_all.Project3DProfile("yx"))
NGC_proj_yx_protons_prompt_cut_all = ROOT.TProfile2D(P_NGC_xy_npe_protons_prompt_cut_all.Project3DProfile("yx"))
Aero_proj_yx_protons_prompt_cut_all = ROOT.TProfile2D(P_Aero_xy_npe_protons_prompt_cut_all.Project3DProfile("yx"))
HGC_proj_yx_protons_random_cut_all = ROOT.TProfile2D(P_HGC_xy_npe_protons_random_cut_all.Project3DProfile("yx"))
NGC_proj_yx_protons_random_cut_all = ROOT.TProfile2D(P_NGC_xy_npe_protons_random_cut_all.Project3DProfile("yx"))
Aero_proj_yx_protons_random_cut_all = ROOT.TProfile2D(P_Aero_xy_npe_protons_random_cut_all.Project3DProfile("yx"))
HGC_proj_yx_protons_cut_all = ROOT.TProfile2D(P_HGC_xy_npe_protons_cut_all.Project3DProfile("yx"))
NGC_proj_yx_protons_cut_all = ROOT.TProfile2D(P_NGC_xy_npe_protons_cut_all.Project3DProfile("yx"))
Aero_proj_yx_protons_cut_all = ROOT.TProfile2D(P_Aero_xy_npe_protons_cut_all.Project3DProfile("yx"))

############################################################################################################################################

# Removes stat box
ROOT.gStyle.SetOptStat(0)

# Saving histograms in PDF
c1_pid1 = TCanvas("c1_pid1", "MM and Detector Distributions", 100, 0, 1400, 1000)
c1_pid1.Divide(2,2)
c1_pid1.cd(1)
# Format the text string with the "p" format
#text_str = 'Beam Energy = {}, Q^2 = {}, W = {}, SHMS_theta = {}'.format(BEAM_ENERGY, Q2, W, ptheta)
text_str = '{}, {}, {}, {}'.format(BEAM_ENERGY, Q2, W, ptheta)
c1_pid1_text_lines = [
    ROOT.TText(0.5, 0.9, "Pion Physics Production Setting"),
    ROOT.TText(0.5, 0.8, text_str),
    ROOT.TText(0.5, 0.6, "PID Cuts"),
    ROOT.TText(0.5, 0.5, "H_cer_npeSum > 1.5"),
    ROOT.TText(0.5, 0.4, "H_cal_etotnorm > 0.7"),
    ROOT.TText(0.5, 0.3, 'P_aero_npeSum > {}'.format(P_aero_npeSum_cut_value)),
#    ROOT.TText(0.5, 0.2, '{} < P_RF_Dist < {}'.format(P_RF_Dist_low_cut_value, P_RF_Dist_high_cut_value)),
    ROOT.TText(0.5, 0.1, 'P_hgcer_npeSum > {}'.format(P_hgcer_npeSum_cut_value)),

]
for c1_pid1_text in c1_pid1_text_lines:
    c1_pid1_text.SetTextSize(0.07)
    c1_pid1_text.SetTextAlign(22)
#    c1_pid1_text.SetTextColor(ROOT.kGreen + 4)
#    if c1_pid1_text.GetTitle() == "Red = SIMC":
#       c1_pid1_text.SetTextColor(ROOT.kRed)  # Setting text color to red
#    if c1_pid1_text.GetTitle() == "Blue = DATA":
#       c1_pid1_text.SetTextColor(ROOT.kBlue)  # Setting text color to red
    c1_pid1_text.Draw()
c1_pid1.cd(2)
P_kin_MMp_protons_data_uncut.SetLineColor(1)
P_kin_MMp_protons_data_uncut.Draw("hist")
P_kin_MMp_protons_data_accpt_cut_all.SetLineColor(2)
P_kin_MMp_protons_data_accpt_cut_all.Draw("hist same")
P_kin_MMp_protons_data_cut_all.SetLineColor(4)
P_kin_MMp_protons_data_cut_all.Draw("hist same")
# Section for Neutron Peak Events Selection
shadedpeak_protons = P_kin_MMp_protons_data_cut_all.Clone()
shadedpeak_protons.SetFillColor(2)
shadedpeak_protons.SetFillStyle(3244)
shadedpeak_protons.GetXaxis().SetRangeUser(minbin, maxbin)
shadedpeak_protons.Draw("samehist")
NeutronEvt_protons = TPaveText(0.58934,0.675,0.95,0.75,"NDC")
BinLow_protons = P_kin_MMp_protons_data_cut_all.GetXaxis().FindBin(minbin)
BinHigh_protons = P_kin_MMp_protons_data_cut_all.GetXaxis().FindBin(maxbin)
BinIntegral_protons = int(P_kin_MMp_protons_data_cut_all.Integral(BinLow_protons, BinHigh_protons))
NeutronEvt_protons.SetLineColor(2)
NeutronEvt_protons.AddText("e #pi n Events: %i" %(BinIntegral_protons))
NeutronEvt_protons.Draw()
# End of Neutron Peak Events Selection Section
legend1_protons = ROOT.TLegend(0.115, 0.835, 0.43, 0.9)
legend1_protons.AddEntry("P_kin_MMp_protons_data_uncut", "without cuts", "l")
legend1_protons.AddEntry("P_kin_MMp_protons_data_accpt_cut_all", "with cuts (acpt)", "l")
legend1_protons.AddEntry("P_kin_MMp_protons_data_cut_all", "with cuts (acpt/CT/PID)", "l")
legend1_protons.Draw("same")
c1_pid1.cd(3)
P_RFTime_Dist_protons_data_uncut.SetLineColor(1)
P_RFTime_Dist_protons_data_uncut.Draw()
c1_pid1.cd(4)
#gPad.SetLogy()
P_RFTime_Dist_protons_data_accpt_cut_all.SetLineColor(2)
P_RFTime_Dist_protons_data_accpt_cut_all.Draw()
P_RFTime_Dist_protons_data_cut_all.SetLineColor(4)
P_RFTime_Dist_protons_data_cut_all.Draw("same")
legend2_protons = ROOT.TLegend(0.115, 0.835, 0.43, 0.9)
legend2_protons.AddEntry("P_RFTime_Dist_protons_data_uncut", "with cuts (acpt)", "l")
legend2_protons.AddEntry("P_RFTime_Dist_protons_data_cut_all", "with cuts (acpt/CT/PID)", "l")
legend2_protons.Draw("same")
c1_pid1.Print(Pion_Analysis_Distributions + '(')

c1_pid2 = TCanvas("c1_pid2", "2D Detector Distributions", 100, 0, 1400, 1400)
c1_pid2.Divide(2,3)
c1_pid2.cd(1)
gPad.SetLogy()
P_ngcer_npeSum_protons_data_uncut.SetLineColor(1)
P_ngcer_npeSum_protons_data_uncut.Draw()
c1_pid2.cd(2)
gPad.SetLogy()
P_ngcer_npeSum_protons_data_accpt_cut_all.SetLineColor(2)
P_ngcer_npeSum_protons_data_accpt_cut_all.Draw()
P_ngcer_npeSum_protons_data_cut_all.SetLineColor(4)
P_ngcer_npeSum_protons_data_cut_all.Draw("same")
legend3_protons = ROOT.TLegend(0.115, 0.835, 0.43, 0.9)
legend3_protons.AddEntry("P_ngcer_npeSum_protons_data_accpt_cut_all", "with cuts (acpt)", "l")
legend3_protons.AddEntry("P_ngcer_npeSum_protons_data_cut_all", "with cuts (acpt/CT/PID)", "l")
legend3_protons.Draw("same")
c1_pid2.cd(3)
gPad.SetLogy()
P_hgcer_npeSum_protons_data_uncut.SetLineColor(1)
P_hgcer_npeSum_protons_data_uncut.Draw()
c1_pid2.cd(4)
gPad.SetLogy()
P_hgcer_npeSum_protons_data_accpt_cut_all.SetLineColor(2)
P_hgcer_npeSum_protons_data_accpt_cut_all.Draw()
P_hgcer_npeSum_protons_data_cut_all.SetLineColor(4)
P_hgcer_npeSum_protons_data_cut_all.Draw("same")
legend4_protons = ROOT.TLegend(0.115, 0.835, 0.43, 0.9)
legend4_protons.AddEntry("P_hgcer_npeSum_protons_data_accpt_cut_all", "with cuts (acpt)", "l")
legend4_protons.AddEntry("P_hgcer_npeSum_protons_data_cut_all", "with cuts (acpt/CT/PID)", "l")
legend4_protons.Draw("same")
c1_pid2.cd(5)
gPad.SetLogy()
P_aero_npeSum_protons_data_uncut.SetLineColor(1)
P_aero_npeSum_protons_data_uncut.Draw()
c1_pid2.cd(6)
gPad.SetLogy()
P_aero_npeSum_protons_data_accpt_cut_all.SetLineColor(2)
P_aero_npeSum_protons_data_accpt_cut_all.Draw()
P_aero_npeSum_protons_data_cut_all.SetLineColor(4)
P_aero_npeSum_protons_data_cut_all.Draw("same")
legend5_protons = ROOT.TLegend(0.115, 0.835, 0.43, 0.9)
legend5_protons.AddEntry("P_aero_npeSum_protons_data_accpt_cut_all", "with cuts (acpt)", "l")
legend5_protons.AddEntry("P_aero_npeSum_protons_data_cut_all", "with cuts (acpt/CT/PID)", "l")
legend5_protons.Draw("same")
c1_pid2.Print(Pion_Analysis_Distributions)

c1_pid3 = TCanvas("c1_pid3", "2D Detector Distributions", 100, 0, 1400, 1400)
c1_pid3.Divide(2,3)
c1_pid3.cd(1)
gPad.SetLogz()
P_hgcer_vs_aero_npe_protons_uncut.Draw("COLZ")
c1_pid3.cd(2)
gPad.SetLogz()
P_hgcer_vs_aero_npe_protons_cut_all.Draw("COLZ")
c1_pid3.cd(3)
gPad.SetLogz()
P_ngcer_vs_hgcer_npe_protons_uncut.Draw("COLZ")
c1_pid3.cd(4)
gPad.SetLogz()
P_ngcer_vs_hgcer_npe_protons_cut_all.Draw("COLZ")
c1_pid3.cd(5)
gPad.SetLogz()
P_ngcer_vs_aero_npe_protons_uncut.Draw("COLZ")
c1_pid3.cd(6)
gPad.SetLogz()
P_ngcer_vs_aero_npe_protons_cut_all.Draw("COLZ")
c1_pid3.Print(Pion_Analysis_Distributions)

c1_pid4 = TCanvas("c1_pid4", "CT Distributions", 100, 0, 1400, 1400)
c1_pid4.Divide(2,3)
c1_pid4.cd(1)
CTime_epCoinTime_ROC1_protons_data_uncut.SetLineColor(1)
CTime_epCoinTime_ROC1_protons_data_uncut.Draw()
c1_pid4.cd(2)
CTime_epCoinTime_ROC1_protons_data_accpt_cut_all.SetLineColor(4)
CTime_epCoinTime_ROC1_protons_data_accpt_cut_all.Draw()
CTime_epCoinTime_ROC1_protons_data_prompt_cut_all.SetLineColor(6)
CTime_epCoinTime_ROC1_protons_data_prompt_cut_all.Draw("same")
CTime_epCoinTime_ROC1_protons_data_random_unsub_cut_all.SetLineColor(8)
CTime_epCoinTime_ROC1_protons_data_random_unsub_cut_all.Draw("same")
legend6_protons = ROOT.TLegend(0.1, 0.815, 0.48, 0.9)
legend6_protons.AddEntry("ePiCoinTime_protons_uncut", "CT_without cuts (acpt only)", "l")
legend6_protons.AddEntry("ePiCoinTime_protons_cut_prompt", "CT_prompt with cuts (acpt/PID)", "l")
legend6_protons.AddEntry("ePiCoinTime_protons_cut_randm", "CT_randoms with cuts (acpt/PID)", "l")
legend6_protons.Draw("same")
c1_pid4.cd(3)
gPad.SetLogz()
CTime_epCoinTime_vs_MMp_protons_uncut.Draw("COLZ")
LowerPrompt1_protons = TLine(PromptWindow[0],gPad.GetUymin(),PromptWindow[0],2)
LowerPrompt1_protons.SetLineColor(2)
LowerPrompt1_protons.SetLineWidth(2)
LowerPrompt1_protons.Draw("same")
UpperPrompt1_protons = TLine(PromptWindow[1],gPad.GetUymin(),PromptWindow[1],2)
UpperPrompt1_protons.SetLineColor(2)
UpperPrompt1_protons.SetLineWidth(2)
UpperPrompt1_protons.Draw("same")
LowerRandomL1_protons = TLine(RandomWindows[0],gPad.GetUymin(),RandomWindows[0],2)
LowerRandomL1_protons.SetLineColor(8)
LowerRandomL1_protons.SetLineWidth(2)
LowerRandomL1_protons.Draw("same")
UpperRandomL1_protons = TLine(RandomWindows[1],gPad.GetUymin(),RandomWindows[1],2)
UpperRandomL1_protons.SetLineColor(8)
UpperRandomL1_protons.SetLineWidth(2)
UpperRandomL1_protons.Draw("same")
LowerRandomR1_protons = TLine(RandomWindows[2],gPad.GetUymin(),RandomWindows[2],2)
LowerRandomR1_protons.SetLineColor(8)
LowerRandomR1_protons.SetLineWidth(2)
LowerRandomR1_protons.Draw("same")
UpperRandomR1_protons = TLine(RandomWindows[3],gPad.GetUymin(),RandomWindows[3],2)
UpperRandomR1_protons.SetLineColor(8)
UpperRandomR1_protons.SetLineWidth(2)
UpperRandomR1_protons.Draw("same")
c1_pid4.cd(4)
gPad.SetLogz()
CTime_epCoinTime_vs_MMp_protons_cut_all.GetYaxis().SetRangeUser(0.4, 1.4)
CTime_epCoinTime_vs_MMp_protons_cut_all.GetXaxis().SetRangeUser(-5, 5)
CTime_epCoinTime_vs_MMp_protons_cut_all.Draw("COLZ")
c1_pid4.cd(5)
P_RFTime_Dist_protons_data_accpt_cut_all.SetLineColor(2)
P_RFTime_Dist_protons_data_accpt_cut_all.Draw()
P_RFTime_Dist_protons_data_cut_all.SetLineColor(4)
P_RFTime_Dist_protons_data_cut_all.Draw("same")
legend7_protons = ROOT.TLegend(0.115, 0.835, 0.43, 0.9)
legend7_protons.AddEntry("P_RFTime_Dist_protons_data_accpt_cut_all", "with cuts (acpt)", "l")
legend7_protons.AddEntry("P_RFTime_Dist_protons_data_cut_all", "with cuts (acpt/CT/PID)", "l")
legend7_protons.Draw("same")
c1_pid4.cd(6)
gPad.SetLogz()
#CTime_epCoinTime_vs_RFTime_protons_cut_all.GetXaxis().SetRangeUser(-5, 5)
#CTime_epCoinTime_vs_RFTime_protons_cut_all.Draw("COLZ")
P_RFTime_vs_gtr_dp_protons_cut_all.GetYaxis().SetRangeUser(-15, 25)
P_RFTime_vs_gtr_dp_protons_cut_all.Draw("COLZ")
c1_pid4.Print(Pion_Analysis_Distributions)

c1_pid5 = TCanvas("c1_pid5", "RF Distributions", 100, 0, 1400, 1400)
c1_pid5.Divide(2,3)
c1_pid5.cd(1)
gPad.SetLogz()
P_RFTime_vs_MMp_protons_uncut.Draw("COLZ")
c1_pid5.cd(2)
gPad.SetLogz()
P_RFTime_vs_MMp_protons_cut_all.GetYaxis().SetRangeUser(0.4, 1.4)
P_RFTime_vs_MMp_protons_cut_all.Draw("COLZ")
c1_pid5.cd(3)
P_cal_etottracknorm_protons_data_uncut.SetLineColor(2)
P_cal_etottracknorm_protons_data_uncut.Draw()
P_cal_etottracknorm_protons_data_cut_all.SetLineColor(4)
P_cal_etottracknorm_protons_data_cut_all.Draw("same")
legend8_protons = ROOT.TLegend(0.1, 0.815, 0.48, 0.9)
legend8_protons.AddEntry("P_cal_etottracknorm_protons_data_uncut", "without cuts", "l")
legend8_protons.AddEntry("P_cal_etottracknorm_protons_data_cut_all", "with cuts (acpt/CT/PID)", "l")
legend8_protons.Draw("same")
c1_pid5.cd(4)
gPad.SetLogy()
H_cal_etottracknorm_protons_data_uncut.SetLineColor(2)
H_cal_etottracknorm_protons_data_uncut.Draw()
H_cal_etottracknorm_protons_data_cut_all.SetLineColor(4)
H_cal_etottracknorm_protons_data_cut_all.Draw("same")
legend9_protons = ROOT.TLegend(0.1, 0.815, 0.48, 0.9)
legend9_protons.AddEntry("H_cal_etottracknorm_protons_data_uncut", "without cuts", "l")
legend9_protons.AddEntry("H_cal_etottracknorm_protons_data_cut_all", "with cuts (acpt/CT/PID)", "l")
legend9_protons.Draw("same")
c1_pid5.cd(5)
gPad.SetLogy()
H_cer_npeSum_protons_data_uncut.SetLineColor(2)
H_cer_npeSum_protons_data_uncut.Draw()
H_cer_npeSum_protons_data_cut_all.SetLineColor(4)
H_cer_npeSum_protons_data_cut_all.Draw("same")
legend11_protons = ROOT.TLegend(0.1, 0.815, 0.48, 0.9)
legend11_protons.AddEntry("H_cer_npeSum_protons_data_uncut", "without cuts", "l")
legend11_protons.AddEntry("H_cer_npeSum_protons_data_cut_all", "with cuts (acpt/CT/PID)", "l")
legend11_protons.Draw("same")
c1_pid5.cd(6)
gPad.SetLogz()
H_cal_etottracknorm_vs_cer_npeSum_protons_random_cut_all.Draw("COLZ")
c1_pid5.Print(Pion_Analysis_Distributions)

c1_pid6 = TCanvas("c1_pid6", "Detector XY Distributions", 100, 0, 1400, 1400)
c1_pid6.Divide(2,3)
c1_pid6.cd(1)
gPad.SetLogz()
HGC_proj_yx_protons_uncut.Draw("COLZ")
c1_pid6.cd(2)
gPad.SetLogz()
HGC_proj_yx_protons_cut_all.Draw("COLZ")
c1_pid6.cd(3)
gPad.SetLogz()
NGC_proj_yx_protons_uncut.Draw("COLZ")
c1_pid6.cd(4)
gPad.SetLogz()
NGC_proj_yx_protons_cut_all.Draw("COLZ")
c1_pid6.cd(5)
gPad.SetLogz()
Aero_proj_yx_protons_uncut.Draw("COLZ")
c1_pid6.cd(6)
gPad.SetLogz()
Aero_proj_yx_protons_cut_all.Draw("COLZ")
c1_pid6.Print(Pion_Analysis_Distributions + ')')

#############################################################################################################################################

# Making directories in output file
outHistFile = ROOT.TFile.Open("%s/%s_%s_%s_%s_%s_ProdCoin_PID_Output_Data.root" % (OUTPATH, BEAM_ENERGY, Q2, W, ptheta, MaxEvent) , "RECREATE")
d_Uncut_Pion_Events_Data = outHistFile.mkdir("Uncut_Pion_Events_Data")
d_Cut_Pion_Events_Accpt_Data = outHistFile.mkdir("Cut_Pion_Events_Accpt_Data")
d_Cut_Pion_Events_Prompt_Data = outHistFile.mkdir("Cut_Pion_Events_Prompt_Data")
d_Cut_Pion_Events_Random_Data = outHistFile.mkdir("Cut_Pion_Events_Random_Data")
d_Cut_Pion_Events_All_Data = outHistFile.mkdir("Cut_Pion_Events_All_Data")

# Writing Histograms for protons
d_Uncut_Pion_Events_Data.cd()
H_gtr_beta_protons_data_uncut.Write()
H_cal_etottracknorm_protons_data_uncut.Write()
H_cer_npeSum_protons_data_uncut.Write()
H_RFTime_Dist_protons_data_uncut.Write()
P_gtr_beta_protons_data_uncut.Write()
P_gtr_dp_protons_data_uncut.Write()
P_cal_etottracknorm_protons_data_uncut.Write()
P_hgcer_npeSum_protons_data_uncut.Write()
P_hgcer_xAtCer_protons_data_uncut.Write()
P_hgcer_yAtCer_protons_data_uncut.Write()
P_ngcer_npeSum_protons_data_uncut.Write()
P_ngcer_xAtCer_protons_data_uncut.Write()
P_ngcer_yAtCer_protons_data_uncut.Write()
P_aero_npeSum_protons_data_uncut.Write()
P_aero_xAtAero_protons_data_uncut.Write()
P_aero_yAtAero_protons_data_uncut.Write()
P_kin_MMp_protons_data_uncut.Write()
P_RFTime_Dist_protons_data_uncut.Write()
CTime_epCoinTime_ROC1_protons_data_uncut.Write()
H_cal_etottracknorm_vs_cer_npeSum_protons_uncut.Write()
P_hgcer_vs_aero_npe_protons_uncut.Write()
P_ngcer_vs_hgcer_npe_protons_uncut.Write()
P_ngcer_vs_aero_npe_protons_uncut.Write()
P_hgcer_yAtCer_vs_hgcer_xAtCer_protons_uncut.Write()
P_aero_yAtAero_vs_aero_xAtAero_protons_uncut.Write()
P_ngcer_yAtCer_vs_ngcer_xAtCer_protons_uncut.Write()
P_cal_etottracknorm_vs_ngcer_npe_protons_uncut.Write()
CTime_epCoinTime_vs_MMp_protons_uncut.Write()
CTime_epCoinTime_vs_beta_protons_uncut.Write()
P_RFTime_vs_MMp_protons_uncut.Write()
CTime_epCoinTime_vs_RFTime_protons_uncut.Write()
P_HGC_xy_npe_protons_uncut.Write()
P_Aero_xy_npe_protons_uncut.Write()
P_NGC_xy_npe_protons_uncut.Write()
P_RFTime_vs_gtr_dp_protons_uncut.Write()

d_Cut_Pion_Events_Accpt_Data.cd()
H_gtr_beta_protons_data_accpt_cut_all.Write()
H_cal_etottracknorm_protons_data_accpt_cut_all.Write()
H_cer_npeSum_protons_data_accpt_cut_all.Write()
H_RFTime_Dist_protons_data_accpt_cut_all.Write()
P_gtr_beta_protons_data_accpt_cut_all.Write()
P_gtr_dp_protons_data_accpt_cut_all.Write()
P_cal_etottracknorm_protons_data_accpt_cut_all.Write()
P_hgcer_npeSum_protons_data_accpt_cut_all.Write()
P_hgcer_xAtCer_protons_data_accpt_cut_all.Write()
P_hgcer_yAtCer_protons_data_accpt_cut_all.Write()
P_ngcer_npeSum_protons_data_accpt_cut_all.Write()
P_ngcer_xAtCer_protons_data_accpt_cut_all.Write()
P_ngcer_yAtCer_protons_data_accpt_cut_all.Write()
P_aero_npeSum_protons_data_accpt_cut_all.Write()
P_aero_xAtAero_protons_data_accpt_cut_all.Write()
P_aero_yAtAero_protons_data_accpt_cut_all.Write()
P_kin_MMp_protons_data_accpt_cut_all.Write()
P_RFTime_Dist_protons_data_accpt_cut_all.Write()
CTime_epCoinTime_ROC1_protons_data_accpt_cut_all.Write()
H_cal_etottracknorm_vs_cer_npeSum_protons_accpt_cut_all.Write()
P_hgcer_vs_aero_npe_protons_accpt_cut_all.Write()
P_ngcer_vs_hgcer_npe_protons_accpt_cut_all.Write()
P_ngcer_vs_aero_npe_protons_accpt_cut_all.Write()
P_hgcer_yAtCer_vs_hgcer_xAtCer_protons_accpt_cut_all.Write()
P_aero_yAtAero_vs_aero_xAtAero_protons_accpt_cut_all.Write()
P_ngcer_yAtCer_vs_ngcer_xAtCer_protons_accpt_cut_all.Write()
P_cal_etottracknorm_vs_ngcer_npe_protons_accpt_cut_all.Write()
CTime_epCoinTime_vs_MMp_protons_accpt_cut_all.Write()
CTime_epCoinTime_vs_beta_protons_accpt_cut_all.Write()
P_RFTime_vs_MMp_protons_accpt_cut_all.Write()
CTime_epCoinTime_vs_RFTime_protons_accpt_cut_all.Write()
P_HGC_xy_npe_protons_accpt_cut_all.Write()
P_Aero_xy_npe_protons_accpt_cut_all.Write()
P_NGC_xy_npe_protons_accpt_cut_all.Write()
P_RFTime_vs_gtr_dp_protons_accpt_cut_all.Write()

d_Cut_Pion_Events_Prompt_Data.cd()
H_gtr_beta_protons_data_prompt_cut_all.Write()
H_cal_etottracknorm_protons_data_prompt_cut_all.Write()
H_cer_npeSum_protons_data_prompt_cut_all.Write()
H_RFTime_Dist_protons_data_prompt_cut_all.Write()
P_gtr_beta_protons_data_prompt_cut_all.Write()
P_gtr_dp_protons_data_prompt_cut_all.Write()
P_cal_etottracknorm_protons_data_prompt_cut_all.Write()
P_hgcer_npeSum_protons_data_prompt_cut_all.Write()
P_hgcer_xAtCer_protons_data_prompt_cut_all.Write()
P_hgcer_yAtCer_protons_data_prompt_cut_all.Write()
P_ngcer_npeSum_protons_data_prompt_cut_all.Write()
P_ngcer_xAtCer_protons_data_prompt_cut_all.Write()
P_ngcer_yAtCer_protons_data_prompt_cut_all.Write()
P_aero_npeSum_protons_data_prompt_cut_all.Write()
P_aero_xAtAero_protons_data_prompt_cut_all.Write()
P_aero_yAtAero_protons_data_prompt_cut_all.Write()
P_kin_MMp_protons_data_prompt_cut_all.Write()
P_RFTime_Dist_protons_data_prompt_cut_all.Write()
CTime_epCoinTime_ROC1_protons_data_prompt_cut_all.Write()
H_cal_etottracknorm_vs_cer_npeSum_protons_prompt_cut_all.Write()
P_hgcer_vs_aero_npe_protons_prompt_cut_all.Write()
P_ngcer_vs_hgcer_npe_protons_prompt_cut_all.Write()
P_ngcer_vs_aero_npe_protons_prompt_cut_all.Write()
P_hgcer_yAtCer_vs_hgcer_xAtCer_protons_prompt_cut_all.Write()
P_aero_yAtAero_vs_aero_xAtAero_protons_prompt_cut_all.Write()
P_ngcer_yAtCer_vs_ngcer_xAtCer_protons_prompt_cut_all.Write()
P_cal_etottracknorm_vs_ngcer_npe_protons_prompt_cut_all.Write()
CTime_epCoinTime_vs_MMp_protons_prompt_cut_all.Write()
CTime_epCoinTime_vs_beta_protons_prompt_cut_all.Write()
P_RFTime_vs_MMp_protons_prompt_cut_all.Write()
CTime_epCoinTime_vs_RFTime_protons_prompt_cut_all.Write()
P_HGC_xy_npe_protons_prompt_cut_all.Write()
P_Aero_xy_npe_protons_prompt_cut_all.Write()
P_NGC_xy_npe_protons_prompt_cut_all.Write()
P_RFTime_vs_gtr_dp_protons_prompt_cut_all.Write()

d_Cut_Pion_Events_Random_Data.cd()
H_gtr_beta_protons_data_random_cut_all.Write()
H_cal_etottracknorm_protons_data_random_cut_all.Write()
H_cer_npeSum_protons_data_random_cut_all.Write()
H_RFTime_Dist_protons_data_random_cut_all.Write()
P_gtr_beta_protons_data_random_cut_all.Write()
P_gtr_dp_protons_data_random_cut_all.Write()
P_cal_etottracknorm_protons_data_random_cut_all.Write()
P_hgcer_npeSum_protons_data_random_cut_all.Write()
P_hgcer_xAtCer_protons_data_random_cut_all.Write()
P_hgcer_yAtCer_protons_data_random_cut_all.Write()
P_ngcer_npeSum_protons_data_random_cut_all.Write()
P_ngcer_xAtCer_protons_data_random_cut_all.Write()
P_ngcer_yAtCer_protons_data_random_cut_all.Write()
P_aero_npeSum_protons_data_random_cut_all.Write()
P_aero_xAtAero_protons_data_random_cut_all.Write()
P_aero_yAtAero_protons_data_random_cut_all.Write()
P_kin_MMp_protons_data_random_cut_all.Write()
P_RFTime_Dist_protons_data_random_cut_all.Write()
CTime_epCoinTime_ROC1_protons_data_random_cut_all.Write()
H_cal_etottracknorm_vs_cer_npeSum_protons_random_cut_all.Write()
P_hgcer_vs_aero_npe_protons_random_cut_all.Write()
P_ngcer_vs_hgcer_npe_protons_random_cut_all.Write()
P_ngcer_vs_aero_npe_protons_random_cut_all.Write()
P_hgcer_yAtCer_vs_hgcer_xAtCer_protons_random_cut_all.Write()
P_aero_yAtAero_vs_aero_xAtAero_protons_random_cut_all.Write()
P_ngcer_yAtCer_vs_ngcer_xAtCer_protons_random_cut_all.Write()
P_cal_etottracknorm_vs_ngcer_npe_protons_random_cut_all.Write()
CTime_epCoinTime_vs_MMp_protons_random_cut_all.Write()
CTime_epCoinTime_vs_beta_protons_random_cut_all.Write()
P_RFTime_vs_MMp_protons_random_cut_all.Write()
CTime_epCoinTime_vs_RFTime_protons_random_cut_all.Write()
P_HGC_xy_npe_protons_random_cut_all.Write()
P_Aero_xy_npe_protons_random_cut_all.Write()
P_NGC_xy_npe_protons_random_cut_all.Write()
P_RFTime_vs_gtr_dp_protons_random_cut_all.Write()

d_Cut_Pion_Events_All_Data.cd()
H_gtr_beta_protons_data_cut_all.Write()
H_cal_etottracknorm_protons_data_cut_all.Write()
H_cer_npeSum_protons_data_cut_all.Write()
H_RFTime_Dist_protons_data_cut_all.Write()
P_gtr_beta_protons_data_cut_all.Write()
P_gtr_dp_protons_data_cut_all.Write()
P_cal_etottracknorm_protons_data_cut_all.Write()
P_hgcer_npeSum_protons_data_cut_all.Write()
P_hgcer_xAtCer_protons_data_cut_all.Write()
P_hgcer_yAtCer_protons_data_cut_all.Write()
P_ngcer_npeSum_protons_data_cut_all.Write()
P_ngcer_xAtCer_protons_data_cut_all.Write()
P_ngcer_yAtCer_protons_data_cut_all.Write()
P_aero_npeSum_protons_data_cut_all.Write()
P_aero_xAtAero_protons_data_cut_all.Write()
P_aero_yAtAero_protons_data_cut_all.Write()
P_kin_MMp_protons_data_cut_all.Write()
P_RFTime_Dist_protons_data_cut_all.Write()
CTime_epCoinTime_ROC1_protons_data_cut_all.Write()
H_cal_etottracknorm_vs_cer_npeSum_protons_cut_all.Write()
P_hgcer_vs_aero_npe_protons_cut_all.Write()
P_ngcer_vs_hgcer_npe_protons_cut_all.Write()
P_ngcer_vs_aero_npe_protons_cut_all.Write()
P_hgcer_yAtCer_vs_hgcer_xAtCer_protons_cut_all.Write()
P_aero_yAtAero_vs_aero_xAtAero_protons_cut_all.Write()
P_ngcer_yAtCer_vs_ngcer_xAtCer_protons_cut_all.Write()
P_cal_etottracknorm_vs_ngcer_npe_protons_cut_all.Write()
CTime_epCoinTime_vs_MMp_protons_cut_all.Write()
CTime_epCoinTime_vs_beta_protons_cut_all.Write()
P_RFTime_vs_MMp_protons_cut_all.Write()
CTime_epCoinTime_vs_RFTime_protons_cut_all.Write()
P_HGC_xy_npe_protons_cut_all.Write()
P_Aero_xy_npe_protons_cut_all.Write()
P_NGC_xy_npe_protons_cut_all.Write()
P_RFTime_vs_gtr_dp_protons_cut_all.Write()

##################################################################################################################################################################################3

infile_DATA.Close() 
outHistFile.Close()

print ("Processing Complete")
