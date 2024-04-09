#! /usr/bin/pythoa
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
import uproot as up
import numpy as np
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
from ROOT import TCanvas, TPaveLabel, TColor, TGaxis, TH1F, TH2F, TPad, TStyle, gStyle, gPad, TLegend, TGaxis, TLine, TMath, TLatex, TPaveText, TArc, TGraphPolar, TText
from ROOT import kBlack, kCyan, kRed, kGreen, kMagenta, kBlue
from functools import reduce

##################################################################################################################################################

# Defining some constants here
#minbin = 0.0 # minbin for selecting neutrons events in missing mass distribution
#maxbin = 0.05 # maxbin for selecting neutrons events in missing mass distribution

##################################################################################################################################################

# Check the number of arguments provided to the script
if len(sys.argv)-1!=1:
    print("!!!!! ERROR !!!!!\n Expected 5 arguments\n Usage is with - ROOTfileSuffixs Beam Energy MaxEvents RunList CVSFile\n!!!!! ERROR !!!!!")
    sys.exit(1)

##################################################################################################################################################

# Input params - run number and max number of events
RUNNUMBER = sys.argv[1]
#MaxEvent = sys.argv[2]
#DATA_Suffix = sys.argv[3]
MaxEvent = "-1"
DATA_Suffix = "Analysed_Data"
################################################################################################################################################
'''
ltsep package import and pathing definitions
'''

# Import package for cuts
from ltsep import Root

lt=Root(os.path.realpath(__file__), "Plot_HeePCoin")

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
Proton_Analysis_Distributions = "%s/test/%s_%s_HeePCoin_Proton_Analysis_Distributions.pdf" % (OUTPATH, RUNNUMBER, MaxEvent)

# Input file location and variables taking
rootFile_DATA = "%s/%s_%s_%s.root" % (OUTPATH, RUNNUMBER, MaxEvent, DATA_Suffix)
#rootFile_DATA = "%s/run_output_files/%s_%s_%s.root" % (OUTPATH, RUNNUMBER, MaxEvent, DATA_Suffix)
###############################################################################################################################################

# Read stuff from the main event tree
infile_DATA = ROOT.TFile.Open(rootFile_DATA, "READ")

Uncut_Proton_Events_Data_tree = infile_DATA.Get("Uncut_Proton_Events")
Cut_Proton_Events_All_Data_tree = infile_DATA.Get("Cut_Proton_Events_All")
Cut_Proton_Events_Prompt_Data_tree = infile_DATA.Get("Cut_Proton_Events_Prompt")

###################################################################################################################################################

# Defining Histograms for Protons
# Uncut Data Histograms
H_gtr_beta_protons_data_uncut = ROOT.TH1D("H_gtr_beta_protons_data_uncut", "HMS #beta; HMS_gtr_#beta; Counts", 200, 0.8, 1.2)
H_gtr_xp_protons_data_uncut = ROOT.TH1D("H_gtr_xp_protons_data_uncut", "HMS xptar; HMS_gtr_xptar; Counts", 200, -0.2, 0.2)
H_gtr_yp_protons_data_uncut = ROOT.TH1D("H_gtr_yp_protons_data_uncut", "HMS yptar; HMS_gtr_yptar; Counts", 200, -0.2, 0.2)
H_gtr_dp_protons_data_uncut = ROOT.TH1D("H_gtr_dp_protons_data_uncut", "HMS #delta; HMS_gtr_dp; Counts", 200, -15, 15)
H_gtr_p_protons_data_uncut = ROOT.TH1D("H_gtr_p_protons_data_uncut", "HMS p; HMS_gtr_p; Counts", 200, 4, 8)
H_dc_x_fp_protons_data_uncut = ROOT.TH1D("H_dc_x_fp_protons_data_uncut", "HMS x_fp'; HMS_dc_x_fp; Counts", 200, -100, 100)
H_dc_y_fp_protons_data_uncut = ROOT.TH1D("H_dc_y_fp_protons_data_uncut", "HMS y_fp'; HMS_dc_y_fp; Counts", 200, -100, 100)
H_dc_xp_fp_protons_data_uncut = ROOT.TH1D("H_dc_xp_fp_protons_data_uncut", "HMS xp_fp'; HMS_dc_xp_fp; Counts", 200, -0.2, 0.2)
H_dc_yp_fp_protons_data_uncut = ROOT.TH1D("H_dc_yp_fp_protons_data_uncut", "HMS yp_fp'; HMS_dc_yp_fp; Counts", 200, -0.2, 0.2)
H_hod_goodscinhit_protons_data_uncut = ROOT.TH1D("H_hod_goodscinhit_protons_data_uncut", "HMS hod goodscinhit; HMS_hod_goodscinhi; Counts", 200, 0.7, 1.3)
H_hod_goodstarttime_protons_data_uncut = ROOT.TH1D("H_hod_goodstarttime_protons_data_uncut", "HMS hod goodstarttime; HMS_hod_goodstarttime; Counts", 200, 0.7, 1.3)
H_cal_etotnorm_protons_data_uncut = ROOT.TH1D("H_cal_etotnorm_protons_data_uncut", "HMS cal etotnorm; HMS_cal_etotnorm; Counts", 200, 0.2, 1.8)
H_cal_etottracknorm_protons_data_uncut = ROOT.TH1D("H_cal_etottracknorm_protons_data_uncut", "HMS cal etottracknorm; HMS_cal_etottracknorm; Counts", 200, 0.2, 1.8)
H_cer_npeSum_protons_data_uncut = ROOT.TH1D("H_cer_npeSum_protons_data_uncut", "HMS cer npeSum; HMS_cer_npeSum; Counts", 200, 0, 50)
H_RFTime_Dist_protons_data_uncut = ROOT.TH1D("H_RFTime_Dist_protons_data_uncut", "HMS RFTime; HMS_RFTime; Counts", 200, 0, 4)
P_gtr_beta_protons_data_uncut = ROOT.TH1D("P_gtr_beta_protons_data_uncut", "SHMS #beta; SHMS_gtr_#beta; Counts", 200, 0.5, 1.3)
P_gtr_xp_protons_data_uncut = ROOT.TH1D("P_gtr_xp_protons_data_uncut", "SHMS xptar; SHMS_gtr_xptar; Counts", 200, -0.2, 0.2)
P_gtr_yp_protons_data_uncut = ROOT.TH1D("P_gtr_yp_protons_data_uncut", "SHMS yptar; SHMS_gtr_yptar; Counts", 200, -0.2, 0.2)
P_gtr_dp_protons_data_uncut = ROOT.TH1D("P_gtr_dp_protons_data_uncut", "SHMS delta; SHMS_gtr_dp; Counts", 200, -30, 30)
P_gtr_p_protons_data_uncut = ROOT.TH1D("P_gtr_p_protons_data_uncut", "SHMS p; SHMS_gtr_p; Counts", 200, 1, 7)
P_dc_x_fp_protons_data_uncut = ROOT.TH1D("P_dc_x_fp_protons_data_uncut", "SHMS x_fp'; SHMS_dc_x_fp; Counts", 200, -100, 100)
P_dc_y_fp_protons_data_uncut = ROOT.TH1D("P_dc_y_fp_protons_data_uncut", "SHMS y_fp'; SHMS_dc_y_fp; Counts", 200, -100, 100)
P_dc_xp_fp_protons_data_uncut = ROOT.TH1D("P_dc_xp_fp_protons_data_uncut", "SHMS xp_fp'; SHMS_dc_xp_fp; Counts", 200, -0.2, 0.2)
P_dc_yp_fp_protons_data_uncut = ROOT.TH1D("P_dc_yp_fp_protons_data_uncut", "SHMS yp_fp'; SHMS_dc_yp_fp; Counts", 200, -0.2, 0.2)
P_hod_goodscinhit_protons_data_uncut = ROOT.TH1D("P_hod_goodscinhit_protons_data_uncut", "SHMS hod goodscinhit; SHMS_hod_goodscinhit; Counts", 200, 0.7, 1.3)
P_hod_goodstarttime_protons_data_uncut = ROOT.TH1D("P_hod_goodstarttime_protons_data_uncut", "SHMS hod goodstarttime; SHMS_hod_goodstarttime; Counts", 200, 0.7, 1.3)
P_cal_etotnorm_protons_data_uncut = ROOT.TH1D("P_cal_etotnorm_protons_data_uncut", "SHMS cal etotnorm; SHMS_cal_etotnorm; Counts", 200, 0, 1)
P_cal_etottracknorm_protons_data_uncut = ROOT.TH1D("P_cal_etottracknorm_protons_data_uncut", "SHMS cal etottracknorm; SHMS_cal_etottracknorm; Counts", 200, 0, 1.6)
P_hgcer_npeSum_protons_data_uncut = ROOT.TH1D("P_hgcer_npeSum_protons_data_uncut", "SHMS HGC npeSum; SHMS_hgcer_npeSum; Counts", 200, 0, 50)
P_hgcer_xAtCer_protons_data_uncut = ROOT.TH1D("P_hgcer_xAtCer_protons_data_uncut", "SHMS HGC xAtCer; SHMS_hgcer_xAtCer; Counts", 200, -60, 60)
P_hgcer_yAtCer_protons_data_uncut = ROOT.TH1D("P_hgcer_yAtCer_protons_data_uncut", "SHMS HGC yAtCer; SHMS_hgcer_yAtCer; Counts", 200, -50, 50)
P_ngcer_npeSum_protons_data_uncut = ROOT.TH1D("P_ngcer_npeSum_protons_data_uncut", "SHMS NGC npeSum; SHMS_ngcer_npeSum; Counts", 200, 0, 50)
P_ngcer_xAtCer_protons_data_uncut = ROOT.TH1D("P_ngcer_xAtCer_protons_data_uncut", "SHMS NGC xAtCer; SHMS_ngcer_xAtCer; Counts", 200, -60, 60)
P_ngcer_yAtCer_protons_data_uncut = ROOT.TH1D("P_ngcer_yAtCer_protons_data_uncut", "SHMS NGC yAtCer; SHMS_ngcer_yAtCer; Counts", 200, -50, 50)
P_aero_npeSum_protons_data_uncut = ROOT.TH1D("P_aero_npeSum_protons_data_uncut", "SHMS aero npeSum; SHMS_aero_npeSum; Counts", 200, 0, 50)
P_aero_xAtAero_protons_data_uncut = ROOT.TH1D("P_acero_xAtAero_protons_data_uncut", "SHMS aero xAtAero; SHMS_aero_xAtAero; Counts", 200, -60, 60)
P_aero_yAtAero_protons_data_uncut = ROOT.TH1D("P_aero_yAtAero_protons_data_uncut", "SHMS aero yAtAero; SHMS_aero_yAtAero; Counts", 200, -50, 50)
P_kin_MMp_protons_data_uncut = ROOT.TH1D("P_kin_MMp_protons_data_uncut", "MIssing Mass data uncut; MM_{p}; Counts", 200, -1., 1.)
P_RFTime_Dist_protons_data_uncut = ROOT.TH1D("P_RFTime_Dist_protons_data_uncut", "SHMS RFTime; SHMS_RFTime; Counts", 200, 0, 4)
CTime_epCoinTime_ROC1_protons_data_uncut = ROOT.TH1D("CTime_epCoinTime_ROC1_protons_data_uncut", "Electron-Proton CTime; e p Coin_Time; Counts", 200, -50, 50)
P_kin_secondary_pmiss_protons_data_uncut = ROOT.TH1D("P_kin_secondary_pmiss_protons_data_uncut", "Momentum Distribution; pmiss; Counts", 200, -0.8, 0.8)
P_kin_secondary_pmiss_x_protons_data_uncut = ROOT.TH1D("P_kin_secondary_pmiss_x_protons_data_uncut", "Momentum_x Distribution; pmiss_x; Counts", 200, -0.6, 0.6)
P_kin_secondary_pmiss_y_protons_data_uncut = ROOT.TH1D("P_kin_secondary_pmiss_y_protons_data_uncut", "Momentum_y Distribution; pmiss_y; Counts", 200, -0.6, 0.6)
P_kin_secondary_pmiss_z_protons_data_uncut = ROOT.TH1D("P_kin_secondary_pmiss_z_protons_data_uncut", "Momentum_z Distribution; pmiss_z; Counts", 200, -0.6, 0.6)
P_kin_secondary_Erecoil_protons_data_uncut = ROOT.TH1D("P_kin_secondary_Erecoil_protons_data_uncut", "Erecoil Distribution; Erecoil; Counts", 200, -0.8, 0.8)
P_kin_secondary_emiss_protons_data_uncut = ROOT.TH1D("P_kin_secondary_emiss_protons_data_uncut", "Emiss Distribution; emiss; Counts", 200, -0.8, 0.8)
P_kin_secondary_Mrecoil_protons_data_uncut = ROOT.TH1D("P_kin_secondary_Mrecoil_protons_data_uncut", "Mrecoil Distribution; Mrecoil; Counts", 200, -0.8, 0.8)
P_kin_secondary_W_protons_data_uncut = ROOT.TH1D("P_kin_secondary_W_protons_data_uncut", "W Distribution; W; Counts", 200, 0, 2)

# Cut (Acceptance + PID) Data Histograms
H_gtr_beta_protons_data_cut_all = ROOT.TH1D("H_gtr_beta_protons_data_cut_all", "HMS #beta; HMS_gtr_#beta; Counts", 200, 0.8, 1.2)
H_gtr_xp_protons_data_cut_all = ROOT.TH1D("H_gtr_xp_protons_data_cut_all", "HMS xptar; HMS_gtr_xptar; Counts", 200, -0.2, 0.2)
H_gtr_yp_protons_data_cut_all = ROOT.TH1D("H_gtr_yp_protons_data_cut_all", "HMS yptar; HMS_gtr_yptar; Counts", 200, -0.2, 0.2)
H_gtr_dp_protons_data_cut_all = ROOT.TH1D("H_gtr_dp_protons_data_cut_all", "HMS #delta; HMS_gtr_dp; Counts", 200, -15, 15)
H_gtr_p_protons_data_cut_all = ROOT.TH1D("H_gtr_p_protons_data_cut_all", "HMS p; HMS_gtr_p; Counts", 200, 4, 8)
H_dc_x_fp_protons_data_cut_all = ROOT.TH1D("H_dc_x_fp_protons_data_cut_all", "HMS x_fp'; HMS_dc_x_fp; Counts", 200, -100, 100)
H_dc_y_fp_protons_data_cut_all = ROOT.TH1D("H_dc_y_fp_protons_data_cut_all", "HMS y_fp'; HMS_dc_y_fp; Counts", 200, -100, 100)
H_dc_xp_fp_protons_data_cut_all = ROOT.TH1D("H_dc_xp_fp_protons_data_cut_all", "HMS xp_fp'; HMS_dc_xp_fp; Counts", 200, -0.2, 0.2)
H_dc_yp_fp_protons_data_cut_all = ROOT.TH1D("H_dc_yp_fp_protons_data_cut_all", "HMS yp_fp'; HMS_dc_yp_fp; Counts", 200, -0.2, 0.2)
H_hod_goodscinhit_protons_data_cut_all = ROOT.TH1D("H_hod_goodscinhit_protons_data_cut_all", "HMS hod goodscinhit; HMS_hod_goodscinhi; Counts", 200, 0.7, 1.3)
H_hod_goodstarttime_protons_data_cut_all = ROOT.TH1D("H_hod_goodstarttime_protons_data_cut_all", "HMS hod goodstarttime; HMS_hod_goodstarttime; Counts", 200, 0.7, 1.3)
H_cal_etotnorm_protons_data_cut_all = ROOT.TH1D("H_cal_etotnorm_protons_data_cut_all", "HMS cal etotnorm; HMS_cal_etotnorm; Counts", 200, 0.2, 1.8)
H_cal_etottracknorm_protons_data_cut_all = ROOT.TH1D("H_cal_etottracknorm_protons_data_cut_all", "HMS cal etottracknorm; HMS_cal_etottracknorm; Counts", 200, 0.2, 1.8)
H_cer_npeSum_protons_data_cut_all = ROOT.TH1D("H_cer_npeSum_protons_data_cut_all", "HMS cer npeSum; HMS_cer_npeSum; Counts", 200, 0, 50)
H_RFTime_Dist_protons_data_cut_all = ROOT.TH1D("H_RFTime_Dist_protons_data_cut_all", "HMS RFTime; HMS_RFTime; Counts", 200, 0, 4)
P_gtr_beta_protons_data_cut_all = ROOT.TH1D("P_gtr_beta_protons_data_cut_all", "SHMS #beta; SHMS_gtr_#beta; Counts", 200, 0.5, 1.3)
P_gtr_xp_protons_data_cut_all = ROOT.TH1D("P_gtr_xp_protons_data_cut_all", "SHMS xptar; SHMS_gtr_xptar; Counts", 200, -0.2, 0.2)
P_gtr_yp_protons_data_cut_all = ROOT.TH1D("P_gtr_yp_protons_data_cut_all", "SHMS yptar; SHMS_gtr_yptar; Counts", 200, -0.2, 0.2)
P_gtr_dp_protons_data_cut_all = ROOT.TH1D("P_gtr_dp_protons_data_cut_all", "SHMS delta; SHMS_gtr_dp; Counts", 200, -30, 30)
P_gtr_p_protons_data_cut_all = ROOT.TH1D("P_gtr_p_protons_data_cut_all", "SHMS p; SHMS_gtr_p; Counts", 200, 1, 7)
P_dc_x_fp_protons_data_cut_all = ROOT.TH1D("P_dc_x_fp_protons_data_cut_all", "SHMS x_fp'; SHMS_dc_x_fp; Counts", 200, -100, 100)
P_dc_y_fp_protons_data_cut_all = ROOT.TH1D("P_dc_y_fp_protons_data_cut_all", "SHMS y_fp'; SHMS_dc_y_fp; Counts", 200, -100, 100)
P_dc_xp_fp_protons_data_cut_all = ROOT.TH1D("P_dc_xp_fp_protons_data_cut_all", "SHMS xp_fp'; SHMS_dc_xp_fp; Counts", 200, -0.2, 0.2)
P_dc_yp_fp_protons_data_cut_all = ROOT.TH1D("P_dc_yp_fp_protons_data_cut_all", "SHMS yp_fp'; SHMS_dc_yp_fp; Counts", 200, -0.2, 0.2)
P_hod_goodscinhit_protons_data_cut_all = ROOT.TH1D("P_hod_goodscinhit_protons_data_cut_all", "SHMS hod goodscinhit; SHMS_hod_goodscinhit; Counts", 200, 0.7, 1.3)
P_hod_goodstarttime_protons_data_cut_all = ROOT.TH1D("P_hod_goodstarttime_protons_data_cut_all", "SHMS hod goodstarttime; SHMS_hod_goodstarttime; Counts", 200, 0.7, 1.3)
P_cal_etotnorm_protons_data_cut_all = ROOT.TH1D("P_cal_etotnorm_protons_data_cut_all", "SHMS cal etotnorm; SHMS_cal_etotnorm; Counts", 200, 0, 1)
P_cal_etottracknorm_protons_data_cut_all = ROOT.TH1D("P_cal_etottracknorm_protons_data_cut_all", "SHMS cal etottracknorm; SHMS_cal_etottracknorm; Counts", 200, 0, 1.6)
P_hgcer_npeSum_protons_data_cut_all = ROOT.TH1D("P_hgcer_npeSum_protons_data_cut_all", "SHMS HGC npeSum; SHMS_hgcer_npeSum; Counts", 200, 0, 50)
P_hgcer_xAtCer_protons_data_cut_all = ROOT.TH1D("P_hgcer_xAtCer_protons_data_cut_all", "SHMS HGC xAtCer; SHMS_hgcer_xAtCer; Counts", 200, -60, 60)
P_hgcer_yAtCer_protons_data_cut_all = ROOT.TH1D("P_hgcer_yAtCer_protons_data_cut_all", "SHMS HGC yAtCer; SHMS_hgcer_yAtCer; Counts", 200, -50, 50)
P_ngcer_npeSum_protons_data_cut_all = ROOT.TH1D("P_ngcer_npeSum_protons_data_cut_all", "SHMS NGC npeSum; SHMS_ngcer_npeSum; Counts", 200, 0, 50)
P_ngcer_xAtCer_protons_data_cut_all = ROOT.TH1D("P_ngcer_xAtCer_protons_data_cut_all", "SHMS NGC xAtCer; SHMS_ngcer_xAtCer; Counts", 200, -60, 60)
P_ngcer_yAtCer_protons_data_cut_all = ROOT.TH1D("P_ngcer_yAtCer_protons_data_cut_all", "SHMS NGC yAtCer; SHMS_ngcer_yAtCer; Counts", 200, -50, 50)
P_aero_npeSum_protons_data_cut_all = ROOT.TH1D("P_aero_npeSum_protons_data_cut_all", "SHMS aero npeSum; SHMS_aero_npeSum; Counts", 200, 0, 50)
P_aero_xAtAero_protons_data_cut_all = ROOT.TH1D("P_acero_xAtAero_protons_data_cut_all", "SHMS aero xAtAero; SHMS_aero_xAtAero; Counts", 200, -60, 60)
P_aero_yAtAero_protons_data_cut_all = ROOT.TH1D("P_aero_yAtAero_protons_data_cut_all", "SHMS aero yAtAero; SHMS_aero_yAtAero; Counts", 200, -50, 50)
P_kin_MMp_protons_data_cut_all = ROOT.TH1D("P_kin_MMp_protons_data_cut_all", "MIssing Mass data (cut_all); MM_{p}; Counts", 200, -1., 1.)
P_RFTime_Dist_protons_data_cut_all = ROOT.TH1D("P_RFTime_Dist_protons_data_cut_all", "SHMS RFTime; SHMS_RFTime; Counts", 200, 0, 4)
CTime_epCoinTime_ROC1_protons_data_cut_all = ROOT.TH1D("CTime_epCoinTime_ROC1_protons_data_cut_all", "Electron-Proton CTime; e p Coin_Time; Counts", 200, -50, 50)
P_kin_secondary_pmiss_protons_data_cut_all = ROOT.TH1D("P_kin_secondary_pmiss_protons_data_cut_all", "Momentum Distribution; pmiss; Counts", 200, -0.8, 0.8)
P_kin_secondary_pmiss_x_protons_data_cut_all = ROOT.TH1D("P_kin_secondary_pmiss_x_protons_data_cut_all", "Momentum_x Distribution; pmiss_x; Counts", 200, -0.6, 0.6)
P_kin_secondary_pmiss_y_protons_data_cut_all = ROOT.TH1D("P_kin_secondary_pmiss_y_protons_data_cut_all", "Momentum_y Distribution; pmiss_y; Counts", 200, -0.6, 0.6)
P_kin_secondary_pmiss_z_protons_data_cut_all = ROOT.TH1D("P_kin_secondary_pmiss_z_protons_data_cut_all", "Momentum_z Distribution; pmiss_z; Counts", 200, -0.6, 0.6)
P_kin_secondary_Erecoil_protons_data_cut_all = ROOT.TH1D("P_kin_secondary_Erecoil_protons_data_cut_all", "Erecoil Distribution; Erecoil; Counts", 200, -0.8, 0.8)
P_kin_secondary_emiss_protons_data_cut_all = ROOT.TH1D("P_kin_secondary_emiss_protons_data_cut_all", "Energy Distribution; emiss; Counts", 200, -0.8, 0.8)
P_kin_secondary_Mrecoil_protons_data_cut_all = ROOT.TH1D("P_kin_secondary_Mrecoil_protons_data_cut_all", "Mrecoil Distribution; Mrecoil; Counts", 200, -0.8, 0.8)
P_kin_secondary_W_protons_data_cut_all = ROOT.TH1D("P_kin_secondary_W_protons_data_cut_all", "W Distribution; W; Counts", 200, 0, 2)

# Cut (Acceptance + PID + Prompt Selection) Data Histograms
H_gtr_beta_protons_data_prompt_cut_all = ROOT.TH1D("H_gtr_beta_protons_data_prompt_cut_all", "HMS #beta; HMS_gtr_#beta; Counts", 200, 0.8, 1.2)
H_gtr_xp_protons_data_prompt_cut_all = ROOT.TH1D("H_gtr_xp_protons_data_prompt_cut_all", "HMS xptar; HMS_gtr_xptar; Counts", 200, -0.2, 0.2)
H_gtr_yp_protons_data_prompt_cut_all = ROOT.TH1D("H_gtr_yp_protons_data_prompt_cut_all", "HMS yptar; HMS_gtr_yptar; Counts", 200, -0.2, 0.2)
H_gtr_dp_protons_data_prompt_cut_all = ROOT.TH1D("H_gtr_dp_protons_data_prompt_cut_all", "HMS #delta; HMS_gtr_dp; Counts", 200, -15, 15)
H_gtr_p_protons_data_prompt_cut_all = ROOT.TH1D("H_gtr_p_protons_data_prompt_cut_all", "HMS p; HMS_gtr_p; Counts", 200, 4, 8)
H_dc_x_fp_protons_data_prompt_cut_all = ROOT.TH1D("H_dc_x_fp_protons_data_prompt_cut_all", "HMS x_fp'; HMS_dc_x_fp; Counts", 200, -100, 100)
H_dc_y_fp_protons_data_prompt_cut_all = ROOT.TH1D("H_dc_y_fp_protons_data_prompt_cut_all", "HMS y_fp'; HMS_dc_y_fp; Counts", 200, -100, 100)
H_dc_xp_fp_protons_data_prompt_cut_all = ROOT.TH1D("H_dc_xp_fp_protons_data_prompt_cut_all", "HMS xp_fp'; HMS_dc_xp_fp; Counts", 200, -0.2, 0.2)
H_dc_yp_fp_protons_data_prompt_cut_all = ROOT.TH1D("H_dc_yp_fp_protons_data_prompt_cut_all", "HMS yp_fp'; HMS_dc_yp_fp; Counts", 200, -0.2, 0.2)
H_hod_goodscinhit_protons_data_prompt_cut_all = ROOT.TH1D("H_hod_goodscinhit_protons_data_prompt_cut_all", "HMS hod goodscinhit; HMS_hod_goodscinhi; Counts", 200, 0.7, 1.3)
H_hod_goodstarttime_protons_data_prompt_cut_all = ROOT.TH1D("H_hod_goodstarttime_protons_data_prompt_cut_all", "HMS hod goodstarttime; HMS_hod_goodstarttime; Counts", 200, 0.7, 1.3)
H_cal_etotnorm_protons_data_prompt_cut_all = ROOT.TH1D("H_cal_etotnorm_protons_data_prompt_cut_all", "HMS cal etotnorm; HMS_cal_etotnorm; Counts", 200, 0.2, 1.8)
H_cal_etottracknorm_protons_data_prompt_cut_all = ROOT.TH1D("H_cal_etottracknorm_protons_data_prompt_cut_all", "HMS cal etottracknorm; HMS_cal_etottracknorm; Counts", 200, 0.2, 1.8)
H_cer_npeSum_protons_data_prompt_cut_all = ROOT.TH1D("H_cer_npeSum_protons_data_prompt_cut_all", "HMS cer npeSum; HMS_cer_npeSum; Counts", 200, 0, 50)
H_RFTime_Dist_protons_data_prompt_cut_all = ROOT.TH1D("H_RFTime_Dist_protons_data_prompt_cut_all", "HMS RFTime; HMS_RFTime; Counts", 200, 0, 4)
P_gtr_beta_protons_data_prompt_cut_all = ROOT.TH1D("P_gtr_beta_protons_data_prompt_cut_all", "SHMS #beta; SHMS_gtr_#beta; Counts", 200, 0.5, 1.3)
P_gtr_xp_protons_data_prompt_cut_all = ROOT.TH1D("P_gtr_xp_protons_data_prompt_cut_all", "SHMS xptar; SHMS_gtr_xptar; Counts", 200, -0.2, 0.2)
P_gtr_yp_protons_data_prompt_cut_all = ROOT.TH1D("P_gtr_yp_protons_data_prompt_cut_all", "SHMS yptar; SHMS_gtr_yptar; Counts", 200, -0.2, 0.2)
P_gtr_dp_protons_data_prompt_cut_all = ROOT.TH1D("P_gtr_dp_protons_data_prompt_cut_all", "SHMS delta; SHMS_gtr_dp; Counts", 200, -30, 30)
P_gtr_p_protons_data_prompt_cut_all = ROOT.TH1D("P_gtr_p_protons_data_prompt_cut_all", "SHMS p; SHMS_gtr_p; Counts", 200, 1, 7)
P_dc_x_fp_protons_data_prompt_cut_all = ROOT.TH1D("P_dc_x_fp_protons_data_prompt_cut_all", "SHMS x_fp'; SHMS_dc_x_fp; Counts", 200, -100, 100)
P_dc_y_fp_protons_data_prompt_cut_all = ROOT.TH1D("P_dc_y_fp_protons_data_prompt_cut_all", "SHMS y_fp'; SHMS_dc_y_fp; Counts", 200, -100, 100)
P_dc_xp_fp_protons_data_prompt_cut_all = ROOT.TH1D("P_dc_xp_fp_protons_data_prompt_cut_all", "SHMS xp_fp'; SHMS_dc_xp_fp; Counts", 200, -0.2, 0.2)
P_dc_yp_fp_protons_data_prompt_cut_all = ROOT.TH1D("P_dc_yp_fp_protons_data_prompt_cut_all", "SHMS yp_fp'; SHMS_dc_yp_fp; Counts", 200, -0.2, 0.2)
P_hod_goodscinhit_protons_data_prompt_cut_all = ROOT.TH1D("P_hod_goodscinhit_protons_data_prompt_cut_all", "SHMS hod goodscinhit; SHMS_hod_goodscinhit; Counts", 200, 0.7, 1.3)
P_hod_goodstarttime_protons_data_prompt_cut_all = ROOT.TH1D("P_hod_goodstarttime_protons_data_prompt_cut_all", "SHMS hod goodstarttime; SHMS_hod_goodstarttime; Counts", 200, 0.7, 1.3)
P_cal_etotnorm_protons_data_prompt_cut_all = ROOT.TH1D("P_cal_etotnorm_protons_data_prompt_cut_all", "SHMS cal etotnorm; SHMS_cal_etotnorm; Counts", 200, 0, 1)
P_cal_etottracknorm_protons_data_prompt_cut_all = ROOT.TH1D("P_cal_etottracknorm_protons_data_prompt_cut_all", "SHMS cal etottracknorm; SHMS_cal_etottracknorm; Counts", 200, 0, 1.6)
P_hgcer_npeSum_protons_data_prompt_cut_all = ROOT.TH1D("P_hgcer_npeSum_protons_data_prompt_cut_all", "SHMS HGC npeSum; SHMS_hgcer_npeSum; Counts", 200, 0, 50)
P_hgcer_xAtCer_protons_data_prompt_cut_all = ROOT.TH1D("P_hgcer_xAtCer_protons_data_prompt_cut_all", "SHMS HGC xAtCer; SHMS_hgcer_xAtCer; Counts", 200, -60, 60)
P_hgcer_yAtCer_protons_data_prompt_cut_all = ROOT.TH1D("P_hgcer_yAtCer_protons_data_prompt_cut_all", "SHMS HGC yAtCer; SHMS_hgcer_yAtCer; Counts", 200, -50, 50)
P_ngcer_npeSum_protons_data_prompt_cut_all = ROOT.TH1D("P_ngcer_npeSum_protons_data_prompt_cut_all", "SHMS NGC npeSum; SHMS_ngcer_npeSum; Counts", 200, 0, 50)
P_ngcer_xAtCer_protons_data_prompt_cut_all = ROOT.TH1D("P_ngcer_xAtCer_protons_data_prompt_cut_all", "SHMS NGC xAtCer; SHMS_ngcer_xAtCer; Counts", 200, -60, 60)
P_ngcer_yAtCer_protons_data_prompt_cut_all = ROOT.TH1D("P_ngcer_yAtCer_protons_data_prompt_cut_all", "SHMS NGC yAtCer; SHMS_ngcer_yAtCer; Counts", 200, -50, 50)
P_aero_npeSum_protons_data_prompt_cut_all = ROOT.TH1D("P_aero_npeSum_protons_data_prompt_cut_all", "SHMS aero npeSum; SHMS_aero_npeSum; Counts", 200, 0, 50)
P_aero_xAtAero_protons_data_prompt_cut_all = ROOT.TH1D("P_acero_xAtAero_protons_data_prompt_cut_all", "SHMS aero xAtAero; SHMS_aero_xAtAero; Counts", 200, -60, 60)
P_aero_yAtAero_protons_data_prompt_cut_all = ROOT.TH1D("P_aero_yAtAero_protons_data_prompt_cut_all", "SHMS aero yAtAero; SHMS_aero_yAtAero; Counts", 200, -50, 50)
P_kin_MMp_protons_data_prompt_cut_all = ROOT.TH1D("P_kin_MMp_protons_data_prompt_cut_all", "MIssing Mass data (prompt_cut_all); MM_{p}; Counts", 200, -1., 1.)
P_RFTime_Dist_protons_data_prompt_cut_all = ROOT.TH1D("P_RFTime_Dist_protons_data_prompt_cut_all", "SHMS RFTime; SHMS_RFTime; Counts", 200, 0, 4)
CTime_epCoinTime_ROC1_protons_data_prompt_cut_all = ROOT.TH1D("CTime_epCoinTime_ROC1_protons_data_prompt_cut_all", "Electron-Proton CTime; e p Coin_Time; Counts", 200, -50, 50)
P_kin_secondary_pmiss_protons_data_prompt_cut_all = ROOT.TH1D("P_kin_secondary_pmiss_protons_data_prompt_cut_all", "Momentum Distribution; pmiss; Counts", 200, -0.8, 0.8)
P_kin_secondary_pmiss_x_protons_data_prompt_cut_all = ROOT.TH1D("P_kin_secondary_pmiss_x_protons_data_prompt_cut_all", "Momentum_x Distribution; pmiss_x; Counts", 200, -0.6, 0.6)
P_kin_secondary_pmiss_y_protons_data_prompt_cut_all = ROOT.TH1D("P_kin_secondary_pmiss_y_protons_data_prompt_cut_all", "Momentum_y Distribution; pmiss_y; Counts", 200, -0.6, 0.6)
P_kin_secondary_pmiss_z_protons_data_prompt_cut_all = ROOT.TH1D("P_kin_secondary_pmiss_z_protons_data_prompt_cut_all", "Momentum_z Distribution; pmiss_z; Counts", 200, -0.6, 0.6)
P_kin_secondary_Erecoil_protons_data_prompt_cut_all = ROOT.TH1D("P_kin_secondary_Erecoil_protons_data_prompt_cut_all", "Erecoil Distribution; Erecoil; Counts", 200, -0.8, 0.8)
P_kin_secondary_emiss_protons_data_prompt_cut_all = ROOT.TH1D("P_kin_secondary_emiss_protons_data_prompt_cut_all", "Energy Distribution; emiss; Counts", 200, -0.8, 0.8)
P_kin_secondary_Mrecoil_protons_data_prompt_cut_all = ROOT.TH1D("P_kin_secondary_Mrecoil_protons_data_prompt_cut_all", "Mrecoil Distribution; Mrecoil; Counts", 200, -0.8, 0.8)
P_kin_secondary_W_protons_data_prompt_cut_all = ROOT.TH1D("P_kin_secondary_W_protons_data_prompt_cut_all", "W Distribution; W; Counts", 200, 0, 2)
MMsquared_data_prompt_cut_all = ROOT.TH1D("MMsquared_data_prompt_cut_all", "Missing Mass Squared; MM^{2}_{p}; Counts", 200, -1., 1.)

#################################################################################################################################################
#2D test histograms
P_kin_secondary_emiss_vs_H_gtr_dp_protons_data_prompt_cut_all = ROOT.TH2D("P_kin_secondary_emiss_vs_H_gtr_dp_protons_data_prompt_cut_all","emiss vs H_gtr_dp (cut all); emiss; H_gtr_dp", 200, -0.4, 0.4, 200, -15, 15)
P_kin_secondary_emiss_vs_P_gtr_dp_protons_data_prompt_cut_all = ROOT.TH2D("P_kin_secondary_emiss_vs_P_gtr_dp_protons_data_prompt_cut_all","emiss vs P_gtr_dp (cut all); emiss; P_gtr_dp", 200, -0.4, 0.4, 200, -20, 20)
P_kin_secondary_emiss_vs_H_dc_xp_fp_protons_data_prompt_cut_all = ROOT.TH2D("P_kin_secondary_emiss_vs_H_dc_xp_fp_protons_data_prompt_cut_all","emiss vs H_dc_xp_fp (cut all); emiss; H_xp_fp", 200, -0.4, 0.4, 200, -0.1, 0.1)
P_kin_secondary_emiss_vs_H_dc_yp_fp_protons_data_prompt_cut_all = ROOT.TH2D("P_kin_secondary_emiss_vs_H_dc_yp_fp_protons_data_prompt_cut_all","emiss vs H_dc_yp_fp (cut all); emiss; H_yp_fp", 200, -0.4, 0.4, 200, -0.1, 0.1)
P_kin_secondary_emiss_vs_P_dc_xp_fp_protons_data_prompt_cut_all = ROOT.TH2D("P_kin_secondary_emiss_vs_P_dc_xp_fp_protons_data_prompt_cut_all","emiss vs P_dc_xp_fp (cut all); emiss; P_xp_fp", 200, -0.4, 0.4, 200, -0.1, 0.1)
P_kin_secondary_emiss_vs_P_dc_yp_fp_protons_data_prompt_cut_all = ROOT.TH2D("P_kin_secondary_emiss_vs_P_dc_yp_fp_protons_data_prompt_cut_all","emiss vs P_dc_yp_fp (cut all); emiss; P_yp_fp", 200, -0.4, 0.4, 200, -0.1, 0.1)

#################################################################################################################################################

# Filling Histograms from DATA ROOT File
ibin = 1
for event in Uncut_Proton_Events_Data_tree:
    H_gtr_beta_protons_data_uncut.Fill(event.H_gtr_beta)
    H_gtr_xp_protons_data_uncut.Fill(event.H_gtr_xp)
    H_gtr_yp_protons_data_uncut.Fill(event.H_gtr_yp)
    H_gtr_dp_protons_data_uncut.Fill(event.H_gtr_dp)
    H_gtr_p_protons_data_uncut.Fill(event.H_gtr_p)
    H_dc_x_fp_protons_data_uncut.Fill(event.H_dc_x_fp)
    H_dc_y_fp_protons_data_uncut.Fill(event.H_dc_y_fp)
    H_dc_xp_fp_protons_data_uncut.Fill(event.H_dc_xp_fp)
    H_dc_yp_fp_protons_data_uncut.Fill(event.H_dc_yp_fp)
    H_hod_goodscinhit_protons_data_uncut.Fill(event.H_hod_goodscinhit)
    H_hod_goodstarttime_protons_data_uncut.Fill(event.H_hod_goodstarttime)
    H_cal_etotnorm_protons_data_uncut.Fill(event.H_cal_etotnorm)
    H_cal_etottracknorm_protons_data_uncut.Fill(event.H_cal_etottracknorm)
    H_cer_npeSum_protons_data_uncut.Fill(event.H_cer_npeSum)
    H_RFTime_Dist_protons_data_uncut.Fill(event.H_RF_Dist)
    P_gtr_beta_protons_data_uncut.Fill(event.P_gtr_beta)
    P_gtr_xp_protons_data_uncut.Fill(event.P_gtr_xp)
    P_gtr_yp_protons_data_uncut.Fill(event.P_gtr_yp)
    P_gtr_dp_protons_data_uncut.Fill(event.P_gtr_dp)
    P_gtr_p_protons_data_uncut.Fill(event.P_gtr_p)
    P_dc_x_fp_protons_data_uncut.Fill(event.P_dc_x_fp)
    P_dc_y_fp_protons_data_uncut.Fill(event.P_dc_y_fp)
    P_dc_xp_fp_protons_data_uncut.Fill(event.P_dc_xp_fp)
    P_dc_yp_fp_protons_data_uncut.Fill(event.P_dc_yp_fp)
    P_hod_goodscinhit_protons_data_uncut.Fill(event.P_hod_goodscinhit)
    P_hod_goodstarttime_protons_data_uncut.Fill(event.P_hod_goodstarttime)
    P_cal_etotnorm_protons_data_uncut.Fill(event.P_cal_etotnorm)
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
    P_kin_secondary_pmiss_protons_data_uncut.Fill(event.pmiss)
    P_kin_secondary_pmiss_x_protons_data_uncut.Fill(event.pmiss_x)
    P_kin_secondary_pmiss_y_protons_data_uncut.Fill(event.pmiss_y)
    P_kin_secondary_pmiss_z_protons_data_uncut.Fill(event.pmiss_z)
    P_kin_secondary_Erecoil_protons_data_uncut.Fill(event.Erecoil)
    P_kin_secondary_emiss_protons_data_uncut.Fill(event.emiss)
    P_kin_secondary_Mrecoil_protons_data_uncut.Fill(event.Mrecoil)
    P_kin_secondary_W_protons_data_uncut.Fill(event.W)
    ibin += 1

ibin = 1
for event in Cut_Proton_Events_All_Data_tree:
    H_gtr_beta_protons_data_cut_all.Fill(event.H_gtr_beta)
    H_gtr_xp_protons_data_cut_all.Fill(event.H_gtr_xp)
    H_gtr_yp_protons_data_cut_all.Fill(event.H_gtr_yp)
    H_gtr_dp_protons_data_cut_all.Fill(event.H_gtr_dp)
    H_gtr_p_protons_data_cut_all.Fill(event.H_gtr_p)
    H_dc_x_fp_protons_data_cut_all.Fill(event.H_dc_x_fp)
    H_dc_y_fp_protons_data_cut_all.Fill(event.H_dc_y_fp)
    H_dc_xp_fp_protons_data_cut_all.Fill(event.H_dc_xp_fp)
    H_dc_yp_fp_protons_data_cut_all.Fill(event.H_dc_yp_fp)
    H_hod_goodscinhit_protons_data_cut_all.Fill(event.H_hod_goodscinhit)
    H_hod_goodstarttime_protons_data_cut_all.Fill(event.H_hod_goodstarttime)
    H_cal_etotnorm_protons_data_cut_all.Fill(event.H_cal_etotnorm)
    H_cal_etottracknorm_protons_data_cut_all.Fill(event.H_cal_etottracknorm)
    H_cer_npeSum_protons_data_cut_all.Fill(event.H_cer_npeSum)
    H_RFTime_Dist_protons_data_cut_all.Fill(event.H_RF_Dist)
    P_gtr_beta_protons_data_cut_all.Fill(event.P_gtr_beta)
    P_gtr_xp_protons_data_cut_all.Fill(event.P_gtr_xp)
    P_gtr_yp_protons_data_cut_all.Fill(event.P_gtr_yp)
    P_gtr_dp_protons_data_cut_all.Fill(event.P_gtr_dp)
    P_gtr_p_protons_data_cut_all.Fill(event.P_gtr_p)
    P_dc_x_fp_protons_data_cut_all.Fill(event.P_dc_x_fp)
    P_dc_y_fp_protons_data_cut_all.Fill(event.P_dc_y_fp)
    P_dc_xp_fp_protons_data_cut_all.Fill(event.P_dc_xp_fp)
    P_dc_yp_fp_protons_data_cut_all.Fill(event.P_dc_yp_fp)
    P_hod_goodscinhit_protons_data_cut_all.Fill(event.P_hod_goodscinhit)
    P_hod_goodstarttime_protons_data_cut_all.Fill(event.P_hod_goodstarttime)
    P_cal_etotnorm_protons_data_cut_all.Fill(event.P_cal_etotnorm)
    P_cal_etottracknorm_protons_data_cut_all.Fill(event.P_cal_etottracknorm)
    P_hgcer_npeSum_protons_data_cut_all.Fill(event.P_hgcer_npeSum)
    P_hgcer_xAtCer_protons_data_cut_all.Fill(event.P_hgcer_xAtCer)
    P_hgcer_yAtCer_protons_data_cut_all.Fill(event.P_hgcer_yAtCer)
    P_ngcer_npeSum_protons_data_cut_all.Fill(event.P_ngcer_npeSum)
    P_ngcer_xAtCer_protons_data_cut_all.Fill(event.P_ngcer_xAtCer)
    P_ngcer_yAtCer_protons_data_cut_all.Fill(event.P_ngcer_yAtCer)
    P_aero_npeSum_protons_data_cut_all.Fill(event.P_aero_npeSum)
    P_aero_xAtAero_protons_data_cut_all.Fill(event.P_aero_xAtAero)
    P_aero_yAtAero_protons_data_cut_all.Fill(event.P_aero_yAtAero)
    P_kin_MMp_protons_data_cut_all.Fill(event.MMp)
    P_RFTime_Dist_protons_data_cut_all.Fill(event.P_RF_Dist)
    CTime_epCoinTime_ROC1_protons_data_cut_all.Fill(event.CTime_epCoinTime_ROC1)
    P_kin_secondary_pmiss_protons_data_cut_all.Fill(event.pmiss)
    P_kin_secondary_pmiss_x_protons_data_cut_all.Fill(event.pmiss_x)
    P_kin_secondary_pmiss_y_protons_data_cut_all.Fill(event.pmiss_y)
    P_kin_secondary_pmiss_z_protons_data_cut_all.Fill(event.pmiss_z)
    P_kin_secondary_Erecoil_protons_data_cut_all.Fill(event.Erecoil)
    P_kin_secondary_emiss_protons_data_cut_all.Fill(event.emiss)
    P_kin_secondary_Mrecoil_protons_data_cut_all.Fill(event.Mrecoil)
    P_kin_secondary_W_protons_data_cut_all.Fill(event.W)
    ibin += 1

ibin = 1
for event in Cut_Proton_Events_Prompt_Data_tree:
    H_gtr_beta_protons_data_prompt_cut_all.Fill(event.H_gtr_beta)
    H_gtr_xp_protons_data_prompt_cut_all.Fill(event.H_gtr_xp)
    H_gtr_yp_protons_data_prompt_cut_all.Fill(event.H_gtr_yp)
    H_gtr_dp_protons_data_prompt_cut_all.Fill(event.H_gtr_dp)
    H_gtr_p_protons_data_prompt_cut_all.Fill(event.H_gtr_p)
    H_dc_x_fp_protons_data_prompt_cut_all.Fill(event.H_dc_x_fp)
    H_dc_y_fp_protons_data_prompt_cut_all.Fill(event.H_dc_y_fp)
    H_dc_xp_fp_protons_data_prompt_cut_all.Fill(event.H_dc_xp_fp)
    H_dc_yp_fp_protons_data_prompt_cut_all.Fill(event.H_dc_yp_fp)
    H_hod_goodscinhit_protons_data_prompt_cut_all.Fill(event.H_hod_goodscinhit)
    H_hod_goodstarttime_protons_data_prompt_cut_all.Fill(event.H_hod_goodstarttime)
    H_cal_etotnorm_protons_data_prompt_cut_all.Fill(event.H_cal_etotnorm)
    H_cal_etottracknorm_protons_data_prompt_cut_all.Fill(event.H_cal_etottracknorm)
    H_cer_npeSum_protons_data_prompt_cut_all.Fill(event.H_cer_npeSum)
    H_RFTime_Dist_protons_data_prompt_cut_all.Fill(event.H_RF_Dist)
    P_gtr_beta_protons_data_prompt_cut_all.Fill(event.P_gtr_beta)
    P_gtr_xp_protons_data_prompt_cut_all.Fill(event.P_gtr_xp)
    P_gtr_yp_protons_data_prompt_cut_all.Fill(event.P_gtr_yp)
    P_gtr_dp_protons_data_prompt_cut_all.Fill(event.P_gtr_dp)
    P_gtr_p_protons_data_prompt_cut_all.Fill(event.P_gtr_p)
    P_dc_x_fp_protons_data_prompt_cut_all.Fill(event.P_dc_x_fp)
    P_dc_y_fp_protons_data_prompt_cut_all.Fill(event.P_dc_y_fp)
    P_dc_xp_fp_protons_data_prompt_cut_all.Fill(event.P_dc_xp_fp)
    P_dc_yp_fp_protons_data_prompt_cut_all.Fill(event.P_dc_yp_fp)
    P_hod_goodscinhit_protons_data_prompt_cut_all.Fill(event.P_hod_goodscinhit)
    P_hod_goodstarttime_protons_data_prompt_cut_all.Fill(event.P_hod_goodstarttime)
    P_cal_etotnorm_protons_data_prompt_cut_all.Fill(event.P_cal_etotnorm)
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
    P_kin_secondary_pmiss_protons_data_prompt_cut_all.Fill(event.pmiss)
    P_kin_secondary_pmiss_x_protons_data_prompt_cut_all.Fill(event.pmiss_x)
    P_kin_secondary_pmiss_y_protons_data_prompt_cut_all.Fill(event.pmiss_y)
    P_kin_secondary_pmiss_z_protons_data_prompt_cut_all.Fill(event.pmiss_z)
    P_kin_secondary_Erecoil_protons_data_prompt_cut_all.Fill(event.Erecoil)
    P_kin_secondary_emiss_protons_data_prompt_cut_all.Fill(event.emiss)
    P_kin_secondary_Mrecoil_protons_data_prompt_cut_all.Fill(event.Mrecoil)
    P_kin_secondary_W_protons_data_prompt_cut_all.Fill(event.W)
    MMsquared_data_prompt_cut_all.Fill(event.MMp*event.MMp)
    ibin += 1

ibin = 1
for event in Cut_Proton_Events_All_Data_tree:
    P_kin_secondary_emiss_vs_H_gtr_dp_protons_data_prompt_cut_all.Fill(event.emiss, event.H_gtr_dp)
    P_kin_secondary_emiss_vs_P_gtr_dp_protons_data_prompt_cut_all.Fill(event.emiss, event.P_gtr_dp)
    P_kin_secondary_emiss_vs_H_dc_xp_fp_protons_data_prompt_cut_all.Fill(event.emiss, event.H_dc_xp_fp)
    P_kin_secondary_emiss_vs_H_dc_yp_fp_protons_data_prompt_cut_all.Fill(event.emiss, event.H_dc_yp_fp)
    P_kin_secondary_emiss_vs_P_dc_xp_fp_protons_data_prompt_cut_all.Fill(event.emiss, event.P_dc_xp_fp)
    P_kin_secondary_emiss_vs_P_dc_yp_fp_protons_data_prompt_cut_all.Fill(event.emiss, event.P_dc_yp_fp)
    ibin += 1

#################################################################################################################################################

ROOT.gStyle.SetOptStat(0)
c1_emiss1 = TCanvas("c1_emiss1", "Missing Energy Distributions", 100, 0, 800,800)
c1_emiss1.Divide(2,2)
#Beam_Energy_S, HMS_p, HMS_theta, SHMS_p, SHMS_theta  = 10.549355, -5.878, 21.655, 5.530, 23.110
#Beam_Energy_S, HMS_p, HMS_theta, SHMS_p, SHMS_theta  = 5.984804, -3.271, 29.170, 3.493, 27.495
#Beam_Energy_S, HMS_p, HMS_theta, SHMS_p, SHMS_theta  = 6.394701, -4.752, 18.595, 2.412, 37.970
#Beam_Energy_S, HMS_p, HMS_theta, SHMS_p, SHMS_theta  = 6.394701, -4.391, 21.095, 2.792, 34.475
#Beam_Energy_S, HMS_p, HMS_theta, SHMS_p, SHMS_theta  = 6.394701, -3.014, 33.350, 4.220, 23.115
#Beam_Energy_S, HMS_p, HMS_theta, SHMS_p, SHMS_theta  = 7.937000, -3.283, 33.640, 5.512, 19.270
#Beam_Energy_S, HMS_p, HMS_theta, SHMS_p, SHMS_theta  = 8.478619, -5.587, 19.560, 3.731, 30.020
#Beam_Energy_S, HMS_p, HMS_theta, SHMS_p, SHMS_theta  = 9.172705, -3.738, 31.645, 6.265, 18.125
Beam_Energy_S, HMS_p, HMS_theta, SHMS_p, SHMS_theta  = 9.878908, -5.366, 23.050, 5.422, 23.050
c1_emiss1.cd(1)
c1_emiss1_text_lines = [
    TText(0.5, 0.9, "HeePCoin Setting"),
    TText(0.5, 0.8, 'Beam Energy = ' + str(Beam_Energy_S)),
    TText(0.5, 0.7, 'HMS_p = ' + str(HMS_p)),
    TText(0.5, 0.6, 'HMS_theta = ' + str(HMS_theta)),
    TText(0.5, 0.5, 'SHMS_p = ' + str(SHMS_p)),
    TText(0.5, 0.4, 'SHMS_theta = ' + str(SHMS_theta))
]
for c1_emiss1_text in c1_emiss1_text_lines:
    c1_emiss1_text.SetTextSize(0.07)
    c1_emiss1_text.SetTextAlign(22)
    c1_emiss1_text.Draw()
c1_emiss1.cd(2)
P_kin_secondary_emiss_protons_data_prompt_cut_all.Draw()
c1_emiss1.cd(3)
P_kin_secondary_emiss_vs_H_gtr_dp_protons_data_prompt_cut_all.Draw("COLZ")
c1_emiss1.cd(4)
P_kin_secondary_emiss_vs_P_gtr_dp_protons_data_prompt_cut_all.Draw("COLZ")
c1_emiss1.Print(Proton_Analysis_Distributions + '(')

c1_emiss2 = TCanvas("c1_emiss2", "Missing Energy Distributions", 100, 0, 800,800)
c1_emiss2.Divide(2,2)
c1_emiss2.cd(1)
P_kin_secondary_emiss_vs_H_dc_xp_fp_protons_data_prompt_cut_all.Draw("COLZ")
c1_emiss2.cd(2)
P_kin_secondary_emiss_vs_H_dc_yp_fp_protons_data_prompt_cut_all.Draw("COLZ")
c1_emiss2.cd(3)
P_kin_secondary_emiss_vs_P_dc_xp_fp_protons_data_prompt_cut_all.Draw("COLZ")
c1_emiss2.cd(4)
P_kin_secondary_emiss_vs_P_dc_yp_fp_protons_data_prompt_cut_all.Draw("COLZ")
c1_emiss2.Print(Proton_Analysis_Distributions + ')')

############################################################################################################################################

# Making directories in output file
outHistFile = ROOT.TFile.Open("%s/test/%s_%s_Output_Data.root" % (OUTPATH, RUNNUMBER, MaxEvent) , "RECREATE")
d_Uncut_Proton_Events_Data = outHistFile.mkdir("Uncut_Proton_Events_Data")
d_Cut_Proton_Events_All_Data = outHistFile.mkdir("Cut_Proton_Events_All_Data")
d_Cut_Proton_Events_Prompt_Data = outHistFile.mkdir("Cut_Proton_Events_Prompt_Data")

# Writing Histograms for protons                                                                  
d_Uncut_Proton_Events_Data.cd()
H_gtr_beta_protons_data_uncut.Write()
H_gtr_xp_protons_data_uncut.Write()
H_gtr_yp_protons_data_uncut.Write()
H_gtr_dp_protons_data_uncut.Write()
H_gtr_p_protons_data_uncut.Write()
H_dc_x_fp_protons_data_uncut.Write()
H_dc_y_fp_protons_data_uncut.Write()
H_dc_xp_fp_protons_data_uncut.Write()
H_dc_yp_fp_protons_data_uncut.Write()
H_hod_goodscinhit_protons_data_uncut.Write()
H_hod_goodstarttime_protons_data_uncut.Write()
H_cal_etotnorm_protons_data_uncut.Write()
H_cal_etottracknorm_protons_data_uncut.Write()
H_cer_npeSum_protons_data_uncut.Write()
H_RFTime_Dist_protons_data_uncut.Write()
P_gtr_beta_protons_data_uncut.Write()
P_gtr_xp_protons_data_uncut.Write()
P_gtr_yp_protons_data_uncut.Write()
P_gtr_dp_protons_data_uncut.Write()
P_gtr_p_protons_data_uncut.Write()
P_dc_x_fp_protons_data_uncut.Write()
P_dc_y_fp_protons_data_uncut.Write()
P_dc_xp_fp_protons_data_uncut.Write()
P_dc_yp_fp_protons_data_uncut.Write()
P_hod_goodscinhit_protons_data_uncut.Write()
P_hod_goodstarttime_protons_data_uncut.Write()
P_cal_etotnorm_protons_data_uncut.Write()
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
P_kin_secondary_pmiss_protons_data_uncut.Write()
P_kin_secondary_pmiss_x_protons_data_uncut.Write()
P_kin_secondary_pmiss_y_protons_data_uncut.Write()
P_kin_secondary_pmiss_z_protons_data_uncut.Write()
P_kin_secondary_Erecoil_protons_data_uncut.Write()
P_kin_secondary_emiss_protons_data_uncut.Write()
P_kin_secondary_Mrecoil_protons_data_uncut.Write()
P_kin_secondary_W_protons_data_uncut.Write()

d_Cut_Proton_Events_All_Data.cd()
H_gtr_beta_protons_data_cut_all.Write()
H_gtr_xp_protons_data_cut_all.Write()
H_gtr_yp_protons_data_cut_all.Write()
H_gtr_dp_protons_data_cut_all.Write()
H_gtr_p_protons_data_cut_all.Write()
H_dc_x_fp_protons_data_cut_all.Write()
H_dc_y_fp_protons_data_cut_all.Write()
H_dc_xp_fp_protons_data_cut_all.Write()
H_dc_yp_fp_protons_data_cut_all.Write()
H_hod_goodscinhit_protons_data_cut_all.Write()
H_hod_goodstarttime_protons_data_cut_all.Write()
H_cal_etotnorm_protons_data_cut_all.Write()
H_cal_etottracknorm_protons_data_cut_all.Write()
H_cer_npeSum_protons_data_cut_all.Write()
H_RFTime_Dist_protons_data_cut_all.Write()
P_gtr_beta_protons_data_cut_all.Write()
P_gtr_xp_protons_data_cut_all.Write()
P_gtr_yp_protons_data_cut_all.Write()
P_gtr_dp_protons_data_cut_all.Write()
P_gtr_p_protons_data_cut_all.Write()
P_dc_x_fp_protons_data_cut_all.Write()
P_dc_y_fp_protons_data_cut_all.Write()
P_dc_xp_fp_protons_data_cut_all.Write()
P_dc_yp_fp_protons_data_cut_all.Write()
P_hod_goodscinhit_protons_data_cut_all.Write()
P_hod_goodstarttime_protons_data_cut_all.Write()
P_cal_etotnorm_protons_data_cut_all.Write()
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
P_kin_secondary_pmiss_protons_data_cut_all.Write()
P_kin_secondary_pmiss_x_protons_data_cut_all.Write()
P_kin_secondary_pmiss_y_protons_data_cut_all.Write()
P_kin_secondary_pmiss_z_protons_data_cut_all.Write()
P_kin_secondary_Erecoil_protons_data_cut_all.Write()
P_kin_secondary_emiss_protons_data_cut_all.Write()
P_kin_secondary_Mrecoil_protons_data_cut_all.Write()
P_kin_secondary_W_protons_data_cut_all.Write()

d_Cut_Proton_Events_Prompt_Data.cd()
H_gtr_beta_protons_data_prompt_cut_all.Write()
H_gtr_xp_protons_data_prompt_cut_all.Write()
H_gtr_yp_protons_data_prompt_cut_all.Write()
H_gtr_dp_protons_data_prompt_cut_all.Write()
H_gtr_p_protons_data_prompt_cut_all.Write()
H_dc_x_fp_protons_data_prompt_cut_all.Write()
H_dc_y_fp_protons_data_prompt_cut_all.Write()
H_dc_xp_fp_protons_data_prompt_cut_all.Write()
H_dc_yp_fp_protons_data_prompt_cut_all.Write()
H_hod_goodscinhit_protons_data_prompt_cut_all.Write()
H_hod_goodstarttime_protons_data_prompt_cut_all.Write()
H_cal_etotnorm_protons_data_prompt_cut_all.Write()
H_cal_etottracknorm_protons_data_prompt_cut_all.Write()
H_cer_npeSum_protons_data_prompt_cut_all.Write()
H_RFTime_Dist_protons_data_prompt_cut_all.Write()
P_gtr_beta_protons_data_prompt_cut_all.Write()
P_gtr_xp_protons_data_prompt_cut_all.Write()
P_gtr_yp_protons_data_prompt_cut_all.Write()
P_gtr_dp_protons_data_prompt_cut_all.Write()
P_gtr_p_protons_data_prompt_cut_all.Write()
P_dc_x_fp_protons_data_prompt_cut_all.Write()
P_dc_y_fp_protons_data_prompt_cut_all.Write()
P_dc_xp_fp_protons_data_prompt_cut_all.Write()
P_dc_yp_fp_protons_data_prompt_cut_all.Write()
P_hod_goodscinhit_protons_data_prompt_cut_all.Write()
P_hod_goodstarttime_protons_data_prompt_cut_all.Write()
P_cal_etotnorm_protons_data_prompt_cut_all.Write()
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
P_kin_secondary_pmiss_protons_data_prompt_cut_all.Write()
P_kin_secondary_pmiss_x_protons_data_prompt_cut_all.Write()
P_kin_secondary_pmiss_y_protons_data_prompt_cut_all.Write()
P_kin_secondary_pmiss_z_protons_data_prompt_cut_all.Write()
P_kin_secondary_Erecoil_protons_data_prompt_cut_all.Write()
P_kin_secondary_emiss_protons_data_prompt_cut_all.Write()
P_kin_secondary_Mrecoil_protons_data_prompt_cut_all.Write()
P_kin_secondary_W_protons_data_prompt_cut_all.Write()
MMsquared_data_prompt_cut_all.Write()

infile_DATA.Close() 
print ("Processing Complete")
