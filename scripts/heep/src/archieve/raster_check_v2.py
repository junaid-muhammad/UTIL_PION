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

##################################################################################################################################################

# Check the number of arguments provided to the script
if len(sys.argv)-1!=1:
    print("!!!!! ERROR !!!!!\n Expected 8 arguments\n Usage is with - Beam Energy \n!!!!! ERROR !!!!!")
    sys.exit(1)

##################################################################################################################################################

# Input params - run number and max number of events
BEAM_ENERGY = sys.argv[1]

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
Proton_Analysis_Distributions_Uncut = "%s/test/%s_HeePCoin_RasterCheck_Uncut_Distributions.pdf" % (OUTPATH, BEAM_ENERGY)
Proton_Analysis_Distributions_Cut = "%s/test/%s_HeePCoin_RasterCheck_Cut_Distributions.pdf" % (OUTPATH, BEAM_ENERGY)

# Input file location and variables taking
rootFile_DATA = "%s/test/%s_-1_Analysed_RasterCheck_Data.root" % (OUTPATH, BEAM_ENERGY)

###############################################################################################################################################

# Read stuff from the main event tree

infile_DATA = ROOT.TFile.Open(rootFile_DATA, "READ")
Uncut_Proton_Events_Data_tree = infile_DATA.Get("Uncut_Proton_Events")
Cut_Proton_Events_All_Data_tree = infile_DATA.Get("Cut_Proton_Events_All")

###################################################################################################################################################

n_bins = 200

# 1D Uncut Data Histograms
H_gtr_xp_protons_data_uncut = ROOT.TH1D("H_gtr_xp_protons_data_uncut", "HMS xptar; HMS_gtr_xptar; Counts", n_bins, -0.2, 0.2)
H_gtr_yp_protons_data_uncut = ROOT.TH1D("H_gtr_yp_protons_data_uncut", "HMS yptar; HMS_gtr_yptar; Counts", n_bins, -0.2, 0.2)
H_gtr_dp_protons_data_uncut = ROOT.TH1D("H_gtr_dp_protons_data_uncut", "HMS #delta; HMS_gtr_dp; Counts", n_bins, -15, 15)
H_gtr_p_protons_data_uncut = ROOT.TH1D("H_gtr_p_protons_data_uncut", "HMS p; HMS_gtr_p; Counts", n_bins, 4, 8)
H_dc_x_fp_protons_data_uncut = ROOT.TH1D("H_dc_x_fp_protons_data_uncut", "HMS x_fp'; HMS_dc_x_fp; Counts", n_bins, -60, 60)
H_dc_y_fp_protons_data_uncut = ROOT.TH1D("H_dc_y_fp_protons_data_uncut", "HMS y_fp'; HMS_dc_y_fp; Counts", n_bins, -40, 40)
H_dc_xp_fp_protons_data_uncut = ROOT.TH1D("H_dc_xp_fp_protons_data_uncut", "HMS xp_fp'; HMS_dc_xp_fp; Counts", n_bins, -0.2, 0.2)
H_dc_yp_fp_protons_data_uncut = ROOT.TH1D("H_dc_yp_fp_protons_data_uncut", "HMS yp_fp'; HMS_dc_yp_fp; Counts", n_bins, -0.2, 0.2)
P_gtr_xp_protons_data_uncut = ROOT.TH1D("P_gtr_xp_protons_data_uncut", "SHMS xptar; SHMS_gtr_xptar; Counts", n_bins, -0.2, 0.2)
P_gtr_yp_protons_data_uncut = ROOT.TH1D("P_gtr_yp_protons_data_uncut", "SHMS yptar; SHMS_gtr_yptar; Counts", n_bins, -0.2, 0.2)
P_gtr_dp_protons_data_uncut = ROOT.TH1D("P_gtr_dp_protons_data_uncut", "SHMS delta; SHMS_gtr_dp; Counts", n_bins, -30, 30)
P_gtr_p_protons_data_uncut = ROOT.TH1D("P_gtr_p_protons_data_uncut", "SHMS p; SHMS_gtr_p; Counts", n_bins, 4, 7)
P_dc_x_fp_protons_data_uncut = ROOT.TH1D("P_dc_x_fp_protons_data_uncut", "SHMS x_fp'; SHMS_dc_x_fp; Counts", n_bins, -60, 60)
P_dc_y_fp_protons_data_uncut = ROOT.TH1D("P_dc_y_fp_protons_data_uncut", "SHMS y_fp'; SHMS_dc_y_fp; Counts", n_bins, -40, 40)
P_dc_xp_fp_protons_data_uncut = ROOT.TH1D("P_dc_xp_fp_protons_data_uncut", "SHMS xp_fp'; SHMS_dc_xp_fp; Counts", n_bins, -0.2, 0.2)
P_dc_yp_fp_protons_data_uncut = ROOT.TH1D("P_dc_yp_fp_protons_data_uncut", "SHMS yp_fp'; SHMS_dc_yp_fp; Counts", n_bins, -0.2, 0.2)
P_kin_MMp_protons_data_uncut = ROOT.TH1D("P_kin_MMp_protons_data_uncut", "MIssing Mass data uncut; MM_{p}; Counts", n_bins, -1., 1.)
P_kin_secondary_pmiss_protons_data_uncut = ROOT.TH1D("P_kin_secondary_pmiss_protons_data_uncut", "Momentum Distribution; pmiss; Counts", n_bins, -1, 1)
P_kin_secondary_pmiss_x_protons_data_uncut = ROOT.TH1D("P_kin_secondary_pmiss_x_protons_data_uncut", "Momentum_x Distribution; pmiss_x; Counts", n_bins, -0.6, 0.6)
P_kin_secondary_pmiss_y_protons_data_uncut = ROOT.TH1D("P_kin_secondary_pmiss_y_protons_data_uncut", "Momentum_y Distribution; pmiss_y; Counts", n_bins, -0.6, 0.6)
P_kin_secondary_pmiss_z_protons_data_uncut = ROOT.TH1D("P_kin_secondary_pmiss_z_protons_data_uncut", "Momentum_z Distribution; pmiss_z; Counts", n_bins, -1, 1)
P_kin_secondary_emiss_protons_data_uncut = ROOT.TH1D("P_kin_secondary_emiss_protons_data_uncut", "Emiss Distribution; emiss; Counts", n_bins, -1, 1)
P_kin_secondary_W_protons_data_uncut = ROOT.TH1D("P_kin_secondary_W_protons_data_uncut", "W Distribution; W; Counts", n_bins, 0, 2)
P_rb_x_protons_data_uncut = ROOT.TH1D("P_rb_x_protons_data_uncut", "Raster x Distribution; Raster x; Counts", n_bins, -0.4, 0.4)
P_rb_y_protons_data_uncut = ROOT.TH1D("P_rb_y_protons_data_uncut", "Raster y Distribution; Raster y; Counts", n_bins, -0.4, 0.4)
P_rb_raster_fr_xbpm_tar_protons_data_uncut = ROOT.TH1D("P_rb_raster_fr_xbpm_tar_protons_data_uncut", "Raster xbpm tar Distribution; Raster xbpm tar; Counts", n_bins, -0.06, 0.0)
P_rb_raster_fr_ybpm_tar_protons_data_uncut = ROOT.TH1D("P_rb_raster_fr_ybpm_tar_protons_data_uncut", "Raster ybpm tar Distribution; Raster ybpm tar; Counts", n_bins, -0.05, -0.01)

# 1D Acceptance Cut Data Histograms
H_gtr_xp_protons_data_cut = ROOT.TH1D("H_gtr_xp_protons_data_cut", "HMS xptar; HMS_gtr_xptar; Counts", n_bins, -0.2, 0.2)
H_gtr_yp_protons_data_cut = ROOT.TH1D("H_gtr_yp_protons_data_cut", "HMS yptar; HMS_gtr_yptar; Counts", n_bins, -0.2, 0.2)
H_gtr_dp_protons_data_cut = ROOT.TH1D("H_gtr_dp_protons_data_cut", "HMS #delta; HMS_gtr_dp; Counts", n_bins, -15, 15)
H_gtr_p_protons_data_cut = ROOT.TH1D("H_gtr_p_protons_data_cut", "HMS p; HMS_gtr_p; Counts", n_bins, 4, 8)
H_dc_x_fp_protons_data_cut = ROOT.TH1D("H_dc_x_fp_protons_data_cut", "HMS x_fp'; HMS_dc_x_fp; Counts", n_bins, -60, 60)
H_dc_y_fp_protons_data_cut = ROOT.TH1D("H_dc_y_fp_protons_data_cut", "HMS y_fp'; HMS_dc_y_fp; Counts", n_bins, -40, 40)
H_dc_xp_fp_protons_data_cut = ROOT.TH1D("H_dc_xp_fp_protons_data_cut", "HMS xp_fp'; HMS_dc_xp_fp; Counts", n_bins, -0.2, 0.2)
H_dc_yp_fp_protons_data_cut = ROOT.TH1D("H_dc_yp_fp_protons_data_cut", "HMS yp_fp'; HMS_dc_yp_fp; Counts", n_bins, -0.2, 0.2)
P_gtr_xp_protons_data_cut = ROOT.TH1D("P_gtr_xp_protons_data_cut", "SHMS xptar; SHMS_gtr_xptar; Counts", n_bins, -0.2, 0.2)
P_gtr_yp_protons_data_cut = ROOT.TH1D("P_gtr_yp_protons_data_cut", "SHMS yptar; SHMS_gtr_yptar; Counts", n_bins, -0.2, 0.2)
P_gtr_dp_protons_data_cut = ROOT.TH1D("P_gtr_dp_protons_data_cut", "SHMS delta; SHMS_gtr_dp; Counts", n_bins, -30, 30)
P_gtr_p_protons_data_cut = ROOT.TH1D("P_gtr_p_protons_data_cut", "SHMS p; SHMS_gtr_p; Counts", n_bins, 4, 7)
P_dc_x_fp_protons_data_cut = ROOT.TH1D("P_dc_x_fp_protons_data_cut", "SHMS x_fp'; SHMS_dc_x_fp; Counts", n_bins, -60, 60)
P_dc_y_fp_protons_data_cut = ROOT.TH1D("P_dc_y_fp_protons_data_cut", "SHMS y_fp'; SHMS_dc_y_fp; Counts", n_bins, -40, 40)
P_dc_xp_fp_protons_data_cut = ROOT.TH1D("P_dc_xp_fp_protons_data_cut", "SHMS xp_fp'; SHMS_dc_xp_fp; Counts", n_bins, -0.2, 0.2)
P_dc_yp_fp_protons_data_cut = ROOT.TH1D("P_dc_yp_fp_protons_data_cut", "SHMS yp_fp'; SHMS_dc_yp_fp; Counts", n_bins, -0.2, 0.2)
P_kin_MMp_protons_data_cut = ROOT.TH1D("P_kin_MMp_protons_data_cut", "MIssing Mass data cut; MM_{p}; Counts", n_bins, -1., 1.)
P_kin_secondary_pmiss_protons_data_cut = ROOT.TH1D("P_kin_secondary_pmiss_protons_data_cut", "Momentum Distribution; pmiss; Counts", n_bins, -1, 1)
P_kin_secondary_pmiss_x_protons_data_cut = ROOT.TH1D("P_kin_secondary_pmiss_x_protons_data_cut", "Momentum_x Distribution; pmiss_x; Counts", n_bins, -0.6, 0.6)
P_kin_secondary_pmiss_y_protons_data_cut = ROOT.TH1D("P_kin_secondary_pmiss_y_protons_data_cut", "Momentum_y Distribution; pmiss_y; Counts", n_bins, -0.6, 0.6)
P_kin_secondary_pmiss_z_protons_data_cut = ROOT.TH1D("P_kin_secondary_pmiss_z_protons_data_cut", "Momentum_z Distribution; pmiss_z; Counts", n_bins, -1, 1)
P_kin_secondary_emiss_protons_data_cut = ROOT.TH1D("P_kin_secondary_emiss_protons_data_cut", "Emiss Distribution; emiss; Counts", n_bins, -1, 1)
P_kin_secondary_W_protons_data_cut = ROOT.TH1D("P_kin_secondary_W_protons_data_cut", "W Distribution; W; Counts", n_bins, 0, 2)
P_rb_x_protons_data_cut = ROOT.TH1D("P_rb_x_protons_data_cut", "Raster x Distribution; Raster x; Counts", n_bins, -0.4, 0.4)
P_rb_y_protons_data_cut = ROOT.TH1D("P_rb_y_protons_data_cut", "Raster y Distribution; Raster y; Counts", n_bins, -0.4, 0.4)
P_rb_raster_fr_xbpm_tar_protons_data_cut = ROOT.TH1D("P_rb_raster_fr_xbpm_tar_protons_data_cut", "Raster xbpm tar Distribution; Raster xbpm tar; Counts", n_bins, -0.06, 0.0)
P_rb_raster_fr_ybpm_tar_protons_data_cut = ROOT.TH1D("P_rb_raster_fr_ybpm_tar_protons_data_cut", "Raster ybpm tar Distribution; Raster ybpm tar; Counts", n_bins, -0.05, -0.01)

# 2D Uncut Data Histograms
P_kin_secondary_emiss_vs_H_gtr_dp_protons_data_uncut = ROOT.TH2D("P_kin_secondary_emiss_vs_H_gtr_dp_protons_data_uncut", "emiss vs HMS delta; emiss; HMS delta",n_bins, -0.2, 0.2, n_bins, -10, 10)
P_kin_secondary_emiss_vs_P_gtr_dp_protons_data_uncut = ROOT.TH2D("P_kin_secondary_emiss_vs_P_gtr_dp_protons_data_uncut", "emiss vs SHMS delta; emiss; SHMS delta",n_bins, -0.2, 0.2, n_bins, -10, 10)
P_kin_secondary_pmiss_vs_H_gtr_dp_protons_data_uncut = ROOT.TH2D("P_kin_secondary_pmiss_vs_H_gtr_dp_protons_data_uncut", "pmiss vs HMS delta; pmiss; HMS delta",n_bins, -0.2, 0.2, n_bins, -10, 10)
P_kin_secondary_pmiss_vs_P_gtr_dp_protons_data_uncut = ROOT.TH2D("P_kin_secondary_pmiss_vs_P_gtr_dp_protons_data_uncut", "pmiss vs SHMS delta; pmiss; SHMS delta",n_bins, -0.2, 0.2, n_bins, -10, 10)
P_gtr_dp_vs_H_gtr_dp_protons_data_uncut = ROOT.TH2D("P_gtr_dp_vs_H_gtr_dp_protons_data_uncut", "SHMS delta vs HMS delta; SHMS delta; HMS delta",n_bins, -10, 10, n_bins, -10, 10)
P_rb_x_vs_P_rb_y_protons_data_uncut = ROOT.TH2D("P_rb_x_vs_P_rb_y_protons_data_uncut", "Raster x vs Raster y; Raster x; Raster y",n_bins, -0.2, 0.2, n_bins, -0.2, 0.2)
P_rb_x_vs_P_kin_secondary_emiss_protons_data_uncut = ROOT.TH2D("P_rb_x_vs_P_kin_secondary_emiss_protons_data_uncut", "Raster x vs emiss; Raster x; emiss",n_bins, -0.2, 0.2, n_bins, -0.2, 0.2)
P_rb_x_vs_P_kin_secondary_pmiss_protons_data_uncut = ROOT.TH2D("P_rb_x_vs_P_kin_secondary_pmiss_protons_data_uncut", "Raster x vs pmiss; Raster x; pmiss",n_bins, -0.2, 0.2, n_bins, -0.2, 0.2)
P_rb_y_vs_P_kin_secondary_emiss_protons_data_uncut = ROOT.TH2D("P_rb_y_vs_P_kin_secondary_emiss_protons_data_uncut", "Raster y vs emiss; Raster y; emiss",n_bins, -0.2, 0.2, n_bins, -0.2, 0.2)
P_rb_y_vs_P_kin_secondary_pmiss_protons_data_uncut = ROOT.TH2D("P_rb_y_vs_P_kin_secondary_pmiss_protons_data_uncut", "Raster y vs pmiss; Raster y; pmiss",n_bins, -0.2, 0.2, n_bins, -0.2, 0.2)
P_rb_x_vs_P_kin_secondary_W_protons_data_uncut = ROOT.TH2D("P_rb_x_vs_P_kin_secondary_W_protons_data_uncut", "Raster x vs W; Raster x; W",n_bins, -0.2, 0.2, n_bins, 0.0, 2.0)
P_rb_y_vs_P_kin_secondary_W_protons_data_uncut = ROOT.TH2D("P_rb_y_vs_P_kin_secondary_W_protons_data_uncut", "Raster y vs W; Raster y; W",n_bins, -0.2, 0.2, n_bins, 0.0, 2.0)

# 2D Acceptance Cut Data Histograms
P_kin_secondary_emiss_vs_H_gtr_dp_protons_data_cut = ROOT.TH2D("P_kin_secondary_emiss_vs_H_gtr_dp_protons_data_cut", "emiss vs HMS delta; emiss; HMS delta",n_bins, -0.2, 0.2, n_bins, -10, 10)
P_kin_secondary_emiss_vs_P_gtr_dp_protons_data_cut = ROOT.TH2D("P_kin_secondary_emiss_vs_P_gtr_dp_protons_data_cut", "emiss vs SHMS delta; emiss; SHMS delta",n_bins, -0.2, 0.2, n_bins, -10, 10)
P_kin_secondary_pmiss_vs_H_gtr_dp_protons_data_cut = ROOT.TH2D("P_kin_secondary_pmiss_vs_H_gtr_dp_protons_data_cut", "pmiss vs HMS delta; pmiss; HMS delta",n_bins, -0.2, 0.2, n_bins, -10, 10)
P_kin_secondary_pmiss_vs_P_gtr_dp_protons_data_cut = ROOT.TH2D("P_kin_secondary_pmiss_vs_P_gtr_dp_protons_data_cut", "pmiss vs SHMS delta; pmiss; SHMS delta",n_bins, -0.2, 0.2, n_bins, -10, 10)
P_gtr_dp_vs_H_gtr_dp_protons_data_cut = ROOT.TH2D("P_gtr_dp_vs_H_gtr_dp_protons_data_cut", "SHMS delta vs HMS delta; SHMS delta; HMS delta",n_bins, -10, 10, n_bins, -10, 10)
P_rb_x_vs_P_rb_y_protons_data_cut = ROOT.TH2D("P_rb_x_vs_P_rb_y_protons_data_cut", "Raster x vs Raster y; Raster x; Raster y",n_bins, -0.2, 0.2, n_bins, -0.2, 0.2)
P_rb_x_vs_P_kin_secondary_emiss_protons_data_cut = ROOT.TH2D("P_rb_x_vs_P_kin_secondary_emiss_protons_data_cut", "Raster x vs emiss; Raster x; emiss",n_bins, -0.2, 0.2, n_bins, -0.2, 0.2)
P_rb_x_vs_P_kin_secondary_pmiss_protons_data_cut = ROOT.TH2D("P_rb_x_vs_P_kin_secondary_pmiss_protons_data_cut", "Raster x vs pmiss; Raster x; pmiss",n_bins, -0.2, 0.2, n_bins, -0.2, 0.2)
P_rb_y_vs_P_kin_secondary_emiss_protons_data_cut = ROOT.TH2D("P_rb_y_vs_P_kin_secondary_emiss_protons_data_cut", "Raster y vs emiss; Raster y; emiss",n_bins, -0.2, 0.2, n_bins, -0.2, 0.2)
P_rb_y_vs_P_kin_secondary_pmiss_protons_data_cut = ROOT.TH2D("P_rb_y_vs_P_kin_secondary_pmiss_protons_data_cut", "Raster y vs pmiss; Raster y; pmiss",n_bins, -0.2, 0.2, n_bins, -0.2, 0.2)
P_rb_x_vs_P_kin_secondary_W_protons_data_cut = ROOT.TH2D("P_rb_x_vs_P_kin_secondary_W_protons_data_cut", "Raster x vs W; Raster x; W",n_bins, -0.2, 0.2, n_bins, 0.0, 2.0)
P_rb_y_vs_P_kin_secondary_W_protons_data_cut = ROOT.TH2D("P_rb_y_vs_P_kin_secondary_W_protons_data_cut", "Raster y vs W; Raster y; W",n_bins, -0.2, 0.2, n_bins, 0.0, 2.0)

#################################################################################################################################################

#Fill histograms
for event in Uncut_Proton_Events_Data_tree:
    H_gtr_xp_protons_data_uncut.Fill(event.H_gtr_xp)
    H_gtr_yp_protons_data_uncut.Fill(event.H_gtr_yp)
    H_gtr_dp_protons_data_uncut.Fill(event.H_gtr_dp)
    H_gtr_p_protons_data_uncut.Fill(event.H_gtr_p)
    H_dc_x_fp_protons_data_uncut.Fill(event.H_dc_x_fp)
    H_dc_y_fp_protons_data_uncut.Fill(event.H_dc_y_fp)
    H_dc_xp_fp_protons_data_uncut.Fill(event.H_dc_xp_fp)
    H_dc_yp_fp_protons_data_uncut.Fill(event.H_dc_yp_fp)
    P_gtr_xp_protons_data_uncut.Fill(event.P_gtr_xp)
    P_gtr_yp_protons_data_uncut.Fill(event.P_gtr_yp)
    P_gtr_dp_protons_data_uncut.Fill(event.P_gtr_dp)
    P_gtr_p_protons_data_uncut.Fill(event.P_gtr_p)
    P_dc_x_fp_protons_data_uncut.Fill(event.P_dc_x_fp)
    P_dc_y_fp_protons_data_uncut.Fill(event.P_dc_y_fp)
    P_dc_xp_fp_protons_data_uncut.Fill(event.P_dc_xp_fp)
    P_dc_yp_fp_protons_data_uncut.Fill(event.P_dc_yp_fp)
    P_kin_MMp_protons_data_uncut.Fill(event.MMp)
    P_kin_secondary_pmiss_protons_data_uncut.Fill(event.pmiss)
    P_kin_secondary_pmiss_x_protons_data_uncut.Fill(event.pmiss_x)
    P_kin_secondary_pmiss_y_protons_data_uncut.Fill(event.pmiss_y)
    P_kin_secondary_pmiss_z_protons_data_uncut.Fill(event.pmiss_z)
    P_kin_secondary_emiss_protons_data_uncut.Fill(event.emiss)
    P_kin_secondary_W_protons_data_uncut.Fill(event.W)
    P_rb_x_protons_data_uncut.Fill(event.raster_x)
    P_rb_y_protons_data_uncut.Fill(event.raster_y)
    P_rb_raster_fr_xbpm_tar_protons_data_uncut.Fill(event.bpm_tar_x)
    P_rb_raster_fr_ybpm_tar_protons_data_uncut.Fill(event.bpm_tar_y)
    P_kin_secondary_emiss_vs_H_gtr_dp_protons_data_uncut.Fill(event.emiss , event.H_gtr_dp)
    P_kin_secondary_emiss_vs_P_gtr_dp_protons_data_uncut.Fill(event.emiss , event.P_gtr_dp)
    P_kin_secondary_pmiss_vs_H_gtr_dp_protons_data_uncut.Fill(event.pmiss , event.H_gtr_dp)
    P_kin_secondary_pmiss_vs_P_gtr_dp_protons_data_uncut.Fill(event.pmiss , event.P_gtr_dp)
    P_gtr_dp_vs_H_gtr_dp_protons_data_uncut.Fill(event.H_gtr_dp , event.P_gtr_dp)
    P_rb_x_vs_P_rb_y_protons_data_uncut.Fill(event.raster_x , event.raster_y)
    P_rb_x_vs_P_kin_secondary_emiss_protons_data_uncut.Fill(event.raster_x , event.emiss)
    P_rb_x_vs_P_kin_secondary_pmiss_protons_data_uncut.Fill(event.raster_x , event.pmiss)
    P_rb_y_vs_P_kin_secondary_emiss_protons_data_uncut.Fill(event.raster_y , event.emiss)
    P_rb_y_vs_P_kin_secondary_pmiss_protons_data_uncut.Fill(event.raster_y , event.pmiss)
    P_rb_x_vs_P_kin_secondary_W_protons_data_uncut.Fill(event.raster_x , event.W)
    P_rb_y_vs_P_kin_secondary_W_protons_data_uncut.Fill(event.raster_y , event.W)

for event in Cut_Proton_Events_All_Data_tree:
    H_gtr_xp_protons_data_cut.Fill(event.H_gtr_xp)
    H_gtr_yp_protons_data_cut.Fill(event.H_gtr_yp)
    H_gtr_dp_protons_data_cut.Fill(event.H_gtr_dp)
    H_gtr_p_protons_data_cut.Fill(event.H_gtr_p)
    H_dc_x_fp_protons_data_cut.Fill(event.H_dc_x_fp)
    H_dc_y_fp_protons_data_cut.Fill(event.H_dc_y_fp)
    H_dc_xp_fp_protons_data_cut.Fill(event.H_dc_xp_fp)
    H_dc_yp_fp_protons_data_cut.Fill(event.H_dc_yp_fp)
    P_gtr_xp_protons_data_cut.Fill(event.P_gtr_xp)
    P_gtr_yp_protons_data_cut.Fill(event.P_gtr_yp)
    P_gtr_dp_protons_data_cut.Fill(event.P_gtr_dp)
    P_gtr_p_protons_data_cut.Fill(event.P_gtr_p)
    P_dc_x_fp_protons_data_cut.Fill(event.P_dc_x_fp)
    P_dc_y_fp_protons_data_cut.Fill(event.P_dc_y_fp)
    P_dc_xp_fp_protons_data_cut.Fill(event.P_dc_xp_fp)
    P_dc_yp_fp_protons_data_cut.Fill(event.P_dc_yp_fp)
    P_kin_MMp_protons_data_cut.Fill(event.MMp)
    P_kin_secondary_pmiss_protons_data_cut.Fill(event.pmiss)
    P_kin_secondary_pmiss_x_protons_data_cut.Fill(event.pmiss_x)
    P_kin_secondary_pmiss_y_protons_data_cut.Fill(event.pmiss_y)
    P_kin_secondary_pmiss_z_protons_data_cut.Fill(event.pmiss_z)
    P_kin_secondary_emiss_protons_data_cut.Fill(event.emiss)
    P_kin_secondary_W_protons_data_cut.Fill(event.W)
    P_rb_x_protons_data_cut.Fill(event.raster_x)
    P_rb_y_protons_data_cut.Fill(event.raster_y)
    P_rb_raster_fr_xbpm_tar_protons_data_cut.Fill(event.bpm_tar_x)
    P_rb_raster_fr_ybpm_tar_protons_data_cut.Fill(event.bpm_tar_y)
    P_kin_secondary_emiss_vs_H_gtr_dp_protons_data_cut.Fill(event.emiss , event.H_gtr_dp)
    P_kin_secondary_emiss_vs_P_gtr_dp_protons_data_cut.Fill(event.emiss , event.P_gtr_dp)
    P_kin_secondary_pmiss_vs_H_gtr_dp_protons_data_cut.Fill(event.pmiss , event.H_gtr_dp)
    P_kin_secondary_pmiss_vs_P_gtr_dp_protons_data_cut.Fill(event.pmiss , event.P_gtr_dp)
    P_gtr_dp_vs_H_gtr_dp_protons_data_cut.Fill(event.H_gtr_dp , event.P_gtr_dp)
    P_rb_x_vs_P_rb_y_protons_data_cut.Fill(event.raster_x , event.raster_y)
    P_rb_x_vs_P_kin_secondary_emiss_protons_data_cut.Fill(event.raster_x , event.emiss)
    P_rb_x_vs_P_kin_secondary_pmiss_protons_data_cut.Fill(event.raster_x , event.pmiss)
    P_rb_y_vs_P_kin_secondary_emiss_protons_data_cut.Fill(event.raster_y , event.emiss)
    P_rb_y_vs_P_kin_secondary_pmiss_protons_data_cut.Fill(event.raster_y , event.pmiss)
    P_rb_x_vs_P_kin_secondary_W_protons_data_cut.Fill(event.raster_x , event.W)
    P_rb_y_vs_P_kin_secondary_W_protons_data_cut.Fill(event.raster_y , event.W)

#################################################################################################################################################

ROOT.gROOT.SetBatch(ROOT.kTRUE) # Set ROOT to batch mode explicitly, does not splash anything to screen
# Removes stat boxi
ROOT.gStyle.SetOptStat(0)

# Saving histograms in PDF
c1_raster1 = TCanvas("c1_raster1", "Raster and Target Distributions", 100, 0, 1400, 600)
c1_raster1.Divide(2,2)
c1_raster1.cd(1)
P_rb_raster_fr_xbpm_tar_protons_data_uncut.Draw()
c1_raster1.cd(2)
P_rb_raster_fr_ybpm_tar_protons_data_uncut.Draw()
c1_raster1.cd(3)
P_rb_x_vs_P_rb_y_protons_data_uncut.Draw("COLZ")
c1_raster1.cd(4)
P_gtr_dp_vs_H_gtr_dp_protons_data_uncut.Draw("COLZ")
c1_raster1.Print(Proton_Analysis_Distributions_Uncut + '(')

c1_raster2 = TCanvas("c1_raster2", "Raster Distributions", 100, 0, 1400, 600)
c1_raster2.Divide(2,2)
c1_raster2.cd(1)
P_rb_x_vs_P_kin_secondary_emiss_protons_data_uncut.Draw("COLZ")
c1_raster2.cd(2)
P_rb_x_vs_P_kin_secondary_pmiss_protons_data_uncut.Draw("COLZ")
c1_raster2.cd(3)
P_rb_y_vs_P_kin_secondary_emiss_protons_data_uncut.Draw("COLZ")
c1_raster2.cd(4)
P_rb_y_vs_P_kin_secondary_pmiss_protons_data_uncut.Draw("COLZ")
c1_raster2.Print(Proton_Analysis_Distributions_Uncut)

c1_delta = TCanvas("c1_delta", "Delta and Target Distributions", 100, 0, 1400, 600)
c1_delta.Divide(2,2)
c1_delta.cd(1)
P_kin_secondary_emiss_vs_H_gtr_dp_protons_data_uncut.Draw("COLZ")
c1_delta.cd(2)
P_kin_secondary_emiss_vs_P_gtr_dp_protons_data_uncut.Draw("COLZ")
c1_delta.cd(3)
P_kin_secondary_pmiss_vs_H_gtr_dp_protons_data_uncut.Draw("COLZ")
c1_delta.cd(4)
P_kin_secondary_pmiss_vs_P_gtr_dp_protons_data_uncut.Draw("COLZ")
c1_delta.Print(Proton_Analysis_Distributions_Uncut )

c1_W = TCanvas("c1_W", "Delta and Target Distributions", 100, 0, 1400, 600)
c1_W.Divide(2,2)
c1_W.cd(1)
P_rb_x_vs_P_kin_secondary_W_protons_data_uncut.Draw("COLZ")
c1_W.cd(2)
P_rb_y_vs_P_kin_secondary_W_protons_data_uncut.Draw("COLZ")
c1_W.cd(3)

c1_W.cd(4)

c1_W.Print(Proton_Analysis_Distributions_Uncut + ')')

#-------------------------------------------------------------------------------------------------------------------------------------

c1_raster1p = TCanvas("c1_raster1p", "Raster and Target Distributions", 100, 0, 1400, 600)
c1_raster1p.Divide(2,2)
c1_raster1p.cd(1)
P_rb_raster_fr_xbpm_tar_protons_data_cut.Draw()
c1_raster1p.cd(2)
P_rb_raster_fr_ybpm_tar_protons_data_cut.Draw()
c1_raster1p.cd(3)
P_rb_x_vs_P_rb_y_protons_data_cut.Draw("COLZ")
c1_raster1p.cd(4)
P_gtr_dp_vs_H_gtr_dp_protons_data_cut.Draw("COLZ")
c1_raster1p.Print(Proton_Analysis_Distributions_Cut + '(')

c1_raster2p = TCanvas("c1_raster2p", "Raster Distributions", 100, 0, 1400, 600)
c1_raster2p.Divide(2,2)
c1_raster2p.cd(1)
P_rb_x_vs_P_kin_secondary_emiss_protons_data_cut.Draw("COLZ")
c1_raster2p.cd(2)
P_rb_x_vs_P_kin_secondary_pmiss_protons_data_cut.Draw("COLZ")
c1_raster2p.cd(3)
P_rb_y_vs_P_kin_secondary_emiss_protons_data_cut.Draw("COLZ")
c1_raster2p.cd(4)
P_rb_y_vs_P_kin_secondary_pmiss_protons_data_cut.Draw("COLZ")
c1_raster2p.Print(Proton_Analysis_Distributions_Cut)

c1_deltap = TCanvas("c1_deltap", "Delta and Target Distributions", 100, 0, 1400, 600)
c1_deltap.Divide(2,2)
c1_deltap.cd(1)
P_kin_secondary_emiss_vs_H_gtr_dp_protons_data_cut.Draw("COLZ")
c1_deltap.cd(2)
P_kin_secondary_emiss_vs_P_gtr_dp_protons_data_cut.Draw("COLZ")
c1_deltap.cd(3)
P_kin_secondary_pmiss_vs_H_gtr_dp_protons_data_cut.Draw("COLZ")
c1_deltap.cd(4)
P_kin_secondary_pmiss_vs_P_gtr_dp_protons_data_cut.Draw("COLZ")
c1_deltap.Print(Proton_Analysis_Distributions_Cut )

c1_Wp = TCanvas("c1_Wp", "Delta and Target Distributions", 100, 0, 1400, 600)
c1_Wp.Divide(2,2)
c1_Wp.cd(1)
P_rb_x_vs_P_kin_secondary_W_protons_data_cut.Draw("COLZ")
c1_Wp.cd(2)
P_rb_y_vs_P_kin_secondary_W_protons_data_cut.Draw("COLZ")
c1_Wp.cd(3)

c1_Wp.cd(4)

c1_Wp.Print(Proton_Analysis_Distributions_Cut + ')')

#############################################################################################################################################

# Making directories in output file
outHistFile = ROOT.TFile.Open("%s/test/%s_Output_RasterCheck_Data.root" % (OUTPATH, BEAM_ENERGY) , "RECREATE")
d_Uncut_Proton_Events_Data = outHistFile.mkdir("Uncut_Proton_Events_Data")
d_Uncut_2D_Proton_Events_Data = outHistFile.mkdir("Uncut_2D_Proton_Events_Data")
d_Cut_Proton_Events_Data = outHistFile.mkdir("Cut_Proton_Events_Data")
d_Cut_2D_Proton_Events_Data = outHistFile.mkdir("Cut_2D_Proton_Events_Data")

# Writing Histograms for protons                                                                  
d_Uncut_Proton_Events_Data.cd()
H_gtr_xp_protons_data_uncut.Write()
H_gtr_yp_protons_data_uncut.Write()
H_gtr_dp_protons_data_uncut.Write()
H_gtr_p_protons_data_uncut.Write()
H_dc_x_fp_protons_data_uncut.Write()
H_dc_y_fp_protons_data_uncut.Write()
H_dc_xp_fp_protons_data_uncut.Write()
H_dc_yp_fp_protons_data_uncut.Write()
P_gtr_xp_protons_data_uncut.Write()
P_gtr_yp_protons_data_uncut.Write()
P_gtr_dp_protons_data_uncut.Write()
P_gtr_p_protons_data_uncut.Write()
P_dc_x_fp_protons_data_uncut.Write()
P_dc_y_fp_protons_data_uncut.Write()
P_dc_xp_fp_protons_data_uncut.Write()
P_dc_yp_fp_protons_data_uncut.Write()
P_kin_MMp_protons_data_uncut.Write()
P_kin_secondary_pmiss_protons_data_uncut.Write()
P_kin_secondary_pmiss_x_protons_data_uncut.Write()
P_kin_secondary_pmiss_y_protons_data_uncut.Write()
P_kin_secondary_pmiss_z_protons_data_uncut.Write()
P_kin_secondary_emiss_protons_data_uncut.Write()
P_kin_secondary_W_protons_data_uncut.Write()
P_rb_x_protons_data_uncut.Write()
P_rb_y_protons_data_uncut.Write()
P_rb_raster_fr_xbpm_tar_protons_data_uncut.Write()
P_rb_raster_fr_ybpm_tar_protons_data_uncut.Write()

d_Uncut_2D_Proton_Events_Data.cd()
P_kin_secondary_emiss_vs_H_gtr_dp_protons_data_uncut.Write()
P_kin_secondary_emiss_vs_P_gtr_dp_protons_data_uncut.Write()
P_kin_secondary_pmiss_vs_H_gtr_dp_protons_data_uncut.Write()
P_kin_secondary_pmiss_vs_P_gtr_dp_protons_data_uncut.Write()
P_gtr_dp_vs_H_gtr_dp_protons_data_uncut.Write()
P_rb_x_vs_P_rb_y_protons_data_uncut.Write()
P_rb_x_vs_P_kin_secondary_emiss_protons_data_uncut.Write()
P_rb_x_vs_P_kin_secondary_pmiss_protons_data_uncut.Write()
P_rb_y_vs_P_kin_secondary_emiss_protons_data_uncut.Write()
P_rb_y_vs_P_kin_secondary_pmiss_protons_data_uncut.Write()
P_rb_x_vs_P_kin_secondary_W_protons_data_uncut.Write()
P_rb_y_vs_P_kin_secondary_W_protons_data_uncut.Write()

d_Cut_Proton_Events_Data.cd()
H_gtr_xp_protons_data_cut.Write()
H_gtr_yp_protons_data_cut.Write()
H_gtr_dp_protons_data_cut.Write()
H_gtr_p_protons_data_cut.Write()
H_dc_x_fp_protons_data_cut.Write()
H_dc_y_fp_protons_data_cut.Write()
H_dc_xp_fp_protons_data_cut.Write()
H_dc_yp_fp_protons_data_cut.Write()
P_gtr_xp_protons_data_cut.Write()
P_gtr_yp_protons_data_cut.Write()
P_gtr_dp_protons_data_cut.Write()
P_gtr_p_protons_data_cut.Write()
P_dc_x_fp_protons_data_cut.Write()
P_dc_y_fp_protons_data_cut.Write()
P_dc_xp_fp_protons_data_cut.Write()
P_dc_yp_fp_protons_data_cut.Write()
P_kin_MMp_protons_data_cut.Write()
P_kin_secondary_pmiss_protons_data_cut.Write()
P_kin_secondary_pmiss_x_protons_data_cut.Write()
P_kin_secondary_pmiss_y_protons_data_cut.Write()
P_kin_secondary_pmiss_z_protons_data_cut.Write()
P_kin_secondary_emiss_protons_data_cut.Write()
P_kin_secondary_W_protons_data_cut.Write()
P_rb_x_protons_data_cut.Write()
P_rb_y_protons_data_cut.Write()
P_rb_raster_fr_xbpm_tar_protons_data_cut.Write()
P_rb_raster_fr_ybpm_tar_protons_data_cut.Write()

d_Cut_2D_Proton_Events_Data.cd()
P_kin_secondary_emiss_vs_H_gtr_dp_protons_data_cut.Write()
P_kin_secondary_emiss_vs_P_gtr_dp_protons_data_cut.Write()
P_kin_secondary_pmiss_vs_H_gtr_dp_protons_data_cut.Write()
P_kin_secondary_pmiss_vs_P_gtr_dp_protons_data_cut.Write()
P_gtr_dp_vs_H_gtr_dp_protons_data_cut.Write()
P_rb_x_vs_P_rb_y_protons_data_cut.Write()
P_rb_x_vs_P_kin_secondary_emiss_protons_data_cut.Write()
P_rb_x_vs_P_kin_secondary_pmiss_protons_data_cut.Write()
P_rb_y_vs_P_kin_secondary_emiss_protons_data_cut.Write()
P_rb_y_vs_P_kin_secondary_pmiss_protons_data_cut.Write()
P_rb_x_vs_P_kin_secondary_W_protons_data_cut.Write()
P_rb_y_vs_P_kin_secondary_W_protons_data_cut.Write()

infile_DATA.Close() 
outHistFile.Close()

print ("Processing Complete")
