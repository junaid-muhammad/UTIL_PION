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
    print("!!!!! ERROR !!!!!\n Expected 2 arguments\n Usage is with - Run Number\n!!!!! ERROR !!!!!")
    sys.exit(1)

##################################################################################################################################################

# Input params - run number and max number of events
Run_Number = sys.argv[1]

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
Proton_Analysis_Distributions = "%s/test/%s_HeePCoin_Raster_Check_Distributions.pdf" % (OUTPATH, Run_Number)

# Input file location and variables taking
rootFile_DATA = "%s/test/PionLT_HeePCoin_replay_production_%s_-1.root" % (OUTPATH, Run_Number)

###############################################################################################################################################
ROOT.gROOT.SetBatch(ROOT.kTRUE) # Set ROOT to batch mode explicitly, does not splash anything to screen
###############################################################################################################################################

# Read stuff from the main event tree
Infile_DATA = up.open(rootFile_DATA)["T"]
#Infile_DATA = ROOT.TFile.Open(rootFile_DATA, "READ")
#Infile_DATA= infile_DATA.Get("T")

H_gtr_xp = Infile_DATA["H.gtr.th"].array()
H_gtr_yp = Infile_DATA["H.gtr.ph"].array()
H_gtr_dp = Infile_DATA["H.gtr.dp"].array()              # dp is Delta
H_gtr_p = Infile_DATA["H.gtr.p"].array()
H_dc_x_fp = Infile_DATA["H.dc.x_fp"].array()
H_dc_y_fp = Infile_DATA["H.dc.y_fp"].array()
H_dc_xp_fp = Infile_DATA["H.dc.xp_fp"].array()
H_dc_yp_fp = Infile_DATA["H.dc.yp_fp"].array()
P_gtr_xp = Infile_DATA["P.gtr.th"].array()
P_gtr_yp = Infile_DATA["P.gtr.ph"].array()
P_gtr_dp = Infile_DATA["P.gtr.dp"].array()              # dp is Delta
P_gtr_p = Infile_DATA["P.gtr.p"].array()
P_dc_x_fp = Infile_DATA["P.dc.x_fp"].array()
P_dc_y_fp = Infile_DATA["P.dc.y_fp"].array()
P_dc_xp_fp = Infile_DATA["P.dc.xp_fp"].array()
P_dc_yp_fp = Infile_DATA["P.dc.yp_fp"].array()
MMp = Infile_DATA["P.kin.secondary.MMp"].array()
emiss = Infile_DATA["P.kin.secondary.emiss"].array()       
pmiss = Infile_DATA["P.kin.secondary.pmiss"].array()
pmiss_x = Infile_DATA["P.kin.secondary.pmiss_x"].array()            
pmiss_y = Infile_DATA["P.kin.secondary.pmiss_y"].array()        
pmiss_z = Infile_DATA["P.kin.secondary.pmiss_z"].array()        
W = Infile_DATA["H.kin.primary.W"].array() 
P_rb_x = Infile_DATA["P.rb.x"].array()
P_rb_y = Infile_DATA["P.rb.y"].array()
P_rb_raster_fr_xbpm_tar = Infile_DATA["P.rb.raster.fr_xbpm_tar"].array()
P_rb_raster_fr_ybpm_tar = Infile_DATA["P.rb.raster.fr_ybpm_tar"].array()

###################################################################################################################################################

# Defining Histograms for Protons
# 1D Uncut Data Histograms
H_gtr_xp_protons_data_uncut = ROOT.TH1D("H_gtr_xp_protons_data_uncut", "HMS xptar; HMS_gtr_xptar; Counts", 200, -0.2, 0.2)
H_gtr_yp_protons_data_uncut = ROOT.TH1D("H_gtr_yp_protons_data_uncut", "HMS yptar; HMS_gtr_yptar; Counts", 200, -0.2, 0.2)
H_gtr_dp_protons_data_uncut = ROOT.TH1D("H_gtr_dp_protons_data_uncut", "HMS #delta; HMS_gtr_dp; Counts", 200, -15, 15)
H_gtr_p_protons_data_uncut = ROOT.TH1D("H_gtr_p_protons_data_uncut", "HMS p; HMS_gtr_p; Counts", 200, 4, 8)
H_dc_x_fp_protons_data_uncut = ROOT.TH1D("H_dc_x_fp_protons_data_uncut", "HMS x_fp'; HMS_dc_x_fp; Counts", 200, -60, 60)
H_dc_y_fp_protons_data_uncut = ROOT.TH1D("H_dc_y_fp_protons_data_uncut", "HMS y_fp'; HMS_dc_y_fp; Counts", 200, -40, 40)
H_dc_xp_fp_protons_data_uncut = ROOT.TH1D("H_dc_xp_fp_protons_data_uncut", "HMS xp_fp'; HMS_dc_xp_fp; Counts", 200, -0.2, 0.2)
H_dc_yp_fp_protons_data_uncut = ROOT.TH1D("H_dc_yp_fp_protons_data_uncut", "HMS yp_fp'; HMS_dc_yp_fp; Counts", 200, -0.2, 0.2)
P_gtr_xp_protons_data_uncut = ROOT.TH1D("P_gtr_xp_protons_data_uncut", "SHMS xptar; SHMS_gtr_xptar; Counts", 200, -0.2, 0.2)
P_gtr_yp_protons_data_uncut = ROOT.TH1D("P_gtr_yp_protons_data_uncut", "SHMS yptar; SHMS_gtr_yptar; Counts", 200, -0.2, 0.2)
P_gtr_dp_protons_data_uncut = ROOT.TH1D("P_gtr_dp_protons_data_uncut", "SHMS delta; SHMS_gtr_dp; Counts", 200, -30, 30)
P_gtr_p_protons_data_uncut = ROOT.TH1D("P_gtr_p_protons_data_uncut", "SHMS p; SHMS_gtr_p; Counts", 200, 4, 7)
P_dc_x_fp_protons_data_uncut = ROOT.TH1D("P_dc_x_fp_protons_data_uncut", "SHMS x_fp'; SHMS_dc_x_fp; Counts", 200, -60, 60)
P_dc_y_fp_protons_data_uncut = ROOT.TH1D("P_dc_y_fp_protons_data_uncut", "SHMS y_fp'; SHMS_dc_y_fp; Counts", 200, -40, 40)
P_dc_xp_fp_protons_data_uncut = ROOT.TH1D("P_dc_xp_fp_protons_data_uncut", "SHMS xp_fp'; SHMS_dc_xp_fp; Counts", 200, -0.2, 0.2)
P_dc_yp_fp_protons_data_uncut = ROOT.TH1D("P_dc_yp_fp_protons_data_uncut", "SHMS yp_fp'; SHMS_dc_yp_fp; Counts", 200, -0.2, 0.2)
P_kin_MMp_protons_data_uncut = ROOT.TH1D("P_kin_MMp_protons_data_uncut", "MIssing Mass data uncut; MM_{p}; Counts", 200, -1., 1.)
P_kin_secondary_pmiss_protons_data_uncut = ROOT.TH1D("P_kin_secondary_pmiss_protons_data_uncut", "Momentum Distribution; pmiss; Counts", 200, -1, 1)
P_kin_secondary_pmiss_x_protons_data_uncut = ROOT.TH1D("P_kin_secondary_pmiss_x_protons_data_uncut", "Momentum_x Distribution; pmiss_x; Counts", 200, -0.6, 0.6)
P_kin_secondary_pmiss_y_protons_data_uncut = ROOT.TH1D("P_kin_secondary_pmiss_y_protons_data_uncut", "Momentum_y Distribution; pmiss_y; Counts", 200, -0.6, 0.6)
P_kin_secondary_pmiss_z_protons_data_uncut = ROOT.TH1D("P_kin_secondary_pmiss_z_protons_data_uncut", "Momentum_z Distribution; pmiss_z; Counts", 200, -1, 1)
P_kin_secondary_emiss_protons_data_uncut = ROOT.TH1D("P_kin_secondary_emiss_protons_data_uncut", "Emiss Distribution; emiss; Counts", 200, -1, 1)
P_kin_secondary_W_protons_data_uncut = ROOT.TH1D("P_kin_secondary_W_protons_data_uncut", "W Distribution; W; Counts", 200, 0, 2)
P_rb_x_protons_data_uncut = ROOT.TH1D("P_rb_x_protons_data_uncut", "Raster x Distribution; Raster x; Counts", 200, -0.4, 0.4)
P_rb_y_protons_data_uncut = ROOT.TH1D("P_rb_y_protons_data_uncut", "Raster y Distribution; Raster y; Counts", 200, -0.4, 0.4)
P_rb_raster_fr_xbpm_tar_protons_data_uncut = ROOT.TH1D("P_rb_raster_fr_xbpm_tar_protons_data_uncut", "Raster xbpm tar Distribution; Raster xbpm tar; Counts", 200, -0.06, 0.0)
P_rb_raster_fr_ybpm_tar_protons_data_uncut = ROOT.TH1D("P_rb_raster_fr_ybpm_tar_protons_data_uncut", "Raster ybpm tar Distribution; Raster ybpm tar; Counts", 200, -0.03, 0.0)

# 2D Uncut Data Histograms
P_kin_secondary_emiss_vs_H_gtr_dp_protons_data_uncut = ROOT.TH2D("P_kin_secondary_emiss_vs_H_gtr_dp_protons_data_uncut", "emiss vs HMS delta; emiss; HMS delta",200, -0.2, 0.2, 200, -10, 10)
P_kin_secondary_emiss_vs_P_gtr_dp_protons_data_uncut = ROOT.TH2D("P_kin_secondary_emiss_vs_P_gtr_dp_protons_data_uncut", "emiss vs SHMS delta; emiss; SHMS delta",200, -0.2, 0.2, 200, -10, 10)
P_kin_secondary_pmiss_vs_H_gtr_dp_protons_data_uncut = ROOT.TH2D("P_kin_secondary_pmiss_vs_H_gtr_dp_protons_data_uncut", "pmiss vs HMS delta; pmiss; HMS delta",200, -0.2, 0.2, 200, -10, 10)
P_kin_secondary_pmiss_vs_P_gtr_dp_protons_data_uncut = ROOT.TH2D("P_kin_secondary_pmiss_vs_P_gtr_dp_protons_data_uncut", "pmiss vs SHMS delta; pmiss; SHMS delta",200, -0.2, 0.2, 200, -10, 10)
P_gtr_dp_vs_H_gtr_dp_protons_data_uncut = ROOT.TH2D("P_gtr_dp_vs_H_gtr_dp_protons_data_uncut", "SHMS delta vs HMS delta; SHMS delta; HMS delta",200, -10, 10, 200, -10, 10)
P_rb_x_vs_P_rb_y_protons_data_uncut = ROOT.TH2D("P_rb_x_vs_P_rb_y_protons_data_uncut", "Raster x vs Raster y; Raster x; Raster y",200, -0.2, 0.2, 200, -0.2, 0.2)
P_rb_x_vs_P_kin_secondary_emiss_protons_data_uncut = ROOT.TH2D("P_rb_x_vs_P_kin_secondary_emiss_protons_data_uncut", "Raster x vs emiss; Raster x; emiss",200, -0.2, 0.2, 200, -0.2, 0.2)
P_rb_x_vs_P_kin_secondary_pmiss_protons_data_uncut = ROOT.TH2D("P_rb_x_vs_P_kin_secondary_pmiss_protons_data_uncut", "Raster x vs pmiss; Raster x; pmiss",200, -0.2, 0.2, 200, -0.2, 0.2)
P_rb_y_vs_P_kin_secondary_emiss_protons_data_uncut = ROOT.TH2D("P_rb_y_vs_P_kin_secondary_emiss_protons_data_uncut", "Raster y vs emiss; Raster y; emiss",200, -0.2, 0.2, 200, -0.2, 0.2)
P_rb_y_vs_P_kin_secondary_pmiss_protons_data_uncut = ROOT.TH2D("P_rb_y_vs_P_kin_secondary_pmiss_protons_data_uncut", "Raster y vs pmiss; Raster y; pmiss",200, -0.2, 0.2, 200, -0.2, 0.2)

#################################################################################################################################################

for H_xp, H_yp, H_dp, H_p, H_xfp, H_yfp, H_xpfp, H_ypfp, P_xp, P_yp, P_dp, P_p, P_xfp, P_yfp, P_xpfp, P_ypfp, mmp, em, pm, pm_x, pm_y, pm_z, w, rb_x, rb_y, rb_xbpm_tar, rb_ybpm_tar in zip (H_gtr_xp, H_gtr_yp, H_gtr_dp, H_gtr_p, H_dc_x_fp, H_dc_y_fp, H_dc_xp_fp, H_dc_yp_fp, P_gtr_xp, P_gtr_yp, P_gtr_dp, P_gtr_p, P_dc_x_fp, P_dc_y_fp, P_dc_xp_fp, P_dc_yp_fp, MMp, emiss, pmiss, pmiss_x, pmiss_y, pmiss_z, W, P_rb_x, P_rb_y, P_rb_raster_fr_xbpm_tar, P_rb_raster_fr_ybpm_tar):

    H_gtr_xp_protons_data_uncut.Fill(H_xp)
    H_gtr_yp_protons_data_uncut.Fill(H_yp)
    H_gtr_dp_protons_data_uncut.Fill(H_dp)
    H_gtr_p_protons_data_uncut.Fill(H_p)
    H_dc_x_fp_protons_data_uncut.Fill(H_xfp)
    H_dc_y_fp_protons_data_uncut.Fill(H_yfp)
    H_dc_xp_fp_protons_data_uncut.Fill(H_xpfp)
    H_dc_yp_fp_protons_data_uncut.Fill(H_ypfp)
    P_gtr_xp_protons_data_uncut.Fill(P_xp)
    P_gtr_yp_protons_data_uncut.Fill(P_yp)
    P_gtr_dp_protons_data_uncut.Fill(P_dp)
    P_gtr_p_protons_data_uncut.Fill(P_p)
    P_dc_x_fp_protons_data_uncut.Fill(P_xfp)
    P_dc_y_fp_protons_data_uncut.Fill(P_yfp)
    P_dc_xp_fp_protons_data_uncut.Fill(P_xpfp)
    P_dc_yp_fp_protons_data_uncut.Fill(P_ypfp)
    P_kin_MMp_protons_data_uncut.Fill(mmp)
    P_kin_secondary_pmiss_protons_data_uncut.Fill(pm)
    P_kin_secondary_pmiss_x_protons_data_uncut.Fill(pm_x)
    P_kin_secondary_pmiss_y_protons_data_uncut.Fill(pm_y)
    P_kin_secondary_pmiss_z_protons_data_uncut.Fill(pm_z)
    P_kin_secondary_emiss_protons_data_uncut.Fill(em)
    P_kin_secondary_W_protons_data_uncut.Fill(w)
    P_rb_x_protons_data_uncut.Fill(rb_x)
    P_rb_y_protons_data_uncut.Fill(rb_y)
    P_rb_raster_fr_xbpm_tar_protons_data_uncut.Fill(rb_xbpm_tar)
    P_rb_raster_fr_ybpm_tar_protons_data_uncut.Fill(rb_ybpm_tar)
    P_kin_secondary_emiss_vs_H_gtr_dp_protons_data_uncut.Fill(em, H_dp)
    P_kin_secondary_emiss_vs_P_gtr_dp_protons_data_uncut.Fill(em, P_dp)
    P_kin_secondary_pmiss_vs_H_gtr_dp_protons_data_uncut.Fill(pm, H_dp)
    P_kin_secondary_pmiss_vs_P_gtr_dp_protons_data_uncut.Fill(pm, P_dp)
    P_gtr_dp_vs_H_gtr_dp_protons_data_uncut.Fill(P_dp, H_dp)
    P_rb_x_vs_P_rb_y_protons_data_uncut.Fill(rb_x, rb_y)
    P_rb_x_vs_P_kin_secondary_emiss_protons_data_uncut.Fill(rb_x, em)
    P_rb_x_vs_P_kin_secondary_pmiss_protons_data_uncut.Fill(rb_x, pm)
    P_rb_y_vs_P_kin_secondary_emiss_protons_data_uncut.Fill(rb_y, em)
    P_rb_y_vs_P_kin_secondary_pmiss_protons_data_uncut.Fill(rb_y, pm)


#################################################################################################################################################

# Removes stat box
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
c1_raster1.Print(Proton_Analysis_Distributions + '(')

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
c1_raster2.Print(Proton_Analysis_Distributions)

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
c1_delta.Print(Proton_Analysis_Distributions + ')')

#############################################################################################################################################

# Making directories in output file
OutHistFile = ROOT.TFile.Open("%s/test/%s_Output_Data.root" % (OUTPATH, Run_Number) , "RECREATE")
d_Uncut_Proton_Events_Data = OutHistFile.mkdir("Uncut_Proton_Events_Data")
d_2D_Uncut_Proton_Events_Data = OutHistFile.mkdir("Cut_Proton_Events_All_Data")

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

d_2D_Uncut_Proton_Events_Data.cd()
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

#rootFile_DATA.Close() 
OutHistFile.Close()

print ("Processing Complete")
