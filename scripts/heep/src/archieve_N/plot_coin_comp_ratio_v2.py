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

# Defining some constants here
#minbin = 0.0 # minbin for selecting neutrons events in missing mass distribution
#maxbin = 0.05 # maxbin for selecting neutrons events in missing mass distribution

##################################################################################################################################################

# Check the number of arguments provided to the script
if len(sys.argv)-1!=8:
    print("!!!!! ERROR !!!!!\n Expected 8 arguments\n Usage is with - ROOTfileSuffixs Beam Energy MaxEvents RunList CVSFile\n!!!!! ERROR !!!!!")
    sys.exit(1)

##################################################################################################################################################

# Input params - run number and max number of events
BEAM_ENERGY = sys.argv[1]
MaxEvent = sys.argv[2]
DATA_Suffix = sys.argv[3]
DUMMY_Suffix = sys.argv[4]
SIMC_Suffix = sys.argv[5]
DATA_RUN_LIST = sys.argv[6]
DUMMY_RUN_LIST = sys.argv[7]
CSV_FILE = sys.argv[8]
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
SIMCPATH= "/volatile/hallc/c-pionlt/junaid/OUTPUT/Analysis/SIMC"
RUNLISTPATH="/u/group/c-pionlt/USERS/junaid/hallc_replay_lt/UTIL_BATCH/InputRunLists/heep_runlist"
CSVPATH="/u/group/c-pionlt/USERS/junaid/hallc_replay_lt/UTIL_PION/efficiencies"
#################################################################################################################################################

# Output PDF File Name
print("Running as %s on %s, hallc_replay_lt path assumed as %s" % (USER, HOST, REPLAYPATH))
Proton_Analysis_Distributions = "%s/%s_%s_HeePCoin_Proton_Analysis_Distributions.pdf" % (OUTPATH, BEAM_ENERGY, MaxEvent)

# Input file location and variables taking
rootFile_DATA = "%s/%s_%s_%s.root" % (OUTPATH, BEAM_ENERGY, MaxEvent, DATA_Suffix)
rootFile_DUMMY = "%s/%s_%s_%s.root" % (OUTPATH, BEAM_ENERGY, MaxEvent, DUMMY_Suffix)
rootFile_SIMC = "%s/%s_%s.root" % (SIMCPATH, BEAM_ENERGY, SIMC_Suffix)
data_run_list = "%s/%s" % (RUNLISTPATH, DATA_RUN_LIST)
dummy_run_list = "%s/%s" % (RUNLISTPATH, DUMMY_RUN_LIST)
csv_file = "%s/%s.csv" % (CSVPATH, CSV_FILE)
###############################################################################################################################################

# Read CSV File and calculate total charge in mC
# Input runlists and csv files
data_run_list_file = (data_run_list)
dummy_run_list_file = (dummy_run_list)
csv_file_name = (csv_file)

print ('\nBeam Energy = ',BEAM_ENERGY, '\n')
print("-"*40)

# Read run numbers from the run list file
with open(data_run_list_file, 'r') as run_list_file_1:
    data_runs = [line.strip() for line in run_list_file_1 if line.strip()]
with open(dummy_run_list_file, 'r') as run_list_file_2:
    dummy_runs = [line.strip() for line in run_list_file_2 if line.strip()]

# Read CSV file using pandas
df = pd.read_csv(csv_file_name)

# Filter DataFrame to include only rows with run numbers in the run list
filtered_data_df = df[df['Run_Number'].astype(str).str.replace('.0', '', regex=False).isin(data_runs)]
filtered_dummy_df = df[df['Run_Number'].astype(str).str.replace('.0', '', regex=False).isin(dummy_runs)]

# Print charge values for each run and calculate the product of efficiency and charge
total_data_effective_charge = 0
total_dummy_effective_charge = 0

# Detector (HMS Cer + HMS Cal) Efficiencies
hms_Cer_detector_efficiency = 0.9981
hms_Cal_detector_efficiency = 0.9981

# Print charge values for each run
for index, row in filtered_data_df.iterrows():
    data_charge = row['BCM2_Charge']  # Assuming 'BCM2_Charge' is a column in your CSV
    data_hms_tracking_efficiency = row['HMS_Elec_SING_TRACK_EFF']  # Assuming 'HMS Tracking_Efficiency' is a column in your CSV
    data_shms_tracking_efficiency = row['SHMS_Prot_SING_TRACK_EFF']  # Assuming 'SHMS Tracking_Efficiency' is a column in your CSV
    data_hms_hodo_3_of_4_efficiency = row['HMS_Hodo_3_of_4_EFF']
    data_shms_hodo_3_of_4_efficiency = row['SHMS_Hodo_3_of_4_EFF']
    data_edtm_livetime_Corr = row['Non_Scaler_EDTM_Live_Time']
    data_BCM2_Beam_Cut_Current = row['BCM2_Beam_Cut_Current']
    # Richard's and Vijay's Boiling Correction
    data_Boiling_factor = 1 + (-0.0007899 * data_BCM2_Beam_Cut_Current)
#    data_Boiling_factor = 1 + (-0.0005296 * data_BCM2_Beam_Cut_Current)
    data_product = (data_charge * data_hms_tracking_efficiency * data_shms_tracking_efficiency * hms_Cer_detector_efficiency * hms_Cal_detector_efficiency * data_hms_hodo_3_of_4_efficiency * data_shms_hodo_3_of_4_efficiency * data_edtm_livetime_Corr * data_Boiling_factor)
    total_data_effective_charge += data_product
    print('Charge for data run: {:<10} Data Charge: {:.3f} HMS Tracking Eff: {:.3f} SHMS Tracking Eff: {:.3f} HMS Cer Detector Eff: {:.3f} HMS Cal Detector_Eff: {:.3f} HMS Hodo 3/4 Eff: {:.3f} SHMS Hodo 3/4 Eff: {:.3f} EDTM Live Time: {:.3f} Data Current: {:.3f} Boiling Correction: {:.3f} Product: {:.3f}'.format(row["Run_Number"], data_charge, data_hms_tracking_efficiency, data_shms_tracking_efficiency, hms_Cer_detector_efficiency, hms_Cal_detector_efficiency, data_hms_hodo_3_of_4_efficiency, data_shms_hodo_3_of_4_efficiency, data_edtm_livetime_Corr, data_BCM2_Beam_Cut_Current, data_Boiling_factor, data_product))
print("-"*40)

for index, row in filtered_dummy_df.iterrows():
    dummy_charge = row['BCM2_Charge']  # Assuming 'BCM2_Charge' is a column in your CSV
    dummy_hms_tracking_efficiency = row['HMS_Elec_SING_TRACK_EFF']  # Assuming 'HMS Efficiency' is a column in your CSV
    dummy_shms_tracking_efficiency = row['SHMS_Prot_SING_TRACK_EFF']  # Assuming 'SHMS Efficiency' is a column in your CSV
    dummy_hms_hodo_3_of_4_efficiency = row['HMS_Hodo_3_of_4_EFF']
    dummy_shms_hodo_3_of_4_efficiency = row['SHMS_Hodo_3_of_4_EFF']
    dummy_edtm_livetime_Corr = row['Non_Scaler_EDTM_Live_Time']
    dummy_product = (dummy_charge * dummy_hms_tracking_efficiency * dummy_shms_tracking_efficiency * hms_Cer_detector_efficiency * hms_Cal_detector_efficiency * dummy_hms_hodo_3_of_4_efficiency * dummy_shms_hodo_3_of_4_efficiency * dummy_edtm_livetime_Corr)
    total_dummy_effective_charge += dummy_product
    print('Charge for dummy run: {:<10} Dummy Charge: {:.3f} HMS Tracking Eff: {:.3f} SHMS Tracking Eff: {:.3f} HMS Cer Detector Eff: {:.3f} HMS Cal Detector_Eff: {:.3f} HMS Hodo 3/4 Eff: {:.3f} SHMS Hodo 3/4 Eff: {:.3f} EDTM Live Time: {:.3f} Product: {:.3f}'.format(row["Run_Number"], dummy_charge, dummy_hms_tracking_efficiency, dummy_shms_tracking_efficiency, hms_Cer_detector_efficiency, hms_Cal_detector_efficiency, dummy_hms_hodo_3_of_4_efficiency, dummy_shms_hodo_3_of_4_efficiency, dummy_edtm_livetime_Corr, dummy_product))
print("-"*40)

print('\nTotal effective charge for the data run list: ',total_data_effective_charge)
print('\nTotal effective charge for the dummy run list: ',total_dummy_effective_charge)

###############################################################################################################################################
ROOT.gROOT.SetBatch(ROOT.kTRUE) # Set ROOT to batch mode explicitly, does not splash anything to screen
###############################################################################################################################################

# Grabs simc number of events and normalizaton factor
simc_hist = "%s/%s_%s.hist" % (SIMCPATH, BEAM_ENERGY, SIMC_Suffix)

f_simc = open(simc_hist)
for line in f_simc:
#    print(line)
    if "Ncontribute" in line:
        val = line.split("=")
        simc_nevents = int(val[1])
    if "normfac" in line:
        val = line.split("=")
        simc_normfactor = float(val[1])
if 'simc_nevents' and 'simc_normfactor' in locals():
    print('\nsimc_nevents = ',simc_nevents,'\nsimc_normfactor = ',simc_normfactor,'\n')
#    print('\n\ndata_charge = {:.4f} +/- {:.4f}'.format(data_charge, data_charge*eff_errProp_data),'\ndummy_charge = {:.4f} +/- {:.4f}'.format(dummy_charge, dummy_charge*eff_errProp_dummy),'\n\n')
else:
    print("ERROR: Invalid simc hist file %s" % simc_hist)
    sys.exit(1)
f_simc.close()
print("-"*40)

##################################################################################################################################################
# Read stuff from the main event tree

infile_DATA = ROOT.TFile.Open(rootFile_DATA, "READ")
infile_DUMMY = ROOT.TFile.Open(rootFile_DUMMY, "READ")
infile_SIMC = ROOT.TFile.Open(rootFile_SIMC, "READ")

#Uncut_Proton_Events_Data_tree = infile_DATA.Get("Uncut_Proton_Events")
#Cut_Proton_Events_All_Data_tree = infile_DATA.Get("Cut_Proton_Events_All")
Cut_Proton_Events_Prompt_Data_tree = infile_DATA.Get("Cut_Proton_Events_Prompt")
nEntries_TBRANCH_DATA  = Cut_Proton_Events_Prompt_Data_tree.GetEntries()

#Uncut_Proton_Events_Dummy_tree = infile_DUMMY.Get("Uncut_Proton_Events")
#Cut_Proton_Events_All_Dummy_tree = infile_DUMMY.Get("Cut_Proton_Events_All")
Cut_Proton_Events_Prompt_Dummy_tree = infile_DUMMY.Get("Cut_Proton_Events_Prompt")
nEntries_TBRANCH_DUMMY  = Cut_Proton_Events_Prompt_Dummy_tree.GetEntries()

Uncut_Proton_Events_SIMC_tree = infile_SIMC.Get("h10")
nEntries_TBRANCH_SIMC  = Uncut_Proton_Events_SIMC_tree.GetEntries()

###################################################################################################################################################

# Defining Histograms for Protons
# Cut (Acceptance + PID + Prompt Selection) Dummy Subtraction Data Histograms
H_gtr_beta_protons_dummysub_data_cut_all = ROOT.TH1D("H_gtr_beta_protons_dummysub_data_cut_all", "HMS #beta; HMS_gtr_#beta; Counts", 200, 0.8, 1.2)
H_gtr_xp_protons_dummysub_data_cut_all = ROOT.TH1D("H_gtr_xp_protons_dummysub_data_cut_all", "HMS xptar; HMS_gtr_xptar; Counts", 200, -0.2, 0.2)
H_gtr_yp_protons_dummysub_data_cut_all = ROOT.TH1D("H_gtr_yp_protons_dummysub_data_cut_all", "HMS yptar; HMS_gtr_yptar; Counts", 200, -0.2, 0.2)
H_gtr_dp_protons_dummysub_data_cut_all = ROOT.TH1D("H_gtr_dp_protons_dummysub_data_cut_all", "HMS #delta; HMS_gtr_dp; Counts", 200, -15, 15)
H_gtr_p_protons_dummysub_data_cut_all = ROOT.TH1D("H_gtr_p_protons_dummysub_data_cut_all", "HMS p; HMS_gtr_p; Counts", 200, 4, 8)
H_dc_x_fp_protons_dummysub_data_cut_all = ROOT.TH1D("H_dc_x_fp_protons_dummysub_data_cut_all", "HMS x_fp'; HMS_dc_x_fp; Counts", 200, -100, 100)
H_dc_y_fp_protons_dummysub_data_cut_all = ROOT.TH1D("H_dc_y_fp_protons_dummysub_data_cut_all", "HMS y_fp'; HMS_dc_y_fp; Counts", 200, -100, 100)
H_dc_xp_fp_protons_dummysub_data_cut_all = ROOT.TH1D("H_dc_xp_fp_protons_dummysub_data_cut_all", "HMS xp_fp'; HMS_dc_xp_fp; Counts", 200, -0.2, 0.2)
H_dc_yp_fp_protons_dummysub_data_cut_all = ROOT.TH1D("H_dc_yp_fp_protons_dummysub_data_cut_all", "HMS yp_fp'; HMS_dc_yp_fp; Counts", 200, -0.2, 0.2)
H_hod_goodscinhit_protons_dummysub_data_cut_all = ROOT.TH1D("H_hod_goodscinhit_protons_dummysub_data_cut_all", "HMS hod goodscinhit; HMS_hod_goodscinhi; Counts", 200, 0.7, 1.3)
H_hod_goodstarttime_protons_dummysub_data_cut_all = ROOT.TH1D("H_hod_goodstarttime_protons_dummysub_data_cut_all", "HMS hod goodstarttime; HMS_hod_goodstarttime; Counts", 200, 0.7, 1.3)
H_cal_etotnorm_protons_dummysub_data_cut_all = ROOT.TH1D("H_cal_etotnorm_protons_dummysub_data_cut_all", "HMS cal etotnorm; HMS_cal_etotnorm; Counts", 200, 0.2, 1.8)
H_cal_etottracknorm_protons_dummysub_data_cut_all = ROOT.TH1D("H_cal_etottracknorm_protons_dummysub_data_cut_all", "HMS cal etottracknorm; HMS_cal_etottracknorm; Counts", 200, 0.2, 1.8)
H_cer_npeSum_protons_dummysub_data_cut_all = ROOT.TH1D("H_cer_npeSum_protons_dummysub_data_cut_all", "HMS cer npeSum; HMS_cer_npeSum; Counts", 200, 0, 50)
H_RFTime_Dist_protons_dummysub_data_cut_all = ROOT.TH1D("H_RFTime_Dist_protons_dummysub_data_cut_all", "HMS RFTime; HMS_RFTime; Counts", 200, 0, 4)
P_gtr_beta_protons_dummysub_data_cut_all = ROOT.TH1D("P_gtr_beta_protons_dummysub_data_cut_all", "SHMS #beta; SHMS_gtr_#beta; Counts", 200, 0.5, 1.3)
P_gtr_xp_protons_dummysub_data_cut_all = ROOT.TH1D("P_gtr_xp_protons_dummysub_data_cut_all", "SHMS xptar; SHMS_gtr_xptar; Counts", 200, -0.2, 0.2)
P_gtr_yp_protons_dummysub_data_cut_all = ROOT.TH1D("P_gtr_yp_protons_dummysub_data_cut_all", "SHMS yptar; SHMS_gtr_yptar; Counts", 200, -0.2, 0.2)
P_gtr_dp_protons_dummysub_data_cut_all = ROOT.TH1D("P_gtr_dp_protons_dummysub_data_cut_all", "SHMS #delta; SHMS_gtr_dp; Counts", 200, -30, 30)
P_gtr_p_protons_dummysub_data_cut_all = ROOT.TH1D("P_gtr_p_protons_dummysub_data_cut_all", "SHMS p; SHMS_gtr_p; Counts", 200, 1, 7)
P_dc_x_fp_protons_dummysub_data_cut_all = ROOT.TH1D("P_dc_x_fp_protons_dummysub_data_cut_all", "SHMS x_fp'; SHMS_dc_x_fp; Counts", 200, -100, 100)
P_dc_y_fp_protons_dummysub_data_cut_all = ROOT.TH1D("P_dc_y_fp_protons_dummysub_data_cut_all", "SHMS y_fp'; SHMS_dc_y_fp; Counts", 200, -100, 100)
P_dc_xp_fp_protons_dummysub_data_cut_all = ROOT.TH1D("P_dc_xp_fp_protons_dummysub_data_cut_all", "SHMS xp_fp'; SHMS_dc_xp_fp; Counts", 200, -0.2, 0.2)
P_dc_yp_fp_protons_dummysub_data_cut_all = ROOT.TH1D("P_dc_yp_fp_protons_dummysub_data_cut_all", "SHMS yp_fp'; SHMS_dc_yp_fp; Counts", 200, -0.2, 0.2)
P_hod_goodscinhit_protons_dummysub_data_cut_all = ROOT.TH1D("P_hod_goodscinhit_protons_dummysub_data_cut_all", "SHMS hod goodscinhit; SHMS_hod_goodscinhit; Counts", 200, 0.7, 1.3)
P_hod_goodstarttime_protons_dummysub_data_cut_all = ROOT.TH1D("P_hod_goodstarttime_protons_dummysub_data_cut_all", "SHMS hod goodstarttime; SHMS_hod_goodstarttime; Counts", 200, 0.7, 1.3)
P_cal_etotnorm_protons_dummysub_data_cut_all = ROOT.TH1D("P_cal_etotnorm_protons_dummysub_data_cut_all", "SHMS cal etotnorm; SHMS_cal_etotnorm; Counts", 200, 0, 1)
P_cal_etottracknorm_protons_dummysub_data_cut_all = ROOT.TH1D("P_cal_etottracknorm_protons_dummysub_data_cut_all", "SHMS cal etottracknorm; SHMS_cal_etottracknorm; Counts", 200, 0, 1.6)
P_hgcer_npeSum_protons_dummysub_data_cut_all = ROOT.TH1D("P_hgcer_npeSum_protons_dummysub_data_cut_all", "SHMS HGC npeSum; SHMS_hgcer_npeSum; Counts", 200, 0, 50)
P_hgcer_xAtCer_protons_dummysub_data_cut_all = ROOT.TH1D("P_hgcer_xAtCer_protons_dummysub_data_cut_all", "SHMS HGC xAtCer; SHMS_hgcer_xAtCer; Counts", 200, -60, 60)
P_hgcer_yAtCer_protons_dummysub_data_cut_all = ROOT.TH1D("P_hgcer_yAtCer_protons_dummysub_data_cut_all", "SHMS HGC yAtCer; SHMS_hgcer_yAtCer; Counts", 200, -50, 50)
P_ngcer_npeSum_protons_dummysub_data_cut_all = ROOT.TH1D("P_ngcer_npeSum_protons_dummysub_data_cut_all", "SHMS NGC npeSum; SHMS_ngcer_npeSum; Counts", 200, 0, 50)
P_ngcer_xAtCer_protons_dummysub_data_cut_all = ROOT.TH1D("P_ngcer_xAtCer_protons_dummysub_data_cut_all", "SHMS NGC xAtCer; SHMS_ngcer_xAtCer; Counts", 200, -60, 60)
P_ngcer_yAtCer_protons_dummysub_data_cut_all = ROOT.TH1D("P_ngcer_yAtCer_protons_dummysub_data_cut_all", "SHMS NGC yAtCer; SHMS_ngcer_yAtCer; Counts", 200, -50, 50)
P_aero_npeSum_protons_dummysub_data_cut_all = ROOT.TH1D("P_aero_npeSum_protons_dummysub_data_cut_all", "SHMS aero npeSum; SHMS_aero_npeSum; Counts", 200, 0, 50)
P_aero_xAtAero_protons_dummysub_data_cut_all = ROOT.TH1D("P_acero_xAtAero_protons_dummysub_data_cut_all", "SHMS aero xAtAero; SHMS_aero_xAtAero; Counts", 200, -60, 60)
P_aero_yAtAero_protons_dummysub_data_cut_all = ROOT.TH1D("P_aero_yAtAero_protons_dummysub_data_cut_all", "SHMS aero yAtAero; SHMS_aero_yAtAero; Counts", 200, -50, 50)
P_kin_MMp_protons_dummysub_data_cut_all = ROOT.TH1D("P_kin_MMp_protons_dummysub_data_cut_all", "MIssing Mass data (dummysub_cut_all); MM_{p}; Counts", 200, -1., 1.)
P_RFTime_Dist_protons_dummysub_data_cut_all = ROOT.TH1D("P_RFTime_Dist_protons_dummysub_data_cut_all", "SHMS RFTime; SHMS_RFTime; Counts", 200, 0, 4)
CTime_epCoinTime_ROC1_protons_dummysub_data_cut_all = ROOT.TH1D("CTime_epCoinTime_ROC1_protons_dummysub_data_cut_all", "Electron-Proton CTime; e p Coin_Time; Counts", 200, -50, 50)
P_kin_secondary_pmiss_protons_dummysub_data_cut_all = ROOT.TH1D("P_kin_secondary_pmiss_protons_dummysub_data_cut_all", "Momentum Distribution; pmiss; Counts", 400, -0.5, 0.5)
P_kin_secondary_pmiss_x_protons_dummysub_data_cut_all = ROOT.TH1D("P_kin_secondary_pmiss_x_protons_dummysub_data_cut_all", "Momentum_x Distribution; pmiss_x; Counts", 400, -0.5, 0.5)
P_kin_secondary_pmiss_y_protons_dummysub_data_cut_all = ROOT.TH1D("P_kin_secondary_pmiss_y_protons_dummysub_data_cut_all", "Momentum_y Distribution; pmiss_y; Counts", 400, -0.5, 0.5)
P_kin_secondary_pmiss_z_protons_dummysub_data_cut_all = ROOT.TH1D("P_kin_secondary_pmiss_z_protons_dummysub_data_cut_all", "Momentum_z Distribution; pmiss_z; Counts", 400, -0.42, 0.42)
P_kin_secondary_Erecoil_protons_dummysub_data_cut_all = ROOT.TH1D("P_kin_secondary_Erecoil_protons_dummysub_data_cut_all", "Erecoil Distribution; Erecoil; Counts", 200, -0.8, 0.8)
P_kin_secondary_emiss_protons_dummysub_data_cut_all = ROOT.TH1D("P_kin_secondary_emiss_protons_dummysub_data_cut_all", "Energy Distribution; emiss; Counts", 400, -0.46, 0.46)
P_kin_secondary_Mrecoil_protons_dummysub_data_cut_all = ROOT.TH1D("P_kin_secondary_Mrecoil_protons_dummysub_data_cut_all", "Mrecoil Distribution; Mrecoil; Counts", 200, -0.8, 0.8)
P_kin_secondary_W_protons_dummysub_data_cut_all = ROOT.TH1D("P_kin_secondary_W_protons_dummysub_data_cut_all", "W Distribution; W; Counts", 200, 0, 2)
MMsquared_dummysub_data_cut_all = ROOT.TH1D("MMsquared_dummysub_data_cut_all", "Missing Mass Squared; MM^{2}_{p}; Counts", 200, -1., 1.)

# Cut (Acceptance + PID + Prompt Selection) Dummy Histograms
H_gtr_beta_protons_dummy_prompt_cut_all = ROOT.TH1D("H_gtr_beta_protons_dummy_prompt_cut_all", "HMS #beta; HMS_gtr_#beta; Counts", 200, 0.8, 1.2)
H_gtr_xp_protons_dummy_prompt_cut_all = ROOT.TH1D("H_gtr_xp_protons_dummy_prompt_cut_all", "HMS xptar; HMS_gtr_xptar; Counts", 200, -0.2, 0.2)
H_gtr_yp_protons_dummy_prompt_cut_all = ROOT.TH1D("H_gtr_yp_protons_dummy_prompt_cut_all", "HMS yptar; HMS_gtr_yptar; Counts", 200, -0.2, 0.2)
H_gtr_dp_protons_dummy_prompt_cut_all = ROOT.TH1D("H_gtr_dp_protons_dummy_prompt_cut_all", "HMS #delta; HMS_gtr_dp; Counts", 200, -15, 15)
H_gtr_p_protons_dummy_prompt_cut_all = ROOT.TH1D("H_gtr_p_protons_dummy_prompt_cut_all", "HMS p; HMS_gtr_p; Counts", 200, 4, 8)
H_dc_x_fp_protons_dummy_prompt_cut_all = ROOT.TH1D("H_dc_x_fp_protons_dummy_prompt_cut_all", "HMS x_fp'; HMS_dc_x_fp; Counts", 200, -100, 100)
H_dc_y_fp_protons_dummy_prompt_cut_all = ROOT.TH1D("H_dc_y_fp_protons_dummy_prompt_cut_all", "HMS y_fp'; HMS_dc_y_fp; Counts", 200, -100, 100)
H_dc_xp_fp_protons_dummy_prompt_cut_all = ROOT.TH1D("H_dc_xp_fp_protons_dummy_prompt_cut_all", "HMS xp_fp'; HMS_dc_xp_fp; Counts", 200, -0.2, 0.2)
H_dc_yp_fp_protons_dummy_prompt_cut_all = ROOT.TH1D("H_dc_yp_fp_protons_dummy_prompt_cut_all", "HMS yp_fp'; HMS_dc_yp_fp; Counts", 200, -0.2, 0.2)
H_hod_goodscinhit_protons_dummy_prompt_cut_all = ROOT.TH1D("H_hod_goodscinhit_protons_dummy_prompt_cut_all", "HMS hod goodscinhit; HMS_hod_goodscinhi; Counts", 200, 0.7, 1.3)
H_hod_goodstarttime_protons_dummy_prompt_cut_all = ROOT.TH1D("H_hod_goodstarttime_protons_dummy_prompt_cut_all", "HMS hod goodstarttime; HMS_hod_goodstarttime; Counts", 200, 0.7, 1.3)
H_cal_etotnorm_protons_dummy_prompt_cut_all = ROOT.TH1D("H_cal_etotnorm_protons_dummy_prompt_cut_all", "HMS cal etotnorm; HMS_cal_etotnorm; Counts", 200, 0.2, 1.8)
H_cal_etottracknorm_protons_dummy_prompt_cut_all = ROOT.TH1D("H_cal_etottracknorm_protons_dummy_prompt_cut_all", "HMS cal etottracknorm; HMS_cal_etottracknorm; Counts", 200, 0.2, 1.8)
H_cer_npeSum_protons_dummy_prompt_cut_all = ROOT.TH1D("H_cer_npeSum_protons_dummy_prompt_cut_all", "HMS cer npeSum; HMS_cer_npeSum; Counts", 200, 0, 50)
H_RFTime_Dist_protons_dummy_prompt_cut_all = ROOT.TH1D("H_RFTime_Dist_protons_dummy_prompt_cut_all", "HMS RFTime; HMS_RFTime; Counts", 200, 0, 4)
P_gtr_beta_protons_dummy_prompt_cut_all = ROOT.TH1D("P_gtr_beta_protons_dummy_prompt_cut_all", "SHMS #beta; SHMS_gtr_#beta; Counts", 200, 0.5, 1.3)
P_gtr_xp_protons_dummy_prompt_cut_all = ROOT.TH1D("P_gtr_xp_protons_dummy_prompt_cut_all", "SHMS xptar; SHMS_gtr_xptar; Counts", 200, -0.2, 0.2)
P_gtr_yp_protons_dummy_prompt_cut_all = ROOT.TH1D("P_gtr_yp_protons_dummy_prompt_cut_all", "SHMS yptar; SHMS_gtr_yptar; Counts", 200, -0.2, 0.2)
P_gtr_dp_protons_dummy_prompt_cut_all = ROOT.TH1D("P_gtr_dp_protons_dummy_prompt_cut_all", "SHMS #delta; SHMS_gtr_dp; Counts", 200, -30, 30)
P_gtr_p_protons_dummy_prompt_cut_all = ROOT.TH1D("P_gtr_p_protons_dummy_prompt_cut_all", "SHMS p; SHMS_gtr_p; Counts", 200, 1, 7)
P_dc_x_fp_protons_dummy_prompt_cut_all = ROOT.TH1D("P_dc_x_fp_protons_dummy_prompt_cut_all", "SHMS x_fp'; SHMS_dc_x_fp; Counts", 200, -100, 100)
P_dc_y_fp_protons_dummy_prompt_cut_all = ROOT.TH1D("P_dc_y_fp_protons_dummy_prompt_cut_all", "SHMS y_fp'; SHMS_dc_y_fp; Counts", 200, -100, 100)
P_dc_xp_fp_protons_dummy_prompt_cut_all = ROOT.TH1D("P_dc_xp_fp_protons_dummy_prompt_cut_all", "SHMS xp_fp'; SHMS_dc_xp_fp; Counts", 200, -0.2, 0.2)
P_dc_yp_fp_protons_dummy_prompt_cut_all = ROOT.TH1D("P_dc_yp_fp_protons_dummy_prompt_cut_all", "SHMS yp_fp'; SHMS_dc_yp_fp; Counts", 200, -0.2, 0.2)
P_hod_goodscinhit_protons_dummy_prompt_cut_all = ROOT.TH1D("P_hod_goodscinhit_protons_dummy_prompt_cut_all", "SHMS hod goodscinhit; SHMS_hod_goodscinhit; Counts", 200, 0.7, 1.3)
P_hod_goodstarttime_protons_dummy_prompt_cut_all = ROOT.TH1D("P_hod_goodstarttime_protons_dummy_prompt_cut_all", "SHMS hod goodstarttime; SHMS_hod_goodstarttime; Counts", 200, 0.7, 1.3)
P_cal_etotnorm_protons_dummy_prompt_cut_all = ROOT.TH1D("P_cal_etotnorm_protons_dummy_prompt_cut_all", "SHMS cal etotnorm; SHMS_cal_etotnorm; Counts", 200, 0, 1)
P_cal_etottracknorm_protons_dummy_prompt_cut_all = ROOT.TH1D("P_cal_etottracknorm_protons_dummy_prompt_cut_all", "SHMS cal etottracknorm; SHMS_cal_etottracknorm; Counts", 200, 0, 1.6)
P_hgcer_npeSum_protons_dummy_prompt_cut_all = ROOT.TH1D("P_hgcer_npeSum_protons_dummy_prompt_cut_all", "SHMS HGC npeSum; SHMS_hgcer_npeSum; Counts", 200, 0, 50)
P_hgcer_xAtCer_protons_dummy_prompt_cut_all = ROOT.TH1D("P_hgcer_xAtCer_protons_dummy_prompt_cut_all", "SHMS HGC xAtCer; SHMS_hgcer_xAtCer; Counts", 200, -60, 60)
P_hgcer_yAtCer_protons_dummy_prompt_cut_all = ROOT.TH1D("P_hgcer_yAtCer_protons_dummy_prompt_cut_all", "SHMS HGC yAtCer; SHMS_hgcer_yAtCer; Counts", 200, -50, 50)
P_ngcer_npeSum_protons_dummy_prompt_cut_all = ROOT.TH1D("P_ngcer_npeSum_protons_dummy_prompt_cut_all", "SHMS NGC npeSum; SHMS_ngcer_npeSum; Counts", 200, 0, 50)
P_ngcer_xAtCer_protons_dummy_prompt_cut_all = ROOT.TH1D("P_ngcer_xAtCer_protons_dummy_prompt_cut_all", "SHMS NGC xAtCer; SHMS_ngcer_xAtCer; Counts", 200, -60, 60)
P_ngcer_yAtCer_protons_dummy_prompt_cut_all = ROOT.TH1D("P_ngcer_yAtCer_protons_dummy_prompt_cut_all", "SHMS NGC yAtCer; SHMS_ngcer_yAtCer; Counts", 200, -50, 50)
P_aero_npeSum_protons_dummy_prompt_cut_all = ROOT.TH1D("P_aero_npeSum_protons_dummy_prompt_cut_all", "SHMS aero npeSum; SHMS_aero_npeSum; Counts", 200, 0, 50)
P_aero_xAtAero_protons_dummy_prompt_cut_all = ROOT.TH1D("P_acero_xAtAero_protons_dummy_prompt_cut_all", "SHMS aero xAtAero; SHMS_aero_xAtAero; Counts", 200, -60, 60)
P_aero_yAtAero_protons_dummy_prompt_cut_all = ROOT.TH1D("P_aero_yAtAero_protons_dummy_prompt_cut_all", "SHMS aero yAtAero; SHMS_aero_yAtAero; Counts", 200, -50, 50)
P_kin_MMp_protons_dummy_prompt_cut_all = ROOT.TH1D("P_kin_MMp_protons_dummy_prompt_cut_all", "MIssing Mass dummy (prompt_cut_all); MM_{p}; Counts", 200, -1., 1.)
P_RFTime_Dist_protons_dummy_prompt_cut_all = ROOT.TH1D("P_RFTime_Dist_protons_dummy_prompt_cut_all", "SHMS RFTime; SHMS_RFTime; Counts", 200, 0, 4)
CTime_epCoinTime_ROC1_protons_dummy_prompt_cut_all = ROOT.TH1D("CTime_epCoinTime_ROC1_protons_dummy_prompt_cut_all", "Electron-Proton CTime; e p Coin_Time; Counts", 200, -50, 50)
P_kin_secondary_pmiss_protons_dummy_prompt_cut_all = ROOT.TH1D("P_kin_secondary_pmiss_protons_dummy_prompt_cut_all", "Momentum Distribution; pmiss; Counts", 400, -0.5, 0.5)
P_kin_secondary_pmiss_x_protons_dummy_prompt_cut_all = ROOT.TH1D("P_kin_secondary_pmiss_x_protons_dummy_prompt_cut_all", "Momentum_x Distribution; pmiss_x; Counts", 400, -0.5, 0.5)
P_kin_secondary_pmiss_y_protons_dummy_prompt_cut_all = ROOT.TH1D("P_kin_secondary_pmiss_y_protons_dummy_prompt_cut_all", "Momentum_y Distribution; pmiss_y; Counts", 400, -0.5, 0.5)
P_kin_secondary_pmiss_z_protons_dummy_prompt_cut_all = ROOT.TH1D("P_kin_secondary_pmiss_z_protons_dummy_prompt_cut_all", "Momentum_z Distribution; pmiss_z; Counts", 400, -0.42, 0.42)
P_kin_secondary_Erecoil_protons_dummy_prompt_cut_all = ROOT.TH1D("P_kin_secondary_Erecoil_protons_dummy_prompt_cut_all", "Erecoil Distribution; Erecoil; Counts", 200, -0.8, 0.8)
P_kin_secondary_emiss_protons_dummy_prompt_cut_all = ROOT.TH1D("P_kin_secondary_emiss_protons_dummy_prompt_cut_all", "Energy Distribution; emiss; Counts", 400, -0.46, 0.46)
P_kin_secondary_Mrecoil_protons_dummy_prompt_cut_all = ROOT.TH1D("P_kin_secondary_Mrecoil_protons_dummy_prompt_cut_all", "Mrecoil Distribution; Mrecoil; Counts", 200, -0.8, 0.8)
P_kin_secondary_W_protons_dummy_prompt_cut_all = ROOT.TH1D("P_kin_secondary_W_protons_dummy_prompt_cut_all", "W Distribution; W; Counts", 200, 0, 2)
MMsquared_dummy_prompt_cut_all = ROOT.TH1D("MMsquared_dummy_prompt_cut_all", "Missing Mass Squared; MM^{2}_{p}; Counts", 200, -1., 1.)

# Uncut SIMC Histograms
H_hsdelta_protons_simc_cut_all = ROOT.TH1D("H_hsdelta_protons_simc_cut_all", "HMS #delta; HMS_#delta; Counts", 200, -15, 15)
H_hsxptar_protons_simc_cut_all = ROOT.TH1D("H_hsxptar_protons_simc_cut_all", "HMS xptar; HMS_xptar; Counts", 200, -0.2, 0.2)
H_hsyptar_protons_simc_cut_all = ROOT.TH1D("H_hsyptar_protons_simc_cut_all", "HMS yptar; HMS_yptar; Counts", 200, -0.2, 0.2)
H_hsytar_protons_simc_cut_all = ROOT.TH1D("H_hsytar_protons_simc_cut_all", "HMS ytar; HMS_ytar; Counts", 200, -20, 20)
H_hsxfp_protons_simc_cut_all = ROOT.TH1D("H_hsxfp_protons_simc_cut_all", "HMS x_fp'; HMS_xfp; Counts", 200, -100, 100)
H_hsyfp_protons_simc_cut_all = ROOT.TH1D("H_hsyfp_protons_simc_cut_all", "HMS y_fp'; HMS_yfp; Counts", 200, -100, 100)
H_hsxpfp_protons_simc_cut_all = ROOT.TH1D("H_hsxpfp_protons_simc_cut_all", "HMS xp_fp'; HMS_xpfp; Counts", 200, -0.2, 0.2)
H_hsypfp_protons_simc_cut_all = ROOT.TH1D("H_hsypfp_protons_simc_cut_all", "HMS yp_fp'; HMS_ypfp; Counts", 200, -0.2, 0.2)
H_hsdeltai_protons_simc_cut_all = ROOT.TH1D("H_hsdeltai_protons_simc_cut_all", "HMS #delta i; HMS_#delta i; Counts", 200, -15, 15)
H_hsxptari_protons_simc_cut_all = ROOT.TH1D("H_hsxptari_protons_simc_cut_all", "HMS xptari; HMS_xptari; Counts", 200, -0.2, 0.2)
H_hsyptari_protons_simc_cut_all = ROOT.TH1D("H_hsyptari_protons_simc_cut_all", "HMS yptari; HMS_yptari; Counts", 200, -0.2, 0.2)
H_hsytari_protons_simc_cut_all = ROOT.TH1D("H_hsytari_protons_simc_cut_all", "HMS ytari; HMS_ytari; Counts", 200, -20, 20)
P_ssdelta_protons_simc_cut_all = ROOT.TH1D("P_ssdelta_protons_simc_cut_all", "SHMS #delta; SHMS_#delta; Counts", 200, -30, 30)
P_ssxptar_protons_simc_cut_all = ROOT.TH1D("P_ssxptar_protons_simc_cut_all", "SHMS xptar; SHMS_xptar; Counts", 200, -0.2, 0.2)
P_ssyptar_protons_simc_cut_all = ROOT.TH1D("P_ssyptar_protons_simc_cut_all", "SHMS yptar; SHMS_yptar; Counts", 200, -0.2, 0.2)
P_ssytar_protons_simc_cut_all = ROOT.TH1D("P_ssytar_protons_simc_cut_all", "SHMS ytar; SHMS_ytar; Counts", 200, -20, 20)
P_ssxfp_protons_simc_cut_all = ROOT.TH1D("P_ssxfp_protons_simc_cut_all", "SHMS x_fp'; SHMS_xfp; Counts", 200, -100, 100)
P_ssyfp_protons_simc_cut_all = ROOT.TH1D("P_ssyfp_protons_simc_cut_all", "SHMS y_fp'; SHMS_yfp; Counts", 200, -100, 100)
P_ssxpfp_protons_simc_cut_all = ROOT.TH1D("P_ssxpfp_protons_simc_cut_all", "SHMS xp_fp'; SHMS_xpfp; Counts", 200, -0.2, 0.2)
P_ssypfp_protons_simc_cut_all = ROOT.TH1D("P_ssypfp_protons_simc_cut_all", "SHMS yp_fp'; SHMS_ypfp; Counts", 200, -0.2, 0.2)
P_ssdeltai_protons_simc_cut_all = ROOT.TH1D("P_ssdeltai_protons_simc_cut_all", "SHMS #delta i; SHMS_#delta i; Counts", 200, -30, 30)
P_ssxptari_protons_simc_cut_all = ROOT.TH1D("P_ssxptari_protons_simc_cut_all", "SHMS xptari; SHMS_xptari; Counts", 200, -0.2, 0.2)
P_ssyptari_protons_simc_cut_all = ROOT.TH1D("P_ssyptari_protons_simc_cut_all", "SHMS yptari; SHMS_yptari; Counts", 200, -0.2, 0.2)
P_ssytari_protons_simc_cut_all = ROOT.TH1D("P_ssytari_protons_simc_cut_all", "SHMS ytari; SHMS_ytari; Counts", 200, -20, 20)
q_protons_simc_cut_all = ROOT.TH1D("q_protons_simc_cut_all", "q Distribution; q; Counts", 200, 0, 10)
nu_protons_simc_cut_all = ROOT.TH1D("nu_protons_simc_cut_all", "nu Distribution; nu; Counts", 200, 0, 10)
Q2_protons_simc_cut_all = ROOT.TH1D("Q2_protons_simc_cut_all", "#Q^2 Distribution; #Q^2; Counts", 200, 0, 15)
epsilon_protons_simc_cut_all = ROOT.TH1D("epsilon_protons_simc_cut_all", "epsilon Distribution; #epsilon; Counts", 200, 0, 2)
thetapq_protons_simc_cut_all = ROOT.TH1D("thetapq_protons_simc_cut_all", "thetapq Distribution; thetapq; Counts", 200, -1, 1)
phipq_protons_simc_cut_all = ROOT.TH1D("phipq_protons_simc_cut_all", "phipq Distribution; phipq; Counts", 200, -2, 8)
pmiss_protons_simc_cut_all = ROOT.TH1D("pmiss_protons_simc_cut_all", "Momentum Distribution; pmiss; Counts", 400, -0.5, 0.5)
pmiss_x_protons_simc_cut_all = ROOT.TH1D("pmiss_x_protons_simc_cut_all", "Momentum_x Distribution; pmiss_x; Counts", 400, -0.5, 0.5)
pmiss_y_protons_simc_cut_all = ROOT.TH1D("pmiss_y_protons_simc_cut_all", "Momentum_y Distribution; pmiss_y; Counts", 400, -0.5, 0.5)
pmiss_z_protons_simc_cut_all = ROOT.TH1D("pmiss_z_protons_simc_cut_all", "Momentum_z Distribution; pmiss_z; Counts", 400, -0.42, 0.42)
emiss_protons_simc_cut_all = ROOT.TH1D("emiss_protons_simc_cut_all", "Energy Distribution; emiss; Counts",400, -0.46, 0.46)
W_protons_simc_cut_all = ROOT.TH1D("W_protons_simc_cut_all", "W Distribution; W; Counts", 200, 0, 2)
MMp_protons_simc_cut_all = ROOT.TH1D("MMp_protons_simc_cut_all", "MIssing Mass SIMC (cut_all); MM_{p}; Counts", 200, -1, 1)

#################################################################################################################################################

#Fill histograms for Dummy Subtraction
#ibin = 1
for event in Cut_Proton_Events_Prompt_Data_tree:
    H_gtr_beta_protons_dummysub_data_cut_all.Fill(event.H_gtr_beta)
    H_gtr_xp_protons_dummysub_data_cut_all.Fill(event.H_gtr_xp)
    H_gtr_yp_protons_dummysub_data_cut_all.Fill(event.H_gtr_yp)
    H_gtr_dp_protons_dummysub_data_cut_all.Fill(event.H_gtr_dp)
    H_gtr_p_protons_dummysub_data_cut_all.Fill(event.H_gtr_p)
    H_dc_x_fp_protons_dummysub_data_cut_all.Fill(event.H_dc_x_fp)
    H_dc_y_fp_protons_dummysub_data_cut_all.Fill(event.H_dc_y_fp)
    H_dc_xp_fp_protons_dummysub_data_cut_all.Fill(event.H_dc_xp_fp)
    H_dc_yp_fp_protons_dummysub_data_cut_all.Fill(event.H_dc_yp_fp)
    H_hod_goodscinhit_protons_dummysub_data_cut_all.Fill(event.H_hod_goodscinhit)
    H_hod_goodstarttime_protons_dummysub_data_cut_all.Fill(event.H_hod_goodstarttime)
    H_cal_etotnorm_protons_dummysub_data_cut_all.Fill(event.H_cal_etotnorm)
    H_cal_etottracknorm_protons_dummysub_data_cut_all.Fill(event.H_cal_etottracknorm)
    H_cer_npeSum_protons_dummysub_data_cut_all.Fill(event.H_cer_npeSum)
    H_RFTime_Dist_protons_dummysub_data_cut_all.Fill(event.H_RF_Dist)
    P_gtr_beta_protons_dummysub_data_cut_all.Fill(event.P_gtr_beta)
    P_gtr_xp_protons_dummysub_data_cut_all.Fill(event.P_gtr_xp)
    P_gtr_yp_protons_dummysub_data_cut_all.Fill(event.P_gtr_yp)
    P_gtr_dp_protons_dummysub_data_cut_all.Fill(event.P_gtr_dp)
    P_gtr_p_protons_dummysub_data_cut_all.Fill(event.P_gtr_p)
    P_dc_x_fp_protons_dummysub_data_cut_all.Fill(event.P_dc_x_fp)
    P_dc_y_fp_protons_dummysub_data_cut_all.Fill(event.P_dc_y_fp)
    P_dc_xp_fp_protons_dummysub_data_cut_all.Fill(event.P_dc_xp_fp)
    P_dc_yp_fp_protons_dummysub_data_cut_all.Fill(event.P_dc_yp_fp)
    P_hod_goodscinhit_protons_dummysub_data_cut_all.Fill(event.P_hod_goodscinhit)
    P_hod_goodstarttime_protons_dummysub_data_cut_all.Fill(event.P_hod_goodstarttime)
    P_cal_etotnorm_protons_dummysub_data_cut_all.Fill(event.P_cal_etotnorm)
    P_cal_etottracknorm_protons_dummysub_data_cut_all.Fill(event.P_cal_etottracknorm)
    P_hgcer_npeSum_protons_dummysub_data_cut_all.Fill(event.P_hgcer_npeSum)
    P_hgcer_xAtCer_protons_dummysub_data_cut_all.Fill(event.P_hgcer_xAtCer)
    P_hgcer_yAtCer_protons_dummysub_data_cut_all.Fill(event.P_hgcer_yAtCer)
    P_ngcer_npeSum_protons_dummysub_data_cut_all.Fill(event.P_ngcer_npeSum)
    P_ngcer_xAtCer_protons_dummysub_data_cut_all.Fill(event.P_ngcer_xAtCer)
    P_ngcer_yAtCer_protons_dummysub_data_cut_all.Fill(event.P_ngcer_yAtCer)
    P_aero_npeSum_protons_dummysub_data_cut_all.Fill(event.P_aero_npeSum)
    P_aero_xAtAero_protons_dummysub_data_cut_all.Fill(event.P_aero_xAtAero)
    P_aero_yAtAero_protons_dummysub_data_cut_all.Fill(event.P_aero_yAtAero)
    P_kin_MMp_protons_dummysub_data_cut_all.Fill(event.MMp)
    P_RFTime_Dist_protons_dummysub_data_cut_all.Fill(event.P_RF_Dist)
    CTime_epCoinTime_ROC1_protons_dummysub_data_cut_all.Fill(event.CTime_epCoinTime_ROC1)
    P_kin_secondary_pmiss_protons_dummysub_data_cut_all.Fill(event.pmiss)
    P_kin_secondary_pmiss_x_protons_dummysub_data_cut_all.Fill(event.pmiss_x)
    P_kin_secondary_pmiss_y_protons_dummysub_data_cut_all.Fill(event.pmiss_y)
    P_kin_secondary_pmiss_z_protons_dummysub_data_cut_all.Fill(event.pmiss_z)
    P_kin_secondary_Erecoil_protons_dummysub_data_cut_all.Fill(event.Erecoil)
    P_kin_secondary_emiss_protons_dummysub_data_cut_all.Fill(event.emiss)
    P_kin_secondary_Mrecoil_protons_dummysub_data_cut_all.Fill(event.Mrecoil)
    P_kin_secondary_W_protons_dummysub_data_cut_all.Fill(event.W)
    MMsquared_dummysub_data_cut_all.Fill(event.MMp*event.MMp)
#    ibin += 1

# Fill histograms from DUMMY ROOT File
#ibin = 1
for event in Cut_Proton_Events_Prompt_Dummy_tree:
    H_gtr_beta_protons_dummy_prompt_cut_all.Fill(event.H_gtr_beta)
    H_gtr_xp_protons_dummy_prompt_cut_all.Fill(event.H_gtr_xp)
    H_gtr_yp_protons_dummy_prompt_cut_all.Fill(event.H_gtr_yp)
    H_gtr_dp_protons_dummy_prompt_cut_all.Fill(event.H_gtr_dp)
    H_gtr_p_protons_dummy_prompt_cut_all.Fill(event.H_gtr_p)
    H_dc_x_fp_protons_dummy_prompt_cut_all.Fill(event.H_dc_x_fp)
    H_dc_y_fp_protons_dummy_prompt_cut_all.Fill(event.H_dc_y_fp)
    H_dc_xp_fp_protons_dummy_prompt_cut_all.Fill(event.H_dc_xp_fp)
    H_dc_yp_fp_protons_dummy_prompt_cut_all.Fill(event.H_dc_yp_fp)
    H_hod_goodscinhit_protons_dummy_prompt_cut_all.Fill(event.H_hod_goodscinhit)
    H_hod_goodstarttime_protons_dummy_prompt_cut_all.Fill(event.H_hod_goodstarttime)
    H_cal_etotnorm_protons_dummy_prompt_cut_all.Fill(event.H_cal_etotnorm)
    H_cal_etottracknorm_protons_dummy_prompt_cut_all.Fill(event.H_cal_etottracknorm)
    H_cer_npeSum_protons_dummy_prompt_cut_all.Fill(event.H_cer_npeSum)
    H_RFTime_Dist_protons_dummy_prompt_cut_all.Fill(event.H_RF_Dist)
    P_gtr_beta_protons_dummy_prompt_cut_all.Fill(event.P_gtr_beta)
    P_gtr_xp_protons_dummy_prompt_cut_all.Fill(event.P_gtr_xp)
    P_gtr_yp_protons_dummy_prompt_cut_all.Fill(event.P_gtr_yp)
    P_gtr_dp_protons_dummy_prompt_cut_all.Fill(event.P_gtr_dp)
    P_gtr_p_protons_dummy_prompt_cut_all.Fill(event.P_gtr_p)
    P_dc_x_fp_protons_dummy_prompt_cut_all.Fill(event.P_dc_x_fp)
    P_dc_y_fp_protons_dummy_prompt_cut_all.Fill(event.P_dc_y_fp)
    P_dc_xp_fp_protons_dummy_prompt_cut_all.Fill(event.P_dc_xp_fp)
    P_dc_yp_fp_protons_dummy_prompt_cut_all.Fill(event.P_dc_yp_fp)
    P_hod_goodscinhit_protons_dummy_prompt_cut_all.Fill(event.P_hod_goodscinhit)
    P_hod_goodstarttime_protons_dummy_prompt_cut_all.Fill(event.P_hod_goodstarttime)
    P_cal_etotnorm_protons_dummy_prompt_cut_all.Fill(event.P_cal_etotnorm)
    P_cal_etottracknorm_protons_dummy_prompt_cut_all.Fill(event.P_cal_etottracknorm)
    P_hgcer_npeSum_protons_dummy_prompt_cut_all.Fill(event.P_hgcer_npeSum)
    P_hgcer_xAtCer_protons_dummy_prompt_cut_all.Fill(event.P_hgcer_xAtCer)
    P_hgcer_yAtCer_protons_dummy_prompt_cut_all.Fill(event.P_hgcer_yAtCer)
    P_ngcer_npeSum_protons_dummy_prompt_cut_all.Fill(event.P_ngcer_npeSum)
    P_ngcer_xAtCer_protons_dummy_prompt_cut_all.Fill(event.P_ngcer_xAtCer)
    P_ngcer_yAtCer_protons_dummy_prompt_cut_all.Fill(event.P_ngcer_yAtCer)
    P_aero_npeSum_protons_dummy_prompt_cut_all.Fill(event.P_aero_npeSum)
    P_aero_xAtAero_protons_dummy_prompt_cut_all.Fill(event.P_aero_xAtAero)
    P_aero_yAtAero_protons_dummy_prompt_cut_all.Fill(event.P_aero_yAtAero)
    P_kin_MMp_protons_dummy_prompt_cut_all.Fill(event.MMp)
    P_RFTime_Dist_protons_dummy_prompt_cut_all.Fill(event.P_RF_Dist)
    CTime_epCoinTime_ROC1_protons_dummy_prompt_cut_all.Fill(event.CTime_epCoinTime_ROC1)
    P_kin_secondary_pmiss_protons_dummy_prompt_cut_all.Fill(event.pmiss)
    P_kin_secondary_pmiss_x_protons_dummy_prompt_cut_all.Fill(event.pmiss_x)
    P_kin_secondary_pmiss_y_protons_dummy_prompt_cut_all.Fill(event.pmiss_y)
    P_kin_secondary_pmiss_z_protons_dummy_prompt_cut_all.Fill(event.pmiss_z)
    P_kin_secondary_Erecoil_protons_dummy_prompt_cut_all.Fill(event.Erecoil)
    P_kin_secondary_emiss_protons_dummy_prompt_cut_all.Fill(event.emiss)
    P_kin_secondary_Mrecoil_protons_dummy_prompt_cut_all.Fill(event.Mrecoil)
    P_kin_secondary_W_protons_dummy_prompt_cut_all.Fill(event.W)
    MMsquared_dummy_prompt_cut_all.Fill(event.MMp*event.MMp)
#    ibin += 1

#Fill histograms for Dummy Subtraction
#ibin = 1
for event in Cut_Proton_Events_Prompt_Data_tree:
    H_gtr_beta_protons_dummysub_data_cut_all.Fill(event.H_gtr_beta)
    H_gtr_xp_protons_dummysub_data_cut_all.Fill(event.H_gtr_xp)
    H_gtr_yp_protons_dummysub_data_cut_all.Fill(event.H_gtr_yp)
    H_gtr_dp_protons_dummysub_data_cut_all.Fill(event.H_gtr_dp)
    H_gtr_p_protons_dummysub_data_cut_all.Fill(event.H_gtr_p)
    H_dc_x_fp_protons_dummysub_data_cut_all.Fill(event.H_dc_x_fp)
    H_dc_y_fp_protons_dummysub_data_cut_all.Fill(event.H_dc_y_fp)
    H_dc_xp_fp_protons_dummysub_data_cut_all.Fill(event.H_dc_xp_fp)
    H_dc_yp_fp_protons_dummysub_data_cut_all.Fill(event.H_dc_yp_fp)
    H_hod_goodscinhit_protons_dummysub_data_cut_all.Fill(event.H_hod_goodscinhit)
    H_hod_goodstarttime_protons_dummysub_data_cut_all.Fill(event.H_hod_goodstarttime)
    H_cal_etotnorm_protons_dummysub_data_cut_all.Fill(event.H_cal_etotnorm)
    H_cal_etottracknorm_protons_dummysub_data_cut_all.Fill(event.H_cal_etottracknorm)
    H_cer_npeSum_protons_dummysub_data_cut_all.Fill(event.H_cer_npeSum)
    H_RFTime_Dist_protons_dummysub_data_cut_all.Fill(event.H_RF_Dist)
    P_gtr_beta_protons_dummysub_data_cut_all.Fill(event.P_gtr_beta)
    P_gtr_xp_protons_dummysub_data_cut_all.Fill(event.P_gtr_xp)
    P_gtr_yp_protons_dummysub_data_cut_all.Fill(event.P_gtr_yp)
    P_gtr_dp_protons_dummysub_data_cut_all.Fill(event.P_gtr_dp)
    P_gtr_p_protons_dummysub_data_cut_all.Fill(event.P_gtr_p)
    P_dc_x_fp_protons_dummysub_data_cut_all.Fill(event.P_dc_x_fp)
    P_dc_y_fp_protons_dummysub_data_cut_all.Fill(event.P_dc_y_fp)
    P_dc_xp_fp_protons_dummysub_data_cut_all.Fill(event.P_dc_xp_fp)
    P_dc_yp_fp_protons_dummysub_data_cut_all.Fill(event.P_dc_yp_fp)
    P_hod_goodscinhit_protons_dummysub_data_cut_all.Fill(event.P_hod_goodscinhit)
    P_hod_goodstarttime_protons_dummysub_data_cut_all.Fill(event.P_hod_goodstarttime)
    P_cal_etotnorm_protons_dummysub_data_cut_all.Fill(event.P_cal_etotnorm)
    P_cal_etottracknorm_protons_dummysub_data_cut_all.Fill(event.P_cal_etottracknorm)
    P_hgcer_npeSum_protons_dummysub_data_cut_all.Fill(event.P_hgcer_npeSum)
    P_hgcer_xAtCer_protons_dummysub_data_cut_all.Fill(event.P_hgcer_xAtCer)
    P_hgcer_yAtCer_protons_dummysub_data_cut_all.Fill(event.P_hgcer_yAtCer)
    P_ngcer_npeSum_protons_dummysub_data_cut_all.Fill(event.P_ngcer_npeSum)
    P_ngcer_xAtCer_protons_dummysub_data_cut_all.Fill(event.P_ngcer_xAtCer)
    P_ngcer_yAtCer_protons_dummysub_data_cut_all.Fill(event.P_ngcer_yAtCer)
    P_aero_npeSum_protons_dummysub_data_cut_all.Fill(event.P_aero_npeSum)
    P_aero_xAtAero_protons_dummysub_data_cut_all.Fill(event.P_aero_xAtAero)
    P_aero_yAtAero_protons_dummysub_data_cut_all.Fill(event.P_aero_yAtAero)
    P_kin_MMp_protons_dummysub_data_cut_all.Fill(event.MMp)
    P_RFTime_Dist_protons_dummysub_data_cut_all.Fill(event.P_RF_Dist)
    CTime_epCoinTime_ROC1_protons_dummysub_data_cut_all.Fill(event.CTime_epCoinTime_ROC1)
    P_kin_secondary_pmiss_protons_dummysub_data_cut_all.Fill(event.pmiss)
    P_kin_secondary_pmiss_x_protons_dummysub_data_cut_all.Fill(event.pmiss_x)
    P_kin_secondary_pmiss_y_protons_dummysub_data_cut_all.Fill(event.pmiss_y)
    P_kin_secondary_pmiss_z_protons_dummysub_data_cut_all.Fill(event.pmiss_z)
    P_kin_secondary_Erecoil_protons_dummysub_data_cut_all.Fill(event.Erecoil)
    P_kin_secondary_emiss_protons_dummysub_data_cut_all.Fill(event.emiss)
    P_kin_secondary_Mrecoil_protons_dummysub_data_cut_all.Fill(event.Mrecoil)
    P_kin_secondary_W_protons_dummysub_data_cut_all.Fill(event.W)
    MMsquared_dummysub_data_cut_all.Fill(event.MMp*event.MMp)
#    ibin += 1

# Fill histograms from SIMC ROOT File
for event in Uncut_Proton_Events_SIMC_tree:
    # Define the acceptance cuts
    HMS_Acceptance = (event.hsdelta>=-8.0) & (event.hsdelta<=8.0) & (event.hsxpfp>=-0.08) & (event.hsxpfp<=0.08) & (event.hsypfp>=-0.045) & (event.hsypfp<=0.045)
    SHMS_Acceptance = (event.ssdelta>=-10.0) & (event.ssdelta<=20.0) & (event.ssxpfp>=-0.06) & (event.ssxpfp<=0.06) & (event.ssypfp>=-0.04) & (event.ssypfp<=0.04)
    if(HMS_Acceptance & SHMS_Acceptance):

       H_hsdelta_protons_simc_cut_all.Fill(event.hsdelta, event.Weight)
       H_hsxptar_protons_simc_cut_all.Fill(event.hsxptar, event.Weight)
       H_hsyptar_protons_simc_cut_all.Fill(event.hsyptar, event.Weight)
       H_hsytar_protons_simc_cut_all.Fill(event.hsytar, event.Weight)
       H_hsxfp_protons_simc_cut_all.Fill(event.hsxfp, event.Weight)
       H_hsyfp_protons_simc_cut_all.Fill(event.hsyfp, event.Weight)
       H_hsxpfp_protons_simc_cut_all.Fill(event.hsxpfp, event.Weight)
       H_hsypfp_protons_simc_cut_all.Fill(event.hsypfp, event.Weight)
       H_hsdeltai_protons_simc_cut_all.Fill(event.hsdeltai, event.Weight)
       H_hsxptari_protons_simc_cut_all.Fill(event.hsxptari, event.Weight)
       H_hsyptari_protons_simc_cut_all.Fill(event.hsyptari, event.Weight)
       H_hsytari_protons_simc_cut_all.Fill(event.hsytari, event.Weight)
       P_ssdelta_protons_simc_cut_all.Fill(event.ssdelta, event.Weight)
       P_ssxptar_protons_simc_cut_all.Fill(event.ssxptar, event.Weight)
       P_ssyptar_protons_simc_cut_all.Fill(event.ssyptar, event.Weight)
       P_ssytar_protons_simc_cut_all.Fill(event.ssytar, event.Weight)
       P_ssxfp_protons_simc_cut_all.Fill(event.ssxfp, event.Weight)
       P_ssyfp_protons_simc_cut_all.Fill(event.ssyfp, event.Weight)
       P_ssxpfp_protons_simc_cut_all.Fill(event.ssxpfp, event.Weight)
       P_ssypfp_protons_simc_cut_all.Fill(event.ssypfp, event.Weight)
       P_ssdeltai_protons_simc_cut_all.Fill(event.ssdeltai, event.Weight)
       P_ssxptari_protons_simc_cut_all.Fill(event.ssxptari, event.Weight)
       P_ssyptari_protons_simc_cut_all.Fill(event.ssyptari, event.Weight)
       P_ssytari_protons_simc_cut_all.Fill(event.ssytari, event.Weight)
       q_protons_simc_cut_all.Fill(event.q, event.Weight)
       nu_protons_simc_cut_all.Fill(event.nu, event.Weight)
       Q2_protons_simc_cut_all.Fill(event.Q2, event.Weight)
       epsilon_protons_simc_cut_all.Fill(event.epsilon, event.Weight)
       thetapq_protons_simc_cut_all.Fill(event.thetapq, event.Weight)
       phipq_protons_simc_cut_all.Fill(event.phipq, event.Weight)
       pmiss_protons_simc_cut_all.Fill(event.Pm, event.Weight)
       pmiss_x_protons_simc_cut_all.Fill(event.Pmx, event.Weight)
       pmiss_y_protons_simc_cut_all.Fill(event.Pmy, event.Weight)
       pmiss_z_protons_simc_cut_all.Fill(event.Pmz, event.Weight)
       emiss_protons_simc_cut_all.Fill(event.Em, event.Weight)
       W_protons_simc_cut_all.Fill(event.W, event.Weight)
       MMp_protons_simc_cut_all.Fill(pow(event.Em, 2) - pow(event.Pm, 2), event.Weight)

#################################################################################################################################################
'''
# Random subtraction from missing mass
for event in Cut_Proton_Events_Random_tree:
    P_kin_MMp_protons_cut_random_scaled.Fill(event.MMp)
    P_kin_MMp_protons_cut_random_scaled.Scale(1.0/nWindows)
P_kin_MMp_protons_cut_random_sub.Add(P_kin_MMp_protons_cut_prompt, P_kin_MMp_protons_cut_random_scaled, 1, -1)
'''
############################################################################################################################################

# Normalize simc by normfactor/nevents
# Normalize data by effective charge

# Proton Absorption Correction
#proton_absorption_correction = 0.0856

normfac_data = 1.0/(total_data_effective_charge)

# Dummy Target Thickness Correction
dummy_target_corr = 4.8579
normfac_dummy = 1.0/(total_dummy_effective_charge * dummy_target_corr)

normfac_simc = (simc_normfactor)/(simc_nevents)

print("-"*40)
print ("normfac_data :", normfac_data)
print ("normfac_dummy: ", normfac_dummy)
print ("normfac_simc: ", normfac_simc)
print("-"*40)

'''
num_events_data = int(H_W_DATA.Integral()) - int(H_W_DUMMY.Integral())
yield_data = num_events_data*normfac_data
num_events_simc = int(H_W_SIMC.Integral())
yield_simc = num_events_simc*normfac_simc
rel_yield = yield_data/yield_simc
'''
############################################################################################################################################

# Data Normalization
H_gtr_beta_protons_dummysub_data_cut_all.Scale(normfac_data)
H_gtr_xp_protons_dummysub_data_cut_all.Scale(normfac_data)
H_gtr_yp_protons_dummysub_data_cut_all.Scale(normfac_data)
H_gtr_dp_protons_dummysub_data_cut_all.Scale(normfac_data)
H_gtr_p_protons_dummysub_data_cut_all.Scale(normfac_data)
H_dc_x_fp_protons_dummysub_data_cut_all.Scale(normfac_data)
H_dc_y_fp_protons_dummysub_data_cut_all.Scale(normfac_data)
H_dc_xp_fp_protons_dummysub_data_cut_all.Scale(normfac_data)
H_dc_yp_fp_protons_dummysub_data_cut_all.Scale(normfac_data)
H_hod_goodscinhit_protons_dummysub_data_cut_all.Scale(normfac_data)
H_hod_goodstarttime_protons_dummysub_data_cut_all.Scale(normfac_data)
H_cal_etotnorm_protons_dummysub_data_cut_all.Scale(normfac_data)
H_cal_etottracknorm_protons_dummysub_data_cut_all.Scale(normfac_data)
H_cer_npeSum_protons_dummysub_data_cut_all.Scale(normfac_data)
H_RFTime_Dist_protons_dummysub_data_cut_all.Scale(normfac_data)
P_gtr_beta_protons_dummysub_data_cut_all.Scale(normfac_data)
P_gtr_xp_protons_dummysub_data_cut_all.Scale(normfac_data)
P_gtr_yp_protons_dummysub_data_cut_all.Scale(normfac_data)
P_gtr_dp_protons_dummysub_data_cut_all.Scale(normfac_data)
P_gtr_p_protons_dummysub_data_cut_all.Scale(normfac_data)
P_dc_x_fp_protons_dummysub_data_cut_all.Scale(normfac_data)
P_dc_y_fp_protons_dummysub_data_cut_all.Scale(normfac_data)
P_dc_xp_fp_protons_dummysub_data_cut_all.Scale(normfac_data)
P_dc_yp_fp_protons_dummysub_data_cut_all.Scale(normfac_data)
P_hod_goodscinhit_protons_dummysub_data_cut_all.Scale(normfac_data)
P_hod_goodstarttime_protons_dummysub_data_cut_all.Scale(normfac_data)
P_cal_etotnorm_protons_dummysub_data_cut_all.Scale(normfac_data)
P_cal_etottracknorm_protons_dummysub_data_cut_all.Scale(normfac_data)
P_hgcer_npeSum_protons_dummysub_data_cut_all.Scale(normfac_data)
P_hgcer_xAtCer_protons_dummysub_data_cut_all.Scale(normfac_data)
P_hgcer_yAtCer_protons_dummysub_data_cut_all.Scale(normfac_data)
P_ngcer_npeSum_protons_dummysub_data_cut_all.Scale(normfac_data)
P_ngcer_xAtCer_protons_dummysub_data_cut_all.Scale(normfac_data)
P_ngcer_yAtCer_protons_dummysub_data_cut_all.Scale(normfac_data)
P_aero_npeSum_protons_dummysub_data_cut_all.Scale(normfac_data)
P_aero_xAtAero_protons_dummysub_data_cut_all.Scale(normfac_data)
P_aero_yAtAero_protons_dummysub_data_cut_all.Scale(normfac_data)
P_kin_MMp_protons_dummysub_data_cut_all.Scale(normfac_data)
P_RFTime_Dist_protons_dummysub_data_cut_all.Scale(normfac_data)
CTime_epCoinTime_ROC1_protons_dummysub_data_cut_all.Scale(normfac_data)
P_kin_secondary_pmiss_protons_dummysub_data_cut_all.Scale(normfac_data)
P_kin_secondary_pmiss_x_protons_dummysub_data_cut_all.Scale(normfac_data)
P_kin_secondary_pmiss_y_protons_dummysub_data_cut_all.Scale(normfac_data)
P_kin_secondary_pmiss_z_protons_dummysub_data_cut_all.Scale(normfac_data)
P_kin_secondary_Erecoil_protons_dummysub_data_cut_all.Scale(normfac_data)
P_kin_secondary_emiss_protons_dummysub_data_cut_all.Scale(normfac_data)
P_kin_secondary_Mrecoil_protons_dummysub_data_cut_all.Scale(normfac_data)
P_kin_secondary_W_protons_dummysub_data_cut_all.Scale(normfac_data)
MMsquared_dummysub_data_cut_all.Scale(normfac_data)

# Dummy Normalization
H_gtr_beta_protons_dummy_prompt_cut_all.Scale(normfac_dummy)
H_gtr_xp_protons_dummy_prompt_cut_all.Scale(normfac_dummy)
H_gtr_yp_protons_dummy_prompt_cut_all.Scale(normfac_dummy)
H_gtr_dp_protons_dummy_prompt_cut_all.Scale(normfac_dummy)
H_gtr_p_protons_dummy_prompt_cut_all.Scale(normfac_dummy)
H_dc_x_fp_protons_dummy_prompt_cut_all.Scale(normfac_dummy)
H_dc_y_fp_protons_dummy_prompt_cut_all.Scale(normfac_dummy)
H_dc_xp_fp_protons_dummy_prompt_cut_all.Scale(normfac_dummy)
H_dc_yp_fp_protons_dummy_prompt_cut_all.Scale(normfac_dummy)
H_hod_goodscinhit_protons_dummy_prompt_cut_all.Scale(normfac_dummy)
H_hod_goodstarttime_protons_dummy_prompt_cut_all.Scale(normfac_dummy)
H_cal_etotnorm_protons_dummy_prompt_cut_all.Scale(normfac_dummy)
H_cal_etottracknorm_protons_dummy_prompt_cut_all.Scale(normfac_dummy)
H_cer_npeSum_protons_dummy_prompt_cut_all.Scale(normfac_dummy)
H_RFTime_Dist_protons_dummy_prompt_cut_all.Scale(normfac_dummy)
P_gtr_beta_protons_dummy_prompt_cut_all.Scale(normfac_dummy)
P_gtr_xp_protons_dummy_prompt_cut_all.Scale(normfac_dummy)
P_gtr_yp_protons_dummy_prompt_cut_all.Scale(normfac_dummy)
P_gtr_dp_protons_dummy_prompt_cut_all.Scale(normfac_dummy)
P_gtr_p_protons_dummy_prompt_cut_all.Scale(normfac_dummy)
P_dc_x_fp_protons_dummy_prompt_cut_all.Scale(normfac_dummy)
P_dc_y_fp_protons_dummy_prompt_cut_all.Scale(normfac_dummy)
P_dc_xp_fp_protons_dummy_prompt_cut_all.Scale(normfac_dummy)
P_dc_yp_fp_protons_dummy_prompt_cut_all.Scale(normfac_dummy)
P_hod_goodscinhit_protons_dummy_prompt_cut_all.Scale(normfac_dummy)
P_hod_goodstarttime_protons_dummy_prompt_cut_all.Scale(normfac_dummy)
P_cal_etotnorm_protons_dummy_prompt_cut_all.Scale(normfac_dummy)
P_cal_etottracknorm_protons_dummy_prompt_cut_all.Scale(normfac_dummy)
P_hgcer_npeSum_protons_dummy_prompt_cut_all.Scale(normfac_dummy)
P_hgcer_xAtCer_protons_dummy_prompt_cut_all.Scale(normfac_dummy)
P_hgcer_yAtCer_protons_dummy_prompt_cut_all.Scale(normfac_dummy)
P_ngcer_npeSum_protons_dummy_prompt_cut_all.Scale(normfac_dummy)
P_ngcer_xAtCer_protons_dummy_prompt_cut_all.Scale(normfac_dummy)
P_ngcer_yAtCer_protons_dummy_prompt_cut_all.Scale(normfac_dummy)
P_aero_npeSum_protons_dummy_prompt_cut_all.Scale(normfac_dummy)
P_aero_xAtAero_protons_dummy_prompt_cut_all.Scale(normfac_dummy)
P_aero_yAtAero_protons_dummy_prompt_cut_all.Scale(normfac_dummy)
P_kin_MMp_protons_dummy_prompt_cut_all.Scale(normfac_dummy)
P_RFTime_Dist_protons_dummy_prompt_cut_all.Scale(normfac_dummy)
CTime_epCoinTime_ROC1_protons_dummy_prompt_cut_all.Scale(normfac_dummy)
P_kin_secondary_pmiss_protons_dummy_prompt_cut_all.Scale(normfac_dummy)
P_kin_secondary_pmiss_x_protons_dummy_prompt_cut_all.Scale(normfac_dummy)
P_kin_secondary_pmiss_y_protons_dummy_prompt_cut_all.Scale(normfac_dummy)
P_kin_secondary_pmiss_z_protons_dummy_prompt_cut_all.Scale(normfac_dummy)
P_kin_secondary_Erecoil_protons_dummy_prompt_cut_all.Scale(normfac_dummy)
P_kin_secondary_emiss_protons_dummy_prompt_cut_all.Scale(normfac_dummy)
P_kin_secondary_Mrecoil_protons_dummy_prompt_cut_all.Scale(normfac_dummy)
P_kin_secondary_W_protons_dummy_prompt_cut_all.Scale(normfac_dummy)
MMsquared_dummy_prompt_cut_all.Scale(normfac_dummy)

############################################################################################################################################

# SIMC Normalization
H_hsdelta_protons_simc_cut_all.Scale(normfac_simc)
H_hsxptar_protons_simc_cut_all.Scale(normfac_simc)
H_hsyptar_protons_simc_cut_all.Scale(normfac_simc)
H_hsytar_protons_simc_cut_all.Scale(normfac_simc)
H_hsxfp_protons_simc_cut_all.Scale(normfac_simc)
H_hsyfp_protons_simc_cut_all.Scale(normfac_simc)
H_hsxpfp_protons_simc_cut_all.Scale(normfac_simc)
H_hsypfp_protons_simc_cut_all.Scale(normfac_simc)
H_hsdeltai_protons_simc_cut_all.Scale(normfac_simc)
H_hsxptari_protons_simc_cut_all.Scale(normfac_simc)
H_hsyptari_protons_simc_cut_all.Scale(normfac_simc)
H_hsytari_protons_simc_cut_all.Scale(normfac_simc)
P_ssdelta_protons_simc_cut_all.Scale(normfac_simc)
P_ssxptar_protons_simc_cut_all.Scale(normfac_simc)
P_ssyptar_protons_simc_cut_all.Scale(normfac_simc)
P_ssytar_protons_simc_cut_all.Scale(normfac_simc)
P_ssxfp_protons_simc_cut_all.Scale(normfac_simc)
P_ssyfp_protons_simc_cut_all.Scale(normfac_simc)
P_ssxpfp_protons_simc_cut_all.Scale(normfac_simc)
P_ssypfp_protons_simc_cut_all.Scale(normfac_simc)
P_ssdeltai_protons_simc_cut_all.Scale(normfac_simc)
P_ssxptari_protons_simc_cut_all.Scale(normfac_simc)
P_ssyptari_protons_simc_cut_all.Scale(normfac_simc)
P_ssytari_protons_simc_cut_all.Scale(normfac_simc)
q_protons_simc_cut_all.Scale(normfac_simc)
nu_protons_simc_cut_all.Scale(normfac_simc)
Q2_protons_simc_cut_all.Scale(normfac_simc)
epsilon_protons_simc_cut_all.Scale(normfac_simc)
thetapq_protons_simc_cut_all.Scale(normfac_simc)
phipq_protons_simc_cut_all.Scale(normfac_simc)
pmiss_protons_simc_cut_all.Scale(normfac_simc)
pmiss_x_protons_simc_cut_all.Scale(normfac_simc)
pmiss_y_protons_simc_cut_all.Scale(normfac_simc)
pmiss_z_protons_simc_cut_all.Scale(normfac_simc)
emiss_protons_simc_cut_all.Scale(normfac_simc)
W_protons_simc_cut_all.Scale(normfac_simc)
MMp_protons_simc_cut_all.Scale(normfac_simc)

############################################################################################################################################

#Dummy Subtraction
H_gtr_beta_protons_dummysub_data_cut_all.Add(H_gtr_beta_protons_dummy_prompt_cut_all,-1)
H_gtr_xp_protons_dummysub_data_cut_all.Add(H_gtr_xp_protons_dummy_prompt_cut_all,-1)
H_gtr_yp_protons_dummysub_data_cut_all.Add(H_gtr_yp_protons_dummy_prompt_cut_all,-1)
H_gtr_dp_protons_dummysub_data_cut_all.Add(H_gtr_dp_protons_dummy_prompt_cut_all,-1)
H_gtr_p_protons_dummysub_data_cut_all.Add(H_gtr_p_protons_dummy_prompt_cut_all,-1)
H_dc_x_fp_protons_dummysub_data_cut_all.Add(H_dc_x_fp_protons_dummy_prompt_cut_all,-1)
H_dc_y_fp_protons_dummysub_data_cut_all.Add(H_dc_y_fp_protons_dummy_prompt_cut_all,-1)
H_dc_xp_fp_protons_dummysub_data_cut_all.Add(H_dc_xp_fp_protons_dummy_prompt_cut_all,-1)
H_dc_yp_fp_protons_dummysub_data_cut_all.Add(H_dc_yp_fp_protons_dummy_prompt_cut_all,-1)
H_hod_goodscinhit_protons_dummysub_data_cut_all.Add(H_hod_goodscinhit_protons_dummy_prompt_cut_all,-1)
H_hod_goodstarttime_protons_dummysub_data_cut_all.Add(H_hod_goodstarttime_protons_dummy_prompt_cut_all,-1)
H_cal_etotnorm_protons_dummysub_data_cut_all.Add(H_cal_etotnorm_protons_dummy_prompt_cut_all,-1)
H_cal_etottracknorm_protons_dummysub_data_cut_all.Add(H_cal_etottracknorm_protons_dummy_prompt_cut_all,-1)
H_cer_npeSum_protons_dummysub_data_cut_all.Add(H_cer_npeSum_protons_dummy_prompt_cut_all,-1)
H_RFTime_Dist_protons_dummysub_data_cut_all.Add(H_RFTime_Dist_protons_dummy_prompt_cut_all,-1)
P_gtr_beta_protons_dummysub_data_cut_all.Add(P_gtr_beta_protons_dummy_prompt_cut_all,-1)
P_gtr_xp_protons_dummysub_data_cut_all.Add(P_gtr_xp_protons_dummy_prompt_cut_all,-1)
P_gtr_yp_protons_dummysub_data_cut_all.Add(P_gtr_yp_protons_dummy_prompt_cut_all,-1)
P_gtr_dp_protons_dummysub_data_cut_all.Add(P_gtr_dp_protons_dummy_prompt_cut_all,-1)
P_gtr_p_protons_dummysub_data_cut_all.Add(P_gtr_p_protons_dummy_prompt_cut_all,-1)
P_dc_x_fp_protons_dummysub_data_cut_all.Add(P_dc_x_fp_protons_dummy_prompt_cut_all,-1)
P_dc_y_fp_protons_dummysub_data_cut_all.Add(P_dc_y_fp_protons_dummy_prompt_cut_all,-1)
P_dc_xp_fp_protons_dummysub_data_cut_all.Add(P_dc_xp_fp_protons_dummy_prompt_cut_all,-1)
P_dc_yp_fp_protons_dummysub_data_cut_all.Add(P_dc_yp_fp_protons_dummy_prompt_cut_all,-1)
P_hod_goodscinhit_protons_dummysub_data_cut_all.Add(P_hod_goodscinhit_protons_dummy_prompt_cut_all,-1)
P_hod_goodstarttime_protons_dummysub_data_cut_all.Add(P_hod_goodstarttime_protons_dummy_prompt_cut_all,-1)
P_cal_etotnorm_protons_dummysub_data_cut_all.Add(P_cal_etotnorm_protons_dummy_prompt_cut_all,-1)
P_cal_etottracknorm_protons_dummysub_data_cut_all.Add(P_cal_etottracknorm_protons_dummy_prompt_cut_all,-1)
P_hgcer_npeSum_protons_dummysub_data_cut_all.Add(P_hgcer_npeSum_protons_dummy_prompt_cut_all,-1)
P_hgcer_xAtCer_protons_dummysub_data_cut_all.Add(P_hgcer_xAtCer_protons_dummy_prompt_cut_all,-1)
P_hgcer_yAtCer_protons_dummysub_data_cut_all.Add(P_hgcer_yAtCer_protons_dummy_prompt_cut_all,-1)
P_ngcer_npeSum_protons_dummysub_data_cut_all.Add(P_ngcer_npeSum_protons_dummy_prompt_cut_all,-1)
P_ngcer_xAtCer_protons_dummysub_data_cut_all.Add(P_ngcer_xAtCer_protons_dummy_prompt_cut_all,-1)
P_ngcer_yAtCer_protons_dummysub_data_cut_all.Add(P_ngcer_yAtCer_protons_dummy_prompt_cut_all,-1)
P_aero_npeSum_protons_dummysub_data_cut_all.Add(P_aero_npeSum_protons_dummy_prompt_cut_all,-1)
P_aero_xAtAero_protons_dummysub_data_cut_all.Add(P_aero_xAtAero_protons_dummy_prompt_cut_all,-1)
P_aero_yAtAero_protons_dummysub_data_cut_all.Add(P_aero_yAtAero_protons_dummy_prompt_cut_all,-1)
P_kin_MMp_protons_dummysub_data_cut_all.Add(P_kin_MMp_protons_dummy_prompt_cut_all,-1)
P_RFTime_Dist_protons_dummysub_data_cut_all.Add(P_RFTime_Dist_protons_dummy_prompt_cut_all,-1)
CTime_epCoinTime_ROC1_protons_dummysub_data_cut_all.Add(CTime_epCoinTime_ROC1_protons_dummy_prompt_cut_all,-1)
P_kin_secondary_pmiss_protons_dummysub_data_cut_all.Add(P_kin_secondary_pmiss_protons_dummy_prompt_cut_all,-1)
P_kin_secondary_pmiss_x_protons_dummysub_data_cut_all.Add(P_kin_secondary_pmiss_x_protons_dummy_prompt_cut_all,-1)
P_kin_secondary_pmiss_y_protons_dummysub_data_cut_all.Add(P_kin_secondary_pmiss_y_protons_dummy_prompt_cut_all,-1)
P_kin_secondary_pmiss_z_protons_dummysub_data_cut_all.Add(P_kin_secondary_pmiss_z_protons_dummy_prompt_cut_all,-1)
P_kin_secondary_Erecoil_protons_dummysub_data_cut_all.Add(P_kin_secondary_Erecoil_protons_dummy_prompt_cut_all,-1)
P_kin_secondary_emiss_protons_dummysub_data_cut_all.Add(P_kin_secondary_emiss_protons_dummy_prompt_cut_all,-1)
P_kin_secondary_Mrecoil_protons_dummysub_data_cut_all.Add(P_kin_secondary_Mrecoil_protons_dummy_prompt_cut_all,-1)
P_kin_secondary_W_protons_dummysub_data_cut_all.Add(P_kin_secondary_W_protons_dummy_prompt_cut_all,-1)
MMsquared_dummysub_data_cut_all.Add(MMsquared_dummy_prompt_cut_all,-1)

###########################################################################################################################################

# Define a function for fitting a Gaussian with dynamically determined FWHM range
def fit_gaussian(hist, x_min, x_max, dtype):

#    print(hist.GetName(),dtype,"-"*25)

    # Find the corresponding bin numbers
    bin_min = hist.GetXaxis().FindBin(x_min)
    bin_max = hist.GetXaxis().FindBin(x_max)

    # Find the maximum value within the specified range
    max_bin = bin_min
    max_value = hist.GetBinContent(max_bin)
    for i in range(bin_min, bin_max):
        if hist.GetBinContent(i) > max_value:
            max_bin = i
            max_value = hist.GetBinContent(i)

    # Print the results
#    print("max_bin", max_bin)
#    print("max_value", max_value)
#    print("bin_center",hist.GetBinCenter(max_bin))

    half_max = max_value*0.75

    # Find left and right bins closest to half-max value
    left_bin = max_bin
    right_bin = max_bin
    while hist.GetBinContent(left_bin) > half_max and left_bin > 1:
        left_bin -= 1
    while hist.GetBinContent(right_bin) > half_max and right_bin < hist.GetNbinsX():
        right_bin += 1

    #min_range = hist.GetBinCenter(max_bin-100)
    #max_range = hist.GetBinCenter(max_bin+100)

    min_range = hist.GetBinCenter(left_bin)
#    print("min_range",min_range)
    max_range = hist.GetBinCenter(right_bin)
#    print("max_range",max_range)
#    print("="*40)

    hist.Fit("gaus", "Q", "", min_range, max_range)
    fit_func = hist.GetFunction('gaus')

    if dtype == "simc":
        fit_func.SetLineColor(kRed)
    if dtype == "data":
        fit_func.SetLineColor(kBlue)
#    if dtype == "dummy":
#        fit_func.SetLineColor(kGreen)

#    print("="*40)
    mean = fit_func.GetParameter(1)
    mean_err = fit_func.GetParError(1)
#    print("mean value",mean)
#    print("mean error",mean_err)
#    print("="*40)
    return [mean, mean_err]

###########################################################################################################################################

# Plot ratios between Data/SIMC
N_data_pmiss_x = int(P_kin_secondary_pmiss_x_protons_dummysub_data_cut_all.Integral())
N_data_pmiss_y = int(P_kin_secondary_pmiss_y_protons_dummysub_data_cut_all.Integral())
N_data_pmiss_z = int(P_kin_secondary_pmiss_z_protons_dummysub_data_cut_all.Integral())
N_data_pmiss = int(P_kin_secondary_pmiss_protons_dummysub_data_cut_all.Integral())
N_data_emiss = int(P_kin_secondary_emiss_protons_dummysub_data_cut_all.Integral())
N_data_W = int(P_kin_secondary_W_protons_dummysub_data_cut_all.Integral())

N_simc_pmiss_x = int(pmiss_x_protons_simc_cut_all.Integral())
N_simc_pmiss_y = int(pmiss_y_protons_simc_cut_all.Integral())
N_simc_pmiss_z = int(pmiss_z_protons_simc_cut_all.Integral())
N_simc_pmiss = int(pmiss_protons_simc_cut_all.Integral())
N_simc_emiss = int(emiss_protons_simc_cut_all.Integral())
N_simc_W = int(W_protons_simc_cut_all.Integral())

dataSimcRatio_pmiss_x = N_data_pmiss_x/N_simc_pmiss_x
dataSimcRatio_pmiss_y = N_data_pmiss_y/N_simc_pmiss_y
dataSimcRatio_pmiss_z = N_data_pmiss_z/N_simc_pmiss_z
dataSimcRatio_pmiss = N_data_pmiss/N_simc_pmiss
dataSimcRatio_emiss = N_data_emiss/N_simc_emiss
dataSimcRatio_W = N_data_W/N_simc_W

#dataSimcRatio_pmiss_x = N_simc_pmiss_x/N_data_pmiss_x
#dataSimcRatio_pmiss_y = N_simc_pmiss_y/N_data_pmiss_y
#dataSimcRatio_pmiss_z = N_simc_pmiss_z/N_data_pmiss_z
#dataSimcRatio_pmiss = N_simc_pmiss/N_data_pmiss
#dataSimcRatio_emiss = N_simc_emiss/N_data_emiss
#dataSimcRatio_W = N_simc_W/N_data_W

# Calculate errors (square root of counts for Poisson statistics)
dN_data_pmiss_x = ma.sqrt(N_data_pmiss_x)
dN_data_pmiss_y = ma.sqrt(N_data_pmiss_y)
dN_data_pmiss_z = ma.sqrt(N_data_pmiss_z)
dN_data_pmiss = ma.sqrt(N_data_pmiss)
dN_data_emiss = ma.sqrt(N_data_emiss)
dN_data_W = ma.sqrt(N_data_W)

dN_simc_pmiss_x = ma.sqrt(N_simc_pmiss_x)
dN_simc_pmiss_y = ma.sqrt(N_simc_pmiss_y)
dN_simc_pmiss_z = ma.sqrt(N_simc_pmiss_z)
dN_simc_pmiss = ma.sqrt(N_simc_pmiss)
dN_simc_emiss = ma.sqrt(N_simc_emiss)
dN_simc_W = ma.sqrt(N_simc_W)

dataSimcRatio_err_pmiss_x = dataSimcRatio_pmiss_x * ma.sqrt((dN_data_pmiss_x/N_data_pmiss_x)**2 + (dN_simc_pmiss_x/N_simc_pmiss_x)**2)
dataSimcRatio_err_pmiss_y = dataSimcRatio_pmiss_y * ma.sqrt((dN_data_pmiss_y/N_data_pmiss_y)**2 + (dN_simc_pmiss_y/N_simc_pmiss_y)**2)
dataSimcRatio_err_pmiss_z = dataSimcRatio_pmiss_z * ma.sqrt((dN_data_pmiss_z/N_data_pmiss_z)**2 + (dN_simc_pmiss_z/N_simc_pmiss_z)**2)
dataSimcRatio_err_pmiss = dataSimcRatio_pmiss * ma.sqrt((dN_data_pmiss / N_data_pmiss)**2 + (dN_simc_pmiss / N_simc_pmiss)**2)
dataSimcRatio_err_emiss = dataSimcRatio_emiss * ma.sqrt((dN_data_emiss/N_data_emiss)**2 + (dN_simc_emiss/N_simc_emiss)**2)
dataSimcRatio_err_W = dataSimcRatio_W * ma.sqrt((dN_data_W/N_data_W)**2 + (dN_simc_W/N_simc_W)**2)

print("="*40)
print("Data/SIMC ratio pmiss_x = {:.3f} +/- {:.3f}".format(dataSimcRatio_pmiss_x, dataSimcRatio_err_pmiss_x))
print("Data/SIMC ratio pmiss_y = {:.3f} +/- {:.3f}".format(dataSimcRatio_pmiss_y, dataSimcRatio_err_pmiss_y))
print("Data/SIMC ratio pmiss_z = {:.3f} +/- {:.3f}".format(dataSimcRatio_pmiss_z, dataSimcRatio_err_pmiss_z))
print("Data/SIMC ratio pmiss = {:.3f} +/- {:.3f}".format(dataSimcRatio_pmiss, dataSimcRatio_err_pmiss))
print("Data/SIMC ratio emiss = {:.3f} +/- {:.3f}".format(dataSimcRatio_emiss, dataSimcRatio_err_emiss))
print("Data/SIMC ratio W = {:.3f} +/- {:.3f}".format(dataSimcRatio_W, dataSimcRatio_err_W))
print("="*40)

##############################################################################################################################################

# Removes stat box
ROOT.gStyle.SetOptStat(0)

# Saving histograms in PDF
c1_delta = TCanvas("c1_delta", "Delta and Target Distributions", 100, 0, 1400, 600)
c1_delta.Divide(3,2)
Beam_Energy_S, HMS_p, HMS_theta, SHMS_p, SHMS_theta  = filtered_data_df['Beam_Energy'].iloc[0],filtered_data_df['HMS_P_Central'].iloc[0],filtered_data_df['HMS_Angle'].iloc[0],filtered_data_df['SHMS_P_Central'].iloc[0],filtered_data_df['SHMS_Angle'].iloc[0]

c1_delta.cd(1)
c1_delta_text_lines = [
    ROOT.TText(0.5, 0.9, "HeePCoin Setting"),
    ROOT.TText(0.5, 0.8, 'Beam Energy = {:.3f}'.format(Beam_Energy_S)),
    ROOT.TText(0.5, 0.7, 'HMS_p = {:.3f}'.format(HMS_p)),
    ROOT.TText(0.5, 0.6, 'HMS_theta = {:.3f}'.format(HMS_theta)),
    ROOT.TText(0.5, 0.5, 'SHMS_p = {:.3f}'.format(SHMS_p)),
    ROOT.TText(0.5, 0.4, 'SHMS_theta = {:.3f}'.format(SHMS_theta)),
    ROOT.TText(0.5, 0.3, "Red = SIMC"),
    ROOT.TText(0.5, 0.2, "Blue = DATA")
]
for c1_delta_text in c1_delta_text_lines:
    c1_delta_text.SetTextSize(0.07)
    c1_delta_text.SetTextAlign(22)
    c1_delta_text.SetTextColor(ROOT.kGreen + 4)
    if c1_delta_text.GetTitle() == "Red = SIMC":
       c1_delta_text.SetTextColor(ROOT.kRed)  # Setting text color to red
    if c1_delta_text.GetTitle() == "Blue = DATA":
       c1_delta_text.SetTextColor(ROOT.kBlue)  # Setting text color to red
    c1_delta_text.Draw()
c1_delta.cd(2)
H_gtr_dp_protons_dummysub_data_cut_all.GetXaxis().SetRangeUser(-15, 15)
H_gtr_dp_protons_dummysub_data_cut_all.SetLineColor(kBlue)
H_gtr_dp_protons_dummysub_data_cut_all.Draw("E1")
H_hsdelta_protons_simc_cut_all.GetXaxis().SetRangeUser(-15, 15)
H_hsdelta_protons_simc_cut_all.SetLineColor(kRed)
H_hsdelta_protons_simc_cut_all.Draw("same, E1")
c1_delta.cd(3)
P_gtr_dp_protons_dummysub_data_cut_all.GetXaxis().SetRangeUser(-20, 20)
P_gtr_dp_protons_dummysub_data_cut_all.SetLineColor(kBlue)
P_gtr_dp_protons_dummysub_data_cut_all.Draw("E1")
P_ssdelta_protons_simc_cut_all.SetLineColor(kRed)
P_ssdelta_protons_simc_cut_all.Draw("same, E1")
c1_delta.cd(4)

c1_delta.cd(5)
H_gtr_xp_protons_dummysub_data_cut_all.GetXaxis().SetRangeUser(-0.15, 0.15)
H_gtr_xp_protons_dummysub_data_cut_all.SetLineColor(kBlue)
H_gtr_xp_protons_dummysub_data_cut_all.Draw("E1")
H_hsxptar_protons_simc_cut_all.SetLineColor(kRed)
H_hsxptar_protons_simc_cut_all.Draw("same, E1")
c1_delta.cd(6)
H_gtr_yp_protons_dummysub_data_cut_all.GetXaxis().SetRangeUser(-0.1, 0.1)
H_gtr_yp_protons_dummysub_data_cut_all.SetLineColor(kBlue)
H_gtr_yp_protons_dummysub_data_cut_all.Draw("E1")
H_hsyptar_protons_simc_cut_all.SetLineColor(kRed)
H_hsyptar_protons_simc_cut_all.Draw("same, E1")
c1_delta.Print(Proton_Analysis_Distributions + '(')

c1_tar = TCanvas("c1_tar", "Focal Plane and Target Distributions", 100, 0, 1400, 600)
c1_tar.Divide(3,2)
c1_tar.cd(1)
P_gtr_xp_protons_dummysub_data_cut_all.GetXaxis().SetRangeUser(-0.1, 0.1)
P_gtr_xp_protons_dummysub_data_cut_all.SetLineColor(kBlue)
P_gtr_xp_protons_dummysub_data_cut_all.Draw("E1")
P_ssxptar_protons_simc_cut_all.SetLineColor(kRed)
P_ssxptar_protons_simc_cut_all.Draw("same, E1")
c1_tar.cd(2)
P_gtr_yp_protons_dummysub_data_cut_all.GetXaxis().SetRangeUser(-0.1, 0.1)
P_gtr_yp_protons_dummysub_data_cut_all.SetLineColor(kBlue)
P_gtr_yp_protons_dummysub_data_cut_all.Draw("E1")
P_ssyptar_protons_simc_cut_all.SetLineColor(kRed)
P_ssyptar_protons_simc_cut_all.Draw("same, E1")
c1_tar.cd(3)
H_dc_xp_fp_protons_dummysub_data_cut_all.GetXaxis().SetRangeUser(-0.1, 0.1)
H_dc_xp_fp_protons_dummysub_data_cut_all.SetLineColor(kBlue)
H_dc_xp_fp_protons_dummysub_data_cut_all.Draw("E1")
H_hsxpfp_protons_simc_cut_all.SetLineColor(kRed)
H_hsxpfp_protons_simc_cut_all.Draw("same, E1")
c1_tar.cd(4)
H_dc_yp_fp_protons_dummysub_data_cut_all.GetXaxis().SetRangeUser(-0.1, 0.1)
H_dc_yp_fp_protons_dummysub_data_cut_all.SetLineColor(kBlue)
H_dc_yp_fp_protons_dummysub_data_cut_all.Draw("E1")
H_hsypfp_protons_simc_cut_all.SetLineColor(kRed)
H_hsypfp_protons_simc_cut_all.Draw("same, E1")
c1_tar.cd(5)
P_ssxpfp_protons_simc_cut_all.GetXaxis().SetRangeUser(-0.1, 0.1)
P_dc_xp_fp_protons_dummysub_data_cut_all.SetLineColor(kBlue)
P_dc_xp_fp_protons_dummysub_data_cut_all.Draw("E1")
P_ssxpfp_protons_simc_cut_all.SetLineColor(kRed)
P_ssxpfp_protons_simc_cut_all.Draw("same, E1")
c1_tar.cd(6)
P_dc_yp_fp_protons_dummysub_data_cut_all.GetXaxis().SetRangeUser(-0.1, 0.1)
P_dc_yp_fp_protons_dummysub_data_cut_all.SetLineColor(kBlue)
P_dc_yp_fp_protons_dummysub_data_cut_all.Draw("E1")
P_ssypfp_protons_simc_cut_all.SetLineColor(kRed)
P_ssypfp_protons_simc_cut_all.Draw("same, E1")
c1_tar.Print(Proton_Analysis_Distributions)

c1_fit = TCanvas("c1_fit", "Missing Mass and Momentum Distributions", 100, 0, 1400, 600)
c1_fit.Divide(3,2)
c1_fit.cd(1)
tmp_b_mean_pmiss_x_simc = fit_gaussian(pmiss_x_protons_simc_cut_all,-0.1, 0.1, "simc")
tmp_b_mean_pmiss_x_data = fit_gaussian(P_kin_secondary_pmiss_x_protons_dummysub_data_cut_all,-0.1, 0.1, "data")
pmiss_x_protons_simc_cut_all.GetXaxis().SetRangeUser(-0.1, 0.1)
pmiss_x_protons_simc_cut_all.SetLineColor(kRed)
pmiss_x_protons_simc_cut_all.SetMarkerColor(kRed)
pmiss_x_protons_simc_cut_all.Draw("E1")
P_kin_secondary_pmiss_x_protons_dummysub_data_cut_all.GetXaxis().SetRangeUser(-0.2, 0.2)
P_kin_secondary_pmiss_x_protons_dummysub_data_cut_all.SetLineColor(kBlue)
P_kin_secondary_pmiss_x_protons_dummysub_data_cut_all.SetMarkerColor(kBlue)
P_kin_secondary_pmiss_x_protons_dummysub_data_cut_all.Draw("same, E1")
Ratio_pmissx = ROOT.TMarker()
Ratio_pmissx.SetMarkerColor(kBlack)
Ratio_pmissx.SetMarkerStyle(20)
pmiss_x_legend = TLegend(0.58, 0.63, 0.9, 0.9)
pmiss_x_legend.SetBorderSize(1)
pmiss_x_legend.SetFillColor(0)
pmiss_x_legend.SetFillStyle(0)
pmiss_x_legend.SetTextSize(0.04)
pmiss_x_legend.AddEntry(pmiss_x_protons_simc_cut_all, "MC = {:.2f}+/-{:.2f}".format(N_simc_pmiss_x,dN_simc_pmiss_x), "p") # lp for liine and point
pmiss_x_legend.AddEntry(P_kin_secondary_pmiss_x_protons_dummysub_data_cut_all, "Data = {:.2f}+/-{:.2f}".format(N_data_pmiss_x,dN_data_pmiss_x), "p")
pmiss_x_legend.AddEntry(Ratio_pmissx,"Ratio = {:.3f}+/-{:.3f}".format(dataSimcRatio_pmiss_x,dataSimcRatio_err_pmiss_x), "p")
pmiss_x_legend.Draw()
pmiss_x_fit_func_simc = pmiss_x_protons_simc_cut_all.GetFunction('gaus')
pmiss_x_fit_func_simc.SetLineWidth(1)
pmiss_x_fit_func_data = P_kin_secondary_pmiss_x_protons_dummysub_data_cut_all.GetFunction('gaus')
pmiss_x_fit_func_data.SetLineWidth(1)
c1_fit.cd(2)
tmp_b_mean_pmiss_y_simc = fit_gaussian(pmiss_y_protons_simc_cut_all,-0.1, 0.1, "simc")
tmp_b_mean_pmiss_y_data = fit_gaussian(P_kin_secondary_pmiss_y_protons_dummysub_data_cut_all,-0.1, 0.1, "data")
pmiss_y_protons_simc_cut_all.GetXaxis().SetRangeUser(-0.1, 0.1)
pmiss_y_protons_simc_cut_all.SetLineColor(kRed)
pmiss_y_protons_simc_cut_all.SetMarkerColor(kRed)
pmiss_y_protons_simc_cut_all.Draw("E1")
P_kin_secondary_pmiss_y_protons_dummysub_data_cut_all.GetXaxis().SetRangeUser(-0.1, 0.1)
P_kin_secondary_pmiss_y_protons_dummysub_data_cut_all.SetLineColor(kBlue)
P_kin_secondary_pmiss_y_protons_dummysub_data_cut_all.SetMarkerColor(kBlue)
P_kin_secondary_pmiss_y_protons_dummysub_data_cut_all.Draw("same, E1")
Ratio_pmissy = ROOT.TMarker()
Ratio_pmissy.SetMarkerColor(kBlack)
Ratio_pmissy.SetMarkerStyle(20)
pmiss_y_legend = TLegend(0.58, 0.63, 0.9, 0.9)
pmiss_y_legend.SetBorderSize(1)
pmiss_y_legend.SetFillColor(0)
pmiss_y_legend.SetFillStyle(0)
pmiss_y_legend.SetTextSize(0.04)
pmiss_y_legend.AddEntry(pmiss_y_protons_simc_cut_all, "MC = {:.2f}+/-{:.2f}".format(N_simc_pmiss_y,dN_simc_pmiss_y), "p")
pmiss_y_legend.AddEntry(P_kin_secondary_pmiss_y_protons_dummysub_data_cut_all, "Data = {:.2f}+/-{:.2f}".format(N_data_pmiss_y,dN_data_pmiss_y), "p")
pmiss_y_legend.AddEntry(Ratio_pmissy,"Ratio = {:.3f}+/-{:.3f}".format(dataSimcRatio_pmiss_y,dataSimcRatio_err_pmiss_y), "p")
pmiss_y_legend.Draw()
pmiss_y_fit_func_simc = pmiss_y_protons_simc_cut_all.GetFunction('gaus')
pmiss_y_fit_func_simc.SetLineWidth(1)
pmiss_y_fit_func_data = P_kin_secondary_pmiss_y_protons_dummysub_data_cut_all.GetFunction('gaus')
pmiss_y_fit_func_data.SetLineWidth(1)
c1_fit.cd(3)
tmp_b_mean_pmiss_z_simc = fit_gaussian(pmiss_z_protons_simc_cut_all,-0.1, 0.1, "simc")
tmp_b_mean_pmiss_z_data = fit_gaussian(P_kin_secondary_pmiss_z_protons_dummysub_data_cut_all,-0.1, 0.1, "data")
pmiss_z_protons_simc_cut_all.GetXaxis().SetRangeUser(-0.1, 0.1)
pmiss_z_protons_simc_cut_all.SetLineColor(kRed)
pmiss_z_protons_simc_cut_all.SetMarkerColor(kRed)
pmiss_z_protons_simc_cut_all.Draw("E1")
P_kin_secondary_pmiss_z_protons_dummysub_data_cut_all.GetXaxis().SetRangeUser(-0.1, 0.1)
P_kin_secondary_pmiss_z_protons_dummysub_data_cut_all.SetLineColor(kBlue)
P_kin_secondary_pmiss_z_protons_dummysub_data_cut_all.SetMarkerColor(kBlue)
P_kin_secondary_pmiss_z_protons_dummysub_data_cut_all.Draw("same, E1")
Ratio_pmissz = ROOT.TMarker()
Ratio_pmissz.SetMarkerColor(kBlack)
Ratio_pmissz.SetMarkerStyle(20)
pmiss_z_legend = TLegend(0.58, 0.63, 0.9, 0.9)
pmiss_z_legend.SetBorderSize(1)
pmiss_z_legend.SetFillColor(0)
pmiss_z_legend.SetFillStyle(0)
pmiss_z_legend.SetTextSize(0.04)
pmiss_z_legend.AddEntry(pmiss_z_protons_simc_cut_all, "MC = {:.2f}+/-{:.2f}".format(N_simc_pmiss_z,dN_simc_pmiss_z), "p")
pmiss_z_legend.AddEntry(P_kin_secondary_pmiss_z_protons_dummysub_data_cut_all, "Data = {:.2f}+/-{:.2f}".format(N_data_pmiss_z,dN_data_pmiss_z), "p")
pmiss_z_legend.AddEntry(Ratio_pmissz,"Ratio = {:.3f}+/-{:.3f}".format(dataSimcRatio_pmiss_z,dataSimcRatio_err_pmiss_z), "p")
pmiss_z_legend.Draw()
pmiss_z_fit_func_simc = pmiss_z_protons_simc_cut_all.GetFunction('gaus')
pmiss_z_fit_func_simc.SetLineWidth(1)
pmiss_z_fit_func_data = P_kin_secondary_pmiss_z_protons_dummysub_data_cut_all.GetFunction('gaus')
pmiss_z_fit_func_data.SetLineWidth(1)
c1_fit.cd(4)
tmp_b_mean_pmiss_simc = fit_gaussian(pmiss_protons_simc_cut_all,-0.1, 0.1, "simc")
tmp_b_mean_pmiss_data = fit_gaussian(P_kin_secondary_pmiss_protons_dummysub_data_cut_all,-0.1, 0.1, "data")
pmiss_protons_simc_cut_all.GetXaxis().SetRangeUser(-0.1, 0.3)
pmiss_protons_simc_cut_all.SetLineColor(kRed)
pmiss_protons_simc_cut_all.SetMarkerColor(kRed)
pmiss_protons_simc_cut_all.Draw("E1")
P_kin_secondary_pmiss_protons_dummysub_data_cut_all.GetXaxis().SetRangeUser(-0.1, 0.3)
P_kin_secondary_pmiss_protons_dummysub_data_cut_all.SetLineColor(kBlue)
P_kin_secondary_pmiss_protons_dummysub_data_cut_all.SetMarkerColor(kBlue)
P_kin_secondary_pmiss_protons_dummysub_data_cut_all.Draw("same, E1")
Ratio_pmiss = ROOT.TMarker()
Ratio_pmiss.SetMarkerColor(kBlack)
Ratio_pmiss.SetMarkerStyle(20)
pmiss_legend = TLegend(0.58, 0.63, 0.9, 0.9)
pmiss_legend.SetBorderSize(1)
pmiss_legend.SetFillColor(0)
pmiss_legend.SetFillStyle(0)
pmiss_legend.SetTextSize(0.04)
pmiss_legend.AddEntry(pmiss_protons_simc_cut_all, "MC = {:.2f}+/-{:.2f}".format(N_simc_pmiss,dN_simc_pmiss), "p")
pmiss_legend.AddEntry(P_kin_secondary_pmiss_protons_dummysub_data_cut_all, "Data = {:.2f}+/-{:.2f}".format(N_data_pmiss,dN_data_pmiss), "p")
pmiss_legend.AddEntry(Ratio_pmiss,"Ratio = {:.3f}+/-{:.3f}".format(dataSimcRatio_pmiss,dataSimcRatio_err_pmiss), "p")
pmiss_legend.Draw()
pmiss_fit_func_simc = pmiss_protons_simc_cut_all.GetFunction('gaus')
pmiss_fit_func_simc.SetLineWidth(1)
pmiss_fit_func_data = P_kin_secondary_pmiss_protons_dummysub_data_cut_all.GetFunction('gaus')
pmiss_fit_func_data.SetLineWidth(1)
c1_fit.cd(5)
tmp_b_mean_emiss_simc = fit_gaussian(emiss_protons_simc_cut_all,-0.1, 0.1, "simc")
tmp_b_mean_emiss_data = fit_gaussian(P_kin_secondary_emiss_protons_dummysub_data_cut_all,-0.1, 0.1, "data")
emiss_protons_simc_cut_all.GetXaxis().SetRangeUser(-0.1, 0.1)
emiss_protons_simc_cut_all.SetLineColor(kRed)
emiss_protons_simc_cut_all.SetMarkerColor(kRed)
emiss_protons_simc_cut_all.Draw("E1")
P_kin_secondary_emiss_protons_dummysub_data_cut_all.GetXaxis().SetRangeUser(-0.1, 0.1)
P_kin_secondary_emiss_protons_dummysub_data_cut_all.SetLineColor(kBlue)
P_kin_secondary_emiss_protons_dummysub_data_cut_all.SetMarkerColor(kBlue)
P_kin_secondary_emiss_protons_dummysub_data_cut_all.Draw("same, E1")
Ratio_emiss = ROOT.TMarker()
Ratio_emiss.SetMarkerColor(kBlack)
Ratio_emiss.SetMarkerStyle(20)
emiss_legend = TLegend(0.58, 0.63, 0.9, 0.9)
emiss_legend.SetBorderSize(1)
emiss_legend.SetFillColor(0)
emiss_legend.SetFillStyle(0)
emiss_legend.SetTextSize(0.04)
emiss_legend.AddEntry(emiss_protons_simc_cut_all, "MC = {:.2f}+/-{:.2f}".format(N_simc_emiss,dN_simc_emiss), "p")
emiss_legend.AddEntry(P_kin_secondary_emiss_protons_dummysub_data_cut_all, "Data = {:.2f}+/-{:.2f}".format(N_data_emiss,dN_data_emiss), "p")
emiss_legend.AddEntry(Ratio_emiss,"Ratio = {:.3f}+/-{:.3f}".format(dataSimcRatio_emiss,dataSimcRatio_err_emiss), "p")
emiss_legend.Draw()
emiss_fit_func_simc = emiss_protons_simc_cut_all.GetFunction('gaus')
emiss_fit_func_simc.SetLineWidth(1)
emiss_fit_func_data = P_kin_secondary_emiss_protons_dummysub_data_cut_all.GetFunction('gaus')
emiss_fit_func_data.SetLineWidth(1)
c1_fit.cd(6)
tmp_b_mean_W_simc = fit_gaussian(W_protons_simc_cut_all,0.7, 1.1, "simc")
tmp_b_mean_W_data = fit_gaussian(P_kin_secondary_W_protons_dummysub_data_cut_all,0.7, 1.1, "data")
W_protons_simc_cut_all.GetXaxis().SetRangeUser(0.6, 1.4)
W_protons_simc_cut_all.SetLineColor(kRed)
W_protons_simc_cut_all.SetMarkerColor(kRed)
W_protons_simc_cut_all.Draw("E1")
P_kin_secondary_W_protons_dummysub_data_cut_all.GetXaxis().SetRangeUser(0.6, 1.4)
P_kin_secondary_W_protons_dummysub_data_cut_all.SetLineColor(kBlue)
P_kin_secondary_W_protons_dummysub_data_cut_all.SetMarkerColor(kBlue)
P_kin_secondary_W_protons_dummysub_data_cut_all.Draw("same, E1")
Ratio_W = ROOT.TMarker()
Ratio_W.SetMarkerColor(kBlack)
Ratio_W.SetMarkerStyle(20)
W_legend = TLegend(0.58, 0.63, 0.9, 0.9)
W_legend.SetBorderSize(1)
W_legend.SetFillColor(0)
W_legend.SetFillStyle(0)
W_legend.SetTextSize(0.04)
W_legend.AddEntry(W_protons_simc_cut_all, "MC = {:.2f}+/-{:.2f}".format(N_simc_W,dN_simc_W), "p")
W_legend.AddEntry(P_kin_secondary_W_protons_dummysub_data_cut_all, "Data = {:.2f}+/-{:.2f}".format(N_data_W,dN_data_W), "p")
W_legend.AddEntry(Ratio_W,"Ratio = {:.3f}+/-{:.3f}".format(dataSimcRatio_W,dataSimcRatio_err_W), "p")
W_legend.Draw()
W_fit_func_simc = W_protons_simc_cut_all.GetFunction('gaus')
W_fit_func_simc.SetLineWidth(1)
W_fit_func_data = P_kin_secondary_W_protons_dummysub_data_cut_all.GetFunction('gaus')
W_fit_func_data.SetLineWidth(1)
c1_fit.Print(Proton_Analysis_Distributions + ')')

#############################################################################################################################################

# Making directories in output file
outHistFile = ROOT.TFile.Open("%s/%s_%s_Output_Data.root" % (OUTPATH, BEAM_ENERGY, MaxEvent) , "RECREATE")
#d_Uncut_Proton_Events_Data = outHistFile.mkdir("Uncut_Proton_Events_Data")
#d_Cut_Proton_Events_All_Data = outHistFile.mkdir("Cut_Proton_Events_All_Data")
#d_Cut_Proton_Events_Prompt_Data = outHistFile.mkdir("Cut_Proton_Events_Prompt_Data")
#d_Cut_Proton_Events_Random = outHistFile.mkdir("Cut_Proton_Events_Random")
#d_Cut_Proton_Events_Prompt_Dummy = outHistFile.mkdir("Cut_Proton_Events_Prompt_Dummy")
d_Cut_Proton_Events_DummySub_Data = outHistFile.mkdir("Cut_Proton_Events_DummySub_Data")
d_Cut_Proton_Events_All_SIMC = outHistFile.mkdir("Cut_Proton_Events_All_SIMC")

# Writing Histograms for protons                                                                  

d_Cut_Proton_Events_DummySub_Data.cd()
H_gtr_beta_protons_dummysub_data_cut_all.Write()
H_gtr_xp_protons_dummysub_data_cut_all.Write()
H_gtr_yp_protons_dummysub_data_cut_all.Write()
H_gtr_dp_protons_dummysub_data_cut_all.Write()
H_gtr_p_protons_dummysub_data_cut_all.Write()
H_dc_x_fp_protons_dummysub_data_cut_all.Write()
H_dc_y_fp_protons_dummysub_data_cut_all.Write()
H_dc_xp_fp_protons_dummysub_data_cut_all.Write()
H_dc_yp_fp_protons_dummysub_data_cut_all.Write()
H_hod_goodscinhit_protons_dummysub_data_cut_all.Write()
H_hod_goodstarttime_protons_dummysub_data_cut_all.Write()
H_cal_etotnorm_protons_dummysub_data_cut_all.Write()
H_cal_etottracknorm_protons_dummysub_data_cut_all.Write()
H_cer_npeSum_protons_dummysub_data_cut_all.Write()
H_RFTime_Dist_protons_dummysub_data_cut_all.Write()
P_gtr_beta_protons_dummysub_data_cut_all.Write()
P_gtr_xp_protons_dummysub_data_cut_all.Write()
P_gtr_yp_protons_dummysub_data_cut_all.Write()
P_gtr_dp_protons_dummysub_data_cut_all.Write()
P_gtr_p_protons_dummysub_data_cut_all.Write()
P_dc_x_fp_protons_dummysub_data_cut_all.Write()
P_dc_y_fp_protons_dummysub_data_cut_all.Write()
P_dc_xp_fp_protons_dummysub_data_cut_all.Write()
P_dc_yp_fp_protons_dummysub_data_cut_all.Write()
P_hod_goodscinhit_protons_dummysub_data_cut_all.Write()
P_hod_goodstarttime_protons_dummysub_data_cut_all.Write()
P_cal_etotnorm_protons_dummysub_data_cut_all.Write()
P_cal_etottracknorm_protons_dummysub_data_cut_all.Write()
P_hgcer_npeSum_protons_dummysub_data_cut_all.Write()
P_hgcer_xAtCer_protons_dummysub_data_cut_all.Write()
P_hgcer_yAtCer_protons_dummysub_data_cut_all.Write()
P_ngcer_npeSum_protons_dummysub_data_cut_all.Write()
P_ngcer_xAtCer_protons_dummysub_data_cut_all.Write()
P_ngcer_yAtCer_protons_dummysub_data_cut_all.Write()
P_aero_npeSum_protons_dummysub_data_cut_all.Write()
P_aero_xAtAero_protons_dummysub_data_cut_all.Write()
P_aero_yAtAero_protons_dummysub_data_cut_all.Write()
P_kin_MMp_protons_dummysub_data_cut_all.Write()
P_RFTime_Dist_protons_dummysub_data_cut_all.Write()
CTime_epCoinTime_ROC1_protons_dummysub_data_cut_all.Write()
P_kin_secondary_pmiss_protons_dummysub_data_cut_all.Write()
P_kin_secondary_pmiss_x_protons_dummysub_data_cut_all.Write()
P_kin_secondary_pmiss_y_protons_dummysub_data_cut_all.Write()
P_kin_secondary_pmiss_z_protons_dummysub_data_cut_all.Write()
P_kin_secondary_Erecoil_protons_dummysub_data_cut_all.Write()
P_kin_secondary_emiss_protons_dummysub_data_cut_all.Write()
P_kin_secondary_Mrecoil_protons_dummysub_data_cut_all.Write()
P_kin_secondary_W_protons_dummysub_data_cut_all.Write()
MMsquared_dummysub_data_cut_all.Write()

d_Cut_Proton_Events_All_SIMC.cd()
H_hsdelta_protons_simc_cut_all.Write()
H_hsxptar_protons_simc_cut_all.Write()
H_hsyptar_protons_simc_cut_all.Write()
H_hsytar_protons_simc_cut_all.Write()
H_hsxfp_protons_simc_cut_all.Write()
H_hsyfp_protons_simc_cut_all.Write()
H_hsxpfp_protons_simc_cut_all.Write()
H_hsypfp_protons_simc_cut_all.Write()
H_hsdeltai_protons_simc_cut_all.Write()
H_hsxptari_protons_simc_cut_all.Write()
H_hsyptari_protons_simc_cut_all.Write()
H_hsytari_protons_simc_cut_all.Write()
P_ssdelta_protons_simc_cut_all.Write()
P_ssxptar_protons_simc_cut_all.Write()
P_ssyptar_protons_simc_cut_all.Write()
P_ssytar_protons_simc_cut_all.Write()
P_ssxfp_protons_simc_cut_all.Write()
P_ssyfp_protons_simc_cut_all.Write()
P_ssxpfp_protons_simc_cut_all.Write()
P_ssypfp_protons_simc_cut_all.Write()
P_ssdeltai_protons_simc_cut_all.Write()
P_ssxptari_protons_simc_cut_all.Write()
P_ssyptari_protons_simc_cut_all.Write()
P_ssytari_protons_simc_cut_all.Write()
q_protons_simc_cut_all.Write()
nu_protons_simc_cut_all.Write()
Q2_protons_simc_cut_all.Write()
epsilon_protons_simc_cut_all.Write()
thetapq_protons_simc_cut_all.Write()
phipq_protons_simc_cut_all.Write()
pmiss_protons_simc_cut_all.Write()
pmiss_x_protons_simc_cut_all.Write()
pmiss_y_protons_simc_cut_all.Write()
pmiss_z_protons_simc_cut_all.Write()
emiss_protons_simc_cut_all.Write()
W_protons_simc_cut_all.Write()
MMp_protons_simc_cut_all.Write()

outHistFile.Close()
infile_DATA.Close() 
infile_DUMMY.Close()       
infile_SIMC.Close()
print ("Processing Complete")
