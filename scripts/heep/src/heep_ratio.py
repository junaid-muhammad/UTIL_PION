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

np.bool = bool
np.float = float

import root_numpy as rnp
import pandas as pd
import root_pandas as rpd
import ROOT
import scipy
from scipy.optimize import curve_fit
import scipy.integrate as integrate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from uncertainties import ufloat
import sys, math, os, subprocess
import array
import csv
import math
from ROOT import TCanvas, TPaveLabel, TColor, TGaxis, TH1F, TH2F, TPad, TStyle, gStyle, gPad, TLegend, TGaxis, TLine, TMath, TLatex, TPaveText, TArc, TGraphPolar, TText
from ROOT import kBlack, kCyan, kRed, kGreen, kMagenta, kBlue
from functools import reduce

##################################################################################################################################################

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
print("="*40)

# Defining Variables
E5p986_DATASIMCRatio = ufloat(1.054,0.005)
E6p395s1_DATASIMCRatio = ufloat(1.083,0.004)
E6p395s2_DATASIMCRatio = ufloat(0.941,0.004)
E6p395s3_DATASIMCRatio = ufloat(1.094,0.011)
E7p937_DATASIMCRatio = ufloat(1.114,0.009)
E8p479_DATASIMCRatio = ufloat(1.136,0.007)
E9p177_DATASIMCRatio = ufloat(1.258,0.014)
E9p876_DATASIMCRatio = ufloat(1.146,0.013)
E10p549_DATASIMCRatio = ufloat(1.220,0.011)

E5p986_BE = 5.986
E6p395s1_BE = 6.395
E6p395s2_BE = 6.395
E6p395s3_BE = 6.395
E7p937_BE = 7.937
E8p479_BE = 8.479
E9p177_BE = 9.177
E9p876_BE = 9.876
E10p549_BE = 10.549

print("="*40)

#################################################################################################################################################

# Create lists for x and y values along with their uncertainties
x_values = [E5p986_BE, E6p395s1_BE, E6p395s2_BE, E6p395s3_BE, E7p937_BE, E8p479_BE, E9p177_BE, E9p876_BE, E10p549_BE]
y_values = [E5p986_DATASIMCRatio.nominal_value, E6p395s1_DATASIMCRatio.nominal_value, E6p395s2_DATASIMCRatio.nominal_value, E6p395s3_DATASIMCRatio.nominal_value, E7p937_DATASIMCRatio.nominal_value, E8p479_DATASIMCRatio.nominal_value, E9p177_DATASIMCRatio.nominal_value, E9p876_DATASIMCRatio.nominal_value, E10p549_DATASIMCRatio.nominal_value]
yerr_values = [E5p986_DATASIMCRatio.std_dev, E6p395s1_DATASIMCRatio.std_dev, E6p395s2_DATASIMCRatio.std_dev, E6p395s3_DATASIMCRatio.std_dev, E7p937_DATASIMCRatio.std_dev, E8p479_DATASIMCRatio.std_dev, E9p177_DATASIMCRatio.std_dev, E9p876_DATASIMCRatio.std_dev, E10p549_DATASIMCRatio.std_dev]

print(x_values)
print(y_values)
print(yerr_values)
print("="*40)

# Extract nominal values and uncertainties for plotting
x_nominals = [float(x) for x in x_values]
y_nominals = [float(y) for y in y_values]
y_errors = [float(yerr) for yerr in yerr_values]

##################################################################################################################################################

# Output PDF File Name
print("Running as %s on %s, hallc_replay_lt path assumed as %s" % (USER, HOST, REPLAYPATH))

plt.figure(figsize=(12,8))

plt.subplot(111)
#plt.grid(zorder=1)
plt.xlim(4.0,12.0)
plt.ylim(0.9,1.3)
plt.errorbar(x_nominals, y_nominals, yerr=y_errors, fmt='o', markersize=8, color='black', linestyle='None', capsize=4, zorder=3, label='Ratios with uncertainties')
plt.scatter(x_nominals, y_nominals, color='blue', zorder=4, label='Ratios values')
# Add a red reference line at y=1
plt.axhline(y=1, color='red', linestyle='--', label='Reference line at y=1')
plt.xlabel('x values', fontsize=20)
plt.ylabel('y values', fontsize=20)
plt.title('Plot of x vs y with uncertainties', fontsize=18)
plt.legend()
#plt.grid(True)
plt.ylabel(r'Data/Simc Ratio', fontsize=20)
plt.xlabel(r'Beam Energies (GeV)', fontsize=20)
plt.tick_params(axis='x', labelsize=14)  # Increase x-axis tick size
plt.tick_params(axis='y', labelsize=14)  # Increase y-axis tick size
plt.xticks(rotation=90)
plt.title('Data/Simc Ratio vs Beam Energy', fontsize=18)

plt.tight_layout(rect=[0,0.03,1,0.95])
plt.savefig(UTILPATH+'/scripts/heep/DATASIMCRatio_plot.png')

############################################################################################################################################

print ("Processing Complete")
