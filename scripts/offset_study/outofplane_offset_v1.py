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
from scipy.optimize import curve_fit
import scipy.integrate as integrate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from uncertainties import ufloat
import sys, math, os, subprocess
import array
import csv
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
E10p549_Pmy_data = ufloat(-0.008789672803, 0.000437432277)
E10p549_Pmy_simc = ufloat(0.000365312290, 0.000047912750)
E10p549_HMS_Pe = 5.878
E10p549_SHMS_Pp = 5.530
E10p549_y = (E10p549_Pmy_simc - E10p549_Pmy_data)/E10p549_HMS_Pe
E10p549_x = (E10p549_SHMS_Pp/E10p549_HMS_Pe)
#E10p549_x = "{:.5f}".format(E10p549_xx)
print('\nE10p549_y =', E10p549_y, '\n')
print('E10p549_x =', E10p549_x, '\n')
print("="*40)

E5p986_Pmy_data = ufloat(-0.004660606987, 0.000074715872)
E5p986_Pmy_simc = ufloat(0.000493720464, 0.000029524967)
E5p986_HMS_Pe = 3.271
E5p986_SHMS_Pp = 3.493
E5p986_y = (E5p986_Pmy_simc - E5p986_Pmy_data)/E5p986_HMS_Pe
E5p986_x = (E5p986_SHMS_Pp/E5p986_HMS_Pe)
#E5p986_x = "{:.5f}".format(E5p986_xx)
print('\nE5p986_y =', E5p986_y, '\n')
print('E5p986_x =', E5p986_x, '\n')
print("="*40)

E6p395s1_Pmy_data = ufloat(-0.010412832730, 0.000093706462)
E6p395s1_Pmy_simc = ufloat(-0.000096174981, 0.000031128716)
E6p395s1_HMS_Pe = 4.752
E6p395s1_SHMS_Pp = 2.412
E6p395s1_y = (E6p395s1_Pmy_simc - E6p395s1_Pmy_data)/E6p395s1_HMS_Pe
E6p395s1_x = (E6p395s1_SHMS_Pp/E6p395s1_HMS_Pe)
#E6p395s1_x = "{:.5f}".format(E6p395s1_xx)
print('\nE6p395s1_y =', E6p395s1_y, '\n')
print('E6p395s1_x =', E6p395s1_x, '\n')
print("="*40)

E6p395s2_Pmy_data = ufloat(-0.008904864475, 0.000128869268)
E6p395s2_Pmy_simc = ufloat(0.000111033092, 0.000038381215)
E6p395s2_HMS_Pe = 4.391
E6p395s2_SHMS_Pp = 2.792
E6p395s2_y = (E6p395s2_Pmy_simc - E6p395s2_Pmy_data)/E6p395s2_HMS_Pe
E6p395s2_x = (E6p395s2_SHMS_Pp/E6p395s2_HMS_Pe)
#E6p395s2_x = "{:.5f}".format(E6p395s2_xx)
print('\nE6p395s2_y =', E6p395s2_y, '\n')
print('E6p395s2_x =', E6p395s2_x, '\n')
print("="*40)

E6p395s3_Pmy_data = ufloat(-0.005898930395, 0.000176765133)
E6p395s3_Pmy_simc = ufloat(0.000545188195, 0.000031680931)
E6p395s3_HMS_Pe = 3.014
E6p395s3_SHMS_Pp = 4.220
E6p395s3_y = (E6p395s3_Pmy_simc - E6p395s3_Pmy_data)/E6p395s3_HMS_Pe
E6p395s3_x = (E6p395s3_SHMS_Pp/E6p395s3_HMS_Pe)
#E6p395s3_x = "{:.5f}".format(E6p395s3_xx)
print('\nE6p395s3_y =', E6p395s3_y, '\n')
print('E6p395s3_x =', E6p395s3_x, '\n')
print("="*40)

E7p937_Pmy_data = ufloat(-0.006570131560, 0.000167200810)
E7p937_Pmy_simc = ufloat(0.000637043786, 0.000035354292)
E7p937_HMS_Pe = 3.283
E7p937_SHMS_Pp = 5.512
E7p937_y = (E7p937_Pmy_simc - E7p937_Pmy_data)/E7p937_HMS_Pe
E7p937_x = (E7p937_SHMS_Pp/E7p937_HMS_Pe)
#E7p937_x = "{:.5f}".format(E7p937_xx)
print('\nE7p937_y =', E7p937_y, '\n')
print('E7p937_x =', E7p937_x, '\n')
print("="*40)

E8p479_Pmy_data = ufloat(-0.008598029303, 0.000246644841)
E8p479_Pmy_simc = ufloat(0.000121096701, 0.000039194517)
E8p479_HMS_Pe = 5.587
E8p479_SHMS_Pp = 3.731
E8p479_y = (E8p479_Pmy_simc - E8p479_Pmy_data)/E8p479_HMS_Pe
E8p479_x = (E8p479_SHMS_Pp/E8p479_HMS_Pe)
#E8p479_x = "{:.5f}".format(E8p479_xx)
print('\nE8p479_y =', E8p479_y, '\n')
print('E8p479_x =', E8p479_x, '\n')
print("="*40)

E9p177_Pmy_data = ufloat(-0.005707239796, 0.000231934456)
E9p177_Pmy_simc = ufloat(0.000650367288, 0.000037889532)
E9p177_HMS_Pe = 3.738
E9p177_SHMS_Pp = 6.265
E9p177_y = (E9p177_Pmy_simc - E9p177_Pmy_data)/E9p177_HMS_Pe
E9p177_x = (E9p177_SHMS_Pp/E9p177_HMS_Pe)
#E9p177_x = "{:.5f}".format(E9p177_xx)
print('\nE9p177_y =', E9p177_y, '\n')
print('E9p177_x =', E9p177_x, '\n')
print("="*40)

E9p876_Pmy_data = ufloat(-0.009993862240, 0.000220013508)
E9p876_Pmy_simc = ufloat(0.000316157644, 0.000043242884)
E9p876_HMS_Pe = 5.366
E9p876_SHMS_Pp = 5.422
E9p876_y = (E9p876_Pmy_simc - E9p876_Pmy_data)/E9p876_HMS_Pe
E9p876_x = (E9p876_SHMS_Pp/E9p876_HMS_Pe)
#E9p876_x = "{:.5f}".format(E9p876_xx)
print('\nE9p876_y =', E9p876_y, '\n')
print('E9p876_x =', E9p876_x, '\n')
print("="*40)

#################################################################################################################################################

# Create lists for x and y values along with their uncertainties
x_values = [E10p549_x, E5p986_x, E6p395s1_x, E6p395s2_x, E6p395s3_x, E7p937_x, E8p479_x, E9p177_x, E9p876_x]
y_values = [E10p549_y, E5p986_y, E6p395s1_y, E6p395s2_y, E6p395s3_y, E7p937_y, E8p479_y, E9p177_y, E9p876_y]

# Extract nominal values and uncertainties for plotting
x_nominals = [float(x) for x in x_values]
y_nominals = [ufloat(y.nominal_value, y.std_dev) for y in y_values]
y_errors = [y.std_dev for y in y_nominals]

#################################################################################################################################################

# Define the model function (in this case, a linear function)
def linear_model(x, a, b):
    return a * x + b
# Convert the x values to numpy array for numerical operations
x_nominals_array = np.array(x_nominals)
y_nominals_array = np.array([y.nominal_value for y in y_nominals])
y_errors_array = np.array(y_errors)

# Perform the error-weighted fit using curve_fit
popt, pcov = curve_fit(linear_model, x_nominals_array, y_nominals_array, sigma=y_errors_array, absolute_sigma=True)

# Extract the optimized slope and intercept
slope = ufloat(popt[0], np.sqrt(pcov[0, 0]))
intercept = ufloat(popt[1], np.sqrt(pcov[1, 1]))

# Print the extracted slope and intercept values
print('\nSlope (with uncertainty):', slope,'\n')
print('\nIntercept (with uncertainty):', intercept, '\n')

# Generate points for the fitted line
x_fit = np.linspace(min(x_nominals_array), max(x_nominals_array), 100)
y_fit = linear_model(x_fit, *popt)

##################################################################################################################################################

# Output PDF File Name
print("Running as %s on %s, hallc_replay_lt path assumed as %s" % (USER, HOST, REPLAYPATH))

plt.figure(figsize=(12,8))

plt.subplot(111)
#plt.grid(zorder=1)
plt.xlim(0.0,2.25)
plt.ylim(0.001,0.003)
plt.errorbar(x_nominals, [y.nominal_value for y in y_nominals], yerr=y_errors, fmt='o', markersize=8, color='black', linestyle='None', capsize=4, zorder=3, label='Data with uncertainties')
plt.scatter(x_nominals, [y.nominal_value for y in y_nominals], color='blue', zorder=4, label='Data points')
plt.plot(x_fit, y_fit, color='red', label='Error-weighted Fit: y = (' + "{:.5f}".format(slope.nominal_value) + ')*x + (' + "{:.5f}".format(intercept.nominal_value) + ')')
plt.legend()
# Format the slope and intercept values
slope_str = '{:.2f} mr'.format(slope.nominal_value*1000)
intercept_str = '+{:.2f} mr'.format(intercept.nominal_value*1000)
# Use the formatted strings in the LaTeX expression
plt.text(1.75, 0.00263, r'$d\phi_{\mathrm{SHMS}}$ = ' + slope_str, fontsize=16, color='red')
plt.text(1.75, 0.00250, r'$d\phi_{\mathrm{HMS}}$ = ' + intercept_str, fontsize=16, color='green')
plt.ylabel(r'$(PMY_\mathrm{SIMC} - PMY_\mathrm{DATA})/P_{e^\prime}$', fontsize=20)
plt.xlabel(r'$(P_p/P_{e^\prime})$', fontsize=20)
#plt.locator_params(axis='x', nbins=20) ### set number of bins for x axis only
plt.tick_params(axis='x', labelsize=14)  # Increase x-axis tick size
plt.tick_params(axis='y', labelsize=14)  # Increase y-axis tick size
plt.xticks(rotation=90)
plt.title('Out-of-Plane Offset', fontsize=18)

plt.tight_layout(rect=[0,0.03,1,0.95])
plt.savefig(UTILPATH+'/scripts/offset_study/Outofplane_offset_v1.png')

############################################################################################################################################

print ("Processing Complete")
