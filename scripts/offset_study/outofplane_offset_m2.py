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
E5p986_Pmy_simc = ufloat(0.00009762, 0.00001994)
E5p986_Pmy_data = ufloat(-0.00473954, 0.00007789)
E5p986_HMS_Pe = 3.271
E5p986_SHMS_Pp = 3.493
E5p986_y = (E5p986_Pmy_simc.nominal_value - E5p986_Pmy_data.nominal_value)/E5p986_SHMS_Pp
E5p986_x = (E5p986_HMS_Pe/E5p986_SHMS_Pp)
E5p986_yerr = math.sqrt((E5p986_Pmy_simc.std_dev / E5p986_SHMS_Pp)**2 + (E5p986_Pmy_data.std_dev / E5p986_SHMS_Pp)**2 + (0.001291 / E5p986_SHMS_Pp)**2)
E5p986_y = "{:.5f}".format(E5p986_y)
E5p986_yerr = "{:.5f}".format(E5p986_yerr)
E5p986_x = "{:.5f}".format(E5p986_x)
print('\nE5p986_y =', E5p986_y, '\n')
print('E5p986_yerr =', E5p986_yerr, '\n')
print('E5p986_x =', E5p986_x, '\n')
print("="*40)

E6p395s1_Pmy_simc = ufloat(0.00032649, 0.00002834)
E6p395s1_Pmy_data = ufloat(-0.01063562, 0.00005390)
E6p395s1_HMS_Pe = 4.752
E6p395s1_SHMS_Pp = 2.412
E6p395s1_y = (E6p395s1_Pmy_simc.nominal_value - E6p395s1_Pmy_data.nominal_value)/E6p395s1_SHMS_Pp
E6p395s1_x = (E6p395s1_HMS_Pe/E6p395s1_SHMS_Pp)
E6p395s1_yerr = math.sqrt((E6p395s1_Pmy_simc.std_dev / E6p395s1_SHMS_Pp)**2 + (E6p395s1_Pmy_data.std_dev / E6p395s1_SHMS_Pp)**2 + (0.001291 / E6p395s1_SHMS_Pp)**2)
E6p395s1_y = "{:.5f}".format(E6p395s1_y)
E6p395s1_yerr = "{:.5f}".format(E6p395s1_yerr)
E6p395s1_x = "{:.5f}".format(E6p395s1_x)
print('\nE6p395s1_y =', E6p395s1_y, '\n')
print('E6p395s1_yerr =', E6p395s1_yerr, '\n')
print('E6p395s1_x =', E6p395s1_x, '\n')
print("="*40)

E6p395s2_Pmy_simc = ufloat(0.00018243, 0.00002623)
E6p395s2_Pmy_data = ufloat(-0.00894236, 0.00007001)
E6p395s2_HMS_Pe = 4.391
E6p395s2_SHMS_Pp = 2.792
E6p395s2_y = (E6p395s2_Pmy_simc.nominal_value - E6p395s2_Pmy_data.nominal_value)/E6p395s2_SHMS_Pp
E6p395s2_x = (E6p395s2_HMS_Pe/E6p395s2_SHMS_Pp)
E6p395s2_yerr = math.sqrt((E6p395s2_Pmy_simc.std_dev / E6p395s2_SHMS_Pp)**2 + (E6p395s2_Pmy_data.std_dev / E6p395s2_SHMS_Pp)**2 + (0.001291 / E6p395s2_SHMS_Pp)**2)
E6p395s2_y = "{:.5f}".format(E6p395s2_y)
E6p395s2_yerr = "{:.5f}".format(E6p395s2_yerr)
E6p395s2_x = "{:.5f}".format(E6p395s2_x)
print('\nE6p395s2_y =', E6p395s2_y, '\n')
print('E6p395s2_yerr =', E6p395s2_yerr, '\n')
print('E6p395s2_x =', E6p395s2_x, '\n')
print("="*40)

E6p395s3_Pmy_simc = ufloat(0.00043868, 0.00002085)
E6p395s3_Pmy_data = ufloat(-0.00611637, 0.00006552)
E6p395s3_HMS_Pe = 3.014
E6p395s3_SHMS_Pp = 4.220
E6p395s3_y = (E6p395s3_Pmy_simc.nominal_value - E6p395s3_Pmy_data.nominal_value)/E6p395s3_SHMS_Pp
E6p395s3_x = (E6p395s3_HMS_Pe/E6p395s3_SHMS_Pp)
E6p395s3_yerr = math.sqrt((E6p395s3_Pmy_simc.std_dev / E6p395s3_SHMS_Pp)**2 + (E6p395s3_Pmy_data.std_dev / E6p395s3_SHMS_Pp)**2 + (0.001291 / E6p395s3_SHMS_Pp)**2)
E6p395s3_y = "{:.5f}".format(E6p395s3_y)
E6p395s3_yerr = "{:.5f}".format(E6p395s3_yerr)
E6p395s3_x = "{:.5f}".format(E6p395s3_x)
print('\nE6p395s3_y =', E6p395s3_y, '\n')
print('E6p395s3_yerr =', E6p395s3_yerr, '\n')
print('E6p395s3_x =', E6p395s3_x, '\n')
print("="*40)

E7p937_Pmy_simc = ufloat(0.00053542, 0.00002352)
E7p937_Pmy_data = ufloat(-0.00658927, 0.00014882)
E7p937_HMS_Pe = 3.283
E7p937_SHMS_Pp = 5.512
E7p937_y = (E7p937_Pmy_simc.nominal_value - E7p937_Pmy_data.nominal_value)/E7p937_SHMS_Pp
E7p937_x = (E7p937_HMS_Pe/E7p937_SHMS_Pp)
E7p937_yerr = math.sqrt((E7p937_Pmy_simc.std_dev / E7p937_SHMS_Pp)**2 + (E7p937_Pmy_data.std_dev / E7p937_SHMS_Pp)**2 + (0.001291 / E7p937_SHMS_Pp)**2)
E7p937_y = "{:.5f}".format(E7p937_y)
E7p937_yerr = "{:.5f}".format(E7p937_yerr)
E7p937_x = "{:.5f}".format(E7p937_x)
print('\nE7p937_y =', E7p937_y, '\n')
print('E7p937_yerr =', E7p937_yerr, '\n')
print('E7p937_x =', E7p937_x, '\n')
print("="*40)

E8p479_Pmy_simc = ufloat(0.00028272, 0.00003185)
E8p479_Pmy_data = ufloat(-0.00880353, 0.00013019)
E8p479_HMS_Pe = 5.587
E8p479_SHMS_Pp = 3.731
E8p479_y = (E8p479_Pmy_simc.nominal_value - E8p479_Pmy_data.nominal_value)/E8p479_SHMS_Pp
E8p479_x = (E8p479_HMS_Pe/E8p479_SHMS_Pp)
E8p479_yerr = math.sqrt((E8p479_Pmy_simc.std_dev / E8p479_SHMS_Pp)**2 + (E8p479_Pmy_data.std_dev / E8p479_SHMS_Pp)**2 + (0.001291 / E8p479_SHMS_Pp)**2)
E8p479_y = "{:.5f}".format(E8p479_y)
E8p479_yerr = "{:.5f}".format(E8p479_yerr)
E8p479_x = "{:.5f}".format(E8p479_x)
print('\nE8p479_y =', E8p479_y, '\n')
print('E8p479_yerr =', E8p479_yerr, '\n')
print('E8p479_x =', E8p479_x, '\n')
print("="*40)

E9p177_Pmy_simc = ufloat(0.00062394, 0.00002701)
E9p177_Pmy_data = ufloat(-0.00577871, 0.00020413)
E9p177_HMS_Pe = 3.738
E9p177_SHMS_Pp = 6.265
E9p177_y = (E9p177_Pmy_simc.nominal_value - E9p177_Pmy_data.nominal_value)/E9p177_SHMS_Pp
E9p177_x = (E9p177_HMS_Pe/E9p177_SHMS_Pp)
E9p177_yerr = math.sqrt((E9p177_Pmy_simc.std_dev / E9p177_SHMS_Pp)**2 + (E9p177_Pmy_data.std_dev / E9p177_SHMS_Pp)**2 + (0.001291 / E9p177_SHMS_Pp)**2)
E9p177_y = "{:.5f}".format(E9p177_y)
E9p177_yerr = "{:.5f}".format(E9p177_yerr)
E9p177_x = "{:.5f}".format(E9p177_x)
print('\nE9p177_y =', E9p177_y, '\n')
print('E9p177_yerr =', E9p177_yerr, '\n')
print('E9p177_x =', E9p177_x, '\n')
print("="*40)

E9p876_Pmy_simc = ufloat(0.00042803, 0.00003255)
E9p876_Pmy_data = ufloat(-0.01010448, 0.00017909)
E9p876_HMS_Pe = 5.366
E9p876_SHMS_Pp = 5.422
E9p876_y = (E9p876_Pmy_simc.nominal_value - E9p876_Pmy_data.nominal_value)/E9p876_SHMS_Pp
E9p876_x = (E9p876_HMS_Pe/E9p876_SHMS_Pp)
E9p876_yerr = math.sqrt((E9p876_Pmy_simc.std_dev / E9p876_SHMS_Pp)**2 + (E9p876_Pmy_data.std_dev / E9p876_SHMS_Pp)**2 + (0.001291 / E9p876_SHMS_Pp)**2)
E9p876_y = "{:.5f}".format(E9p876_y)
E9p876_yerr = "{:.5f}".format(E9p876_yerr)
E9p876_x = "{:.5f}".format(E9p876_x)
print('\nE9p876_y =', E9p876_y, '\n')
print('E9p876_yerr =', E9p876_yerr, '\n')
print('E9p876_x =', E9p876_x, '\n')
print("="*40)

E10p549_Pmy_simc = ufloat(0.00037031, 0.00003566)
E10p549_Pmy_data = ufloat(-0.00960802, 0.00022925)
E10p549_HMS_Pe = 5.878
E10p549_SHMS_Pp = 5.530
E10p549_y = (E10p549_Pmy_simc.nominal_value - E10p549_Pmy_data.nominal_value)/E10p549_SHMS_Pp
E10p549_x = (E10p549_HMS_Pe/E10p549_SHMS_Pp)
E10p549_yerr = math.sqrt((E10p549_Pmy_simc.std_dev / E10p549_SHMS_Pp)**2 + (E10p549_Pmy_data.std_dev / E10p549_SHMS_Pp)**2 + (0.001291 / E10p549_SHMS_Pp)**2)
E10p549_y = "{:.5f}".format(E10p549_y)
E10p549_yerr = "{:.5f}".format(E10p549_yerr)
E10p549_x = "{:.5f}".format(E10p549_x)
print('\nE10p549_y =', E10p549_y, '\n')
print('E10p549_yerr =', E10p549_yerr, '\n')
print('E10p549_x =', E10p549_x, '\n')
print("="*40)

#################################################################################################################################################

# Create lists for x and y values along with their uncertainties
x_values = [E5p986_x, E6p395s1_x, E6p395s2_x, E6p395s3_x, E7p937_x, E8p479_x, E9p177_x, E9p876_x, E10p549_x]
y_values = [E5p986_y, E6p395s1_y, E6p395s2_y, E6p395s3_y, E7p937_y, E8p479_y, E9p177_y, E9p876_y, E10p549_y]
yerr_values = [E5p986_yerr, E6p395s1_yerr, E6p395s2_yerr, E6p395s3_yerr, E7p937_yerr, E8p479_yerr, E9p177_yerr, E9p876_yerr, E10p549_yerr]

# Extract nominal values and uncertainties for plotting
x_nominals = [float(x) for x in x_values]
y_nominals = [float(y) for y in y_values]
y_errors = [float(yerr) for yerr in yerr_values]

#################################################################################################################################################

# Define the model function (in this case, a linear function)
def linear_model(x, a, b):
    return a * x + b
# Convert the x values to numpy array for numerical operations
x_nominals_array = np.array(x_nominals)
y_nominals_array = np.array(y_nominals)
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

#This code snippet calculates the residuals by subtracting the fitted values from the observed values. Then, it calculates the chi-square value by summing the squared residuals divided by the squared uncertainties. This gives you a measure of how well the model fits the data, with lower values indicating a better fit.

residuals = y_nominals_array - linear_model(x_nominals_array, *popt)

# Calculate the chi-square value
chi_square = np.sum((residuals / y_errors_array)**2)

# Degrees of freedom is the difference between the number of data points and the number of parameters in the model.
degrees_of_freedom = len(x_nominals_array) - len(popt)

# Reduced chi-square value 
reduced_chi_square = chi_square / degrees_of_freedom

# Print the chi-square value
print('Chi-square value:', chi_square,'\n')
print('Reduced chi-square value:', reduced_chi_square, '\n')
print("="*40)

##################################################################################################################################################

# Output PDF File Name
print("Running as %s on %s, hallc_replay_lt path assumed as %s" % (USER, HOST, REPLAYPATH))

plt.figure(figsize=(12,8))

plt.subplot(111)
#plt.grid(zorder=1)
plt.xlim(0.0,2.25)
plt.ylim(0.000,0.007)
plt.errorbar(x_nominals, y_nominals, yerr=y_errors, fmt='o', markersize=8, color='black', linestyle='None', capsize=4, zorder=3, label='Data with uncertainties')
plt.scatter(x_nominals, y_nominals, color='blue', zorder=4, label='Data points')
plt.plot(x_fit, y_fit, color='red', label='Error-weighted Fit: y = (' + "{:.5f}".format(slope.nominal_value) + ')*x + (' + "{:.5f}".format(intercept.nominal_value) + ')')
plt.legend()
# Format the slope and intercept values
slope_str = '+{:.3f} mr'.format(slope.nominal_value*1000)
intercept_str = '{:.3f} mr'.format(intercept.nominal_value*1000)
# Use the formatted strings in the LaTeX expression
plt.text(0.05, 0.0065, r'$d\phi_{\mathrm{HMS}}$ = ' + slope_str, fontsize=16, color='red')
plt.text(0.05, 0.0060, r'$d\phi_{\mathrm{SHMS}}$ = ' + intercept_str, fontsize=16, color='green')
#plt.ylabel(r'$(PMY_\mathrm{SIMC} - PMY_\mathrm{DATA})/P_p}$', fontsize=20)
plt.xlabel('$(PMY_\\mathrm{SIMC} - PMY_\\mathrm{DATA})/P_p$', fontsize=20)
plt.xlabel(r'$(P_{e^\prime}/P_p)$', fontsize=20)
#plt.locator_params(axis='x', nbins=20) ### set number of bins for x axis only
plt.tick_params(axis='x', labelsize=14)  # Increase x-axis tick size
plt.tick_params(axis='y', labelsize=14)  # Increase y-axis tick size
plt.xticks(rotation=90)
plt.title('Out-of-Plane Offset', fontsize=18)

plt.tight_layout(rect=[0,0.03,1,0.95])
plt.savefig(UTILPATH+'/scripts/offset_study/Outofplane_offset_m2.png')

############################################################################################################################################

print ("Processing Complete")
