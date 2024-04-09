#! /usr/bin/python

#
# Description:
# ================================================================
# Time-stamp: "2024-02-09 12:05:00 junaid"
# ================================================================
#
# Created: Muhammad junaid  <mjo147@uregina.ca>
# Copyright (c) trottar & junaid
#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from csv import DictReader
import sys, os

################################################################################################################################################
'''
User Inputs
'''
ROOTPrefix = sys.argv[1]
runType = sys.argv[2]
timestmp = sys.argv[3]

################################################################################################################################################
'''
ltsep package import and pathing definitions
'''

# Import package for cuts
import ltsep as lt 

p=lt.SetPath(os.path.realpath(__file__))

# Add this to all files for more dynamic pathing
USER=p.getPath("USER") # Grab user info for file finding
HOST=p.getPath("HOST")
REPLAYPATH=p.getPath("REPLAYPATH")
UTILPATH=p.getPath("UTILPATH")
ANATYPE=p.getPath("ANATYPE")
SCRIPTPATH=p.getPath("SCRIPTPATH")

################################################################################################################################################

#inp_f = UTILPATH+"/scripts/efficiency/OUTPUTS/%s_%s_efficiency_data_%s.csv"  % (ROOTPrefix.replace("replay_",""),runType,timestmp)
inp_f = UTILPATH+"/efficiencies/%s_%s_efficiency_data_%s.csv"  % (ROOTPrefix.replace("replay_",""),runType,timestmp)

# Converts csv data to dataframe
try:
    efficiency_data = pd.read_csv(inp_f)
except IOError:
    print("Error: %s does not appear to exist." % inp_f)
#print(efficiency_data.keys())

################################################################################################################################################################################################

plt.figure(figsize=(12,8))

plt.subplot(121)    
plt.grid(zorder=1)
#plt.xlim(0,100)
plt.ylim(0.9,1.02)
plt.errorbar(efficiency_data["Run_Number"],efficiency_data["HMS_Cer_SING_Elec_Eff"],yerr=efficiency_data["HMS_Cer_SING_Elec_Eff_ERROR"],color='black',linestyle='None',zorder=3)
plt.scatter(efficiency_data["Run_Number"],efficiency_data["HMS_Cer_SING_Elec_Eff"],color='red',zorder=4)
plt.ylabel('HMS_Cer_SING_Elec_Eff', fontsize=12)
plt.xlabel('Run_Number', fontsize=12)
plt.title('HMS %s-%s' % (int(min(efficiency_data["Run_Number"])),int(max(efficiency_data["Run_Number"]))), fontsize=12)

plt.subplot(122)
plt.grid(zorder=1)
#plt.xlim(0,100)
plt.ylim(0.9,1.02)
plt.errorbar(efficiency_data["Run_Number"],efficiency_data["HMS_Cal_SING_Elec_Eff"],yerr=efficiency_data["HMS_Cal_SING_Elec_Eff_ERROR"],color='black',linestyle='None',zorder=3)
plt.scatter(efficiency_data["Run_Number"],efficiency_data["HMS_Cal_SING_Elec_Eff"],color='red',zorder=4)
plt.ylabel('HMS_Cal_SING_Elec_Eff', fontsize=12)
plt.xlabel('Run_Number', fontsize=12)
plt.title('HMS %s-%s' % (int(min(efficiency_data["Run_Number"])),int(max(efficiency_data["Run_Number"]))), fontsize=12)

plt.tight_layout(rect=[0,0.03,1,0.95])   
plt.savefig(UTILPATH+'/scripts/efficiency/OUTPUTS/plots/HMS_ELec_%s.png' % (ROOTPrefix.replace("replay_","")))

########################################################################################################################################################################################

#plt.show()

print("Plotting Complete")
