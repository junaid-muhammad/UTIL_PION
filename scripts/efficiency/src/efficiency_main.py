#! /usr/bin/python
#
# Description:
# ================================================================
# Time-stamp: "2025-03-13 01:29:19 junaid"
# ================================================================
#
# Author:  Muhammad Junaid III <mjo147@uregina.ca>
#
# Copyright (c) junaid
#
###################################################################################################################################################

import pandas as pd
import sys, os, time

################################################################################################################################################
'''
User Inputs
'''
ROOTPrefix = sys.argv[1]
runType = sys.argv[2]
runNum = sys.argv[3]
MaxEvent=sys.argv[4]

################################################################################################################################################
'''
ltsep package import and pathing definitions
'''

# Import package for cuts
import ltsep as lt 

# Import package for cuts
from ltsep import Root

lt=Root(os.path.realpath(__file__),"efficiency", DEBUG=False)

# Add this to all files for more dynamic pathing
USER=lt.USER # Grab user info for file finding
HOST=lt.HOST
REPLAYPATH=lt.REPLAYPATH
UTILPATH=lt.UTILPATH
SIMCPATH=lt.SIMCPATH
ANATYPE=lt.ANATYPE
OUTPATH=lt.OUTPATH

################################################################################################################################################

print("Running as %s on %s, hallc_replay_lt path assumed as %s" % (USER, HOST, REPLAYPATH))

timestmp = time.strftime("%Y_%m_%d")

# Output for luminosity table
out_f = UTILPATH+"/scripts/efficiency/OUTPUTS/%s_%s_efficiency_data_%s.csv"  % (ROOTPrefix.replace("replay_",""),runType,timestmp)
#out_f = UTILPATH+"/scripts/efficiency/OUTPUTS/%s_efficiency_data_%s.csv"  % (ROOTPrefix.replace("replay_",""),timestmp)

################################################################################################################################################

#import efficiency_hgcer
import efficiency_report

DEBUG=True

if "coin" in ROOTPrefix:
    #hgcerDict = efficiency_hgcer.dictionary(UTILPATH,runNum,MaxEvent)
    hgcerDict = {}
    reportDict = efficiency_report.dictionary(UTILPATH,ROOTPrefix,runNum,MaxEvent)
else:
    hgcerDict = {}
    reportDict = efficiency_report.dictionary(UTILPATH,ROOTPrefix,runNum,MaxEvent)

################################################################################################################################################

data = {}
for d in (hgcerDict, reportDict): 
    data.update(d)

eff_data = {i : data[i] for i in sorted(data.keys())}
#out_data = pd.DataFrame()
# Convert merged dictionary to a pandas dataframe then sort it
table  = pd.DataFrame([eff_data], columns=eff_data.keys())
table = table.reindex(sorted(table.columns), axis=1)

file_exists = os.path.isfile(out_f)

# Updates csv file with efficiency values for later analysis (see plot_yield.py)
if file_exists:
    try:
        out_data = pd.read_csv(out_f)
    except IOError:
        print("Error: %s does not appear to exist." % out_f)
    # Checks if run number is alread in csv and replaces it if it is there
    run_index = out_data.index[out_data["Run_Number"] == int(runNum)].tolist()
    out_data.drop(run_index, inplace=True)
#    out_data = out_data.append(table,ignore_index=True)  # ---- For new python version added next line
    out_data = pd.concat([out_data, table], ignore_index=True)
#print("Output efficiency values\n",out_data)
    out_data.to_csv(out_f, index = False, header=True, mode='w+',)
else:
    table.to_csv(out_f, index = False, header=True, mode='a',)            
