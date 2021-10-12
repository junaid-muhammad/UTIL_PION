#! /usr/bin/python

#
# Description:
# ================================================================
# Time-stamp: "2021-10-06 05:40:11 trottar"
# ================================================================
#
# Author:  Richard L. Trotta III <trotta@cua.edu>
#
# Copyright (c) trottar
#
import os

l_flag = "run"

if l_flag == "all":
    lumi_list = [12126,12128,12129,12130,12131,12132,12133,12135,12137,12140,12141,12142,12143,12144,12145,12147,12148,12150,
                 12151,12152,12153,12154,12156,12157,12158,12159,12160,12161,12162,12163,12164,12166,12167,12170,12171,12172,
                 12173,12174,12175,12178,12179,12181,12184,12185,12186,12187,12189,12190,12191,12192,12193,12194,12195,112196,12197,12198,12199]
elif l_flag == "1":
    lumi_list = [12126]
else:
    lumi_list = [12126,12128,12129,12130,12131,12132,12133,12135,12137,12140,12141,12142]

for l in lumi_list:
    print("Plotting run %s" % l)
    os.system("../trigWindows.sh -p %s" % (l))