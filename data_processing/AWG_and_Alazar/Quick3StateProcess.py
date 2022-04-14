# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 11:11:55 2022

@author: Hatlab-RRK

Purpose: create a neat, almost-executable file that can quickly plot a 3-state pulse file, and have the option to do just histograms, 
or additionally try to fit using majority vote and give classification accuracy
"""
from Pulse_Processing_utils import extract_3pulse_histogram_from_filepath, extract_2pulse_histogram_from_filepath
import os
#histogram plotting
datapath = [path for path in os.listdir() if path.find('.ddh5')!= -1]
fid = extract_3pulse_histogram_from_filepath(datapath, plot = False, hist_scale = None, numRecords = 3840*2, numRecordsUsed = 3840*2, IQ_offset = (0,0), fit = False, lpf = True, lpf_wc = 50e6, boxcar = False, bc_window = [50, 150], record_track = True, tuneup_plots = True, debug = False, tstart_index = 0, tstop_index = -1)