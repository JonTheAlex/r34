"""
utils.py

Authors: 
Jonathon Alexander
William Schoenhals
zachandfox
Spring, 2020
"""

from datetime import datetime
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial as poly
from scipy.signal import find_peaks, lfilter, peak_widths
from scipy.interpolate import CubicSpline
import pandas as pd
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QHBoxLayout, QLabel, QGridLayout, QTextBrowser, QWidget, QPushButton
from PyQt5.QtCore import QSize
import sys, re, os

# [Jon] Read in ECG Data
def read_ecg_csv(flip_indicator, fileDirectory, fileName = None):
    tone = 'Tone'
    prev_out = False

    dir_name = os.path.dirname(fileDirectory)
    
    os.chdir(dir_name)

    if fileName==None:
        ecg_fileName = os.path.basename(fileDirectory)
    else:
        ecg_fileName = fileName
        
    sub_id = re.findall('\w+\d\d\d', ecg_fileName)
    cond = re.findall(r'Pre Trial|Post Trial|Trial', ecg_fileName)
    task = re.findall(r'Task\d{1}', ecg_fileName)
    task = task[0]
    df_ecg = pd.read_csv(ecg_fileName,sep='\s*,\s*',
                        header=0, encoding='ascii', engine='python')

    if ecg_fileName.find(tone) != -1:
        df_ecg_detected = df_ecg.reset_index()
    elif len(df_ecg.Data) <= 1:
        # this trial will be skipped
        no_data = np.zeros((1,3))
        no_data_df = pd.DataFrame(data=no_data, columns=['index', 'Data', 'Detections'])
        return no_data_df, no_data_df, sub_id, cond, task
    else:
        df_ecg_formatted = df_ecg.reset_index()
        if flip_indicator == True:
            df_ecg_formatted = flip_ecg_in_df(df_ecg_formatted)
        df_ecg_detrended = detrend_ecg(df_ecg_formatted)    
        df_ecg_detrended = detrend_ecg(df_ecg_formatted)
        df_ecg_detected = detect_rpeaks(df_ecg_detrended)

    squeeze_fileName = ecg_fileName.replace('BioPatch', 'Squeeze')
    df_squeeze = read_squeeze_csv(squeeze_fileName)

    if len(df_squeeze.Data)<=1:
        no_data = np.zeros((1,3))
        no_data_df = pd.DataFrame(data=no_data, columns=['index', 'Data', 'Detections'])
        return no_data_df, no_data_df, sub_id, cond, task

    return df_ecg_detected, df_squeeze, sub_id, cond, task

# [Jon] Check for existing file
def overwrite_check(fileDirectory):
    fileBase = os.path.basename(fileDirectory)
    fileDir = os.path.dirname(fileDirectory)

    sub_id = ' '.join(re.findall(r'\w{2}\d{3}', fileBase))
    cond = ' '.join(re.findall(r'Pre Trial|Post Trial|Trial', fileDir))
    task = ' '.join(re.findall(r'Task\d{1}', fileBase))
    results_file = sub_id + '-' + cond + '-' + task + '-OUT' + '.csv'
    os.chdir(fileDir)

    if os.path.exists(results_file):
        return True
    else :
        return False

def flip_ecg_in_df(df):
    df.Data = df.Data * -1
    return df


# [Will] Detrend ECG Data
def detrend_ecg(df):
    '''
    Input: experiment dataframe
    Return: experiment dataframe
    Modifies ecg data contained in dataframe.Data
    Fit a 3rd order polynomial and subtract to detrend
    '''
    poly_coef = poly.polyfit(df.index, df.Data, 3)
    poly_trend = poly.polyval(df.index, poly_coef)
    df.Data = df.Data - poly_trend
    return df

# [Will] Detrend Squeeze Data
def detrend_squeeze(df):
    '''
    Input: experiment dataframe
    Return: experiment dataframe
    Removes DC bias from data
    '''
    squeeze = df.Data
    squeeze_detrend = squeeze - min(squeeze) # squeeze data will NEVER be <0
    df.Data = squeeze_detrend
    return df

def interp_squeeze(df):
    y = df.Data
    x = np.arange(0, len(y))
    cs = CubicSpline(x, y)
    x_new = np.arange(0, len(y)-1, 0.1)
    y_new = cs(x_new)
    return y_new

def smooth_squeeze(x):
    # from scipy cookbook!
    window_len = 50
    buffer_add_pre = np.zeros(int(window_len/2)); 
    buffer_add_pre.fill(x[0])
    buffer_add_post = np.zeros(int(window_len/2)); 
    buffer_add_pre.fill(x[-1])
    scaled_x = np.r_[buffer_add_pre, x, buffer_add_post]
    w = np.hamming(window_len)
    # perform convolution to smooth
    y = np.convolve(w/w.sum(), scaled_x, mode='valid')
    return y


# [Will] Detect R-Peaks
def detect_rpeaks(df):
    '''
    Input df with detrended ecg data
    Returns df with populated rpeaks series
    Modifies rpeak data contained in dataframe.rpeaks
    '''
    # build Finite Impulse Response Filter
    fir_4 = build_fir(4)
    # convolute FIR Filter and ECG data
    module_4 = lfilter(fir_4, [1.0], df.Data)

    # create arrays of min and max locations
    max_prom = abs(np.amax(module_4)*0.5)
    min_prom = abs(np.amin(module_4)*0.15)
    module_4_max_info = find_peaks(module_4, prominence=max_prom, wlen=300)
    module_4_min_info = find_peaks(-module_4, prominence=min_prom, wlen=300)
    mod_4_max_loc = np.array(module_4_max_info[0], dtype=int)
    mod_4_min_loc = np.array(module_4_min_info[0], dtype=int)
    min_range = 50
    valid_mod_max = np.empty(0, dtype=int)
    valid_mod_min = np.empty(0, dtype=int)

    # find all valid pairs from all points
    for max_loc in mod_4_max_loc:
        if (max_loc - min_range) < 0:
            pass # edge case. max is detected within 30 of start of data. invalid
        else:
            valid_mins = mod_4_min_loc[((max_loc-min_range) < mod_4_min_loc) & (mod_4_min_loc < max_loc)] 
            if len(valid_mins) > 0:
                best_min = max(valid_mins)
                valid_mod_min = np.append(valid_mod_min, best_min) 
                valid_mod_max = np.append(valid_mod_max, max_loc)
            else:
                pass # no min can be paired with this max. invalid.

    # min/max pairs with amp that exceeds min_amp are likely R-peaks
    # find all likely min/max pairs
    min_amp = 4 * min_prom
    certain_mod_max = np.empty(0, dtype=int)
    certain_mod_min = np.empty(0, dtype=int)
    for pair_index in range(0, np.size(valid_mod_max)):
        valid_pair_min = valid_mod_min[pair_index]
        valid_pair_max = valid_mod_max[pair_index]
        mod_amp = module_4[valid_pair_max] + abs(module_4[valid_pair_min])
        if mod_amp > min_amp:
            certain_mod_max = np.append(certain_mod_max, valid_pair_max)
            certain_mod_min = np.append(certain_mod_min, valid_pair_min)

    # use min/max pairs to locate r-peaks in ecg data
    r_peaks = np.empty(0, dtype=int)
    for certain_index in range(0, np.size(certain_mod_max)):
        r_hood = np.arange(certain_mod_min[certain_index], certain_mod_max[certain_index])
        ecg_y = df.Data[r_hood]
        ecg_x = df.index[r_hood]
        new_r_peak_height = max(ecg_y)
        # new_r_peak_height can have multiple locations. always select the earlies occurance.
        new_r_peak_loc = min(ecg_x[ecg_y == new_r_peak_height]) 
        r_peaks = np.append(r_peaks, new_r_peak_loc)

    # rebuild the df.Detections series
    detections = np.zeros(np.size(df.Detections), dtype=int)
    detections[r_peaks] = 1

    df.Detections = detections
    return df  


def build_fir(scale):
    '''
    Input scale
    Build FIR scale coefficents for wavelet R peak detection
    Returns filter coefficents
    '''
    high_len = 1 + scale
    low_len = 4 + (scale - 2)*3

    h_coeff = np.zeros(low_len)
    g_coeff = np.zeros(high_len)

    if scale == 1:
        filter_coeff = [-2, 2]
    else:
        g_coeff[0] = -2
        g_coeff[-1] = 2
        h_coeff[0] = 0.125
        h_coeff[scale] = 0.375
        h_coeff[scale*2-1] = 0.375
        h_coeff[-1] = 0.125

        filter_coeff = np.convolve(g_coeff, h_coeff)
    
    return filter_coeff

# Create ECG Subplot

# [Jon] Display ECG + R-peak data

# [Zach] Edit R-peak detections

# Inspect R-Peak detections

# [Jon] Submit and Write R-Peak Edits

# Create R-Peak subplot

# Read in Squeeze Data

def read_squeeze_csv(squeeze_fileName):
    df_squeeze = pd.read_csv(squeeze_fileName, sep='\s*,\s*',
                           header=0, encoding='ascii', engine='python')
    df_squeeze =  df_squeeze.reset_index()

    return df_squeeze

def message():

    msg = str(r'''      ,_
    >' )
    ( ( \ 
        ''|\
    ''')      


    return msg

def write_empty_output(fileDirectory, t1000_df, classic_df):
    fileBase = os.path.basename(fileDirectory)
    fileDir = os.path.dirname(fileDirectory)

    sub_id = ' '.join(re.findall(r'\w{2}\d{3}', fileBase))
    cond = ' '.join(re.findall(r'Pre Trial|Post Trial|Trial', fileDir))
    task = ' '.join(re.findall(r'Task\d{1}', fileBase))
    file_prefix = os.path.dirname(fileDirectory) + '/'
    file_suffix = sub_id +'-' + cond + '-' + task
    err = 0
    msg = ''
    try:
        t1000_df.to_csv(file_prefix +  file_suffix + '-T1000' + '.CSV')
        classic_df.to_csv(file_prefix +  file_suffix + '-Classic' + '.CSV')
    except PermissionError:
        err = 1
        msg = 'Failure. Files cannot be saved while the files are open. Close all data files and redo.'
    except FileNotFoundError:
        err = 1
        msg = 'Failure. Unable to write to location: ' + file_prefix + ' check your connection or device'

    return msg, err

def write_output(fileDirectory, merged_df, t1000_df, classic_df, sqz_df, ecg_df, classic_pairs, quality='High'):
    err = 0
    msg = ''
    fileBase = os.path.basename(fileDirectory)
    fileDir = os.path.dirname(fileDirectory)

    sub_id = ' '.join(re.findall(r'\w{2}\d{3}', fileBase))
    cond = ' '.join(re.findall(r'Trial|Pre Trial|Post Trial', fileDir))
    task = ' '.join(re.findall(r'Task\d{1}', fileBase))

    file_prefix = os.path.dirname(fileDirectory) + '/'
    file_suffix = sub_id +'-' + cond + '-' + task
    try:
        merged_df.to_csv(file_prefix + file_suffix + '-OUT'+ '.CSV')
        t1000_df.to_csv(file_prefix +  file_suffix + '-T1000' + '.CSV')
        classic_df.to_csv(file_prefix +  file_suffix + '-Classic' + '.CSV')
        save_plot(ecg_df, sqz_df, merged_df, classic_pairs, file_prefix, file_suffix, quality)
    except PermissionError:
        err = 1
        msg = 'Failure. Files cannot be saved while the files are open. Close all data files redo.'
    except FileNotFoundError:
        err = 1
        msg = 'Failure. Unable to write to location: ' + file_prefix + ' check your connection or device'

    return msg, err

def merge_ecg_squeeze(ecg, squeeze):
    df_ecg = ecg
    df_squeeze = squeeze

    df_squeeze['index'] = df_squeeze['index'] * 20
    df_merge = df_ecg.merge(df_squeeze, how='left', on='index', suffixes=('_ECG', '_Squeeze'))

    return df_merge

# [Will] Detect Squeezes
def detect_squeezes(df):
    # interpolate
    interpolated_sq = interp_squeeze(df)
    # smooooth
    smoothed_sq = smooth_squeeze(interpolated_sq)
    # constants
    # a trial with no squeezes will still have variable pressure reading due to ambient pressure from sub or temperature
    min_width = 50
    min_height = round(0.2*max(smoothed_sq)+1)
    # find all peaks and widths of peaks in squeeze data  
    peak_info = find_peaks(smoothed_sq, height=min_height)
    peak_locs = peak_info[0]
    peak_heights = peak_info[1]['peak_heights']
    widths, width_heights, left_ips, right_ips = peak_widths(smoothed_sq, peak_locs, rel_height=0.5)
    # remove peaks that do not meet the minimum width
    mask = widths > min_width
    widths = widths[mask]
    peak_locs = peak_locs[mask]
    print('range is: ', (max(smoothed_sq) - min(smoothed_sq)))
    if len(peak_locs) == 0 or ((max(df.Data) - min(df.Data)) <= 6): # skip calculations if there is nothing to calculate!
        sq_detection_series = np.zeros(np.size(df.Data), dtype=int)
        df.Detections = sq_detection_series
        return df, peak_heights
    peak_heights = peak_heights[mask]
    left_ips = left_ips[mask]
    right_ips = right_ips[mask]
    # convert type of _ips into int for use as indices
    left_ips = [int(i) for i in left_ips]
    right_ips = [0] + [int(i) for i in right_ips] # create a dummy right_imps to pair with 1st left_ips
    # interval is between peaks
    intervals = [(right_ips[i], left_ips[i]) for i in range(len(left_ips))]
    # look for the minima between peaks to determine where the next squeeze starts
    # if several minima exits, choose the right-most minima
    minima = []
    #minima = [find_peaks(-smoothed_sq[left:right])[0] for (left, right) in intervals]
    for indx in intervals:
        left, right = indx
        all_mins = find_peaks(-smoothed_sq[left:right])[0]
        if len(all_mins) == 0: #if there are no mins, all_mins is empty. no min found. default to latest index
            minima.append(np.array([right-left]))
        else:
            minima.append(all_mins)
    rightestminidx = [minimum[-1] for minimum in minima]
    rightestmin = [right_ips[idx] + rightestminidx[idx] for idx in range(len(rightestminidx))]
    # a 'detect' is defined as being the point 30% of the x distance from the minima to the peak
    sq_detects = np.array([rightestmin[i] + int(0.3*(peak_locs[i] - rightestmin[i])) for i in range(len(rightestmin))])
    # remove noise based on Y diff between peak_loc and sq_detect, =1 is always noise
    y_diff = peak_heights - smoothed_sq[rightestmin]
    remove_small_sq = y_diff >= 2
    peak_heights = peak_heights[remove_small_sq]
    sq_detects = sq_detects[remove_small_sq]
    # map interpolated detections onto original squeeze data
    # from interp_squeeze() we know that the data was interpolated at a scale of 0.1
    sq_un_interped = [int(i/10) for i in sq_detects]
    # reform dataframe and return
    sq_detection_series = np.zeros(np.size(df.Data), dtype=int)
    sq_detection_series[sq_un_interped] = 1
    df.Detections = sq_detection_series

    return df, peak_heights

# Create Squeeze Detection Subplot

# Write Squeeze Data

### Analyze Trial ###
# Merge together R-peak and Squeeze Data
# Associate ECG to Squeeze Data
def pair_classic(merged_df):
    '''
    Input: Dataframe
    Return: 2-D array
        Each row is a detection pair
        1st column is index of r-peak det
        2nd column is index of squeeze det
        Per Classic Definition:
            Pairs will not overlap. Len() of columns are ==
        A trail with no pairs will return np.zeros((2, 1)) 
    '''
    classic = np.empty((1, 2), dtype=int)
    end_of_data = len(merged_df.Data_Squeeze.values)

    if np.nansum(merged_df.Detections_Squeeze) == 0:
        # no squeezes detected, nothing to pair
        return np.delete(classic, 0, 0)

    sq_det = np.asarray(merged_df.index[merged_df.Detections_Squeeze == 1])
    r_det = np.asarray(merged_df.index[merged_df.Detections_ECG == 1])

    for hb in range(0, np.size(r_det)):
        # build rr_interval
        # ignore time between start of data and 1st r peak
        if hb == np.size(r_det)-1: # invterval last r_peak and end of data
            rr_interval = range(r_det[hb], end_of_data) 
        else: # interval between r peak and next
            rr_interval = range(r_det[hb], r_det[hb + 1])
        valid_sq = sq_det[(rr_interval[0] < sq_det) & (sq_det < rr_interval[-1])] 
        if len(valid_sq) > 0:
            best_sq = min(valid_sq)
            next_row = np.array([[r_det[hb], best_sq]])
            classic = np.concatenate((classic, next_row), axis=0)
            # add on to peak heights as well
        else:
            pass # no squeezes for this hb. no pair.
    # first row is empty data from classic initialization   
    classic = np.delete(classic, 0, 0)
    return classic

# Calculate Time Delays
def classic_calc(df_merged, classic_pair_array, quality, peak_heights):
    '''
    Inputs: merged dataframe
            2-D np.array() of indicies of paired detections [rpeak, sq]
    Outputs: df of metrics, ready for csv write
        Per Documentation, return is:
        [MAX, MIN, STD, MEAN, Num HB, Num Pairs, Num Squeezes, Num Squeeze Omitted]
        An input of no pairs will result in NAN for all except results[4] and results[5]
    '''
    diff = np.zeros(0)
    r_det = classic_pair_array[:,0]
    sq_det = classic_pair_array[:,1]
    results = np.empty(10)
    results[:] = np.nan

    if len(sq_det) == 0:
        # no pairs
        return create_classic_results_df(results)

    # reults calculated no matter high or med quality
    results[4] = sum(df_merged.Detections_ECG) # num hbs
    results[5] = np.nansum(df_merged.Detections_Squeeze, dtype=int) # num squeezes
    results[8] = np.round(np.mean(peak_heights)) # average squeeze height
    results[9] = np.round(np.std(peak_heights)) # std of squeeze height

    # Fill in results[0:3] only if our quality is high. otherwise default values stay as NaN
    if quality == 'High':
        diff = np.round(np.add(sq_det, -r_det), -1) # sq is accurate to only 10ms. Round to nearest 10 ms
        results[0] = np.round(np.min(diff), -1) # min
        results[1] = np.round(np.max(diff), -1) # max
        results[2] = np.round(np.std(diff), -1) # std
        results[3] = np.round(np.mean(diff), -1) # mean
        results[6] = np.size(sq_det) # num squeeze detections
        results[7] = results[5] - np.size(sq_det) # num squeezes omitted
    
    return create_classic_results_df(results)

def create_classic_results_df(classic_results):
    '''
    Input: array of results from classic pairing and calculations
    Output: dataframe of headers and results ready for writing to csv
    '''
    results_df = {'MIN': [classic_results[0]],
    'MAX': [classic_results[1]],
    'STD': [classic_results[2]],
    'MEAN': [classic_results[3]],
    'NUM_HBS': [classic_results[4]],
    'NUM_SQUEEZES': [classic_results[5]],
    'NUM_SQ_DET': [classic_results[6]],
    'SQUEEZE_OMIT': [classic_results[7]],
    'MEAN_SQ_HEIGHT': [classic_results[8]],
    'STD_SQ_HEIGHT': [classic_results[9]]}

    return pd.DataFrame(data=results_df)

def pair_t1000(ptt_indx, det_array, index):
    '''
    Input: hb_array, detection array, index
    hb_array, detection array, index should be the same size
    Output: 1D array of any/all paired detection delay times
    '''
    # Pulse Transit Time (ptt) shifts rpeaks 200ms to the right
    #TODO: add 200 to index, cut off indices >60K
    det_indx = index[det_array == 1]

    latency_ar = np.array(0)

    for hb in range(0, np.size(ptt_indx)-1,1):
        # TODO: use pd.Series.diff() here
        half = int(round((ptt_indx[hb+1] - ptt_indx[hb])/2, 0))
        lagging_win = range(ptt_indx[hb], ptt_indx[hb] + half, 1)
        leading_win = range(ptt_indx[hb] + half + 1, ptt_indx[hb+1])
        # add any detections that fall within leading or lagging window
        # detections will only be assigned to one window
        # every detection is assigned except a detection between 0 and first ptt and last ptt and end of data
        # howevver, not every _win will have detections
        lagg_dets = det_indx[ ((lagging_win[0] < det_indx) & (det_indx < lagging_win[-1])) ]
        lead_dets = det_indx[ ((leading_win[0] < det_indx) & (det_indx < leading_win[-1])) ]
        # TODO: move this outside loop, if possible
        # calculate latency
        if len(lagg_dets > 0):
            new_lagg_vals = lagg_dets - ptt_indx[hb] # will always yield positive
            latency_ar = np.append(latency_ar, new_lagg_vals)
        if len(lead_dets > 0):
            new_lead_vals = lead_dets - ptt_indx[hb+1] # will always yield negative
            latency_ar = np.append(latency_ar, new_lead_vals)
        
    return latency_ar

# Export PDF of results
def save_plot(ecg_df, sqz_df, merge_df, classic_pairs, file_prefix, file_suffix, quality):
    plt.clf()  
    plt.cla()
    classic_pair_df = pd.DataFrame(classic_pairs)
    # Create df for ecg detections
    detected_ecg = classic_pair_df.iloc[:, [0]].rename(columns={0:'index'})
    detected_ecg = pd.merge(detected_ecg, merge_df, how='left', on='index')[['index', 'Data_ECG']]
    # Create df for sqz detections
    detected_sqz = classic_pair_df.iloc[:, [1]].rename(columns={1:'index'})
    detected_sqz = pd.merge(detected_sqz, merge_df, how='left', on='index')[['index', 'Data_Squeeze']]
    # Merge them together #
    detected = detected_ecg.join(detected_sqz, how='inner', lsuffix='_ECG', rsuffix='_Squeeze')
    detected_segs = detected.to_numpy()
    detected_segs = np.reshape(detected_segs, (-1, 2, 2))
    # Generate Plot components
    fig_out = plt.figure(figsize=(36,3))
    ax_out = fig_out.add_subplot(111)
    ax_out.plot(ecg_df['index'], ecg_df.Data, 'r-', lw=0.25)
    ax_out.plot(sqz_df['index'], sqz_df.Data, 'b-', lw=0.25)
    ax_out.plot(detected_ecg['index'], detected_ecg.Data_ECG, 'ko', ms=0.5)
    ax_out.plot(detected_sqz['index'], detected_sqz.Data_Squeeze, 'ko', ms=0.5)
    line_segments = LineCollection(detected_segs, 
        linewidths=0.25,
        colors='g')
    ax_out.add_collection(line_segments)
    fig_out.savefig(file_prefix + 'Image-' + file_suffix +  '.pdf', dpi=1000)
    plt.clf()  
    plt.cla() 

        
# Compute trial statistics
def calc_t1000(latency_ar, hb_array, det_array, index, quality):
    '''
    Input: latency array from subject
            hb_array
            detections array 
            index of arrays
    Output: dataframe ready for csv writing
    '''
    # create empty array
    t1000_results = np.empty(10)
    t1000_results[:] = np.nan
    # check quality. skip T1000 if data is medium or if there are no squeezes.
    if quality == 'Medium' or (np.nansum(det_array) == 0):
        return create_t1000_results_df(t1000_results)
    # determine subj score
    SD_sub = round(np.std(latency_ar), 1)
    resampled_std = np.array([])
    n = 1000
    # PPT = pulse transit time of 200ms
    ptt = np.zeros(200)
    ptt_shifted = np.append(ptt, hb_array)[:-200]
    ptt_indx = index[ptt_shifted == 1]
    for precision_sample in range(0, n, 1):
        resampled_det_array = resampled_experiment(sum(hb_array), det_array)
        sampled_latency = pair_t1000(ptt_indx, resampled_det_array, index)
        SD_resample = round(np.std(sampled_latency), 2)
        resampled_std = np.append(resampled_std, SD_resample)
    # determine resampled precision score
    ESD = round(np.mean(resampled_std), 2)
    SESD = round(np.std(resampled_std), 2)
    if ESD == 0 or SESD == 0:
        resampled_precision = 0
    else:
        resampled_precision = round(((ESD - SD_sub) / SESD), 2)
    # build dataframe
    t1000_results[0] = round(np.mean(latency_ar), 1)
    t1000_results[1] = SD_sub
    t1000_results[2] = ESD
    t1000_results[3] = SESD
    t1000_results[4] = resampled_precision
    t1000_results[5] = sum(hb_array) - np.nansum(det_array)

    return create_t1000_results_df(t1000_results)


def create_t1000_results_df(t1000_results):
    '''
    Input: array of results from cacl_t1000
    Output: dataframe of headers and results ready for writing to CSV
    '''
    results_df = {'MEAN': [t1000_results[0]],
    'SD': [t1000_results[1]],
    'ESD': [t1000_results[2]],
    'SESD': [t1000_results[3]],
    'RESAMPLED_PRECISION': [t1000_results[4]],
    'HB_WITH_NO_DETS': [t1000_results[5]]}

    return pd.DataFrame(data=results_df)
# Write analytical output

def resampled_experiment(num_hbs, det_array):
    '''
    Input: hh_array and detection array
    Output: resampled detection array
    Create a random, uniform dist of detections
    Number of detections is equal to number of subject created detections
    '''
    # TODO: allow for third argument 'n_resamples', return that many re-samples
    num_detects = np.nansum(det_array)
    resampled_array = np.zeros(len(det_array))
    resampled_detects = np.random.randint(0, int(np.size(det_array)), int(num_detects))
    resampled_array[resampled_detects] = 1
    return resampled_array
# Create analytical results subplot

def check_for_outliers(hb_array, index):
    '''
    Input: hb array
    Output: message, err
    '''
    normal_hb_rate = 60 # bpm
    min_hbs_expected = int(normal_hb_rate * 0.75)
    max_hbs_expected = int(normal_hb_rate * 1.25)
    min_rpeak_dist_ms = 500
    max_rpeak_dist_ms = 1500
    err = 0
    msg = ''

    hb_indx = np.sort(index[hb_array == 1])
    hb_diff = np.diff(hb_indx)
    hb_diff = np.append(hb_diff, 1000) # append a normal rr interval so len(hb_diff) == len(hb_indx)
    # check if there are fewer than 10 hbs
    if len(hb_indx) <= 10:
        err = 1
        msg = msg + "Fewer than 10 r peaks detected!"
        return msg, err
    if any(hb_diff < min_rpeak_dist_ms):
        err = 1
        min_index = hb_indx[hb_diff == min(hb_diff)]
        msg = msg + 'err: small rr interval at loc ' + str(min_index) + '\n'
    if any(hb_diff > max_rpeak_dist_ms):
        err = 1
        max_index = hb_indx[hb_diff == max(hb_diff)]
        msg = msg + 'err: large rr interval at loc' + str(max_index) + '\n'
    # check if last r peak is too far away from end of data
    if hb_indx[-1] < (len(hb_array) - max_rpeak_dist_ms):
        err = 1
        msg = msg + "err: missing last r peak"
    msg = msg + ' You can bypass all outliers and directly save. Warning! Might cause unexpected behavior!!'
    return msg, err

def amalgamate(folderPath):
    # Save list of subids, conditions, tasks
    folderPath = folderPath.replace('/', '\\')
    subids = [subid for subid in os.listdir(folderPath)
            if os.path.isdir(os.path.join(folderPath, subid))]
    conditions = ['Pre Trial', 'Trial', 'Post Trial']
    tasks = ['Task{}'.format(task) for task in range(1,6)]
    # Get list of completed outputs
    outpaths = [[path_creator(folderPath, subid, cond, task), subid, cond, task] for subid in subids for cond in conditions for task in tasks
            if (os.path.isfile(path_creator(folderPath, subid, cond, task)[0])
            and os.path.isfile(path_creator(folderPath, subid, cond, task)[1]))]
    # Create dictionary of output dataframes
    dfs = {}
    for i, outpath in enumerate(outpaths):
        classic_df = pd.read_csv(outpath[0][0]).drop(columns=['Unnamed: 0'])
        t1000_df = pd.read_csv(outpath[0][1]).drop(columns=['Unnamed: 0'])
        response_df = getResponse(outpath[0][0], outpath[3])
        df = classic_df.join(t1000_df, lsuffix='_CLASSIC', rsuffix='_T1000')
        df = df.join(response_df, rsuffix='_RESPONSE')
        df.insert(0, 'SUBJECTID', [outpath[1]])
        df.insert(1, 'CONDITION', [outpath[2]])
        df.insert(2, 'TASK', [outpath[3]])
        dfs[i]=df.copy()
    df = pd.concat(dfs)
    # Save time of export creation as timestamp
    timestamp = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    df.to_csv('{}/amalgamate_{}.csv'.format(folderPath, timestamp), index=False)

def path_creator(folderPath, subid, cond, task):
    outpath1 = os.path.join(folderPath, subid, cond, 
            '{}-{}-{}-Classic.csv'.format(subid, cond, task))
    outpath2 = os.path.join(folderPath, subid, cond, 
            '{}-{}-{}-T1000.csv'.format(subid, cond, task))
    return (outpath1, outpath2)

def getResponse(outputs_path, task_id):
    # read data from TaskTimes.CSV which is found under /inputs/ folder
    response_int_array = np.empty(0, dtype=int)
    response_folder_path = outputs_path.replace('Outputs', 'Inputs')
    response_folder_path = response_folder_path.split('\\')[0:-1]
    response_folder_path = '\\'.join(response_folder_path)
    response_folder_path = os.path.join(response_folder_path)
    file_names = os.listdir(response_folder_path)
    for file in file_names:
        if 'TaskTimes' in file:
            TaskTimes_file = os.path.join(response_folder_path, file)
            break
    # open folder and find tasktimes.csv
    taskTimeList = []
    with open(TaskTimes_file) as fp:
        line = fp.readline()
        while line:
            taskTimeList.append(line)
            line = fp.readline()
    if task_id == 'Task1':
        task_range = range(16, 19, 1)
    elif task_id == 'Task2':
        task_range = range(19, 22, 1)
    elif task_id == 'Task3':
        task_range = range(22, 25, 1)
    elif task_id == 'Task4':
        task_range = range(25, 28, 1)
    elif task_id == 'Task5':
        task_range = range(28, 31, 1)
    else:
        return create_response_df(np.zeros(3))

    for line in task_range:
        response_str = taskTimeList[line].split(':')[1][:-1]
        if response_str.isdigit():
            response = int(response_str)
        else: # if data is missing, or an invalid repsonse is given, enter value as None
            response = None
        response_int_array = np.append(response_int_array, response)

    response_dataframe = create_response_df(response_int_array)
    return response_dataframe

def create_response_df(response_array):
    # create useful headers for all response data
    response_df = {
    'INTENSITY': [response_array[0]],
    'DIFFICULTY': [response_array[1]],
    'PERFORMANCE': [response_array[2]]
    }

    return pd.DataFrame(data=response_df)
