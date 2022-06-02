# Authors:
# Jonathon Alexander
# Zachary Osman
# William Schoenhals
#
# R34 Interoception GUI
# Float Clinic and Research Center
# Laureate Institute for Brain Research
#
# Fall, 2019
# Spring, 2020
#
import sys, os, utils, re, glob, traceback, logging
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QHBoxLayout, QLabel, QGridLayout, QTextBrowser, QWidget, QPushButton, QFileDialog, QProgressBar, QScrollBar, QMessageBox
from PyQt5.QtCore import QSize, Qt, QObject, QEvent
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector, Slider
from datetime import datetime
import numpy as np
import pandas as pd
import time

# define font style and size
TEXT_ATTR = QtGui.QFont("Times", 20)
VERSION = 'V1.44'

class Top_Module(QMainWindow):

    def __init__(self):
        QMainWindow.__init__(self).__init__(\
            flags=QtCore.Qt.WindowMinimizeButtonHint |
            QtCore.Qt.WindowMaximizeButtonHint |
            QtCore.Qt.WindowCloseButtonHint)

        # initialize data elements #
        self.ecg_df = pd.DataFrame()
        self.sqz_df = pd.DataFrame()
        self.merge_df = pd.DataFrame()
        self.fileDirectory = None
        self.taskFiles = []
        self.taskFilesCount = 0
        self.task_id = ''
        self.qualityAssurance = []
        self.quality = ''
        self.bypass_save = False
        self.flip_flipper = False

        #initialize logging
        logFilePath = os.path.dirname(os.path.abspath(__file__))
        self.LOG_FILENAME = logFilePath + '/error_log.out'
        logging.basicConfig(filename=self.LOG_FILENAME, level=logging.ERROR)
        logging.error(datetime.now())

        # set properties of main window
        self.setMinimumSize(QSize(700, 435))    
        self.setWindowTitle("R34 Interoception Analysis")

        # create our grid layout
        self.mainLayout = QGridLayout()

        # create layout to hold the interactive ECG plot
        self.plotLayout = QHBoxLayout()
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.draw()
        self.plotLayout.addWidget(self.canvas) # add matplotlib canvas as widget
        self.canvas.installEventFilter(self)
        self.ax = self.figure.add_subplot(111) # create axis ax

        # Create QtWidgets
        self.browserButton = QPushButton(font=TEXT_ATTR, text='Browse for file',
                clicked=self.clickedBrowser) # file browser button
        self.outputText = QTextBrowser(font=TEXT_ATTR)
        self.outputText.setText('Feeling curious? Press a button') # output text box
        self.outputText.setFixedWidth(425)
        self.outputTextSubID = QTextBrowser(font=TEXT_ATTR)
        self.outputTextSubID.setText('None') # 'sub_id' text output box
        self.outputTextSubID.setFixedWidth(150)
        self.addRemoveStatus = QLabel(font=TEXT_ATTR, text='Toggle Status:')
        self.versionStatus = QLabel(font=TEXT_ATTR, text=VERSION)
        self.addRemoveIndicator = QLabel(font=TEXT_ATTR, text='Add')
        self.addRemoveIndicator.setStyleSheet('color : green')
        self.addRemoveIndicator.setFixedWidth(90)
        self.addRemoveButton = QPushButton(font=TEXT_ATTR, 
                text='Mode', clicked=self.add_remove_toggle) # 'Toggle Add/Remove' button
        self.scrollbar = QScrollBar(orientation=Qt.Horizontal, maximum=100, 
                sliderMoved=self.updatePlot) # scrollbar
        self.prog_bar = QProgressBar(font=TEXT_ATTR, value=0) # progress bar
        self.saveButton = QPushButton(font=TEXT_ATTR, 
                text='Save', clicked=self.clickedSave) # save button 
        self.amalgamButton = QPushButton(font=TEXT_ATTR, 
                text='Amalgamate', clicked=self.clicked_amalgam) # amalgamate button
        self.flip_ecg = QPushButton(font=TEXT_ATTR,
                text='Flip ECG', clicked=self.clicked_flip)

        # add widgets and layouts to our grid layout
        self.mainLayout.addWidget(self.browserButton, 1, 1, 1, 3)
        self.mainLayout.addWidget(self.flip_ecg, 1, 1, 2, 4)
        self.mainLayout.addWidget(self.versionStatus, 18, 1, 1, 1)
        self.mainLayout.addWidget(self.outputText, 1, 10, 1, 30)
        self.mainLayout.addWidget(self.outputTextSubID, 2, 10, 1, 8)
        self.mainLayout.addWidget(self.addRemoveStatus, 2, 20, 1, 1)
        self.mainLayout.addWidget(self.addRemoveIndicator, 2, 21, 1, 1)
        self.mainLayout.addWidget(self.addRemoveButton, 2, 24, 1, 3)
        self.mainLayout.addLayout(self.plotLayout, 3, 0, 1, 32)
        self.mainLayout.addWidget(self.scrollbar, 4, 0, 1, 32)
        self.mainLayout.addWidget(self.prog_bar, 18, 10, 1, 12)
        self.mainLayout.addWidget(self.saveButton, 17, 24, 1, 3)
        self.mainLayout.addWidget(self.amalgamButton, 18, 24, 1, 3)
        # set grid layout to our main window
        # requires use of a 'dummy' widget for setCentralWidget()
        self.dummyWidget = QWidget()
        self.dummyWidget.setLayout(self.mainLayout)
        self.setCentralWidget(self.dummyWidget)

        # Initialize default add/remove state
        self.set_add_remove_interactivity(self.addRemoveIndicator.text())
        self.GUI_default()

    def eventFilter(self, obj, event):
        if obj == self.canvas:
            #print(event.type())
            if event.type() == QEvent.Wheel:
                wheelEvent = event
                displacey = -wheelEvent.angleDelta().y()*10
                scrollval = self.scrollbar.value()
                self.scrollbar.setValue(scrollval+displacey)
                self.updatePlot()
                return True
            else:
                return False
        else:
            return QMainWindow.eventFilter(self, obj, event)

    def add_remove_toggle(self):
        self.saveButton.setText('Save')
        self.bypass_save = False
        try:
            text = self.addRemoveIndicator.text()
            if text == 'Remove':
                self.addRemoveIndicator.setText('Add')
                self.addRemoveIndicator.setStyleSheet('color : green')
            else:
                self.addRemoveIndicator.setText('Remove')
                self.addRemoveIndicator.setStyleSheet('color : red')
            self.set_add_remove_interactivity(self.addRemoveIndicator.text())
        except:
            if self.fileDirectory == None:
                output_dir = 'no viable file directory'
            else:
                output_dir = self.fileDirectory
            logging.exception('Exception on add_remove_toggle' + ' ' + output_dir)
            var = traceback.format_exc()
            self.showWarningDialog(var)
    def set_add_remove_interactivity(self, text):
        self.saveButton.setText('Save')
        self.bypass_save = False
        if text == 'Remove':
            self.rectprops = dict(edgecolor=None, alpha=0.4, facecolor='red')
            self.rs = SpanSelector(self.ax, self.RemovePeaks, direction='horizontal',
                useblit=False, rectprops=self.rectprops, minspan=1)
        else:
            self.rectprops = dict(edgecolor=None, alpha=0.4, facecolor='green')
            self.rs = SpanSelector(self.ax, self.AddPeaks, direction='horizontal',
                useblit=False, rectprops=self.rectprops, minspan=1)

    def AddPeaks(self, x1, x2):
        try:
            if self.saveButton.isEnabled() == False:
                # save button is disabled when no data is present to detect
                # do not allow user to interact with graph with no data
                return 
            mask = self.ecg_df['index'].between(x1, x2, inclusive=False)
            selected = self.ecg_df[mask]
            if not selected.empty:
                new_max_idx = selected.set_index('index')['Data'].idxmax()
                self.ecg_df.at[new_max_idx, 'Detections'] = 1
                self.updatePlot()
                self.set_add_remove_interactivity('Add')
        except:
            if self.fileDirectory == None:
                output_dir = 'no viable file directory'
            else:
                output_dir = self.fileDirectory
            logging.exception('Exception on save button' + ' ' + output_dir)
            var = traceback.format_exc()
            self.showWarningDialog(var)

    def RemovePeaks(self, x1, x2):
        try:
            if self.saveButton.isEnabled() == False:
                # save button is disabled when no data is present to detect
                # do not allow user to interact with graph with no data
                self.writeMessage(utils.message())
                return 
            mask = self.ecg_df['index'].between(x1, x2, inclusive=False)
            selected = self.ecg_df[mask]
            if not selected.empty:
                self.ecg_df.loc[mask, 'Detections'] = 0
                self.updatePlot()
                self.set_add_remove_interactivity('Remove')
        except:
            if self.fileDirectory == None:
                output_dir = 'no viable file directory'
            else:
                output_dir = self.fileDirectory
            logging.exception('Exception on save button' + ' ' + output_dir)
            var = traceback.format_exc()
            self.showWarningDialog(var)

    def clickedBrowser(self):
        #QtWidgets.QApplication.processEvents()
        self.flip_ecg.setEnabled(False)
        self.flip_flipper = False
        title = "Open Trial .csv OR Open Quality Text File" 
        file_filter = "Biopatch (*TrialBioPatch*);; CSV Files (*.csv);; Bad Quality (*Quality*)"
        try:
            possible_file, _ = QFileDialog.getOpenFileName(self, title, "", file_filter)
            if len(possible_file) == 0:
                return
            self.GUI_ready_for_new_sub()
            self.fileDirectory = possible_file
            if "Quality Assurance.txt" in possible_file:
                # this trial comes from an experiment with bad data
                # user seletect a *Quality Assurance.txt
                # build analysis files with ONLY subjective data
                self.create_low_qual_exp_results()
                return
            os.chdir(os.path.dirname(self.fileDirectory))
            self.taskFiles = glob.glob('*BioPatch*.csv')
            qualityAssurance_list = glob.glob('*Quality*.txt')
            self.qualityAssurance = qualityAssurance_list[0]
            self.quality = self.quality_check()
            self.file_check()
            self.updateProg(0)
        except:
            if self.fileDirectory == None:
                output_dir = 'no viable file directory'
            else:
                output_dir = self.fileDirectory
            logging.exception('Exception on clicked_browswer' + ' ' + output_dir)
            var = traceback.format_exc()
            self.showWarningDialog(var)
            self.GUI_ready_for_new_sub()

    def quality_check(self):
        with open(self.qualityAssurance, 'r') as f:
            quality_string = f.read()
            matches = re.findall('[0-9]+', quality_string)
            if len(matches) == 4:
                return 'High'
            else:
                return 'Medium'


    def clicked_flip(self):
        if self.ecg_df.empty:
            return
        if self.flip_flipper == False:
            self.ecg_df, self.sqz_df, sub_id, cond, self.task_id = utils.read_ecg_csv(True, self.fileDirectory)
            self.flip_flipper = True
        else:
            self.ecg_df, self.sqz_df, sub_id, cond, self.task_id = utils.read_ecg_csv(False, self.fileDirectory)
            self.flip_flipper = False
        self.updatePlot(loadstate='Initial')
        self.bypass_save = False      


    def file_check(self):
        try:
            msg = "The source file you have opened has an existing output file\nDo you wish to continue with analysis, and overwrite the existing output?"

            overwrite_possible = utils.overwrite_check(self.fileDirectory)
            if overwrite_possible:
                user_wants_overwrite = self.showDialog(msg)
                if user_wants_overwrite != True:
                    self.writeMessage("You decided not to overwrite. Select new task") 
                    self.GUI_ready_for_new_sub()
                    return

            self.ecg_df, self.sqz_df, sub_id, cond, self.task_id = utils.read_ecg_csv(False, self.fileDirectory)
            self.writeSubID(sub_id[0], cond[0], self.task_id)
            self.flip_ecg.setEnabled(True)
            self.flip_flipper = False

            if len(self.ecg_df) == 1 or self.task_id == 'Task2':
                # skip manual detections
                self.GUI_ready_for_auto_tasks()
                self.updatePlot()
                self.clickedSave()
                return
    
            self.updatePlot(loadstate='Initial')
            self.GUI_ready_for_interact()
        except:
            if self.fileDirectory == None:
                output_dir = 'no viable file directory'
            else:
                output_dir = self.fileDirectory
            logging.exception('Exception on file_check' + ' ' + output_dir)
            var = traceback.format_exc()
            self.showWarningDialog(var)


    def showDialog(self, msg_text):
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Information)
        msgBox.setText(msg_text)
        msgBox.setWindowTitle("Warning Message")
        msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        return_val = msgBox.exec()

        if return_val == QMessageBox.Yes:
            return True
        else:
            return False

    def showWarningDialog(self, msg_text):
        self.GUI_ready_for_new_sub()
        err_dialog = QtWidgets.QErrorMessage()
        err_dialog.resize(550, 200)
        err_dialog.setWindowTitle("Error! Unable to complete operation. Please save message and report error.")
        err_dialog.showMessage(msg_text)
        err_dialog.exec()


    def writeMessage(self, msg):
        # Display 'msg' arg in outputText widget. Clears previous message
        # TODO: protect function from crash if attempting to print unprintable msg
        self.outputText.clear()
        self.outputText.append(msg)

    def writeSubID(self, msg, cond, task):
        # Display 'msg' arg in outputText widget. Clears previous message
        # TODO: protect function from crash if attempting to print unprintable msg
        self.outputTextSubID.clear()
        self.outputTextSubID.append(msg + ' ' + cond + ' ' + task)

    def updatePlot(self, loadstate=''):
        try: 
            # How many samples to see in viewing window
            view_width = 6000
            # Allow room on either end of plot
            view_buffer = view_width/10
            val = self.scrollbar.value()
            # Define Window view limits
            if loadstate == 'Initial':
                scale = 1.10
                idxmin = self.ecg_df['Data'].idxmin()
                idxmax = self.ecg_df['Data'].idxmax()
                datamin = self.ecg_df['Data'][idxmin]
                datamax = self.ecg_df['Data'][idxmax]
                center = (datamax + datamin)/2
                width = (datamax - datamin)
                # Set scroll-bar range
                scroll_min = self.ecg_df.index.min() - view_buffer
                scroll_max = self.ecg_df.index.max() - view_width + view_buffer
                self.scrollbar.setMinimum(scroll_min)
                self.scrollbar.setMaximum(scroll_max)
                self.scrollbar.setValue(scroll_min)
                # Set initial viewing limits
                xlims = (scroll_min, scroll_min+view_width)
                ylims = (center - width*scale/2, 
                        center + width*scale/2)
            else:
                xlims = (val, val+view_width)
                ylims = self.ax.get_ylim()
            # Clear plot
            self.ax.clear()
            # Conditionally re-draw plots
            if not self.ecg_df.empty:
                self.ax.plot(self.ecg_df['index'], self.ecg_df['Data'], 'k-')
                self.ax.plot(
                    self.ecg_df[self.ecg_df['Detections']==1]['index'], 
                    self.ecg_df[self.ecg_df['Detections']==1]['Data'], 'ro')
                # Set x-y labels
                self.ax.set_xlabel('mSec')
                self.ax.set_ylabel('mVolt')
                self.ax.set_yticklabels([])
            # Set viewing window
            self.ax.set_xlim(xlims)
            self.ax.set_ylim(ylims)
            # Draw canvas
            self.canvas.draw()
        except:
            if self.fileDirectory == None:
                output_dir = 'no viable file directory'
            else:
                output_dir = self.fileDirectory
            logging.exception('Exception on click detect' + ' ' + output_dir)
            var = traceback.format_exc()
            self.showWarningDialog(var)

    def clicked_amalgam(self):
        try:
            # TODO: make them navigate and select the 'Outputs' folder before proceeding
            # TODO: recursive search for specified outputs, then amalgamate
            folderPath = QFileDialog.getExistingDirectory(self, 'Select Outputs folder')
            folderName = os.path.basename(folderPath)
            # Catch if folder name is not 'Outputs'
            if folderName != 'Outputs':
                QMessageBox.warning(self, '', 'Folder choice must be \'Outputs\'')
            if folderName == 'Outputs':
                utils.amalgamate(folderPath)
                self.writeMessage('Amalgamation created')
        except:
            if self.fileDirectory == None:
                output_dir = 'no viable file directory'
            else:
                output_dir = self.fileDirectory
            logging.exception('Exception on clicked_amalgam' + ' ' + output_dir)
            var = traceback.format_exc()
            self.showWarningDialog(var)

    def clickedSave(self):
        try:
            if len(self.ecg_df) <= 1:
                self.writeMessage("saving empty data...")
                empty_classic_ar = np.empty(10)
                empty_classic_ar[:] = np.nan
                empty_t1000_ar = np.empty(6)
                empty_t1000_ar[:] = np.nan    
                empty_classic_df = utils.create_classic_results_df(empty_classic_ar)
                empty_t1000_df = utils.create_t1000_results_df(empty_t1000_ar)
                msg, err = utils.write_empty_output(self.fileDirectory, empty_t1000_df, empty_classic_df)
            else:
                self.writeMessage("saving data...")
                time.sleep(0.5)
                self.updateProg(10)                
                if self.task_id != 'Task2':  
                    user_msg, err = utils.check_for_outliers(self.ecg_df.Detections, self.ecg_df.index)
                    self.writeMessage(user_msg)
                    if err == 1 and not(self.bypass_save):
                        # do not allow save to continue
                        self.updateProg(0)
                        self.saveButton.setText('Bypass Save')
                        self.saveButton.setEnabled(True)
                        self.bypass_save = True
                        return
                    else:
                        self.saveButton.setText('Save')
                        self.bypass_save = False
                self.sqz_df, peak_heights = utils.detect_squeezes(utils.detrend_squeeze(self.sqz_df))
                self.merge_df = utils.merge_ecg_squeeze(self.ecg_df, self.sqz_df)
                classic_pairs = utils.pair_classic(self.merge_df)
                time.sleep(0.5)
                self.updateProg(20)
                self.classic_pair_df = pd.DataFrame(classic_pairs)
                classic_results_df = utils.classic_calc(self.merge_df, classic_pairs, self.quality, peak_heights)
                time.sleep(0.5)
                self.updateProg(30)
                hb_array = self.merge_df.Detections_ECG
                # determine pulse transit time by shifting 200ms
                # find indicies of PPT affect HB's
                ptt = np.zeros(200)
                ptt_shifted = np.append(ptt, hb_array)[:-200]
                ptt_indx = self.merge_df.index[ptt_shifted == 1]
                t1000_pairs = utils.pair_t1000(ptt_indx, self.merge_df.Detections_Squeeze, self.merge_df.index )
                t1000_results_df = utils.calc_t1000(t1000_pairs, self.merge_df.Detections_ECG, self.merge_df.Detections_Squeeze, self.merge_df.index, self.quality)
                self.updateProg(90)
                if self.task_id == 'Task2':
                    # replace ECG with tone values for the plotting element
                    self.merge_df.Data_ECG = self.merge_df.Detections_ECG * max(self.merge_df.Data_Squeeze)
                    self.ecg_df.Data = self.merge_df.Data_ECG
                msg, err = utils.write_output(self.fileDirectory, self.merge_df, t1000_results_df, classic_results_df, self.sqz_df, self.ecg_df, classic_pairs, self.quality)

            if err == 1:
                self.writeMessage(msg)
                self.updateProg(0)
                self.GUI_ready_for_new_task()
            else:
                self.writeMessage('Save succesfull!')
                self.GUI_ready_for_new_task()
                self.updateProg(100)    

                if self.task_id == 'Task1':
                    self.taskFilesCount = 0
                elif self.task_id == 'Task2':
                    self.taskFilesCount = 1
                elif self.task_id == 'Task3':
                    self.taskFilesCount = 2
                elif self.task_id == 'Task4':
                    self.taskFilesCount = 3
                else:
                    self.taskFilesCount = 4

                self.taskFilesCount += 1

                if self.taskFilesCount == 5:
                    self.taskFilesCount = 0
                    self.updateProg(0)
                    self.taskFiles = []
                    self.writeMessage("No more automated tasks available. Choose new experiment.")
                    self.writeSubID("","","")
                    self.GUI_ready_for_new_sub()
                    return
                try: 
                    self.nextTask()
                except:
                    if self.fileDirectory == None:
                        output_dir = 'no viable file directory'
                    else:
                        output_dir = self.fileDirectory
                    logging.exception('Exception on task files' + ' ' + output_dir)                    
                    self.writeMessage("Failed to automatically setup next task. Recheck connection to Ouputs folder and try again.")
                    return
        except:
            if self.fileDirectory == None:
                output_dir = 'no viable file directory'
            else:
                output_dir = self.fileDirectory
            logging.exception('Exception on task files' + ' ' + output_dir)
            var = traceback.format_exc()
            self.showWarningDialog(var)

    def nextTask(self):
        try:   
            #pull up the fileDirectory for next task.
            delim = '/'
            file_parts = [part+delim for part in self.fileDirectory.split(delim)] 
            file_parts[-1] = self.taskFiles[self.taskFilesCount]
            new_task_file = "".join(file_parts)
            self.fileDirectory = new_task_file
            fileBase = os.path.basename(self.fileDirectory)
            sub_id = ' '.join(re.findall(r'\w{2}\d{3}', fileBase))
            task = ' '.join(re.findall(r'Task\d{1}', fileBase))
            task_number = ' '.join(re.findall(r'\d{1}', task))
            msg_text = "Would you like to evaluate the next task for " + sub_id + " " + task + " " + task_number + "?"
                    
            if self.showDialog(msg_text):
                self.file_check()
            else:
                self.GUI_ready_for_new_sub()
                return
        except:
            if self.fileDirectory == None:
                output_dir = 'no viable file directory'
            else:
                output_dir = self.fileDirectory
            logging.exception('Exception on next_task' + ' ' + output_dir)
            var = traceback.format_exc()
            self.showWarningDialog(var)
            self.GUI_ready_for_new_sub()
                                                        
    def updateProg(self, prog_val):
        '''
        Input: int prog_val
        Immediatley updates progress bar
        '''
        self.prog_bar.setValue(prog_val)

    def closeEvent(self, event):
        quit_msg = "Are you sure you want to exit the program?"
        reply = QMessageBox.question(self, 'Message', 
                        quit_msg, QMessageBox.Yes, QMessageBox.No)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def GUI_default(self):
        self.bypass_save = False
        self.browserButton.setEnabled(True)
        self.outputText.setText("Ready for next experiment")
        self.outputTextSubID.setText("No subject selected yet")
        self.saveButton.setEnabled(False)
        self.scrollbar.setEnabled(False)
        self.flip_ecg.setEnabled(False)
        self.flip_flipper = False
    
    def GUI_ready_for_interact(self):        
        self.writeMessage("Detect R Peaks. Save when complete. Or choose new experiment (will lose all unsaved work!)")
        self.saveButton.setEnabled(True)
        self.scrollbar.setEnabled(True)

    def GUI_ready_for_new_sub(self):
        self.bypass_save = False
        self.fileDirectory = None
        self.taskFiles = []
        self.task_id = ''
        self.saveButton.setEnabled(False)
        self.GUI_ready_for_new_task()

    def GUI_ready_for_auto_tasks(self):
        plt.cla()
        plt.clf()
        self.ax.clear()
        self.canvas.draw()
        self.saveButton.setEnabled(False)
        self.scrollbar.setEnabled(False)

    def GUI_ready_for_new_task(self):
        self.bypass_save = False
        self.scrollbar.setEnabled(False)
        plt.cla()
        plt.clf()
        self.ecg_df = pd.DataFrame()
        self.sqz_df = pd.DataFrame()
        self.merge_df = pd.DataFrame()

    def create_low_qual_exp_results(self):
        self.writeMessage("Gathering subjective data for low quality experiment.")
        msg = "You identified this data as Bad Quality. All trial output analysis will be overwritten. Continue?"
        user_wants_continue = self.showDialog(msg)
        if user_wants_continue != True:
            self.writeMessage("You decided not to overwrite. Select new task") 
            self.GUI_ready_for_new_sub()
            return

        empty_classic_ar = np.empty(10)
        empty_classic_ar[:] = np.nan
        empty_t1000_ar = np.empty(6)
        empty_t1000_ar[:] = np.nan    
        empty_classic_df = utils.create_classic_results_df(empty_classic_ar)
        empty_t1000_df = utils.create_t1000_results_df(empty_t1000_ar)

        # create a false self.fileDirectory obj that contains correct sub id, trial, and append task info
        # run write_empty_output for each of the false obj to fully save a bad data trial
        task_names = {'Task1', 'Task2', 'Task3', 'Task4', 'Task5'}
        fileBase = os.path.basename(self.fileDirectory) # Quality Assurance.txt
        i = 0
        for task in task_names:
            false_file_dir = self.fileDirectory + task
            msg, err = utils.write_empty_output(false_file_dir, empty_t1000_df, empty_classic_df)
            if err == 1:
                self.writeMessage(msg)
                self.updateProg(0)
            else:
                msg = 'Saved ' + task
                self.updateProg(20 + 20*i)
                i = i + 1
                time.sleep(0.5)
        if i == 5:
            self.writeMessage('Empty files created for low quality trial. Ready for amalgamate! You can also continue analysis on new sub')
        else:
            self.writeMessage('Progress not 100. Error in saving one or more files. Check connection to save location. Close open trial files. Retry.')
        self.GUI_ready_for_new_sub()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mainWin = Top_Module()
    mainWin.show()
    sys.exit( app.exec_() )

