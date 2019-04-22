import sys
import os
from os.path import basename
import pickle

from PyQt5.QtGui import *
from PyQt5 import QtCore
from PyQt5.uic import loadUi

from PyQt5.QtWidgets import QFileDialog, QWidget, QApplication, QSizePolicy
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QTableView, QVBoxLayout, QHeaderView

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.fftpack

from FFTOperation import fft_with_filter
from sklearn.neural_network import MLPClassifier

from PlotCanvas import PlotCanvas
    
class UITable(QTableWidget):
    def __init__(self, parent=None, row=5, column=3):
        QTableWidget.__init__(self, parent)
        self.table = QTableWidget(self)
        self.setParent(parent)

        self.table.setRowCount(row)
        self.table.setColumnCount(column)
        self.table.setItem(0, 0, QTableWidgetItem("File Name"))
        self.table.setItem(0, 1, QTableWidgetItem("Status"))
        self.table.setItem(0, 2, QTableWidgetItem("Label"))

        header = self.table.horizontalHeader()       
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)

    def load_items(self, items):
        row = 1
        column = 0
        for item in items:
            self.table.setItem(row, column, QTableWidgetItem(basename(item)))
            row += 1

class Widget(QWidget):
    
    def __init__(self):
        
        print('Training patterns...')
        
        super().__init__()
        uifile = os.path.join(os.path.dirname(__file__), 'DataAnalysis.ui')
        self.ui = loadUi(uifile, self)

        # canvas
        self.m = PlotCanvas(self, width=4, height=7)
        self.m.move(500,0)

        # class variables
        self.files_list = []
        self.fft_list = []
        self.target_index = []

        for elem in self.ui.children():
            name = elem.objectName()
            if name == 'radio_choose_model':
                elem.clicked.connect(self.select_model)
            if name == 'radio_choose_train_model':
                elem.clicked.connect(self.select_train_mode)
                
            if name == 'train_frame':
                elem.hide()
                self.train_frame = elem

                for child_elem in elem.children():
                    child_name = child_elem.objectName()
                    if child_name == 'btn_load_file':
                        child_elem.clicked.connect(self.load_file)
                    elif child_name == 'btn_train':
                        child_elem.clicked.connect(self.start_fft)
                        child_elem.clicked.connect(self.train)
                    elif child_name == 'btn_save_model':
                        child_elem.clicked.connect(self.save_model)
                    elif child_name == 'table_widget':
                        self.table = child_elem
                        self.table.setRowCount(5)
                        self.clip = QApplication.clipboard()
                    # elif child_name == 'btn_start_fft':
                    #     child_elem.clicked.connect(self.start_fft)
                    elif child_name == 'txt_low_pass':
                        self.low_pass = child_elem
                    elif child_name == 'txt_high_pass':
                        self.high_pass = child_elem
                    elif child_name == 'txt_model_filename':
                        self.model_file_name = child_elem

            if name == 'model_frame':
                for child_elem in elem.children():
                    child_name = child_elem.objectName()
                    if child_name == 'browse_model':
                        child_elem.clicked.connect(self.load_model)
                    elif child_name == 'browse_model_input':
                        self.browse_model_input = child_elem
            
            if name == 'prediction_frame':
                for child_elem in elem.children():
                    child_name = child_elem.objectName()
                    if child_name == 'btn_load_predict_file':
                        child_elem.clicked.connect(self.load_predict_file)
                    elif child_name == 'txt_predict':
                        self.txt_predict = child_elem
                    elif child_name == 'btn_predict':
                        child_elem.clicked.connect(self.start_prediction)

    def select_model(self):
        self.model_frame.show()
        self.train_frame.hide()

    def load_model(self):
        filename = QFileDialog.getOpenFileName(w, 'Open File', './model/')
        self.browse_model_input.setText(filename[0])
        self.clf = pickle.load(open(filename[0], 'rb'))

    def save_model(self):
        file_name = self.model_file_name.text()
        pickle.dump(self.clf, open('./model/'+file_name, 'wb'))

    def select_train_mode(self):
        self.model_frame.hide()
        self.train_frame.show() 

    def load_items(self):
        row = 0
        for item in self.files_list:
            self.table.setItem(row, 0, QTableWidgetItem(basename(item)))
            self.table.setItem(row, 1, QTableWidgetItem('LOADED'))
            self.table.setItem(row, 2, QTableWidgetItem(str(row)))
            row += 1

    def load_file(self):
        filename = QFileDialog.getOpenFileName(w, 'Open File', './')
        self.files_list.append(filename[0])
        self.load_items()
    
    def load_predict_file(self):
        filename = QFileDialog.getOpenFileName(w, 'Open File', './model/')
        self.predict_file = filename[0]
        self.txt_predict.setText(self.predict_file)
    
    def start_prediction(self):
        df = pd.read_csv(self.predict_file, header=None)
        data_set = fft_with_filter(df, self.low_pass.text(), self.high_pass.text())
        # mean = np.mean(data_set)
        # std = np.std(data_set)
        # median = np.median(data_set)
        # feature = [mean, median, std ]
        features = []
        for f_list in data_set:
            features += [np.ravel(f_list).tolist()]
        result = self.clf.predict(features)
        self.m.plot(result)

    def start_fft(self):
        try:
            self.fft_list =[]
            for file in self.files_list:
                df = pd.read_csv(file, header=None)
                data = fft_with_filter(df, self.low_pass.text(), self.high_pass.text())
                self.target_index.append(len(data))
                self.fft_list += data
        except Exception: 
            pass
        
    def train(self):
        targets = []
        y = []
        index = 0
        for t_i in self.target_index:
            targets += [index]*t_i
            y += [int(self.table.item(index, 2).text())]*t_i
            index +=1
        # for index, data in enumerate(self.fft_list):
        #     mean = np.mean(data)
        #     std = np.std(data)
        #     var = np.var(data)
        #     median = np.median(data)
        #     feature = [mean, median, std ]
        #     features.append(feature)
        #     targets.append([index])

        self.clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(4,), random_state=1)
        X = []
        for f_list in self.fft_list:
            X += [np.ravel(f_list).tolist()]
        
        self.clf.fit(X, y)
        self.m.roc_plot(self.clf, X, y)
        return

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Widget()
    w.show()
    sys.exit(app.exec_())


