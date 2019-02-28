import sys
import os
from os.path import basename

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
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from FFTOperation import fft_with_filter
from sklearn.neural_network import MLPClassifier

import pickle
    
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


class PlotCanvas(FigureCanvas):
 
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
 
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
 
        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
 
    def plot(self, value):
        print('called')
        ax = self.figure.add_subplot(111)
        ax.clear()
        ax.plot(value, 'r-')
        ax.set_title('Predicted Result')
        self.draw()

    def matrixPlot(self, value):
        ax = self.figure.add_subplot(111)
        ax.clear()
        ax.matshow(value)
        ax.set_title('Confusion Matrix')
        self.draw()

class Widget(QWidget):
    
    def __init__(self):
        
        print('Training patterns...')
        
        super().__init__()
        uifile = os.path.join(os.path.dirname(__file__), 'DataAnalysis.ui')
        self.ui = loadUi(uifile, self)

        # canvas
        self.m = PlotCanvas(self, width=4, height=3)
        self.m.move(500,0)

        # class variables
        self.files_list = []
        self.fft_list = []
        self.target_index = []

        for elem in self.ui.children():
            name = elem.objectName()
            if name == 'btn_load_file':
                elem.clicked.connect(self.load_file)
            if name == 'btn_load_predict_file':
                elem.clicked.connect(self.load_predict_file)
            if name == 'btn_predict':
                elem.clicked.connect(self.start_prediction)
            if name == 'table_widget':
                self.table = elem
                self.table.setRowCount(5)
                self.clip = QApplication.clipboard()
            if name == 'btn_start_fft':
                elem.clicked.connect(self.start_fft)
            if name == 'btn_train':
                elem.clicked.connect(self.train)
            if name == 'txt_low_pass':
                self.low_pass = elem
            if name == 'txt_high_pass':
                self.high_pass = elem
            if name == 'btn_load_file':
                elem.clicked.connect(self.load_file)           


    def load_items(self):
        row = 0
        for item in self.files_list:
            self.table.setItem(row, 0, QTableWidgetItem(basename(item)))
            self.table.setItem(row, 1, QTableWidgetItem('LOADED'))
            row += 1

    def load_file(self):
      try:
        filename = QFileDialog.getOpenFileName(w, 'Open File', './')
        self.files_list.append(filename[0])
        self.load_items()
      except Exception: 
        pass
    
    def load_predict_file(self):
        try:
            filename = QFileDialog.getOpenFileName(w, 'Open File', './')
            self.predict_file = filename[0]
        except Exception: 
            pass
    
    def start_prediction(self):
        df = pd.read_csv(self.predict_file, usecols=range(1,4097))
        data_set = fft_with_filter(df, self.low_pass.text(), self.high_pass.text())
        # mean = np.mean(data_set)
        # std = np.std(data_set)
        # median = np.median(data_set)
        # feature = [mean, median, std ]
        features = []
        for f_list in data_set:
            a = np.ravel(f_list)
            features += [np.ravel(f_list).tolist()]
        result = self.clf.predict(features)
        print(result)

    def start_fft(self):
        try:
            for file in self.files_list:
                df = pd.read_csv(file, usecols=range(1,4097))
                data = fft_with_filter(df, self.low_pass.text(), self.high_pass.text())
                self.target_index.append(len(data))
                self.fft_list += data
                print(self.fft_list)
        except Exception: 
            pass
        

    def train(self):
        features = []
        targets = []
        index = 0
        for t_i in self.target_index:
            targets += [index]*t_i
            index +=1
        # for index, data in enumerate(self.fft_list):
        #     mean = np.mean(data)
        #     std = np.std(data)
        #     var = np.var(data)
        #     median = np.median(data)
        #     feature = [mean, median, std ]
        #     features.append(feature)
        #     targets.append([index])

        self.clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(50,15,5), random_state=1)
        features = []
        for f_list in self.fft_list:
            a = np.ravel(f_list)
            features += [np.ravel(f_list).tolist()]
        self.clf.fit(features, targets)
        return

    def plotFinalGraph(self):
      try:
        shuffleData1, shuffleTarget1 = self.generateDataAndTarget(self.testFinalData, 0)
        print(self.clf.predict(shuffleData1))
        self.m2.plot(self.clf.predict(shuffleData1))
      except Exception: 
        print(Exception)
        pass

    def keyPressEvent(self, e):
        if (e.modifiers() & QtCore.Qt.ControlModifier):
            selected = self.table.selectedRanges()

            if e.key() == QtCore.Qt.Key_C: #copy
                s = '\t'+"\t".join([str(self.table.horizontalHeaderItem(i).text()) for i in range(selected[0].leftColumn(), selected[0].rightColumn()+1)])
                s = s + '\n'

                for r in range(selected[0].topRow(), selected[0].bottomRow()+1):
                    s += '\t'
                    for c in range(selected[0].leftColumn(), selected[0].rightColumn()+1):
                        try:
                            s += str(self.table.item(r,c).text()) + "\t"
                        except AttributeError:
                            s += "\t"
                    s = s[:-1] + "\n" #eliminate last '\t'
                self.clip.setText(s)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Widget()
    w.show()
    sys.exit(app.exec_())


