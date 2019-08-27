from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QFileDialog, QWidget, QApplication, QSizePolicy
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from matplotlib import gridspec
from matplotlib.colors import ListedColormap

import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

class PlotCanvas(FigureCanvas):
   
    def __init__(self, parent=None, width=10, height=10, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)

        gs = gridspec.GridSpec(2, 2)
        gs.update(hspace=1)
        self.axes_result = fig.add_subplot(gs[0,:])
        self.axes_result.set_title('Predicted Result')
        # self.axes_result.SubplotParams(bottom=2)

        self.axes_roc = fig.add_subplot(gs[1,:-1])
        self.axes_roc.set_title('Roc')

        self.axes_confusion = fig.add_subplot(gs[1,:])
        self.axes_confusion.set_title('Confusion Matrix')

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
 
        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
 
    def plot(self, value):
        self.axes_result.clear()
        self.axes_result.plot(value, 'r-')
        self.axes_result.set_title('Predicted Result')
        self.draw()
    
    def roc_plot(self, clf, X, y):
        self.axes_roc.clear()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        cm = confusion_matrix(y_pred, y_test)
        self.matrixPlot(cm)
        pred = clf.predict_proba(X_test)[::,1]
        fpr, tpr, _ = metrics.roc_curve(y_test, pred, pos_label=y[0])
        auc = metrics.roc_auc_score(y_test, pred)
        self.axes_roc.plot(fpr,tpr,label="data 1, auc="+str(auc))
        self.axes_roc.legend(loc=4)
        self.axes_roc.set_ylabel('True Positive Rate')
        self.axes_roc.set_xlabel('False Positive Rate')
        self.draw()

    def matrixPlot(self, value):
        self.axes_confusion.clear()
        cmap = ListedColormap(['g', 'w'])
        self.axes_confusion.matshow(value,cmap=cmap)
        self.axes_confusion.set_title('Confusion Matrix')

        for i in range(len(value)):
            for j in range(len(value[0])):
                c = value[j,i]
                self.axes_confusion.text(i, j, str(c), va='center', ha='center')
        self.draw()

    def matrixPlotConfusion(self, y_test, result):
        self.axes_confusion.clear()
        cm = confusion_matrix(y_test, result)
        print(cm)
        sum = np.sum(cm, axis=1)

        precision_CLASS_A = round(precision_score(y_test, result, average='binary',pos_label='CLASS A'),2)
        precision_CLASS_B = round(precision_score(y_test, result, average='binary',pos_label='CLASS B'),2)
        recall_CLASS_A = round(recall_score(y_test, result, average='binary',pos_label='CLASS A'),2)
        recall_CLASS_B = round(recall_score(y_test, result, average='binary',pos_label='CLASS B'),2)
        cm_new = np.append(cm[0], precision_CLASS_A)
        cm_new2 = np.append(cm[1], precision_CLASS_B)
        cm_new3 = np.array([recall_CLASS_A, recall_CLASS_B, accuracy_score(y_test, result)])
        cm = np.array([cm_new,cm_new2,cm_new3])

        sns.heatmap(cm, annot=True, ax = self.axes_confusion,linewidths=.5,fmt='g',cmap="Reds"); #annot=True to annotate cells

        # labels, title and ticks
        self.axes_confusion.set_xlabel('Predicted labels');
        self.axes_confusion.set_ylabel('True labels'); 
        self.axes_confusion.set_title('Confusion Matrix'); 
        counter = 0
        for i in range(0,2):
            for j in range(0,2):
                percentage = cm[i][j]/sum[i]
                if (counter+1)/3 == 1:
                    counter += 1
                t = self.axes_confusion.texts[counter]
                t.set_text(str(cm[i][j]) + '\n' + str(round(percentage*100,2)) + " %")
                counter = counter + 1

        self.axes_confusion.xaxis.set_ticklabels(['CLASS A', 'CLASS B'])
        self.axes_confusion.yaxis.set_ticklabels(['CLASS A','CLASS B'])
        self.draw()
