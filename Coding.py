# -*- coding: utf-8 -*-
from PyQt4 import QtGui
from PyQt4 import QtCore
from PyQt4.QtGui import * 
from PyQt4.QtCore import *
import numpy as np
from PyQt4 import QtCore,QtGui
from tasarim import Ui_Dialog
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import copy
import pickle
import os
import cv2
import copy
import pickle
import os 
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
#from sklearn.svm import SVC
#from sklearn import svm
#from sklearn import  linear_model
#from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

from PIL import Image
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation 
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as  plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use
plt.style.use('ggplot')
import random
from sklearn.cross_validation import StratifiedKFold


def func_X_yukle():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn import datasets
    from sklearn.decomposition import PCA
    
    # import some data to play with
    iris = datasets.load_iris()
    X = iris.data[:, :4]  # we only take the first two features.
    y=iris.target
    print y
    print X
    return X,y

    
"""    
def loadDataset(filename,trainingSet=[] , testSet=[]):
	with open(filename, 'rb') as csvfile:
	    lines = csv.reader(csvfile)
	    dataset = list(lines)
	    for x in range(len(dataset)-1):
	        for y in range(4):
	            dataset[x][y] = float(dataset[x][y])
	        if random.random() < split:
	            trainingSet.append(dataset[x])
	        else:
	            testSet.append(dataset[x])    
"""             
class MainWindow(QtGui.QMainWindow, Ui_Dialog):
   

    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        self.setupUi(self)
        self.statusBar().showMessage(unicode("Hazir\n"))
       
      
        #self.dosyayukle_knn.clicked.connect(self.dosyayukle_knn)
       
        self.normalizetest.addItems(["normalize_MinMax", "normalize_Zscore","Medyan"]) 
        self.normalizetest.currentIndexChanged.connect(self.sec)
        
        self.randomforest.clicked.connect(self.RandomForest)
        
        
        
        self.dosyarandom.clicked.connect(self.dosyaYuklerandomforest)
       
    
        self.btn_traintest.clicked.connect(self.train_and_test_forest)
        self.normalizeyukle.clicked.connect(self.dosyaYuklenormalize)
        #self.normalizetraintest.clicked.connect(self.train_and_test_forest)
        self.btnveridok.clicked.connect(self.dosyaYukletrain)
        self.traintest.clicked.connect(self.train_and_test_yuzde)
        self.liste=[]
        
  
            
                
   
    def  classification(self):
        clf=RandomForestClassifier(max_depth=None,random_state=0)
        clf.fit(self.X_train ,self.y_train)
        results=clf.predict(self.X_test)
        print "başarı:",accuracy_score(self.y_test ,results,normalize=False)
    def  RandomForest(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.25, random_state=42)
        print (self.X_train.shape,self.X_test.shape)
        print (self.X_train.shape,self.X_test.shape)
        clf = RandomForestClassifier(max_depth=2, random_state=0)
        clf.fit(self.X_train, self.y_train)
        results=clf.predict(self.X_test)
        print (results)
        print ( "Random Forest Başarısı:",round(accuracy_score(self.y_test, results)*100,2))
        self.label_6.setText("Random Forest Başarı Basari:" + str(round(accuracy_score(self.y_test, results)*100,2))) 
        
    def k_islemi (self):
        X = np.array(self.X_test)
        y = np.array([self.y_train])
        kf = StratifiedKFold(n_splits=5)
             
        #KFold(n_splits=5, random_state=None, shuffle=False)
        for train_index, test_index in kf.split(X):
               print("TRAIN:", train_index, "TEST:", test_index)
               self.X_train,self. X_test = X[train_index], X[test_index]
               self. y_train,self. y_test = y[train_index], y[test_index]
    @QtCore.pyqtSignature("bool")
    def on_dosyayukle_knn_clicked(self):
        
        f = open('./data/tendataset.txt')
        X=[]
        
        for i,row in enumerate(f.readlines()):
            
            currentline = row.split(",")   
            temp=[]
            for column_value in currentline:
                temp.append(column_value)

            X.append(temp)
            
        for row in X:
            print "Data:",row
        
        X=np.array(X)
        print "Array:",X.shape
        self.X=X[:,:4]
        #self.y=X[:,5]
        self.verileri_dok_knn(self.X,self.table_2)
        
    def dosyaYukle(self):
       
        f = open('./data/x.data')
        X=[]
        
        for i,row in enumerate(f.readlines()):
            
            currentline = row.split(",")   
            temp=[]
            for column_value in currentline:
                temp.append(column_value)

            X.append(temp)
            
        for row in X:
            print "Data:",row
        
        X=np.array(X)
        print "Array:",X.shape
        self.X=X[:,:4]
        self.y=X[:,5]
        self.verileri_dok(self.X,self.table)
        
        
    def dosyaYuklerandomforest(self):
        
        import pandas as pd
        import numpy
        data = pd.read_csv('./data/mammographicmasses1.txt', header = None)
        print data.shape
        data=numpy.array(data)
        print data
        #f = open('./data/x.data')
        X=[]
        for i,row in enumerate(data):
            #currentline = row.split()   
            temp=[]
            for column_value in row:
                temp.append(column_value)

            X.append(temp)

        
        X=np.array(X)
        print "Array:",X.shape
        self.X=X[:,:5]
        self.y=X[:,5]
        
        self.verileri_dokforest(self.X,self.tabloveriler)
                 
    def verileri_dokforest(self,X,tabloveriler):
        num_rows=len(X)

        tabloveriler.clear()    
        tabloveriler.setColumnCount(4)
        tabloveriler.setRowCount(num_rows) ##set number of rows

        for rowNumber,row in enumerate(X):
            #row[1].encode("utf-8")
            tabloveriler.setItem(rowNumber, 0, QtGui.QTableWidgetItem(str(row[0])))
            tabloveriler.setItem(rowNumber, 1, QtGui.QTableWidgetItem(str(row[1])))
            tabloveriler.setItem(rowNumber, 2, QtGui.QTableWidgetItem(str(row[2])))
            tabloveriler.setItem(rowNumber, 3, QtGui.QTableWidgetItem(str(row[3])))
            tabloveriler.setItem(rowNumber, 4, QtGui.QTableWidgetItem(str(row[4]))) 
            #tablo.setItem(rowNumber, 3, QtGui.QTableWidgetItem(str(row[5])))          
    def verileri_dok_knn(self,X,table_2):
        
        num_rows1=len(self.X)

        table_2.clear()    
        table_2.setColumnCount(1)
        table_2.setRowCount(num_rows1) ##set number of rows

        for rowNumber1,row1 in enumerate(X):
            #row[1].encode("utf-8")
            table_2.setItem(rowNumber1, 0, QtGui.QTableWidgetItem(str(row1[0:15])))
          
            
        
          
        
  
        
        
    def verileri_dok(self,X,tablo):
        
        num_rows=len(self.X)

        tablo.clear()    
        tablo.setColumnCount(4)
        tablo.setRowCount(num_rows) ##set number of rows

        for rowNumber,row in enumerate(X):
            #row[1].encode("utf-8")
            tablo.setItem(rowNumber, 0, QtGui.QTableWidgetItem(str(row[0])))
            tablo.setItem(rowNumber, 1, QtGui.QTableWidgetItem(str(row[1])))
            tablo.setItem(rowNumber, 2, QtGui.QTableWidgetItem(str(row[2])))
            tablo.setItem(rowNumber, 3, QtGui.QTableWidgetItem(str(row[3])))
            
   
        
   


         
    def verileri_dok_1(self,X,tablo1):
        
        num_rows=len(self.X)

        tablo1.clear()    
        tablo1.setColumnCount(4)
        tablo1.setRowCount(num_rows) ##set number of rows

        for rowNumber,row in enumerate(X):
            #row[1].encode("utf-8")
            tablo1.setItem(rowNumber, 0, QtGui.QTableWidgetItem(str(row[0])))
            tablo1.setItem(rowNumber, 1, QtGui.QTableWidgetItem(str(row[1])))
            tablo1.setItem(rowNumber, 2, QtGui.QTableWidgetItem(str(row[2])))
            tablo1.setItem(rowNumber, 3, QtGui.QTableWidgetItem(str(row[3])))
           
                       
    def train_and_test(self):

        X_train, X_test,y_train,y_test = train_test_split(self.X, self.y, test_size=0.30, random_state=42)        
        self.verileri_dok(X_train,self.table_train)#train icin
        self.verileri_dok(X_test,self.table_test)#test icin
        
        #self.X_train, self.X_test, self.y_train, self.y_test= X_train, X_test, y_train,y_test 
    def train_and_test_1(self):

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.30, random_state=42)        
        self.verileri_dok_1(X_train,self.table_train_1)#train icin
        self.verileri_dok_1(X_test,self.table_test_1)#test icin
             
    def train_and_test_forest(self):

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.30, random_state=42)        
        self.verileri_dokforest(X_train,self.trainveriler)#train icin
        self.verileri_dokforest(X_test,self.testveriler)#test icin
                 
   
   
   
        
    @QtCore.pyqtSignature("bool")
    def on_knn_clicked(self):
        ins = self.knnAlgoritma("./data/tendataset.txt", 17, 50300, 51000)
        ins.distance(1)
        ins.findClass()
        
        ins.grafik()  
        
        
        image="grafik.png"
        w,h=self.graphicsView.width()-5,self.graphicsView.height()-5
        self.graphicsView.setScene(self.show_image(image,w,h))
        
    @QtCore.pyqtSignature("bool")
    def on_btn_veriler_clicked(self):
        
        import numpy as np
        import pandas as pd
        from matplotlib import pyplot as plt
        plt.rcParams['figure.figsize'] = (16, 9)
        plt.style.use('ggplot')
        # Importing the dataset
        data1 =pd.read_csv('xclara.csv')
        print(data1.shape)
        data1.head()
        # Getting the values and plotting it
        self.f1 = data1['V1'].values
        self.f2 = data1['V2'].values
        self.X1 = np.array(list(zip(self.f1, self.f2)))
        plt.scatter(self.f1, self.f2, c='black', s=7)
            
        
        # Number of clusters
        self.k = 3
        # X coordinates of random centroids
        self.C_x = np.random.randint(0, np.max(self.X1)-20, size=self.k)
        # Y coordinates of random centroids
        self.C_y = np.random.randint(0, np.max(self.X1)-20, size=self.k)
        self.C = np.array(list(zip(self.C_x, self.C_y)), dtype=np.float32)
        self.label_4.setText("Merkezler" + str(self.C))
        # Plotting along with the Centroids
        print(self.C)
        plt.scatter(self.f1, self.f2, c='#050505', s=7)
        plt.scatter(self.C_x, self.C_y, marker='*', s=200, c='g')
        plt.savefig("kmeans_1.png") 
        image="kmeans_1.png"
        w,h=self.graphicsView_2.width()-5,self.graphicsView_2.height()-5
        self.graphicsView_2.setScene(self.show_image(image,w,h))
       
            
        
        
        
    @QtCore.pyqtSignature("bool")
    def on_btn_kmeans_clicked(self):
        from copy import deepcopy
        import numpy as np
        
        from matplotlib import pyplot as plt
       
        def dist(a, b, ax=1):
            return np.linalg.norm(a -b,axis=ax)
        C_old = np.zeros(self.C.shape)
      
        # Cluster Lables(0, 1, 2)
        clusters = np.zeros(len(self.X1))
        # Error func. - Distance between new centroids and old centroids
        error = dist(self.C, C_old, None)
        # Loop will run till the error becomes zero
        while error != 0:
            # Assigning each value to its closest cluster
        
            for i in range(len(self.X1)):
                distances = dist(self.X1[i], self.C)
                cluster = np.argmin(distances)
                clusters[i] = cluster
                    # Storing the old centroid values
            C_old = deepcopy(self.C)
             # Finding the new centroids by taking the average value
            for i in range(self.k):
                 points = [self.X1[j] for j in range(len(self.X1)) if clusters[j] == i]
                 self.C[i] = np.mean(points, axis=0)
            error =dist(self.C, C_old, None)
        colors = ['r', 'g', 'b', 'y', 'c', 'm']
        fig, ax = plt.subplots()
          
       
        for i in range(self.k):
            points = np.array([self.X1[j] for j in range(len(self.X1)) if clusters[j] == i])
            ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
        ax.scatter(self.C[:, 0], self.C[:, 1], marker='*', s=200, c='#050505')
        plt.savefig("kmeans_2.png")
        image="kmeans_2.png"
        w,h=self.graphicsView_3.width()-5,self.graphicsView_3.height()-5
        self.graphicsView_3.setScene(self.show_image(image,w,h))
       
        
        
        
        
    def show_image(self, img_name,width,height):
            pixMap = QtGui.QPixmap(img_name)
            pixMap=pixMap.scaled(width,height)
            pixItem = QtGui.QGraphicsPixmapItem(pixMap)
            scene2 = QGraphicsScene()
            scene2.addItem(pixItem)
            return scene2
        
  
    def veriNormalize(self):
        first_column=self.X[:,0]
        print first_column
        max_value=max(first_column)
        min_value=min(first_column)
        print "max value:",max_value," min value:",min_value
        num_rows=len(self.X)
        for i,value in enumerate(first_column):
            normalize_value=value=(value-min_value)/(max_value-min_value)
            first_column[i]=round(normalize_value,2)
        self.table.clear()    
        self.table.setColumnCount(1)
        self.table.setRowCount(num_rows) ##set number of rows
        
        for rowNumber,row in enumerate(first_column):
            #row[1].encode("utf-8")
            self.table.setItem(rowNumber, 0, QtGui.QTableWidgetItem(str(row)))
       

    def verileriYukle(self):    


        X,y=func_X_yukle()
        self.X=X
        self.y=y
        num_rows=len(X)
        self.table.setColumnCount(4)
        self.table.setRowCount(num_rows+1) ##set number of rows
    

        self.verileri_dok(X,self.table)#ham veri
        
    def verileriYukle_1(self):    


        X,y=func_X_yukle()
        self.X=X
        self.y=y
        num_rows=len(X)
        self.table_1.setColumnCount(4)
        self.table_1.setRowCount(num_rows+1) ##set number of rows
    

        self.verileri_dok_1(X,self.table_1)#ham veri
        
        
        
        
#####################################dengeelenmekkk######################################        
    @QtCore.pyqtSignature("bool")
    def on_btndengelenmemis_clicked(self):
        
            f = open('./data/buyukdataset.txt')
            self.X=[]
            for i,row in enumerate(f.readlines()):
                currentline = row.split(",")
                temp=[]
                for column_value in currentline:
                    temp.append(column_value)
                    
                self.X.append(temp)
            self.X=np.array(self.X)
            print "Array:",self.X.shape    
             
            print self.X    
            f = open('./data/kucukdataset.txt')
            self.Y=[]
            for i,row in enumerate(f.readlines()):
                currentline = row.split(",")
                temp=[]
                for column_value in currentline:
                    temp.append(column_value)
                    
                self.Y.append(temp)
            self.Y=np.array(self.Y)
            print "Array:",self.Y.shape    
            print self.Y 
            colors=["c.","y.","b.","r.","g."]
            print "veriler"
            fig=plt.figure()
            for i in range (len(self.X)):    
                plt.plot(self.X[i][0],self.X[i][1],colors[1], markersize=9, marker=".")
            for i in range (len(self.Y)):
                plt.plot(self.Y[i][0],self.Y[i][1],colors[0], markersize=9, marker="*")
            plt.show()
            fig.savefig("dengelenmemis.png")
            image="dengelenmemis.png"
            w,h=self.graphicsView_4.width()-5,self.graphicsView_4.height()-5
            self.graphicsView_4.setScene(self.show_image(image,w,h))
         
    @QtCore.pyqtSignature("bool")
    def on_btnros_clicked(self):  
        ros=[]
        randomize=[]
        randomizesayilar=[]
        rus=[]
        ROSY=[]
        rosuzunluk=[]
        
        if self.X.shape<self.Y.shape:
            colors=["c.","y.","b.","r.","g."]
            randomizesayilar=random.sample(self.Y,(len(self.Y)-len(self.X)))
            for i in randomizesayilar:
                randomize.append(i)
            print randomize   
            #plt.plot(ros[i][0],ros[i][1],colors[1],markersize=9,marker=".") 
            P=np.array(randomize)
            print "randomizesayilaruzunluğu:",P.shape 
            for n in self.X:
                ROSY.append(n)
            #print "Ydatasetveriler",ROSY
            for m in randomize:
                ROSY.append(m)
            print "ros veriler",ROSY  
            rosuzunluk=np.array(ROSY)
            print "toplamyArray:",rosuzunluk.shape 
            ros=np.array(rosuzunluk)
            print "Xrossayilaruzunluğu:",ros.shape 
            #ros=random.sample(X,len(ROSY)) 
            print "X",self.Y
            xuzunluk=np.array(self.Y)
            print "Xuzunluk:",self.Y.shape
            print" ros"
            fig=plt.figure()
            for i in range(len(ros)):
                plt.plot(ros[i][0],ros[i][1],colors[1],markersize=10,marker=".")
        
            for i in range(len(self.Y)):
                plt.plot(self.Y[i][0],self.Y[i][1],colors[0],markersize=5,marker="*")
         
            plt.show()
            fig.savefig("ros.png")
            image2="ros.png"
            w,h=self.ros.width()-5,self.ros.height()-5
            self.ros.setScene(self.show_image(image2,w,h))
            
        
        if self.X.shape>self.Y.shape:
            colors=["c.","y.","b.","r.","g."]
            randomizesayilar=random.sample(self.X,(len(self.X)-len(self.Y)))
            for i in randomizesayilar:
                randomize.append(i)
            print randomize   
            #plt.plot(ros[i][0],ros[i][1],colors[1],markersize=9,marker=".") 
        
            P=np.array(randomize)
            print "randomizesayilaruzunluğu:",P.shape 
            for n in self.Y:
                ROSY.append(n)
            #print "Ydatasetveriler",ROSY
          
            for m in randomize:
                ROSY.append(m)
            print "ros veriler",ROSY       
            rosuzunluk=np.array(ROSY)
            print "toplamyArray:",rosuzunluk.shape 
            ros=np.array(rosuzunluk)
            print "rossayilaruzunluğu:",ros.shape 
           # ros=random.sample(X,len(ROSY)) 
            print "X",self.X
            xuzunluk=np.array(self.X)
            print "Xuzunluk:",X.shape
            print"rossss"
            for i in range(len(ros)):
                plt.plot(ros[i][0],ros[i][1],colors[1],markersize=10,marker=".")
        
            for i in range(len(self.X)):
                plt.plot(self.X[i][0],self.X[i][1],colors[0],markersize=5,marker="*")
         
            plt.show()
            fig.savefig("ros.png")
            image3="ros.png"
            w,h=self.ros.width()-5,self.ros.height()-5
            self.ros.setScene(self.show_image(image3,w,h))
            
        
        
        
        
        
        
        
    @QtCore.pyqtSignature("bool")
    def on_pushButton_5_clicked(self):  
        colors=["c.","y.","b.","r.","g."]
        if self.X.shape>self.Y.shape:
                print "rus"
                rus=random.sample(self.X,len(self.Y)) 
                fig=plt.figure()
                for i in range(len(rus)):
                    plt.plot(rus[i][0],rus[i][1],colors[1],markersize=5,marker=".")
                for i in range(len(Y)):
                    plt.plot(self.Y[i][0],self.Y[i][1],colors[0],markersize=5,marker="*")
                plt.show() 
                fig.savefig("rus.png")
                image1="rus.png"
                w,h=self.rus.width()-5,self.rus.height()-5
                self.rus.setScene(self.show_image(image1,w,h))
        if self.X.shape<self.Y.shape:
            print "russs"
            fig=plt.figure()
            rus=random.sample(self.Y,len(self.X)) 
            for i in range(len(rus)):
                plt.plot(rus[i][0],rus[i][1],colors[1],markersize=10,marker=".")
                
            for i in range(len(self.X)):
                plt.plot(self.X[i][0],self.X[i][1],colors[0],markersize=10,marker="*")
            plt.show()   
            fig.savefig("rus.png")
            image1="rus.png"
            w,h=self.rus.width()-5,self.rus.height()-5
            self.rus.setScene(self.show_image(image1,w,h))  
     
#########################################traintest#############################
    def dosyaYukletrain(self):
       
            f = open('./data/x.data')
            X=[]
            
            for i,row in enumerate(f.readlines()):
                
                currentline = row.split(",")   
                temp=[]
                for column_value in currentline:
                    temp.append(column_value)
    
                X.append(temp)
                
            for row in X:
                print "Data:",row
            
            X=np.array(X)
            print "Array:",X.shape
            self.X=X[:,:4]
            self.y=X[:,5]
            self.verileri_doktrain(self.X,self.tableWidget)
            
            
       
         
    def verileri_doktrain(self,X,tableWidget):
        num_rows=len(self.X)
        tableWidget.clear()    
        tableWidget.setColumnCount(4)
        tableWidget.setRowCount(num_rows) ##set number of rows

        for rowNumber,row in enumerate(X):
            #row[1].encode("utf-8")
            tableWidget.setItem(rowNumber, 0, QtGui.QTableWidgetItem(str(row[0])))
            tableWidget.setItem(rowNumber, 1, QtGui.QTableWidgetItem(str(row[1])))
            tableWidget.setItem(rowNumber, 2, QtGui.QTableWidgetItem(str(row[2])))
            tableWidget.setItem(rowNumber, 3, QtGui.QTableWidgetItem(str(row[3])))    
        
    def train_and_test_yuzde(self):
        yuzde=float(self.lineEdit_4.text())

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=(yuzde/100), random_state=42)        
        self.verileri_doktrain(X_train,self.normaltrain)#train icin
        self.verileri_doktrain(X_test,self.normaltest)#test icin
        
    def verileri_dokyuzde(self,X,normaltrain):
        num_rows=len(X)

        normaltrain.clear()    
        normaltrain.setColumnCount(4)
        normaltrain.setRowCount(num_rows) ##set number of rows

        for rowNumber,row in enumerate(X):
            #row[1].encode("utf-8")
            tabloveriler.setItem(rowNumber, 0, QtGui.QTableWidgetItem(str(row[0])))
            tabloveriler.setItem(rowNumber, 1, QtGui.QTableWidgetItem(str(row[1])))
            tabloveriler.setItem(rowNumber, 2, QtGui.QTableWidgetItem(str(row[2])))
            tabloveriler.setItem(rowNumber, 3, QtGui.QTableWidgetItem(str(row[3])))
            tabloveriler.setItem(rowNumber, 4, QtGui.QTableWidgetItem(str(row[4])))   
            
   #--------------------------------normalize----------------------
      
  
        
             
    def dosyaYuklenormalize(self):
        
        f = open('./data/diabetes.data')
        X=[]
        for i,row in enumerate(f.readlines()):   
            currentline = row.split(",")   
            temp=[]
            for column_value in currentline:
                temp.append(column_value)
            X.append(temp)
        
        X=np.array(X)
        print "Array:",X.shape
        self.X=X[:,:8]
        self.y=X[:,8]
        self.veriYukle(self.X,self.y,self.tablonormalize) 
        
            
    def veriYukle(self,X,y,tablonormalize):
        num_rows=len(X)
        tablonormalize.clear()    
        tablonormalize.setColumnCount(8)
        tablonormalize.setRowCount(num_rows) ##set number of rows    
        for rowNumber,row in enumerate(X):
            #row[1].encode("utf-8")
            tablonormalize.setItem(rowNumber, 0, QtGui.QTableWidgetItem(str(row[0])))
            tablonormalize.setItem(rowNumber, 1, QtGui.QTableWidgetItem(str(row[1])))
            tablonormalize.setItem(rowNumber, 2, QtGui.QTableWidgetItem(str(row[2])))
            tablonormalize.setItem(rowNumber, 3, QtGui.QTableWidgetItem(str(row[3])))
            tablonormalize.setItem(rowNumber, 4, QtGui.QTableWidgetItem(str(row[4])))
            tablonormalize.setItem(rowNumber, 5, QtGui.QTableWidgetItem(str(row[5])))
            tablonormalize.setItem(rowNumber, 6, QtGui.QTableWidgetItem(str(row[6])))
            tablonormalize.setItem(rowNumber, 7, QtGui.QTableWidgetItem(str(row[7])))
        for rowNumber,row in enumerate(y):
            tablonormalize.setItem(rowNumber, 8, QtGui.QTableWidgetItem(str(row)))
    def sec(self):
        from PIL import Image
        if(self.normalizetest.currentIndex()==0):
           # def normalize_MinMax(self)
                
                self.tablonormalize.clear()         
                for s in range(0,8):
                    first_column=self.X[:,s]
                    max_value=float(max(first_column))
                    min_value=float(min(first_column))
                    print "max value:",max_value," min value:",min_value
                    num_rows=len(self.X)
                    for i,value in enumerate(first_column):
                        normalize_value=((float(value)-min_value)/(max_value-min_value))
                        first_column[i]=round(normalize_value,2)   
                    self.tablonormalize.setColumnCount(8)
                    self.tablonormalize.setRowCount(num_rows) ##set number of rows
                    
                    for rowNumber,row in enumerate(first_column):
                        #row[1].encode("utf-8")
                        self.tablonormalize.setItem(rowNumber, s, QtGui.QTableWidgetItem(str(row)))
        if(self.normalizetest.currentIndex()==1):
               #def normalize_Z_Score(self)
                self.tablonormalize.clear()
                for s in range(0,8):
                    colm= np.array(self.X[:,s]).astype(np.float)
                    ui=np.mean(colm)
                    ai=np.std(colm)
                    print"Aritmetik ortalama:",ui,"  standart sapma:",ai
                    num_rows=len(self.X)
                    for i,value in enumerate(colm):
                        normalize_zscor=float(value)-ui/ai
                        colm[i]=float(round(normalize_zscor,3))
                    self.tablonormalize.setColumnCount(8)
                    self.tablonormalize.setRowCount(num_rows) ##set number of rows
                    
                    for rowNumber,row in enumerate(colm):
                        #row[1].encode("utf-8")
                        self.tablonormalize.setItem(rowNumber, s, QtGui.QTableWidgetItem(str(row)))
        if(self.normalizetest.currentIndex()==2):         
             #   def normalizeMedian(self):
                self.tablonormalize.clear()
                for s in range(0,8):
                    column=np.array(self.X[:,s]).astype(np.float)
                    med=np.median(column)
                    print "Medyan: ",med
                    num_rows=len(self.X)
                    for i,value in enumerate(column):
                        normalize_medyan=float(value)/med
                        column[i]=float(round(normalize_medyan,3))
                    self.tablonormalize.setColumnCount(8)
                    self.tablonormalize.setRowCount(num_rows) ##set number of rows
                    
                    for rowNumber,row in enumerate(column):
                        #row[1].encode("utf-8")
                        self.tablonormalize.setItem(rowNumber, s, QtGui.QTableWidgetItem(str(row)))
 #############################################naviebayes##################
            
    @QtCore.pyqtSignature("bool")
    def on_naviebayes_clicked(self):   
        kelime=str(self.lineEdit_3.text())
        def aranacak(kume, kelime, index):
            counter=0
            for i in range(len(kume)):
                if kume[i][index]==kelime:
                    counter+=1
            return counter
    
        def aranacak_2(kume, kelime):
            counter=0
            for i in range(len(kume)):
                if kume[i]==kelime:
                    counter+=1
            return counter
    
        def kelimeler(kume):
            kelimeler=[]
            silinecek="!@#$.?,"
            for i in range(len(kume)):
                cumle=kume[i][0]
                for char in silinecek:
                    cumle=cumle.replace(char,"")
                parca=cumle.split(' ')
                for c in parca:
                    if aranacak_2(kelimeler, c)==0:
                        kelimeler.append(c)
            return kelimeler
    
        def arama(kume,kumeci,kelime):
            counter=0
            for i in range(len(kume)):
                if kume[i][1]==kumeci and kume[i][0].count(kelime)>0:
                    counter+=kume[i][0].count(kelime)
            return counter
            
        data=[["mac, corner, aerobik, antrenman.","spor"],
            ["saha futbol fitness voleybol basketbol.","spor"],
            ["penalti, ofsayt,sut,tac, masa tenisi","spor"],
            ["ceza sahasi ,kale, top.","spor"],
            ["enflasyon, deflasyon, komisyon, sermaye, endeks","ekonomi"],
            ["lira, kar, zarar, altin, faiz, hisse","ekonomi"],
            ["bonus , piyasa, euro, tl, para, hesap","ekonomi"],
            ["finans, dolar, gelir","ekonomi"]]

        countspor=aranacak(data,"spor",1)
        countekonom=aranacak(data,"ekonomi",1)
        print("ekonomi adet:"+str(countekonom)+" spor Adet:"+str(countspor))
        sporagirlik=float(countspor)/(float(countspor)+float(countekonom))
        ekonomagirlik=float(countekonom)/(float(countspor)+float(countekonom))
        print("spor Ağırlık:"+str(sporagirlik)+" ekonomi Ağırlık:"+str(ekonomagirlik))
        kelimeci=kelimeler(data)
        print(kelimeler(data))
        sportoplam=0
        spordeger=[]
        for i in kelimeci:
            sportoplam+=(arama(data,"spor",i)+1)

        for i in range(len(kelimeci)):
            deger=float(arama(data,"spor",kelimeci[i])+1)/float(sportoplam)
            spordeger.append(deger)
            print(str(kelimeci[i])+" için "+str(deger))
    
        ekonomtoplam=0
        ekonomdeger=[]
        for i in kelimeci:
            ekonomtoplam+=(arama(data,"ekonomi",i)+1)
        for i in range(len(kelimeci)):
            deger=float(arama(data,"ekonomi",kelimeci[i])+1)/float(ekonomtoplam)
            ekonomdeger.append(deger)
            print(str(kelimeci[i])+" için "+str(deger))
        c_kelime=kelime.split(" ")
        print(c_kelime)
        sporcarpim=1
        for i in c_kelime:
            for x in range(len(kelimeci)):
                if kelimeci[x]==i:
                    sporcarpim*=spordeger[x]
        ekonomcarpim=1
        for i in c_kelime:
            for x in range(len(kelimeci)):
                if kelimeci[x]==i:
                    ekonomcarpim*=ekonomdeger[x]
        sporsonuc=sporcarpim*sporagirlik
        ekonomsonuc=ekonomcarpim*ekonomagirlik
        print("spor cümle oran:"+str(sporcarpim)+" Ağırlık*Oran:"+str(sporsonuc))
        print("ekonomi cümle oran:"+str(ekonomcarpim)+" Ağırlık*Oran:"+str(ekonomsonuc))
        
        if sporsonuc<ekonomsonuc:
            print("Kelime ekonomi")
            self.nb_label.setText("Kelime Ekonomi")
        if sporsonuc>ekonomsonuc:
            print("Kelime spor")
            self.nb_label.setText("Kelime spor")
            
        if sporcarpim==1 and ekonomcarpim==1:
            self.nb_label.setText("Kelime yok")                  
                        
###################################parkinsonnnnnnn#########################################################
    @QtCore.pyqtSignature("bool") 
    def on_btnhw_veriler_clicked(self):
                path="./hw_dataset/control/"  
                path1="./hw_dataset/parkinson/"   
                path2="./new_dataset/parkinson/"   
                SST_data=[]      
                SSTy=[] 
                DST_data=[]        
                DSTy=[]     
                STCP_data=[]     
                STCPy=[] 
                SSTp=[]      
                SSTpy=[] 
                DSTp=[]        
                DSTpy=[]
                newdata=[]    
                STCPp=[]     
                STCPpy=[] 
                
                SST_train=[]
                DST_train=[]
                STCP_train=[]
                dosyalar=os.listdir(path)
                for dosya in dosyalar:
                    f = open("./hw_dataset/control/"+dosya)        
                    for i,row in enumerate(f.readlines()):
                        currentline = row.split(";")   
                        temp=[]
                        for column_value in currentline:
                            temp.append(column_value)
                        if(int(temp[len(temp)-1])==0):
                            temp.remove(temp[6])   
                            temp.append("0")
                            SST_data.append(temp)
                        elif(int(temp[len(temp)-1])==1):
                            temp.remove(temp[6])
                            temp.append("0")
                            DST_data.append(temp)
                        elif(int(temp[len(temp)-1])==2):
                            temp.remove(temp[6])
                            temp.append("0")
                            STCP_data.append(temp)   
                #""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                dosyalar1=os.listdir(path1)
                for dosya1 in dosyalar1:
                    f = open("./hw_dataset/parkinson/"+dosya1)        
                    for i,row in enumerate(f.readlines()):
                        currentline = row.split(";")   
                        tempp=[]
                        for column_value in currentline:
                            tempp.append(column_value)
                        if(int(tempp[len(tempp)-1])==0):
                            tempp.remove(tempp[6])
                            tempp.append("1")
                            SSTp.append(tempp)
                        elif(int(tempp[len(tempp)-1])==1):
                            tempp.remove(tempp[6])
                            tempp.append("1")
                            DSTp.append(tempp)
                        elif(int(tempp[len(tempp)-1])==2):
                            tempp.remove(tempp[6])
                            tempp.append("1")
                            STCPp.append(tempp)   
                SST_train=[]
                SST_test=[]
                DST_test=[]
                STCP_test=[]
                self.HW_DATASETLER=[]
                
                SST_data.extend(SSTp)  
                DST_data.extend(DSTp)  
                STCP_data.extend(STCPp)
                self.HW_DATASETLER.extend(SST_data)
                self.HW_DATASETLER.extend(DST_data)
                self.HW_DATASETLER.extend(STCP_data)
                print self.HW_DATASETLER
            
                for row2 in self.HW_DATASETLER:
                    
                   # print "veriler:",row2
        
                    #self.HW_DATASETLER=np.array(self.HW_DATASETLER)
                    #print "Array:",self.HW_DATASETLER.shape 
                    self.verileri_dok_2(self.HW_DATASETLER,self.tableWidget_2)
                for i  in SST_data:
                    SST_test.append(i[6]) 
                for n  in SST_data:
                    SST_train.append(n[0:6])
                for m  in DST_data:
                    DST_test.append(m[6])
                   
                for t  in DST_data:
                    DST_train.append(t[0:6])
                   
                for k in STCP_data:
                    STCP_test.append(k[6])
                    
                for J  in STCP_data:
                    STCP_train.append(J[0:6])
                    
                sst_train=np.array(SST_train)
                print "sst_train:",sst_train.shape
                sst_test=np.array(SST_test)
                print "sst_test:",sst_test.shape 
                dst_train=np.array(DST_train)
                print "dst_train:",dst_train.shape  
                stcp_train=np.array(STCP_train)
                print "STCP_train:",stcp_train.shape            
                stcp_test=np.array(STCP_test)
                print "STCP_train:",stcp_test.shape  
                ##-------------------nw_dataset------------------------------------------------------
                SST_new_test=[]
                DST_new_test=[]
                STCP_new_test=[]
                SST_new_train=[]
                DST_new_train=[]
                STCP_new_train=[]
                SST_new=[]
                DST_new=[]
                STCP_new=[]
                dosyalar2=os.listdir(path2)
                for dosya2 in dosyalar2:
                    f = open("./new_dataset/parkinson/"+dosya2)        
                    for i,row in enumerate(f.readlines()):
                        currentline = row.split(";")   
                        temp2=[]
                        for column_value in currentline:
                            temp2.append(column_value)
                        if(int(temp2[len(temp2)-1])==0):
                            temp2.remove(temp2[6])
                            temp2.append("1")
                            SST_new.append(temp2)
                        elif(int(temp2[len(temp2)-1])==1):
                            temp2.remove(temp2[6])
                            temp2.append("1")
                            DST_new.append(temp2)
                        elif(int(temp2[len(temp)-1])==2):
                            temp2.remove(temp2[6])
                            temp2.append("1")
                            STCP_new.append(temp2)  
                for sst_t in SST_new:
                    SST_new_test.append(sst_t[6])   
                for sstt  in SST_new:
                    SST_new_train.append(sstt[0:6])    
                for dst_t in DST_new:
                    DST_new_test.append(dst_t[6])   
                for dstt  in DST_new:
                    DST_new_train.append(dstt[0:6])    
                for stcp_t in STCP_new:
                    STCP_new_test.append(stcp_t[6])    
                for stcp in STCP_new:
                    STCP_new_train.append(stcp[0:6])     
                #-------------------------boyut----------------------------------------------    
                    
                Xtrain_SST,X_test_SST,yTrain_SST,y_test_SST= train_test_split(SST_train,SST_test,test_size=0.30,random_state=0)
                Xtrain_SST1,X_test_SST1,yTrain_SST1,y_test_SST1= train_test_split(SST_new_train,SST_new_test,test_size=0.30,random_state=0)
                clf = RandomForestClassifier(max_depth=None, random_state=0)
                clf.fit(Xtrain_SST,yTrain_SST)
                results=clf.predict(Xtrain_SST1)
                print ( "SST Random Forest Başarısı:",accuracy_score(yTrain_SST1,results))
                self.label_7.setText("Random Forest Başarı Basari:" + str(round(accuracy_score(yTrain_SST1, results)*100,2)))
                cm=confusion_matrix(yTrain_SST1, results)
                matrix= np.array(cm)
                print matrix
                
                 #SST  BAŞARIM......
                knn = KNeighborsClassifier(n_neighbors=5)
                knn.fit(Xtrain_SST, yTrain_SST)
                y_pred = knn.predict(Xtrain_SST1)
                print ("SST KNeighborsClassifier Basarisi:",metrics.accuracy_score(yTrain_SST1, y_pred)*100)
                self.label_8.setText("Random Forest Başarı Basari:" + str(round(accuracy_score(yTrain_SST1, results)*100,2)))
                
                #----------------------------------------------------------
                
                Xtrain_DST,X_test_DST,yTrain_DST,y_test_DST= train_test_split(DST_train,DST_test,test_size=0.30,random_state=0)
                Xtrain_DST1,X_test_DST1,yTrain_DST1,y_test_DST1= train_test_split(DST_new_train,DST_new_test,test_size=0.30,random_state=0)
                clf = RandomForestClassifier(max_depth=None, random_state=0)
                clf.fit(Xtrain_DST,yTrain_DST)
                results=clf.predict(Xtrain_DST1)
                print ( "DST Random Forest Başarısı:",round(accuracy_score(yTrain_DST1,results)*100,2))
                print ( "DST confusion_matrix:",confusion_matrix(yTrain_DST1,results))
                self.label_9.setText("Random Forest Başarı Basari:" + str(round(accuracy_score(yTrain_DST1, results)*100,2)))
                
                #DST ALGORİTMA BAŞARISI
                knn = KNeighborsClassifier(n_neighbors=5)
                knn.fit(Xtrain_DST, yTrain_DST)
                y_pred = knn.predict(Xtrain_DST1)
                print ("DST KNeighborsClassifier Basarisi",metrics.accuracy_score(yTrain_DST1, y_pred)*100)
                self.label_10.setText("Random Forest Başarı Basari:" + str(round(accuracy_score(yTrain_DST1, results)*100,2)))
                #-------------------------------------------------
                Xtrain_STCP,X_test_STCP,yTrain_STCP,y_test_STCP= train_test_split(STCP_train,STCP_test,test_size=0.30,random_state=0)
                Xtrain_STCP1,X_test_STCP1,yTrain_STCP1,y_test_STCP1= train_test_split(STCP_new_train,STCP_new_test,test_size=0.30,random_state=0)
                #STCP...........
                clf = RandomForestClassifier(max_depth=None, random_state=0)
                clf.fit(Xtrain_STCP,yTrain_STCP)
                results=clf.predict(Xtrain_STCP1)
                print ( "u/STCP Random Forest Başarısı:",accuracy_score(yTrain_STCP1,results))
                self.label_11.setText("Random Forest Başarı Basari:" + str(round(accuracy_score(yTrain_STCP1, results)*100,2)))
                print ( "u/STCP confusion_matrix:",confusion_matrix(yTrain_STCP1,results)) 
                #STCP SVM ALG BAŞARIMI
                knn = KNeighborsClassifier(n_neighbors=5)
                knn.fit(Xtrain_STCP,yTrain_STCP)
                y_pred = knn.predict(Xtrain_STCP1)
                print ("STCP KNeighborsClassifier Basarisi",metrics.accuracy_score(yTrain_STCP1, y_pred)*100)
                self.label_12.setText("Random Forest Başarı Basari:" + str(round(accuracy_score(yTrain_STCP1, results)*100,2)))
                #-------------------------------------------------------
                X_data=[]
                Y_data=[]
                
                for line1 in (SST_train+DST_train+STCP_train+SST_new_train+DST_new_train+STCP_new_train):
                    X_data.append(line1)
                    
                
                X=np.array(X_data)
                print "train data:",X.shape 
                for line2 in (SST_test+DST_test+STCP_test+SST_new_test+DST_new_test+STCP_new_test):
                    Y_data.append(line2)
                    
                X_train, X_test,y_train,y_test = train_test_split(X_data,Y_data,test_size=0.30,random_state=0)
                clf = RandomForestClassifier(max_depth=None, random_state=0)
                clf.fit(X_train, y_train)
                results=clf.predict(X_test)
                #print (results)
                cm=confusion_matrix(y_test, results)
                matrix= np.array(cm)
                print matrix
                print ( "u/Random Forest Başarısıbutun:",accuracy_score(y_test, results))
                self.label_14.setText("Random Forest Başarı Basari:" + str(round(accuracy_score(y_test, results)*100,2)))
                from sklearn.neighbors import KNeighborsClassifier
                from sklearn import metrics
                knn = KNeighborsClassifier(n_neighbors=3)
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)
                print ("KNeighborsClassifier Basarisi",metrics.accuracy_score(y_test, y_pred)*100)
                self.label_14.setText("KNeighborsClassifier Forest Başarı Basari:" + str(round(accuracy_score(y_test, results)*100,2)))
    def verileri_dok_2(self,HW_DATASETLER,tableWidget_2):
                    
                num_rows2=len(self.HW_DATASETLER)
        
                self.tableWidget_2.clear()    
                self.tableWidget_2.setColumnCount(2)
                self.tableWidget_2.setRowCount(num_rows2) ##set number of rows
                
        
                for rowNumber2,row2 in enumerate(self.HW_DATASETLER):
                    
                    #row[1].encode("utf-8")
                    self.tableWidget_2.setItem(rowNumber2, 0, QtGui.QTableWidgetItem(str(row2[0:50])))
                    
                    
                    
                    
                    
                    
                    
        




                        
    
    class knnAlgoritma():
       
       
 
        def __init__(self, dataset, k, nfrom, nto):
            self.b, self.g, self.r, self.class_attr = [], [], [], []
            self.ucboyutlu = [264,124,183]
            self.k = 3
            with open(dataset, "r") as f:
                for i in f.readlines()[nfrom:nto]:
                    self.r.append(int(i.split()[0]))
                    self.g.append(int(i.split()[1]))
                    self.b.append(int(i.split()[2]))
                    self.class_attr.append(i.split()[3])
        def distance(self, dist=1):
            self.dist = []
            for i in range(len(self.class_attr)):
                self.dist.append((pow((pow((
                abs(int(self.b[i]) - int(self.ucboyutlu[0])) +
                abs(int(self.g[i]) - int(self.ucboyutlu[1])) +
                abs(int(self.r[i]) - int(self.ucboyutlu[2]))), 2)), 1/dist), i))
            return self.dist
        def findClass(self):
           self.class_values = []
           self.result = ""
           for i in sorted(self.dist)[:self.k]:
               self.class_values.append(self.class_attr[i[1]])
           self.birinci = self.class_values.count("1")
           self.ikinci = self.class_values.count("2")
           print("Birinci Sınıf:", self.birinci)
           print("İkinci Sınıf:", self.ikinci)
           
          
          
          
          
           if self.birinci > self.ikinci:
               self.result = "1. Sinif"
           else:
               self.result = "2. Sinif"
           print("SONUÇ: "+self.result)
           
           
         
        def grafik(self):
     
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for bi, gj, rk, class_attr in zip(self.b, self.g, self.r, self.class_attr):
                if class_attr == "1":
                    ax.scatter(bi,gj,rk, c='r', marker='.')
                else:
                    ax.scatter(bi,gj,rk, c='g', marker='.')
            ax.scatter(int(self.ucboyutlu[0]), int(self.ucboyutlu[1]), int(self.ucboyutlu[2]), c='b')
            ax.set_xlabel('X Ekseni')
            ax.set_ylabel('Y Ekseni')
            ax.set_zlabel('Z Ekseni')
            
            fig.text(0, 0,"Birinci Sınıf: : " + str(self.birinci) +
                " -- İkinci Sınıf:: " + str(self.ikinci) +
                " -- {{Sonuc : " + self.result + "}}")
                                                                      
            plt.legend()
            plt.show()
            fig.savefig("grafik.png") 
            
            
          
                
             

        
        
        
        

        
        
        
        
        


          
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     