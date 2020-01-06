#-*- coding:utf-8 -*-

import sys
import os

from PyQt5.QtWidgets import *
from PyQt5 import uic

import h5py
import datetime
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
import matplotlib.cm as cm

import pandas as pd
from pyhdf.SD import SD, SDC
import dplython
import conda

conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
os.environ["PROJ_LIB"] = proj_lib

from mpl_toolkits.basemap import Basemap
from dplython import (DplyFrame, X, diamonds, select, sift,
  sample_n, sample_frac, head, arrange, mutate, group_by,
  summarize, DelayFunction)
import numpy as np

import csv

path = '/Library/Fonts/NanumBarunGothic.ttf'
font_name = fm.FontProperties(fname=path, size=50).get_name()
plt.rc('font', family=font_name)

form_class = uic.loadUiType("ui.ui")[0]


class WindowClass(QMainWindow, form_class):
    dirName = ''
    dataFilePath = ''

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.pushButton.clicked.connect(self.pushButtonFunction)
        self.fileListWidget.itemClicked.connect(self.chkItemClicked)
        self.text = self.processingBrowser

        df = pd.read_csv("bridge.csv", converters={"name":str, "lat":float, "lon":float})
        self.name = df['name'].tolist()
        self.lats = df['lat'].tolist()
        self.lons = df['lon'].tolist()

        self.figure = Figure(figsize=(50, 50), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout = self.formLayout
        layout.addWidget(self.canvas)
        layout.addWidget(self.toolbar)

    def pushButtonFunction(self):
        self.fileListWidget.clear()
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        folderUrl = QFileDialog.getExistingDirectory(self, "폴더 선택", "", options=options)
        self.dirName = folderUrl
        if folderUrl:
            fileList = os.listdir(folderUrl)
            fileList.sort()
            for f in fileList:
                self.fileListWidget.addItem(f)

    def chkItemClicked(self):
        self.fileInfoWidget.clear()
        idx = self.fileListWidget.currentRow()
        selectedFile = self.fileListWidget.selectedItems()
        if selectedFile:
            filePath = self.dirName + "/" + selectedFile[0].text()
            WindowClass.dataFilePath = filePath
            fName, ext = os.path.splitext(filePath)
            if ext == '.h5':
                comsDataTypeName = selectedFile[0].text().split('_')[3]
                comsDataTypeName = comsDataTypeName.upper()
                comsDataTypeDate = selectedFile[0].text().split('_')[5].split('.')[0]
                comsDataTypeDate = datetime.datetime.strptime(comsDataTypeDate, '%Y%m%d%H%M')
                file = h5py.File(filePath, 'r')
                dataKeys = list(file['Product/'].keys())
                dataTypeName = 1
                for i in dataKeys:
                    tempList = i.split('_')  # [Land,Surface,Temperature]
                    strInitial = ''
                    for tempListElement in tempList:
                        strInitial = strInitial + tempListElement[0]  # LST
                    if strInitial == comsDataTypeName:
                        dataTypeName = i
                        continue
                    else:
                        print('retry')
                print(dataTypeName)
                dataItem = list(file[f'Product/{dataTypeName}'].attrs.items())
                dataType = dataItem[0][1]
                maxVal = dataItem[1][1]
                minVal = dataItem[2][1]
                numOfCol = dataItem[4][1]
                numOfRow = dataItem[5][1]
                offsetVal = dataItem[6][1]
                scalingFactor = dataItem[7][1]
                unitName = dataItem[9][1]

                self.fileInfoWidget.addItem(f'데이터종류 : {comsDataTypeName}({dataTypeName})')
                self.fileInfoWidget.addItem(f'생산시각 : {comsDataTypeDate}')
                self.fileInfoWidget.addItem(f'데이터타입 : {dataType}')
                self.fileInfoWidget.addItem(f'최대값 : {maxVal}')
                self.fileInfoWidget.addItem(f'최소값 : {minVal}')
                self.fileInfoWidget.addItem(f'열 : {numOfCol}')
                self.fileInfoWidget.addItem(f'행 : {numOfRow}')
                self.fileInfoWidget.addItem(f'오프셋 : {offsetVal}')
                self.fileInfoWidget.addItem(f'스케일링 : {scalingFactor}')
                self.fileInfoWidget.addItem(f'단위 : {unitName}')

                hdffile = 'coms_cn_geos_lonlat.hdf'
                hdffile_sd = SD(hdffile, SDC.READ)

                lat_sds = hdffile_sd.select('Lat')  # select sds
                lon_sds = hdffile_sd.select('Lon')

                lat2D = lat_sds.get()  # get sds data
                lon2D = lon_sds.get()

                dn = file[f'Product/{dataTypeName}'][:]

                lon1D = np.reshape(lon2D, (1, np.product(lon2D.shape)))[0]
                lat1D = np.reshape(lat2D, (1, np.product(lat2D.shape)))[0]

                dn1D = np.reshape(dn, (1, np.product(dn.shape)))[0]

                data = pd.DataFrame(np.column_stack([lat1D, lon1D, dn1D]), columns=['lat', 'lon', 'dn'])
                data_L1 = (DplyFrame(data) >>
                           mutate(val=dplython.X.dn * float(scalingFactor) + float(offsetVal)) >>
                           sift((float(minVal) <= dplython.X.val) & (dplython.X.val <= float(maxVal))) >>
                           sift((-90 <= dplython.X.lat) & (dplython.X.lat <= 90)) >>
                           sift((-180 <= dplython.X.lon) & (dplython.X.lon <= 360))
                           )

                VAL = data_L1.val.values

                # create an canvas

                self.canvas.figure.clf()  # clear current plot
                self.ax = self.canvas.figure.add_subplot(111)

                m = Basemap(projection='cyl', llcrnrlon=122.937, llcrnrlat=32.502, urcrnrlon=131.968, urcrnrlat=43.469,
                            ax=self.ax)
                # m.drawcoastlines(color='black')
                # m.drawcountries(color='black')
                # m.drawmapboundary(fill_color='white')

                bridgeLats, bridgeLons = m(self.lats, self.lons)
                print(bridgeLats, bridgeLons)
                # m.colorbar(location='bottom', label='Land Surface Temperature', pad=0.5, ax=m)

                Lon, Lat = m(data_L1.lon.values, data_L1.lat.values)

                self.ax.scatter(Lon, Lat, c=VAL, s=1, marker="s", zorder=1, cmap=plt.cm.get_cmap('coolwarm'), alpha=0.7)
                self.ax.scatter(bridgeLons, bridgeLats, s=10, marker="o", c='red')

                self.ax.set_title(f'COMS_{comsDataTypeDate}_{dataTypeName}')
                self.ax.set_xticks([122.937, 131.968], minor=False)
                self.ax.set_yticks([32, 43])
                self.ax.set_xlabel('Latitude')
                self.ax.set_ylabel('Longitude')

                # refresh canvas
                self.canvas.draw()

            elif ext == '.csv':
                awsData = pd.read_csv(filePath, engine='python', encoding='euc-kr')
                awsDataHeader = list(awsData)
                print(len(awsDataHeader))
                headerStr = ''
                for i in awsDataHeader:
                    headerStr = headerStr + ' ' + i

                awsDataValueList = awsData.values
                stationName = awsDataValueList[:, 0][0]
                startDate = awsDataValueList[:, 1][0]
                endDate = awsDataValueList[:, 1][awsDataValueList[:, 1].size - 1]

                self.fileInfoWidget.addItem(f'포함데이터 : {headerStr}')
                self.fileInfoWidget.addItem(f'지점명 : {stationName}')
                self.fileInfoWidget.addItem(f'데이터시작날짜 : {startDate}')
                self.fileInfoWidget.addItem(f'데이터종료날짜 : {endDate}')

                dlg = popupDialog(awsDataHeader)
                dlg.exec_()
                axDict = dlg.checkboxDict
                # print(axDict)
                # print(len(axDict))

                self.canvas.figure.clf()  # clear current plot
                self.ax = self.canvas.figure.add_subplot(111)
                self.ax2 = self.ax.twinx()

                axColorList = [cm.rainbow(a) for a in np.linspace(0.0, 1.0, len(axDict))]

                # date : awsDataValueList[:, 1]
                for i, v in enumerate(awsDataHeader):
                    if axDict[i].isChecked():
                        self.ax.scatter(awsDataValueList[:, 1], awsDataValueList[:, i+2], s=1, marker="o", zorder=1, alpha=1.0, label=awsDataHeader[i+2], color=axColorList[i])
                        if i+2 >= len(awsDataHeader) - 1:
                            break
                    else:
                        self.ax2.scatter(awsDataValueList[:, 1], awsDataValueList[:, i+2], s=1, marker="o", zorder=1, alpha=1.0, label=awsDataHeader[i+2], color=axColorList[i])
                        if i+2 >= len(awsDataHeader) - 1:
                            break

                self.ax.set_title('AWS DATA')
                self.ax.set_xlabel('Date')
                # self.ax.set_ylabel('Longitude')
                self.ax.legend(loc='upper left')
                # refresh canvas
                self.ax2.legend(loc='upper right')

                self.canvas.draw()

            else:
                print('will be support')

        else:
            print('Error Occured')


class popupDialog(QDialog):

    def __init__(self, headerList):
        super().__init__()
        self.setupUI(headerList)

    def setupUI(self, headerList):
        self.setGeometry(800, 200, 300, 300)
        self.setWindowTitle("축, 보조축 설정")
        self.label = QLabel(self)

        # self.setWindowIcon(QIcon('icon.png'))

        self.checkboxDict = {}
        for i, v in enumerate(headerList):
            self.checkboxDict[i] = QCheckBox(f"{headerList[i+2]}", self)
            self.checkboxDict[i].move(10, 20 + (i * 30))
            self.checkboxDict[i].resize(150, 30)
            if i + 2 >= len(headerList) - 1:
                break
        labelMove = 20 + (i * 30)

        self.label.resize(300, 20)
        self.label.move(10, 30 + labelMove)
        self.label.setText("같은 그룹으로 묶고 싶은 요소에 체크하세요")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    app.exec_()
