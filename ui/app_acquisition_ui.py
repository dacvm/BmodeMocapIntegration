# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'app_acquisition.ui'
##
## Created by: Qt User Interface Compiler version 6.10.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QLayout, QLineEdit, QMainWindow,
    QMenuBar, QPushButton, QRadioButton, QSizePolicy,
    QStatusBar, QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(800, 1000)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.horizontalLayout_2 = QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.groupBox_1 = QGroupBox(self.centralwidget)
        self.groupBox_1.setObjectName(u"groupBox_1")
        self.verticalLayout_4 = QVBoxLayout(self.groupBox_1)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_4.setContentsMargins(5, 3, 5, 5)
        self.verticalLayout_bmode = QVBoxLayout()
        self.verticalLayout_bmode.setObjectName(u"verticalLayout_bmode")

        self.verticalLayout_4.addLayout(self.verticalLayout_bmode)


        self.verticalLayout.addWidget(self.groupBox_1)

        self.groupBox_2 = QGroupBox(self.centralwidget)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.verticalLayout_6 = QVBoxLayout(self.groupBox_2)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.verticalLayout_6.setContentsMargins(5, 3, 5, 5)
        self.verticalLayout_mocap = QVBoxLayout()
        self.verticalLayout_mocap.setObjectName(u"verticalLayout_mocap")

        self.verticalLayout_6.addLayout(self.verticalLayout_mocap)


        self.verticalLayout.addWidget(self.groupBox_2)

        self.groupBox_3 = QGroupBox(self.centralwidget)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.verticalLayout_8 = QVBoxLayout(self.groupBox_3)
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.verticalLayout_8.setContentsMargins(9, 4, 9, 9)
        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        self.label = QLabel(self.groupBox_3)
        self.label.setObjectName(u"label")

        self.gridLayout.addWidget(self.label, 0, 2, 1, 1)

        self.label_2 = QLabel(self.groupBox_3)
        self.label_2.setObjectName(u"label_2")

        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)

        self.label_status_bmode = QLabel(self.groupBox_3)
        self.label_status_bmode.setObjectName(u"label_status_bmode")

        self.gridLayout.addWidget(self.label_status_bmode, 0, 1, 1, 1)

        self.label_status_mocap = QLabel(self.groupBox_3)
        self.label_status_mocap.setObjectName(u"label_status_mocap")

        self.gridLayout.addWidget(self.label_status_mocap, 1, 1, 1, 1)

        self.label_1 = QLabel(self.groupBox_3)
        self.label_1.setObjectName(u"label_1")

        self.gridLayout.addWidget(self.label_1, 0, 0, 1, 1)

        self.label_3 = QLabel(self.groupBox_3)
        self.label_3.setObjectName(u"label_3")

        self.gridLayout.addWidget(self.label_3, 1, 2, 1, 1)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.radioButton_mha = QRadioButton(self.groupBox_3)
        self.radioButton_mha.setObjectName(u"radioButton_mha")
        self.radioButton_mha.setChecked(True)

        self.horizontalLayout_4.addWidget(self.radioButton_mha)

        self.radioButton_mhacsv = QRadioButton(self.groupBox_3)
        self.radioButton_mhacsv.setObjectName(u"radioButton_mhacsv")

        self.horizontalLayout_4.addWidget(self.radioButton_mhacsv)

        self.radioButton_imagecsv = QRadioButton(self.groupBox_3)
        self.radioButton_imagecsv.setObjectName(u"radioButton_imagecsv")

        self.horizontalLayout_4.addWidget(self.radioButton_imagecsv)


        self.gridLayout.addLayout(self.horizontalLayout_4, 1, 3, 1, 1)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.lineEdit_coupledrecord_recorddir = QLineEdit(self.groupBox_3)
        self.lineEdit_coupledrecord_recorddir.setObjectName(u"lineEdit_coupledrecord_recorddir")

        self.horizontalLayout_5.addWidget(self.lineEdit_coupledrecord_recorddir)

        self.pushButton_coupledrecord_recorddirClear = QPushButton(self.groupBox_3)
        self.pushButton_coupledrecord_recorddirClear.setObjectName(u"pushButton_coupledrecord_recorddirClear")
        icon = QIcon(QIcon.fromTheme(QIcon.ThemeIcon.EditClear))
        self.pushButton_coupledrecord_recorddirClear.setIcon(icon)

        self.horizontalLayout_5.addWidget(self.pushButton_coupledrecord_recorddirClear)


        self.gridLayout.addLayout(self.horizontalLayout_5, 0, 3, 1, 1)

        self.pushButton_coupledrecord_recorddirBrowse = QPushButton(self.groupBox_3)
        self.pushButton_coupledrecord_recorddirBrowse.setObjectName(u"pushButton_coupledrecord_recorddirBrowse")
        icon1 = QIcon(QIcon.fromTheme(QIcon.ThemeIcon.FolderOpen))
        self.pushButton_coupledrecord_recorddirBrowse.setIcon(icon1)

        self.gridLayout.addWidget(self.pushButton_coupledrecord_recorddirBrowse, 0, 4, 1, 2)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.pushButton_coupledrecord_recordStream = QPushButton(self.groupBox_3)
        self.pushButton_coupledrecord_recordStream.setObjectName(u"pushButton_coupledrecord_recordStream")

        self.horizontalLayout_6.addWidget(self.pushButton_coupledrecord_recordStream)

        self.pushButton_coupledrecord_snapshot = QPushButton(self.groupBox_3)
        self.pushButton_coupledrecord_snapshot.setObjectName(u"pushButton_coupledrecord_snapshot")

        self.horizontalLayout_6.addWidget(self.pushButton_coupledrecord_snapshot)


        self.gridLayout.addLayout(self.horizontalLayout_6, 1, 4, 1, 2)

        self.gridLayout.setColumnStretch(1, 1)
        self.gridLayout.setColumnStretch(3, 2)
        self.gridLayout.setColumnStretch(5, 1)

        self.horizontalLayout_3.addLayout(self.gridLayout)


        self.verticalLayout_8.addLayout(self.horizontalLayout_3)


        self.verticalLayout.addWidget(self.groupBox_3)

        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(1, 1)

        self.horizontalLayout.addLayout(self.verticalLayout)

        self.horizontalLayout.setStretch(0, 1)

        self.horizontalLayout_2.addLayout(self.horizontalLayout)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 800, 22))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.groupBox_1.setTitle(QCoreApplication.translate("MainWindow", u"B-mode", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("MainWindow", u"Motion Capture", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("MainWindow", u"Coupled Recording", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Record Path", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Mocap Status", None))
        self.label_status_bmode.setText(QCoreApplication.translate("MainWindow", u"Disconnected", None))
        self.label_status_mocap.setText(QCoreApplication.translate("MainWindow", u"Disconnected", None))
        self.label_1.setText(QCoreApplication.translate("MainWindow", u"B-Mode Status", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"Record Type", None))
        self.radioButton_mha.setText(QCoreApplication.translate("MainWindow", u".mha", None))
        self.radioButton_mhacsv.setText(QCoreApplication.translate("MainWindow", u".mha + .csv", None))
        self.radioButton_imagecsv.setText(QCoreApplication.translate("MainWindow", u"image + .csv", None))
        self.lineEdit_coupledrecord_recorddir.setPlaceholderText(QCoreApplication.translate("MainWindow", u"~/output/", None))
        self.pushButton_coupledrecord_recorddirClear.setText("")
        self.pushButton_coupledrecord_recorddirBrowse.setText(QCoreApplication.translate("MainWindow", u"Record Dir", None))
        self.pushButton_coupledrecord_recordStream.setText(QCoreApplication.translate("MainWindow", u"Record", None))
        self.pushButton_coupledrecord_snapshot.setText(QCoreApplication.translate("MainWindow", u"Snapshot", None))
    # retranslateUi

