import sys

from pathlib import Path

import random
import imageio
import functools
import time
import numpy as np
import cv2 as cv

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt, QBasicTimer, QDate, QMimeData
import PyQt5.QtCore as QtCore
from PyQt5.QtCore import QSize
from PyQt5.Qt import QPainter
from PyQt5 import QtWidgets

from PyQt5.QtCore import QSize
from PyQt5.Qt import QPainter

blue = '#435585'
whiteBlue = '#818FB4'
white = '#F5E8C7'

firstColorName = '#012F34'
secondColorName = '#0E484E'
thirdColorName = '#417C81'
fourthColorName = '#97B9BE'
fourthColorName = '#b1cace'

blue = thirdColorName
whiteBlue = secondColorName
white = fourthColorName


buttonStyleSheet = \
            '''
            border: 4px solid ''' + whiteBlue + ''';
            color: ''' + white + ''';
            font-family: 'shanti';
            font-size: 16px;
            border-radius: 25px;
            padding: 15px 0;
            margin-top: 20px}
            *:hover{
                background:  ''' + whiteBlue + '''
            }
            '''
smallerButtonStyleSheet = \
'''
            border: 3px solid ''' + whiteBlue + ''';
            color: ''' + white + ''';
            /* color: white; */
            font-family: 'shanti';
            border-radius: 10px;
            padding: 2px 0;
            margin-top: 2px}
            *:hover{
                background:  ''' + whiteBlue + ''';
                color: white;
            }
            '''
invertedSmallerButtonStyleSheet = \
'''
            border: 3px solid ''' + whiteBlue + ''';
            color: ''' + white + ''';
            background:  ''' + whiteBlue + ''';
            /* color: white; */
            font-family: 'shanti';
            border-radius: 10px;
            padding: 2px 0;
            margin-top: 2px}
            *:hover{
                background:  ''' + white + ''';
                color: white;
            }
            '''


