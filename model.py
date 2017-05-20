#!/usr/bin/env python

# coding=utf-8
import numpy as np # linear algebra
import pandas as pd
import tensorflow as tf
import os
import cv2


class model(object):
    
    def __init__(self,placeholder_input,placeholder_target):
        self.placeholder_input=placeholder_input
        self.placeholder_target=placeholder_target
