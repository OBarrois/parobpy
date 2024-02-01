#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 14:45:52 2021
@author: wongj

modified by: obarrois 01 Feb 2023

The parody_py package contains the routines needed to postprocess data from
PARODY-JA4.3 dynamo simulations.
Modules
_______
core_properties.py
    Constants of the core problem
data_utils.py --> load_parody.py
    Functions for data input/output
plot_utils.py --> plot_lib.py
    Functions for plotting results
parob_lib.py (from magic and pizza)
    Functions toolbox for derivatives and more
routines.py --> post_processing.py
    Functions for postprocessing data
"""
from .core_properties import *
from .load_parody import *
from .plot_lib import *
from .parob_lib import *
from .post_processing import *
