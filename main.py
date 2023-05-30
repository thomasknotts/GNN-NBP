# -*- coding: utf-8 -*-
"""
Created on Mon May  8 13:11:51 2023

@author: Frank
"""

# import packages

import streamlit as st
from torch_utils import get_predictions

st.set_page_config(
    page_title = "Normal Boiling Point Prediction",
    page_icon = ":new_moon_with_face:"
    )

st.title("Normal Boiling Point Prediction")


get_predictions()















