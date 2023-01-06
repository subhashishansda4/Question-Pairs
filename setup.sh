# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 18:04:01 2023

@author: VAGUE
"""

mkdir -p ~/.streamlit/

echo "\
    [server]\n\
    port = $PORT\n\
    enableCORS = false\n\
    headless = true\n\
    \n\
    " > ~/.streamlit/config.toml