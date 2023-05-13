#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 17:34:35 2023

@author: pylerner
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

## MAC sFTP mount point
# PATH_PREFIX = '/Users/pylerner/.CMVolumes/nii_local_sftp'

# Debian sFTP root
# PATH_PREFIX = '/home/pyler/netdisk'

FILE_NAME = 'training.log'

print(os.listdir())
print(os.getcwd())
print(os.path.abspath(FILE_NAME))

PATH_PREFIX = os.getcwd()

# DATA = str(os.path.abspath(FILE_NAME))
DATA = 'training.log'


# DATA = os.path.join(PATH_PREFIX, FILE_NAME)


def get_history_data(DATA):
      
    df = pd.read_csv(DATA)
    df.epoch += 1
    return df.set_index('epoch')


def data_plot(df, X=None, Y=None):
    return st.pyplot(df.plot(x=X, y=Y).figure)


def get_period(a, b_list, c_list):
    
    idx = b_list.index(a)
    return c_list[idx]
    
        
st.title('Домашнее задание уровня Lite "Вывод датафрейма в streamlite"')
st.subheader('Выполнено Лернером В.')

str_periods = [
    '1 - 20',
    '21 - 40',
    '41 - last',
    'all'
]

idx_periods = [(0, 20), (21, 40), (41, -1), (0, -1)]


try:

    df = get_history_data(DATA)
    
    # Выбор диапазонов строк
    filters = st.sidebar.multiselect(
        'Выбрать диапазоны строк',
        str_periods,
        ['all']         # "все" по умолчаниюs
    )
    
    if not filters:
        st.error('Выбери не менее одного диапазона')
        
    else:
        
        if 'all' in filters:
            data = df
            
        else:
            
            data = pd.DataFrame()
            
            for f in filters:
                
                start, end = get_period(f,
                                        str_periods, 
                                        idx_periods
                                        )
                
                data = pd.concat([data, df[start: end]]).sort_index()
                
        
        st.write(
            '### История обучения модели PSPNet сегментации стройки по 16 классам',
            data
            )
    
            
        fig, ax = plt.subplots(1, 2)
        col1, col2 = st.columns(2)
        
        with col1:
            st.header('График функции потерь')  
            data_plot(data, Y=['loss', 'val_loss'])
            
        
        with col2:
            st.header('График функции точности')
            data_plot(data, 
                      Y=['sparse_categorical_accuracy',
                         'val_sparse_categorical_accuracy']
                      )
                    
    
except FileNotFoundError as e:
    st.error(
        """
        **Data file not found.**
        error: %s
    """
        % e.strerror
    )
    
show_code = st.sidebar.checkbox('Показать код')

if show_code:
    
    PATH = os.path.abspath(__file__)
    
    with open(PATH, 'r') as file:
        file = file.read()
        st.code(file, language='python')
        
