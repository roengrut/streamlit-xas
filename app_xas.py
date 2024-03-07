import streamlit as st 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import sys
import os
import pickle
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

st.set_page_config(page_title='XAS Spectral Data Classification',layout='wide')

def get_filesize(file):
    size_bytes = sys.getsizeof(file)
    size_mb = size_bytes / (1024**2)
    return size_mb

def validate_file(file):
    filename = file.name
    name, ext = os.path.splitext(filename)
    if ext in ('.csv','.xlsx'):
        return ext
    else:
        return False

# sidebar
with st.sidebar:  
    uploaded_file = st.file_uploader("Upload .csv, .xlsx files not exceeding 10 MB")
    if uploaded_file is not None:  
        st.write('---')
        st.header('Select your XAS materials')
        materials_mode = st.radio('Main Materials:',
                                    options=('Cu-O', 'Fe-O', 'Mn-O', 'Ti-O', 'V-O', 'Zn-O'))

if uploaded_file is not None:
    ext = validate_file(uploaded_file)
    if ext:
        filesize = get_filesize(uploaded_file)
        if filesize <= 10:
            if ext == '.csv':
                # time being let load csv
                df = pd.read_csv(uploaded_file, header=None)
            else:
                xl_file = pd.ExcelFile(uploaded_file)
                sheet_tuple = tuple(xl_file.sheet_names)
                sheet_name = st.sidebar.selectbox('Select the sheet',sheet_tuple)
                df = xl_file.parse(sheet_name)
                
                
            # generate report

        else:
            st.error(f'Maximum allowed filesize is 10 MB. But received {filesize} MB')
            
    else:
        st.error('Please upload only .csv or .xlsx file')
  
    if materials_mode == 'Cu-O':
        model = tf.keras.models.load_model('xas_Cu_O.h5')
        database_file = 'Cu_O_spectrum.csv'
        label_file = 'Cu-O_data.csv'
    elif materials_mode == 'Fe-O':
        model = tf.keras.models.load_model('xas_Fe_O.h5')
        database_file = 'Fe_O_spectrum.csv'
        label_file = 'Fe-O_data.csv'
    elif materials_mode == 'Mn-O':
        model = tf.keras.models.load_model('xas_Mn_O.h5')
        database_file = 'Mn_O_spectrum_CN.csv'
        label_file = 'Mn-O_data_CN.csv'
    elif materials_mode == 'Ti-O':
        model = tf.keras.models.load_model('xas_Ti_O.h5')
        database_file = 'Ti_O_spectrum_CN.csv'
        label_file = 'Ti-O_data_CN.csv'
    elif materials_mode == 'V-O':
        model = tf.keras.models.load_model('xas_V_O.h5')
        database_file = 'V_O_spectrum_CN.csv'
        label_file = 'V-O_data_CN.csv'
    elif materials_mode == 'Zn-O':
        model = tf.keras.models.load_model('xas_Zn_O.h5')
        database_file = 'Zn_O_spectrum_CN.csv'
        label_file = 'Zn-O_data_CN.csv'
    else: 
        model = tf.keras.models.load_model('xas_Cu_O.h5')
        database_file = 'Cu_O_spectrum.csv'
        label_file = 'Cu-O_data.csv'

    
    df_database = pd.read_csv(database_file, header=None)
    df_label = pd.read_csv(label_file)
    y = df_label['symmetry_class']

    # concatenating df and df_database along rows
    df_vertical_concat = pd.concat([df, df_database], axis=0)
    #st.dataframe(df_vertical_concat)

    # Scaling the data
    scaler = StandardScaler()
    x = scaler.fit_transform(df_database)
    x_new = scaler.fit_transform(df_vertical_concat)

    # PCA
    pca = PCA(0.95)
    x_pca = pca.fit_transform(x)
    #print(x_pca.shape)
    x_new_pca = pca.transform(x_new)
    #print(x_new_pca.shape)

    # NN Prediction
    ans = model.predict(x_new_pca)

    row = 0
    print(ans[row])
    class_list = ans.tolist()
    my_list = class_list[row]
    max_value = max(my_list)
    max_index = my_list.index(max_value)
    print(max_value, max_index)

    # load and visualize the input data
    with st.container():
        st.subheader('Data Input:')
        st.dataframe(df)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Spectral Input:') 
            plt.figure()
            plt.plot(df.T)
            st.pyplot(plt)

        with col2:
            st.subheader('Classification Result:')
            if max_index == 1:
                true_class, num_index, emoji = 'Cubic', ':one:', ':satisfied:'
            elif max_index == 2:
                true_class, num_index, emoji = 'Hexagonal', ':two:', ':innocent:'
            elif max_index == 3:
                true_class, num_index, emoji = 'Monoclinic', ':three:', ':laughing:'
            elif max_index == 4:
                true_class, num_index, emoji = 'Orthorhombic', ':four:', ':smile:'
            elif max_index == 5:
                true_class, num_index, emoji = 'Tetragonal', ':five:', ':blush:'
            elif max_index == 6:
                true_class, num_index, emoji = 'Triclinic', ':six:', ':kissing_heart:'
            elif max_index == 7:
                true_class, num_index, emoji = 'Trigonal', ':seven:', ':sunglasses:'
            st.subheader(f'The :blue[Symmetry Class] is :red[{true_class}] ({num_index}) {emoji}')
       
else:
    st.title('XAS (XANES) Spectral Data Classification App')
    st.info('Upload your data in the left sidebar to proceed the classification')



