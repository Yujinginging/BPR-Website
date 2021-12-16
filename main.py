import os
import geopandas
import streamlit as st

from dbfread import DBF
import csv
from datetime import date

import numpy as np
import pandas as pd
import warnings;

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.simplefilter('ignore')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso, LassoCV

import Controllers

now = date.today().year





class View:
    def format_color_groups_forPipeStatus(df):
        colors = ['green', 'red']
        x = df.copy()
        factors = list(x['PipeStatus'].unique())
        i = 0
        for factor in factors:
            if factor == 0:

                style = f'background-color: {colors[i]}'
                x.loc[x['PipeStatus'] == factor, :] = style
            else:
                style = f'background-color: {colors[i + 1]}'
                x.loc[x['PipeStatus'] == factor, :] = style
        return x

    def format_color_groups_forPrediction(df):
        colors = ['green', 'red']
        x = df.copy()
        factors = list(x['Predict pipe status'].unique())
        i = 0
        for factor in factors:
            if factor == 0:
                style = f'background-color: {colors[i]}'
                x.loc[x['Predict pipe status'] == factor, :] = style
            else:
                style = f'background-color: {colors[i + 1]}'
                x.loc[x['Predict pipe status'] == factor, :] = style
        return x

    def format_color_groups_forAge(df):
        colors = ['yellow']
        x = df.copy()
        factors = [x['Possible broken year'].unique()]
        i = 0
        for factor in factors:
            if factor != np.NaN:
                style = f'background-color: {colors[i]}'
                x.loc[x['Possible broken year'] == factor, :] = style

        return x


def get_matched_depth_all(select_x, select_y):
    length = SearchedDataAll.loc[SearchedDataAll['XKoordinat'] == select_x]['Laengde'].values[0]
    angle = SearchedDataAll.loc[SearchedDataAll['XKoordinat'] == select_x]['Fald'].values[0]

    # calculate another point by length:
    end_x = select_x + (length * np.cos(angle))
    end_y = select_y + (length * np.sin(angle))
    if (end_x > select_x):
        max_x = end_x
        min_x = select_x
    else:
        min_x = end_x
        max_x = select_x
    if (end_y > select_y):
        max_y = end_y
        min_y = select_y
    else:
        min_y = end_y
        max_y = select_y

    matched_depth_col = df.loc[(df['XKoordinat'] <= max_x) & (df['XKoordinat'] >= min_x)
                               & (df['YKoordinat'] <= max_y) & (df['YKoordinat'] >= min_y)]['Depth']
    # test if there is a value
    if (matched_depth_col.size > 0):
        matched_depth = matched_depth_col.values[0]
    else:
        matched_depth = np.NaN

    return matched_depth


def add_depth_all(datacopy):
    select_x = datacopy['XKoordinat']
    select_y = datacopy['YKoordinat']
    return get_matched_depth_all(select_x, select_y)








st.title("Welcome")

# upload ground water
st.write("Please upload groundwater file here (.xyz): ")
xyz_path = st.text_input("Enter file path: (e.g: C:/Users/JingJing/Desktop/S7/BPR2/GRW_MBS_50m.xyz")
st.write("Please upload pipe file here (.dbf): ")
dbf_path = st.text_input(
    "Enter file path: (e.g.'C:/Users/JingJing/Downloads/Archive (1) (1)/Energi_Viborg_Dandas_data.dbf') Use'/' instead of '\'!)")

# drop down menu
st.markdown('Enter the year (You want to predict pipe status in the following ** years):')
option = st.selectbox(
    'Choose year period:',
    ('10', '20', '30', '50', '100'))

if st.button('Predict', key='Predict'):
    if xyz_path is not None and dbf_path is not None:
        df = Controllers.GroundwaterController.read_xyz(xyz_path) # ('C:/Users/JingJing/Desktop/S7/BPR2/GRW_MBS_50m.xyz')

    # transfer to csv and display
        csv = Controllers.ViewPipeController.to_csv(dbf_path) # ('C:/Users/JingJing/Downloads/Archive (1) (1)/Energi_Viborg_Dandas_data.dbf')
    else:
        st.error("Wrong input")
    # transfer to csv and display
    data = pd.read_csv(csv)
    # data preparation:
    data = Controllers.ViewPipeController.DataFillNan(data)

    # a new data copy for search and predict
    SearchedDataAll = pd.DataFrame(data)
    # add depth
    SearchedDataAll['Depth'] = SearchedDataAll.apply(add_depth_all, axis=1)
    SearchedDataAll = SearchedDataAll.dropna()  # all data with adding column depth

    # prepare data achieved:
    data_with_TVObsAndSaneri_Groundwater = Controllers.SearchPipeController.Get_DatawithTVObsAndSaneri(SearchedDataAll)
    #

    # not broken pipes
    data_not_broken = Controllers.SearchPipeController.GetDataNotBroken(SearchedDataAll)

    data_not_broken_Groundwater = data_not_broken.sample(n=619)
    frames = [data_not_broken_Groundwater, data_with_TVObsAndSaneri_Groundwater]
    dataFinal = pd.concat(frames)  # data with train test data

    # copy data
    datacopy = pd.DataFrame(dataFinal)
    datacopy = Controllers.ViewPipeController.AddColumns(datacopy)  # add age and pipe status columns

    # devide data_features and data_target for Pipe status prediction
    data_features, data_target = Controllers.SearchPipeController.GetFeaturesAndTarget(datacopy)

    # divide data_features and data target for age prediction
    data_features_age, data_target_age = Controllers.SearchPipeController.GetFeaturesAndTarget_Age(datacopy)

    # create features and target for all data
    SearchedDataAll = Controllers.ViewPipeController.AddColumns(SearchedDataAll)  # add age and pipe status columns
    # copy for age
    SearchedDataAll_copy = pd.DataFrame(SearchedDataAll).copy()

    data_features_all, data_target_all = Controllers.SearchPipeController.GetFeaturesAndTarget(
        SearchedDataAll)  # data features and target for pipe status prediction
    # add age for prediction of pipe status
    data_features_all['Age'] = Controllers.PredictPipeController.ChangeAge(data_features_all, option)

    data_features_all_age, data_target_all_age = Controllers.SearchPipeController.GetFeaturesAndTarget_Age(
        SearchedDataAll_copy)  # data features and target for age prediction

    # get data features and target
    st.subheader('data_features:')
    #
    st.dataframe(data_features)
    st.subheader('data_target')
    #
    st.dataframe(data_target)

    X_trainval, X_test, y_trainval, y_test, dec_tree = Controllers.PredictPipeController.DicisionTree(data_features, data_target)

    dec_tree.fit(X_trainval, y_trainval)

    # table
    prediction = dec_tree.predict(data_features_all)
    SearchedDataAll["Predict pipe status"] = prediction


    st.subheader("Pipe status Predict Result: ")
    st.dataframe(pd.DataFrame(SearchedDataAll).style.apply(View.format_color_groups_forPrediction, axis=None))

    # age prediction processing
    age_prediction = Controllers.PredictPipeController.Lasso_AgePrediction(data_features_age, data_target_age, data_features_all_age)
    #

    SearchedDataAll_copy['Predict Age'] = age_prediction

    SearchedDataAll_copy['Possible broken year'] = (now + (
    SearchedDataAll_copy[SearchedDataAll_copy['Predict Age'] > SearchedDataAll_copy['Age']]['Predict Age']) - (
                                                   SearchedDataAll_copy[
                                                       SearchedDataAll_copy['Predict Age'] > SearchedDataAll_copy[
                                                           'Age']]['Age'])).astype(int)

    st.subheader("Pipe possible broken year Predict Result: ")
    st.dataframe(SearchedDataAll_copy)

# Search
id_pipe = st.text_input("Enter ID: ")
X_coordinate = st.text_input("Enter X coordinate: ")
Y_coordinate = st.text_input("Enter Y coordinate: ")

if st.button("Search", key="Search"):
    df = Controllers.GroundwaterController.read_xyz(xyz_path) #('C:/Users/JingJing/Desktop/S7/BPR2/GRW_MBS_50m.xyz')

    # transfer to csv and display
    csv = Controllers.ViewPipeController.to_csv(dbf_path) #('C:/Users/JingJing/Downloads/Archive (1) (1)/Energi_Viborg_Dandas_data.dbf')
    # transfer to csv and display
    data = pd.read_csv(csv)
    # data preparation:
    data = Controllers.ViewPipeController.DataFillNan(data)

    # a new data copy for search and predict
    SearchedDataAll = pd.DataFrame(data)
    # add depth
    SearchedDataAll['Depth'] = SearchedDataAll.apply(add_depth_all, axis=1)
    SearchedDataAll = SearchedDataAll.dropna()  # all data with adding column depth

    # prepare data achieved:
    data_with_TVObsAndSaneri_Groundwater = Controllers.SearchPipeController.Get_DatawithTVObsAndSaneri(SearchedDataAll)
    #

    # not broken pipes
    data_not_broken = Controllers.SearchPipeController.GetDataNotBroken(SearchedDataAll)

    data_not_broken_Groundwater = data_not_broken.sample(n=619)
    frames = [data_not_broken_Groundwater, data_with_TVObsAndSaneri_Groundwater]
    dataFinal = pd.concat(frames)  # data with train test data

    # copy data
    datacopy = pd.DataFrame(dataFinal)
    datacopy = Controllers.ViewPipeController.AddColumns(datacopy)  # add age and pipe status columns

    # devide data_features and data_target for Pipe status prediction
    data_features, data_target = Controllers.SearchPipeController.GetFeaturesAndTarget(datacopy)

    # divide data_features and data target for age prediction
    data_features_age, data_target_age = Controllers.SearchPipeController.GetFeaturesAndTarget_Age(datacopy)

    # create features and target for all data
    SearchedDataAll = Controllers.ViewPipeController.AddColumns(SearchedDataAll)  # add age and pipe status columns
    # copy for age
    SearchedDataAll_copy = pd.DataFrame(SearchedDataAll).copy()

    data_features_all, data_target_all = Controllers.SearchPipeController.GetFeaturesAndTarget(
        SearchedDataAll)  # data features and target for pipe status prediction
    # add age for prediction of pipe status
    data_features_all['Age'] = Controllers.PredictPipeController.ChangeAge(data_features_all, option)

    data_features_all_age, data_target_all_age = Controllers.SearchPipeController.GetFeaturesAndTarget_Age(
        SearchedDataAll_copy)  # data features and target for age prediction

    # # get data features and target
    # st.subheader('data_features:')
    # #
    # st.dataframe(data_features)
    # st.subheader('data_target')
    # #
    # st.dataframe(data_target)

    X_trainval, X_test, y_trainval, y_test, dec_tree = Controllers.PredictPipeController.DicisionTree(data_features, data_target)

    dec_tree.fit(X_trainval, y_trainval)

    # table
    prediction = dec_tree.predict(data_features_all)
    SearchedDataAll["Predict pipe status"] = prediction

    st.subheader("Pipe status Predict Result: ")
    st.dataframe(pd.DataFrame(SearchedDataAll).style.apply(View.format_color_groups_forPrediction, axis=None))

    # age prediction processing
    age_prediction = Controllers.PredictPipeController.Lasso_AgePrediction(data_features_age, data_target_age, data_features_all_age)
    #

    SearchedDataAll_copy['Predict Age'] = age_prediction

    SearchedDataAll_copy['Possible broken year'] = (now + (
        SearchedDataAll_copy[SearchedDataAll_copy['Predict Age'] > SearchedDataAll_copy['Age']]['Predict Age']) - (
                                                        SearchedDataAll_copy[
                                                            SearchedDataAll_copy['Predict Age'] > SearchedDataAll_copy[
                                                                'Age']]['Age'])).astype(int)

    st.subheader("Pipe possible broken year Predict Result: ")
    st.dataframe(SearchedDataAll_copy)

    SearchedDataAll = pd.DataFrame(SearchedDataAll)
    # search result
    if id_pipe is not None and X_coordinate is None and Y_coordinate is None:
        pipe = SearchedDataAll.loc[SearchedDataAll['ID'] == int(id_pipe)]
        st.subheader("Search Result for pipes: ")
        pipe = pd.DataFrame(pipe).style.apply(View.format_color_groups_forPrediction, axis=None)
        st.dataframe(pipe)

        #age
        age = SearchedDataAll_copy.loc[SearchedDataAll_copy['ID'] == int(id_pipe)]
        st.subheader("Search Result for age: ")
        st.dataframe(age)

    elif X_coordinate is not None and Y_coordinate is not None:
        pipe = SearchedDataAll.loc[
            (SearchedDataAll['XKoordinat'].astype(int) == int(float(X_coordinate))) & (
                        SearchedDataAll['YKoordinat'].astype(int) == int(float(Y_coordinate)))]
        st.subheader("Search Result: ")
        pipe = pd.DataFrame(pipe).style.apply(View.format_color_groups_forPrediction, axis=None)

        st.dataframe(pipe)

        #age
        age = SearchedDataAll_copy.loc[(SearchedDataAll_copy['XKoordinat'].astype(int) == int(float(X_coordinate)))&(SearchedDataAll_copy['YKoordinat'].astype(int) == int(float(Y_coordinate)))]
        st.subheader("Search Result for age: ")
        st.dataframe(age)
    else:
        st.write("please enter again!")
