import os

import altair as alt
import geopandas
import streamlit as st

from dbfread import DBF
import csv
import pandas as pd
from datetime import date
import geopandas as gpd

import numpy as np
import pandas as pd
import sklearn as sk
import seaborn as sns
import warnings;

from sklearn.metrics import classification_report, roc_auc_score
from sklearn.tree import DecisionTreeClassifier

warnings.simplefilter('ignore')
import matplotlib.pyplot as plt
import mglearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

now = date.today().year


class PipeController:

    def display_data(filepath):
        for record in DBF(filepath):
            return record

    def to_csv(dbf_table_pth):
        # csv_fn = dbf_table_pth[:-4] + ".csv"  # Set the csv file name
        csv_fn = dbf_table_pth + ".csv"  # Set the csv file name

        table = DBF(dbf_table_pth)  # table variable is a DBF object
        with open(csv_fn, 'w', newline='') as f:  # create a csv file, fill it with dbf content
            writer = csv.writer(f)
            writer.writerow(table.field_names)  # write the column name
            for record in table:  # write the rows
                writer.writerow(list(record.values()))
        return csv_fn  # return the csv name

    def DataFillNan(data):
        columns_to_be_removed = ['mslink', 'LedningID', 'Dobbeltled', 'EjerKompon', 'SystemKode', 'KategoriAf',
                                 'DatoUdf']
        data = data.drop(columns_to_be_removed, axis='columns')

        # in the column DatoSaneri is the date of repairing and if there is no date it means it is not repaired

        data['DatoSaneri'].fillna(0, inplace=True)
        return data

    def Get_DatawithTVObsAndSaneri(data):

        # take only the pipes that are broken(by TV insection) now and the repaired ones

        data_with_TVObsAndSaneri = data[data['TVObsKode'].isin([1]) | data['DatoSaneri'] > 0]

        return data_with_TVObsAndSaneri

    def GetDatawithTVObsAndSaneri_GW(data_with_TVObsAndSaneri):
        data_with_TVObsAndSaneri['Depth'] = data_with_TVObsAndSaneri.apply(add_depth, axis=1)
        data_with_TVObsAndSaneri_Groundwater = data_with_TVObsAndSaneri.dropna()
        return data_with_TVObsAndSaneri_Groundwater

    def GetDataNotBroken(data):

        data_not_broken = data[~data['TVObsKode'].isin([0]) | data['DatoSaneri'] == 0]
        data_not_broken = data_not_broken.sample(n=4000)
        return data_not_broken

    def GetDataNotBroken_GW(data_not_broken):
        data_not_broken_Groundwater = data_not_broken.dropna()
        data_not_broken_Groundwater = data_not_broken_Groundwater.sample(n=619)
        return data_not_broken_Groundwater

    def GetFeaturesAndTarget(datacopy):
        columns_to_be_removed = ['PipeStatus', 'TVObsKode', 'ID', 'TransportK', 'Funktionsk', 'MaterialeK',
                                 'DatoSaneri']
        data_features = datacopy.drop(columns_to_be_removed, axis='columns')
        # columns_to_be_removed = ['fra_kote', 'til_kote', 'Laengde', 'Fald', 'DiameterIn', 'MaterialeK', 'anlag_aar',
        #                          'TransportK', 'Funktionsk', 'DatoSaneri', 'Age', 'Depth', 'TVObsKode', 'YKoordinat',
        #                          'XKoordinat', 'ID']

        columns_to_be_removed = ['fra_kote', 'til_kote', 'Laengde', 'Fald', 'DiameterIn', 'MaterialeK', 'anlag_aar',
                                 'TransportK', 'Funktionsk', 'DatoSaneri', 'Age', 'Depth', 'TVObsKode', 'YKoordinat',
                                 'XKoordinat', 'ID', 'DatoOprett', 'DatoOpdate']

        data_target = datacopy.drop(columns_to_be_removed, axis='columns')
        return data_features, data_target

    def AddColumns(datacopy):
        datacopy['Age'] = datacopy.apply(age_df, axis=1)
        datacopy['PipeStatus'] = datacopy.apply(broken_df, axis=1)
        return datacopy

    # train test, no validation data
    def DicisionTree(dataFeatures, datatarget):
        X_trainval, X_test, y_trainval, y_test = train_test_split(dataFeatures, datatarget, stratify=datatarget,
                                                                  random_state=42)
        best_score = 0

        for max_depth in [3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, None]:
            for criterion in ['gini', 'entropy']:
                for max_features in ['sqrt', 'log2', 'auto', None]:
                    # Learn the model
                    dec_tree = DecisionTreeClassifier(max_depth=max_depth, max_features=max_features,
                                                      criterion=criterion)

                    # Perform cross validation
                    scores = cross_val_score(dec_tree, X_trainval, y_trainval, cv=5, scoring='recall')

                    # Compute the mean score
                    score = scores.mean()

                    # If improvement, store score and parameter
                    if score > best_score:
                        best_score = score
                        best_max_depth = max_depth
                        best_max_features = max_features
                        best_criterion = criterion

        # Build a model on the combine training and valiation data
        dec_tree = DecisionTreeClassifier(max_depth=best_max_depth, max_features=best_max_features,
                                          criterion=best_criterion)
        return X_trainval, X_test, y_trainval, y_test, dec_tree

    def plot_feature_importances_pipes(model, dataFeatures):
        n_features = dataFeatures.shape[1]
        feature_names = dataFeatures.columns
        plt.barh(range(n_features), model.feature_importances_, align='center')
        plt.yticks(np.arange(n_features), feature_names)
        plt.xlabel("Feature importance")
        plt.ylabel("Feature")


class GroundwaterController():

    def read_xyz(fileName):
        xyz_coordinates = []  # put xyz in an array

        with open(fileName, "r") as file:
            for line_number, line in enumerate(file):
                x, y, z = line.split()

                xyz_coordinates.append([int(x), int(y), float(z)])

        my_array = np.array(xyz_coordinates)

        df = pd.DataFrame(my_array, columns=['XKoordinat', 'YKoordinat', 'Depth'])

        return df


def age_df(datacopy):
    if (datacopy['TVObsKode'] == 1) and (datacopy['DatoSaneri'] > 0):
        return (now - datacopy['DatoSaneri'])
    elif (datacopy['TVObsKode'] == 1) and (datacopy['DatoSaneri'] == 0):
        return (now - datacopy['anlag_aar'])
    elif (datacopy['TVObsKode'] == 0) and (datacopy['DatoSaneri'] > 0):
        return (now - datacopy['DatoSaneri'])
    elif (datacopy['TVObsKode'] == 0) and (datacopy['DatoSaneri'] == 0):
        return (now - datacopy['anlag_aar'])


def broken_df(datacopy):
    if (datacopy['TVObsKode'] == 1) and (datacopy['DatoSaneri'] < (datacopy['DatoOpdate'])) and (
            datacopy['DatoSaneri'] != 0):
        return 1
    elif (datacopy['TVObsKode'] == 1) and (datacopy['DatoSaneri'] >= (datacopy['DatoOpdate'])) and (
            datacopy['DatoSaneri'] != 0):
        return 0
    elif (datacopy['TVObsKode'] == 1) and (datacopy['DatoSaneri'] == 0):
        return 1
    elif (datacopy['TVObsKode'] == 0) and (datacopy['DatoSaneri'] > 0):
        return 0
    elif (datacopy['TVObsKode'] == 0) and (datacopy['DatoSaneri'] == 0):
        return 0


def get_matched_depth_unbroken(select_x, select_y):
    #     select_x = data['XKoordinat']
    #     select_y = data['YKoordinat']
    # select_y = data.loc[data['XKoordinat'] == select_x]['YKoordinat'].values[0]
    length = data_not_broken.loc[data_not_broken['XKoordinat'] == select_x]['Laengde'].values[0]
    angle = data_not_broken.loc[data_not_broken['XKoordinat'] == select_x]['Fald'].values[0]
    #     length=data['Laengde']
    #     angle = data['Fald']
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


def add_depth_unbroken(datacopy):
    select_x = datacopy['XKoordinat']
    select_y = datacopy['YKoordinat']
    return get_matched_depth_unbroken(select_x, select_y)


def get_matched_depth(select_x, select_y):
    # select_x = data['XKoordinat']
    #     select_y = data['YKoordinat']
    # select_y = data.loc[data['XKoordinat'] == select_x]['YKoordinat'].values[0]
    length = dataWithTVObsAndSaneri.loc[dataWithTVObsAndSaneri['XKoordinat'] == select_x]['Laengde'].values[0]
    angle = dataWithTVObsAndSaneri.loc[dataWithTVObsAndSaneri['XKoordinat'] == select_x]['Fald'].values[0]
    #     length=data['Laengde']
    #     angle = data['Fald']
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


def add_depth(datacopy):
    select_x = datacopy['XKoordinat']
    select_y = datacopy['YKoordinat']
    return get_matched_depth(select_x, select_y)


def highlighted(data):
    if data['Predict pipe status'] == 1:
        return ['background-color: red'] * len(data)


# upload button
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
    df = GroundwaterController.read_xyz('C:/Users/JingJing/Desktop/S7/BPR2/GRW_MBS_50m.xyz')

    # transfer to csv and display
    csv = PipeController.to_csv('C:/Users/JingJing/Downloads/Archive (1) (1)/Energi_Viborg_Dandas_data.dbf')
    # transfer to csv and display
    data = pd.read_csv(csv)
    # data preparation:
    data = PipeController.DataFillNan(data)

    # prepare data achieved:
    dataWithTVObsAndSaneri = PipeController.Get_DatawithTVObsAndSaneri(data)
    #
    dataWithTVObsAndSaneri['Depth'] = dataWithTVObsAndSaneri.apply(add_depth, axis=1)
    data_with_TVObsAndSaneri_Groundwater = dataWithTVObsAndSaneri.dropna()

    # not broken pipes
    data_not_broken = PipeController.GetDataNotBroken(data)

    data_not_broken['Depth'] = data_not_broken.apply(add_depth_unbroken, axis=1)
    data_not_broken_Groundwater = data_not_broken.dropna()

    data_not_broken_Groundwater = data_not_broken_Groundwater.sample(n=619)
    frames = [data_not_broken_Groundwater, data_with_TVObsAndSaneri_Groundwater]
    dataFinal = pd.concat(frames)

    # copy data
    datacopy = dataFinal
    datacopy = PipeController.AddColumns(datacopy)
    data_features, data_target = PipeController.GetFeaturesAndTarget(datacopy)
    # get data features and target
    st.subheader('data_features:')
    #
    st.dataframe(data_features)
    st.subheader('data_target')
    #
    st.dataframe(data_target)


def highlight_survived(s):
    return ['background-color: green'] * len(s) if s['Predict pipe status'] == 1 else ['background-color: red'] * len(s)
    # get values from above buttons from session state
    # data_features = st.session_state.data_features
    # data_target = st.session_state.data_target
    X_trainval, X_test, y_trainval, y_test, dec_tree = PipeController.DicisionTree(data_features, data_target)
    dec_tree.fit(X_trainval, y_trainval)
    y_pred = dec_tree.predict(X_test)
    st.write(classification_report(y_test, y_pred))
    st.write(roc_auc_score(y_test, y_pred))
    # st.plotly_chart(PipeController.plot_feature_importances_pipes(dec_tree,data_features))

    # table
    new_data = dataFinal
    columns_to_be_removed = ['PipeStatus', 'TVObsKode', 'ID', 'TransportK', 'Funktionsk', 'MaterialeK',
                             'DatoSaneri']
    new_data = new_data.drop(columns_to_be_removed, axis='columns')
    prediction = dec_tree.predict(new_data)
    new_data["Predict pipe status"] = prediction

    st.subheader("Result: ")
    st.dataframe(new_data.style.apply(highlight_survived, axis=1))


id_pipe = st.number_input("Enter ID: ", step=1)
X_coordinate = st.number_input("Enter X coordinate: ", step=1)
Y_coordinate = st.number_input("Enter Y coordinate: ", step=1)

if st.button("Search", key="Search"):
    df = GroundwaterController.read_xyz('C:/Users/JingJing/Desktop/S7/BPR2/GRW_MBS_50m.xyz')

    # transfer to csv and display
    csv = PipeController.to_csv('C:/Users/JingJing/Downloads/Archive (1) (1)/Energi_Viborg_Dandas_data.dbf')
    # transfer to csv and display
    data = pd.read_csv(csv)
    # data preparation:
    data = PipeController.DataFillNan(data)

    # prepare data achieved:
    dataWithTVObsAndSaneri = PipeController.Get_DatawithTVObsAndSaneri(data)
    #
    dataWithTVObsAndSaneri['Depth'] = dataWithTVObsAndSaneri.apply(add_depth, axis=1)
    data_with_TVObsAndSaneri_Groundwater = dataWithTVObsAndSaneri.dropna()

    # not broken pipes
    data_not_broken = PipeController.GetDataNotBroken(data)

    data_not_broken['Depth'] = data_not_broken.apply(add_depth_unbroken, axis=1)
    data_not_broken_Groundwater = data_not_broken.dropna()

    data_not_broken_Groundwater = data_not_broken_Groundwater.sample(n=619)
    frames = [data_not_broken_Groundwater, data_with_TVObsAndSaneri_Groundwater]
    dataFinal = pd.concat(frames)

    # copy data
    datacopy = dataFinal
    datacopy = PipeController.AddColumns(datacopy)
    data_features, data_target = PipeController.GetFeaturesAndTarget(datacopy)
    # get data features and target
    st.subheader('data_features:')
    #
    st.dataframe(data_features)
    st.subheader('data_target')
    #
    st.dataframe(data_target)

    # get values from above buttons from session state
    # data_features = st.session_state.data_features
    # data_target = st.session_state.data_target
    X_trainval, X_test, y_trainval, y_test, dec_tree = PipeController.DicisionTree(data_features, data_target)
    dec_tree.fit(X_trainval, y_trainval)
    y_pred = dec_tree.predict(X_test)
    st.write(classification_report(y_test, y_pred))
    st.write(roc_auc_score(y_test, y_pred))
    # st.plotly_chart(PipeController.plot_feature_importances_pipes(dec_tree,data_features))

    # table
    new_data = dataFinal
    columns_to_be_removed = ['PipeStatus', 'TVObsKode', 'ID', 'TransportK', 'Funktionsk', 'MaterialeK',
                             'DatoSaneri']
    new_data = new_data.drop(columns_to_be_removed, axis='columns')
    prediction = dec_tree.predict(new_data)
    new_data["Predict pipe status"] = prediction

    if (new_data['Predict pipe status'].values[0] == 1):
        new_data.style.set_properties(**{'background-color': 'red'})
    else:
        new_data.style.set_properties(**{'background-color': 'green'})

    st.subheader("Result: ")
    st.dataframe(new_data)

    # color rows with green and red
    if id_pipe is not None:
        pipe = dataFinal.loc[dataFinal['ID'] == id_pipe]
        columns_to_be_removed = ['PipeStatus', 'TVObsKode', 'TransportK', 'Funktionsk', 'MaterialeK',
                                 'DatoSaneri']
        pipe = pipe.drop(columns_to_be_removed, axis='columns')
        st.dataframe(pipe.style.apply(highlight_survived, axis=1))
    elif X_coordinate is not None and Y_coordinate is not None:
        pipe = new_data.loc[(new_data['XKoordinat'] == X_coordinate) & (new_data['YKoordinat'] == Y_coordinate)]
        st.dataframe(pipe.style.apply(highlight_survived, axis=1))
    else:
        st.write("please enter again!")
