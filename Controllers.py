import warnings
from datetime import date

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.simplefilter('ignore')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso, LassoCV

import numpy as np
from dbfread import DBF
import csv


import Models



now = date.today().year
class ViewPipeController:

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



    def AddColumns(datacopy):
        datacopy['Age'] = datacopy.apply(age_df, axis=1)
        datacopy['PipeStatus'] = datacopy.apply(broken_df, axis=1)
        return datacopy

class SearchPipeController:
    def Get_DatawithTVObsAndSaneri(data):

        # take only the pipes that are broken(by TV insection) now and the repaired ones

        data_with_TVObsAndSaneri = data[data['TVObsKode'].isin([1]) | data['DatoSaneri'] > 0]

        return data_with_TVObsAndSaneri

    def GetDataNotBroken(data):

        data_not_broken = data[~data['TVObsKode'].isin([0]) | data['DatoSaneri'] == 0]
        data_not_broken = data_not_broken.sample(n=4000)
        return data_not_broken

    def GetFeaturesAndTarget(datacopy):
        data_features, data_target = Models.Pipe.GetFeaturesAndTarget(datacopy)
        return data_features, data_target

    # get features and target for age prediction data

    def GetFeaturesAndTarget_Age(data):
        data_features, data_target = Models.Pipe.GetFeaturesAndTarget_Age(data)
        return data_features, data_target

class PredictPipeController:

    def Lasso_AgePrediction(datafeatures, datatarget, prefict_datafeatures):
        # Divide the data into training, test and validation
        X_trainval, X_test, y_trainval, y_test = train_test_split(datafeatures, datatarget, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, random_state=43)

        # preprocessing using 0-1 scaling
        scaler = StandardScaler()
        scaler.fit(X_train)

        X_trainval_scaled = scaler.transform(X_trainval)
        prefict_datafeatures_scaled = scaler.transform(prefict_datafeatures)

        best_score = 0
        for alphas in 10 ** np.linspace(-10, 10, 100):
            # Set a certain number of alphas
            lasso1 = Lasso(max_iter=10000, alpha=alphas)

            # Perform cross validation
            scores = cross_val_score(lasso1, X_trainval_scaled, y_trainval, cv=5)

            # Compute the mean score
            score = scores.mean()

            # If improvement, store score and parameter
            if score > best_score:
                best_score = score
                best_alphas = alphas

        # Build a model on the combine training and valiation data
        lasso1 = Lasso(max_iter=10000, alpha=best_alphas)
        lasso1.fit(X_trainval_scaled, y_trainval)

        agePrediction = lasso1.predict(prefict_datafeatures_scaled)
        agePrediction = agePrediction.astype(int)

        return agePrediction

    def GetBrokenYear(data):

        data['Possible broken year'] = now + data[data['Predict Age'] > data['Age']]['Predict Age'] - \
                                       data[data['Predict Age'] > data['Age']]['Age']
        return data['Possible broken year']

    # change age of the pipe age based on user's option
    def ChangeAge(data, year):
        year = int(year)
        data['Age'] = data['Age'] + year
        return data['Age']

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
        df = Models.GroundWater.ReadXYZ(fileName)

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

