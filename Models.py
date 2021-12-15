

import numpy as np
import pandas as pd
import warnings;

class Pipe:
    def __init__(self):
        self.id = self.GetID()
        self.data_features,self.data_target = self.GetFeaturesAndTarget_Age()

    def GetID(self):
        return self.id

    def GetFeaturesAndTarget_Age(data):
        columns_to_be_removed = ['Age', 'ID', 'DatoOprett', 'DatoOpdate']
        data_features = data.drop(columns_to_be_removed, axis='columns')
        columns_to_be_removed = ['fra_kote', 'til_kote', 'Laengde', 'Fald', 'DiameterIn', 'MaterialeK', 'anlag_aar',
                                 'TransportK',
                                 'Funktionsk', 'TVObsKode', 'DatoSaneri', 'PipeStatus', 'ID', 'XKoordinat',
                                 'YKoordinat', 'Depth', 'DatoOprett', 'DatoOpdate']
        data_target = data.drop(columns_to_be_removed, axis='columns')
        return data_features, data_target

    def GetFeaturesAndTarget(datacopy):
        columns_to_be_removed = ['PipeStatus', 'ID', 'TransportK', 'Funktionsk', 'MaterialeK',
                                 'DatoSaneri']
        data_features = datacopy.drop(columns_to_be_removed, axis='columns')

        columns_to_be_removed = ['fra_kote', 'til_kote', 'Laengde', 'Fald', 'DiameterIn', 'MaterialeK', 'anlag_aar',
                                 'TransportK', 'Funktionsk', 'DatoSaneri', 'Age', 'Depth', 'TVObsKode', 'YKoordinat',
                                 'XKoordinat', 'ID', 'DatoOprett', 'DatoOpdate']

        data_target = datacopy.drop(columns_to_be_removed, axis='columns')
        return data_features, data_target

class GroundWater:
    def __init__(self):
        self.df = self.ReadXYZ()


    def ReadXYZ(fileName):
        xyz_coordinates = []  # put xyz in an array

        with open(fileName, "r") as file:
            for line_number, line in enumerate(file):
                x, y, z = line.split()

                xyz_coordinates.append([int(x), int(y), float(z)])

        my_array = np.array(xyz_coordinates)

        df = pd.DataFrame(my_array, columns=['XKoordinat', 'YKoordinat', 'Depth'])
        return df

