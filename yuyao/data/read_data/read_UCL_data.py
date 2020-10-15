import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from numbers import Number
import numpy as np


def read_UCL_data( file_path, header = None, standardization = True ):

    df = pd.read_csv(file_path, header = None)
    #change the value of the attributes to numberical values if is not
    enc = LabelEncoder()
    # drop the rows with missing value 
    df = df.dropna(how='any',axis=0) 
    for i in range (len(df.columns)):
        col = df.columns[i]
        if(isinstance(df[col].iloc[0], Number)):
            continue
        enc.fit(df[col])
        df[col] = enc.transform(df[col])

    if standardization == True:
        scaler = StandardScaler()
        scaled_np = scaler.fit_transform(df)
        scaled_df = pd.DataFrame(scaled_np)
        # the fit_transform will also standardize the label,then it is need to assign the label value back to original.
        scaled_df.iloc[:,-1] = df.iloc[:,-1]
        return scaled_df.values.tolist()
    else: 
        return df.values.tolist()
    