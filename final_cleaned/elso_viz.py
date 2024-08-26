# %%

import pandas as pd
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import seaborn as sns 

# %%

# NOTE: will eventually do some sensitivity analysis to see the effect of deleting patients on our modeling. 

elsoDF = pd.read_csv('elsoOI.csv')

elsoDF.head()

# %%

elsoDF = elsoDF.loc[elsoDF['Out'] == 0]

elsoDF.describe()

# %%
# NOTE: 
# AgeGr -> label encoding. 
# Sex -> one hot 
# Race -> one hot encoding OR frequency encoding 
    # look into using frequency then 
# Mode -> one hot encoding 
# SupType -> one hot encoding, splitting into 3 columns  
# PreECMOAKI -> binary  
# .... 
# need to normalize totPress, probably use a minmaxscaler()
# main outcome is RRTduringECMO 
    # note that the set is imbalanced as outcome is 30 %.

elsoDF = elsoDF.drop(columns=['ID'])

scaler = StandardScaler()
elsoDF['totPress'] = scaler.fit_transform(elsoDF[['totPress']])


label_encoder = LabelEncoder()
elsoDF['AgeGr'] = label_encoder.fit_transform(elsoDF['AgeGr'])

mapping = {'Yes' : 1, 'No' : 0}
elsoDF['PreECMOAKI'] = elsoDF['PreECMOAKI'].map(mapping)

# One-hot encoding
selected_columns = ['Sex', 'Race', 'Mode', 'SupType']
temp_df = pd.get_dummies(elsoDF[selected_columns], dtype='int64')

# %%
elsoDF = pd.concat([elsoDF, temp_df], axis=1)

# If you want to drop the original columns that were one-hot encoded, you can do so
elsoDF = elsoDF.drop(columns=selected_columns)


# %%

elsoDF.describe()

# %%

cov_matrix = elsoDF.cov()

print(cov_matrix)

# %%




df_important = elsoDF.drop(columns=['iNO', 'HypoTerm', 'NMB', 'Plasmaph', 'CPR', 'CultPos', 'Outcome', 'LOSdays', 'VentDur', 'Hem', 'OIout', 'pHout', 'SBPout', 'Out'])
# sns.heatmap(df_important, annot=True, cmap='coolwarm')


# %%

df_important.to_csv('elsoOI_tempML.csv')

elsoDF.to_csv('elsoDF_encoded_total.csv')