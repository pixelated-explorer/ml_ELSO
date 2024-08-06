# load dataset and visualize things 

# %%
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 

# %%

elso_df = pd.read_csv('ELSOcleaner.csv')
elso_df.head()

# %%

# NOTE the outcomes that we can care about: 
    # Discontinuation, 127 
    # SurvECMO, 0
    # SurvHosp, 0
    # LOSdays, 152
    # VentDur, 0
    # NOTE: RRTduringECMO, 0
    # HoursECMO 0 
# we MAINLY care about RRTduringECMO (a binary value)

# other significant features; 
    # without missing values: 
        # Race, AgeYrs, CDH, Mode, SupType, pH, AKIpresent, CKD, 
        # CPR, CultPos, RRTpreECMO, HypoTerm, NMB, Pressors, Inotropes, 
        # totPress, PreRRT, PreECMOAKI
    
    # With missing values
        # Sex, 183
        # OI, 4965

    # similar features, just use AKIpresent for now and ignore the other two which are Yes or No. Not useful atm 
        # AKIpresent = PreRRt = PreECMOAKI
        # RRTpreECMO is similar to the above as well. 

print(elso_df.isna().sum())
print('---------')
print(elso_df.shape)

# %%
# a couple of interesting graphs to show quick linear relationships

feature_list = ['Race', 'AgeYrs', 'CDH', 'Mode', 'SupType', 'pH', 'AKIpresent', 'CKD', 'CPR', 'CultPos', 'RRTpreECMO', 'HypoTerm', 'NMB', 'Pressors', 'Inotropes', 'totPress', 'PreRRT', 'PreECMOAKI']

for i, feature in enumerate(feature_list):
    plt.figure(figsize=(8, 6))
    sns.lineplot(data = elso_df, x= feature, y='RRTduringECMO', marker='o')
    plt.title(f'Line plot of {feature} vs RRTduringECMO')
    plt.xlabel(feature)
    plt.grid(True)
    plt.show()

# %%


