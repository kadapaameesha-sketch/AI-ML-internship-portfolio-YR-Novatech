import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing

# Data Collection (>1000 samples)
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Missing Values Handling
df.loc[10:20, 'MedInc'] = np.nan
df['MedInc'] = df['MedInc'].fillna(df['MedInc'].median())

# Remove Duplicates
df = df.drop_duplicates()

# Data Type Conversion
df['MedHouseVal'] = df['MedHouseVal'].astype(float)

# Outlier Management (IQR)
Q1 = df['MedInc'].quantile(0.25)
Q3 = df['MedInc'].quantile(0.75)
IQR = Q3 - Q1
df['MedInc'] = np.clip(df['MedInc'], Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

# Feature Engineering
df['RoomsPerHousehold'] = df['AveRooms'] / df['AveOccup']

# Save Cleaned Dataset
df.to_csv('cleaned_california_housing.csv', index=False)
print("Data preprocessing completed successfully!")
