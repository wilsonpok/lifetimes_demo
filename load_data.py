import pandas as pd
import numpy as np
from pathlib import Path

# Inputs
DATA_ZIP = 'data/raw/ecommerce-data.zip'

# Outputs
OUT_DIR = 'data/processed/'
DATA_CSV = OUT_DIR + 'data.csv'

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)


################
# Load data
################

df = pd.read_csv(DATA_ZIP,
                 encoding='unicode_escape',
                 dtype={'CustomerID': str, 'InvoiceNo': str})

df.InvoiceDate = pd.to_datetime(df.InvoiceDate, dayfirst=True)

print(df.info())

print(df.describe)

print(df.shape)


################
# Clean data
################

# exclude the orders with 0 value
df = df[df['Quantity'] > 0]

# exclude the Unit Price with 0 value
df = df[df['UnitPrice'] > 0]

# C indicates the returned orders we don't want them as well
df = df[~df['InvoiceNo'].str.contains("C", na=False)]

# Drop nulls
df.dropna(inplace=True)

print(df.shape)


#################
# Remove outliers
#################

def find_boundaries(df, variable, q1=0.05, q2=0.95):
    # the boundaries are the quantiles
    lower_boundary = df[variable].quantile(q1)  # lower quantile
    upper_boundary = df[variable].quantile(q2)  # upper quantile
    return upper_boundary, lower_boundary


def capping_outliers(df, variable):
    upper_boundary, lower_boundary = find_boundaries(df, variable)
    df[variable] = np.where(df[variable] > upper_boundary, upper_boundary,
                            np.where(df[variable] < lower_boundary,
                                     lower_boundary, df[variable]))


capping_outliers(df, 'UnitPrice')
capping_outliers(df, 'Quantity')

print(df.describe())

df['Total_Price'] = df['UnitPrice'] * df['Quantity']

print(df['InvoiceDate'].min())
print(df['InvoiceDate'].max())


#################
# Output
#################

df.to_csv(DATA_CSV, index=False)
