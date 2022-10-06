import pandas as pd
import matplotlib.pyplot as plt
from lifetimes.utils import summary_data_from_transaction_data
from lifetimes import BetaGeoFitter  # BG/NBD
from lifetimes import GammaGammaFitter  # Gamma-Gamma Model
from lifetimes.plotting import plot_frequency_recency_matrix
from pathlib import Path

# Inputs
DATA_CSV = 'data/processed/data.csv'

# Outputs
PLOT_DIR = 'plots/get_clv/'
FREQ_REC_MAT_PNG = PLOT_DIR + 'freq_rec_mat.png'

Path(PLOT_DIR).mkdir(parents=True, exist_ok=True)


#################
# Load data
#################

df = pd.read_csv(DATA_CSV, dtype={'CustomerID': str, 'InvoiceNo': str})


#################
# Get CLV
#################

clv = summary_data_from_transaction_data(df,
                                         'CustomerID',
                                         'InvoiceDate',
                                         'Total_Price',
                                         observation_period_end='2011-03-03')

print(clv.head())

# we want only customers shopped more than 1 times
clv = clv[clv['frequency'] > 0]


#################
# Fit BetaGeo
#################

bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(clv['frequency'], clv['recency'], clv['T'])

plt.figure(figsize=(12, 9))
plot_frequency_recency_matrix(bgf)
plt.savefig(FREQ_REC_MAT_PNG)


#################
# Fit GammaGamma
#################

t = 180  # 30 day period
clv['expected_purc_6_months'] =\
    bgf.conditional_expected_number_of_purchases_up_to_time(t,
                                                            clv['frequency'],
                                                            clv['recency'],
                                                            clv['T'])

print(clv.head())


clv = clv.loc[clv.expected_purc_6_months > 0]


# Check no correlation
print(clv[['frequency', 'monetary_value']].corr())

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(clv["frequency"], clv["monetary_value"])


clv['6_months_clv'] = ggf.customer_lifetime_value(bgf,
                                                  clv["frequency"],
                                                  clv["recency"],
                                                  clv["T"],
                                                  clv["monetary_value"],
                                                  time=6,
                                                  freq='D',
                                                  discount_rate=0.01)

clv.sort_values('6_months_clv', ascending=False)


###################
# Segment customers
###################

clv['Segment'] = pd.qcut(clv['6_months_clv'], 4,
                         labels=['Hibernating', 'Need Attention',
                                 'LoyalCustomers', 'Champions'])

print(clv.head(15))

print(clv.groupby('Segment').mean())
