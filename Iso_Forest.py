# -*- coding: utf-8 -*-
"""
# Isolation Forest: GSA SmartPay 
# Purpose: Run Isolation Forest on transaction-level dataset 
# Date Updated: 2018 December 28 
# Author: Sri Iyer

""" 

"""
# setting options 
pd.set_eng_float_format(accuracy=3, use_eng_prefix=True)
np.set_printoptions(suppress = True)
pd.reset_option("display.float_format")

"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Read in the sample dataset (~10 million rows) 
fulldata = pd.read_csv(filepath_or_buffer = "C:/Users/kimda/Documents/Isolation Forest/sample4.csv", header = 0, sep = ',' , encoding='cp1252')
"""
# utf-8 encoding did not work - had to switch to 'cp1252' 
"""

# investigate input variables 
info = fulldata.describe(include='all')
# list(fulldata)

# Transform variables from numeric to categorical 
changes = ['TRANSACTIONID',  
           'TRANSACTIONDATEID', 
           'BANKNUMBERID', 
           'SIC_MCC_CODE', 
           'CorporateMerchantID', 
           'Merchant_Zip', 
           'ACCOUNTID', 
           'COUNTRYID']
for col in changes: 
    fulldata[col] = fulldata[col].astype('object')
## ------------------------------------------------------------------ ##
    
# THINGS TO DO TO CLEAN / BIN THE DATA 

## ---------------------Negative Amounts ---------------------------- ##

# Remove all transactions that have a negative transaction amount 
fulldata2= fulldata[fulldata['TRANSACTION_AMOUNT'] > 0 ]  
list(fulldata2)

## --------------------------Location ------------------------------ ##

# need to remove last 4 digits of zip code transactions 
fulldata2['ZIP_CODE'] = fulldata2['ZIP_CODE'].astype(str).str[:5]
# info2 = fulldata2.describe(include='all')

# read in the data frame to map zip code to state 
# data set is from: http://federalgovernmentzipcodes.us/ 
location = pd.read_csv("C:/Users/kimda/Documents/free-zipcode-database-Primary.csv")
location = location[['Zipcode', 'State']] 
fulldata2.merge( location, how = 'left', left_on = 'ZIP_CODE', right_on = 'Zipcode')

## -------------------------- Dates--------------------------------- ##
  
# Change date format for Date Opened 
pd.to_datetime(str(fulldata2['Date_Opened']), format='%Y%m%d')

# Change date format for Transaction Date ID and pull out Month and Day 
fulldata2['TRANSACTIONDATEID'] = pd.to_datetime(fulldata2['TRANSACTIONDATEID'].astype(str), format='%Y%m%d')
fulldata2.TRANSACTIONDATEID.dtype

# creating the month variable off of the transaction date id 
fulldata2['TRANSMONTH'] = fulldata2['TRANSACTIONDATEID'].dt.month 

# creating the day of the week variable off of the transaction date id 
fulldata2['TRANSDAYOFWEEK'] = fulldata2['TRANSACTIONDATEID'].dt.dayofweek 

# Transform variables from numeric to categorical 
changes = ['TRANSMONTH', 
           'TRANSDAYOFWEEK']

for col in changes: 
    fulldata2[col] = fulldata2[col].astype('object')

## ----------------- Data Investigation --------------------------- ##

#check distributions on sales tax (a couple of really high values)  
fulldata.hist(column = "SALES_TAX", bins = 25)

# check how many account ID's have less than 10 transactions 
lowtrx = fulldata.groupby(['ACCOUNTID']).size()
lowtrx.value_counts([lowtrx < 10])

''' 
Test with a single agency
''' 

## ------------------------------------------------------------------ ##
    
# Starting with GSA for analysis / POC 

## ------------------------------------------------------------------ ##

print(fulldata['AGENCY_NAME'].unique()) 

gsa = fulldata2.loc[fulldata['AGENCY_NAME'] == "General Services Administration"]
gsa.set_index(['TRANSACTIONID'], inplace = True)

# hhs = fulldata.loc[fulldata['AGENCY_NAME'] == "Department of Health and Human Services"]
# hhs.to_csv("C:/Users/kimda/Documents/hhs.csv")

# select - GSA and HHS to start 
list(gsa)

gsa_info = gsa.describe(include = "all")

list(gsa_model)

## --------------------------Columns to drop --------------------------------- ##

dropcol = ['ACCOUNTID', 
           'TRANSACTIONDATEID',
           'TRANSPOSTINGDATEID',
           'CONVERSION_RATE', 
           'BANKNUMBERID', 
           'ZIP_CODE', 
           'Merchant_Zip',
           'Merchant_Name',
           'MERCHANT_GROUP_CATEGORY', 
           'MERCHANT_MAP_CODE',
           'Merchant_ID_Number', 
           'CorporateMerchantID', 
           'TRANSACTIONDATEID',
           'TRANSACTION_STATUS',
           'TRANSACTION_STATUS_2',
           'TRANSACTION_STATUS_3',
           'TRANSACTION_TYPE_2',
           'Date_Opened', 
           'Date_Inactive',
           'AGENCY_CATEGORY',
           'SALES_TAX_FLAG', 
           'TICKET_NUMBER',
           'SALES_TAX', 
           'RATE_PERCENT', 
           'AGENCY_NAME', 
           'CONV_CHECK', 
           'MEMO_FLAG',
           'CORPORATE_NAME',
           'TRANSFER_TRANSACTION_FLAG']

gsa_model = gsa.drop(labels = dropcol, axis=1)
# gsa_model = gsa_model.sample(n=10000)

gsa_info2 = gsa_model.describe(include = "all")
gsa_info2.sum(axis = 1)['unique']


## ------------------------ Scale continuous variables ------------------- ##


# normalize the continuous variables using the Robust Scaler option as recommend for anomaly detection 
list(gsa_model)
contcol = ['TRANSACTION_AMOUNT',
           'SALES_TAX', 
           'RATE_PERCENT'] 

for col in contcol: 
    gsa_model[col] = gsa_model[col].astype('float')
    fit = RobustScaler().fit(gsa_model[col].values.reshape(-1,1))
    gsa_model[col] = fit.transform(gsa_model[col].values.reshape(-1,1))
    gsa_model[col] = gsa_model[col].abs()
    
gsa_info3 = gsa_model.describe(include = "all")

## -------------------------- One Hot Encoding ----------------------------- ##


# One hot encoding of categorical variables 
list(gsa)
cols = ['AUTHORIZATION_REQUIRED',
        'TERMINAL_ENTRY_MODE',
        'MINORITY_CODE',
        'SIC_MCC_CODE',
        'Account_Category',
        'Account_Type',
        'DEBIT_CREDIT_INDICATOR',
        'TRANS_CODE_TYPE',
        'CARD_TYPE_NAME',
        'COUNTRYID',
        'AGENCY_BUREAU_NAME',
        'TRANS_TYPE_DESC',
        'BANKID',
        'CURRENCY', 
        'TRANSMONTH', 
        'TRANSDAYOFWEEK']

# gsa_model.dtypes

gsa_ohc = pd.get_dummies(data = gsa_model, columns = cols, dummy_na = False, sparse = True, drop_first = True, dtype=np.float32)
# initial drops not sufficient ran with ~1.2 million columns - need to drop / bin additional columns

#validate that the sum of the unique values of the one hot encoded variables matches total number of columns 
gsa_info3[cols].sum(axis = 1)['unique']

gsa_info4 = gsa_ohc.describe(include = 'all')

## -------------------------- Modeling ------------------------------------ ##

# running the Isolation Forest 
seed = 5644
clf = IsolationForest(n_estimators = 100, max_samples=10, random_state=seed, contamination= .05, behaviour = 'new', n_jobs = -1)
model_full = clf.fit(gsa_ohc)
# start time was ~2:04 PM 

y_pred = clf.predict(gsa_ohc)
unique, counts = np.unique(y_pred, return_counts=True)
dict(zip(unique, counts))
# 157 identified for the 10,000 sample from GSA (1.57%)
 # 12,815 for the whole 10% sample at 735K rows(1.74%)
 # 14,628 run with date / additional changes - on 731K rows (minus the negative trx) (2.04%)
anomalyscore = clf.decision_function(gsa_ohc)
 
# gsa_full_predictions = clf.fit_predict(gsa_ohc)
gsa_ohc["Anomaly"] = y_pred
gsa_ohc["Anomaly Score"] = anomalyscore

# join the results to original gsa file for analysis 
gsa_full = pd.merge(gsa, gsa_ohc[['Anomaly', 'Anomaly Score']], how = 'left', left_index = True, right_index=True)
#write out files (one encoded one not to csv)
gsa_ohc.to_csv("C:/Users/kimda/Documents/gsa_sample_results.csv")
gsa_full.to_csv("C:/Users/kimda/Documents/gsa_sample_results_not_encoded_5percent.csv")

 ## ------------------------------------------------------------------------##

#                             VISUALIZATIONS 
 
 ## ------------------------------------------------------------------------##

# run pca for visualization purposes on gsa_ohc 
gsa_ohc = gsa_ohc.drop(labels = ['Anomaly', 'Anomaly Score'], axis=1)
pca = PCA(n_components = 2)
pca.fit(gsa_ohc)
pca_gsa = pca.transform(gsa_ohc)
scale = StandardScaler()
gsaviz = pd.DataFrame(data = pca_gsa, index = gsa_model.index.copy(), columns = ['pca1', 'pca2'])
gsaviz['pca_1'] = gsaviz['pca1'].abs() / gsaviz['pca1'].abs().max()
gsaviz['pca_2'] = gsaviz['pca2'].abs() / gsaviz['pca2'].abs().max()

gsaviz = pd.merge(gsaviz, gsa_full[['Anomaly']], how = 'left', left_index = True, right_index=True)

gsa_anomaly = gsaviz[gsaviz['Anomaly'] == -1]
gsa_notanomaly = gsaviz[gsaviz['Anomaly'] == 1]
gsaviz.to_csv("C:/Users/kimda/Documents/gsa_pca.csv")

p1 = plt.scatter(gsa_anomaly['pca_1'], gsa_anomaly['pca_2'], c = 'red', marker = 'x', s = 1,  cmap= 'Blues')
p2 = plt.scatter(gsa_notanomaly['pca_1'], gsa_notanomaly['pca_2'], c = 'green', marker = '.', s = 1, cmap= 'Blues')
plt.savefig('C:/Users/kimda/Documents/Isolation Forest/test.png')
plt.show()


gsa_full['Anomaly Score'].max()