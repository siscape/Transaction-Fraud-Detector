# #############################################################################
## INSTRUCTIONS
# keep everything in the same folder and change INPUTS 

# #############################################################################
## INPUTS

# folder file path of 'big computer' folder
folder_path = 'C:\\Users\\cfield001\\Documents\\GSA\\compression algorithm\\big computer\\'

# input data filename
input_data_csv = 'sample4_GSA.csv'

# results output filename
output_data_csv = 'sample4_GSA_results.csv'



# input data format (must be: sample5, sample4, or IF_output)
input_data_format = 'sample4' 

# #############################################################################
## IMPORT
import numpy as np
import pandas as pd
import sys
import datetime as dt

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# start timer
start = dt.datetime.now()
# #############################################################################
## LOAD DATA and DROP COLUMNS

# load data as df
df = pd.read_csv(folder_path + input_data_csv, dtype=str, encoding='cp1252')

#gsa = df[df['AGENCY_NAME'] == "General Services Administration"]
#gsa.to_csv(folder_path + 'sample4_GSA.csv', index=True)

# cleaning data based on input format
if input_data_format == 'sample5':
    # drop negative transactions
    df = df[df['TRANSACTION_AMOUNT'].astype(float) > 0]
    # clean zipcode to only have 5 digits
    df['ZIP_CODE'] = df['ZIP_CODE'].str[:5]
    df['Merchant_Zip'] = df['Merchant_Zip'].str[:5] 
    # list of columns to drop
    dropcol = ['ACCOUNTID',  
               'BANKNUMBERID',
               'Merchant_Name',
               'TRANSACTION_STATUS',
               'TRANSACTION_STATUS_2',
               'TRANSACTION_STATUS_3',
               'TRANSACTION_TYPE_2',
               'Date_Opened', 
               'AGENCY_CATEGORY',
               'SALES_TAX_FLAG', 
               'TICKET_NUMBER',
               'TRANSACTION_AMOUNT']
elif input_data_format == 'IF_output':  
    # list of columns to drop
    dropcol = ['Anomaly', 
               'Anomaly Score']
elif input_data_format == 'sample4':  
    # drop negative transactions
    df = df[df['TRANSACTION_AMOUNT'].astype(float) > 0]
    # clean zipcode to only have 5 digits - dropped, ignore
    #df['ZIP_CODE'] = df['ZIP_CODE'].str[:5]
    #df['Merchant_Zip'] = df['Merchant_Zip'].str[:5]
    # Change date format for Transaction Date ID and pull out Month and Day 
    df['TRANSACTIONDATEID'] = pd.to_datetime(df['TRANSACTIONDATEID'].astype(str), format='%Y%m%d')    
    # creating the month variable off of the transaction date id 
    df['TRANSMONTH'] = df['TRANSACTIONDATEID'].dt.month.astype(str)
    # creating the day of the week variable off of the transaction date id 
    df['TRANSDAYOFWEEK'] = df['TRANSACTIONDATEID'].dt.dayofweek.astype(str)
    # list of columns to drop
    dropcol = ['ACCOUNTID', 
               'TRANSACTIONDATEID',
               'CONVERSION_RATE', 
               'BANKNUMBERID', 
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
               # in kimmy's code, but not the sample4 dataset 
               #'TRANSPOSTINGDATEID', 
               #'ZIP_CODE',

# reformat transaction id so that it matches format used in compression (string with .0 removed)
df['TRANSACTIONID'] = df['TRANSACTIONID'].str.replace('\.0', '', regex=True)
# set transaction ID as index
df.set_index('TRANSACTIONID', drop=True, inplace=True)
# drop columns from dataframe 
df.drop(columns=dropcol, inplace=True)

# print update and seconds elapsed
print('Load data and Drop columns - {}'.format(str(dt.timedelta(seconds=(dt.datetime.now() - start).seconds))))

# #############################################################################
## FORMAT DATA FOR COMPRESSION

# convert all data to strings
df = df.astype(str)
# reformat column values to be 'Column_Name: value'
for c in list(df):
    df[c] = c + ': ' + df[c]
# create new variable row_str which concatenates the entire row of reformatted values as one string
def row_str(dframe):
    out_str = ''
    for c in list(dframe):
        out_str += dframe[c] + ','
    return out_str
df['row_str'] = row_str(df)
# encode the row_str variable
df['row_enc'] = df['row_str'].str.encode(encoding="utf-8")

# print update and seconds elapsed
print('format data for compression - {}'.format(str(dt.timedelta(seconds=(dt.datetime.now() - start).seconds))))

# #############################################################################
## COMPRESS

# import compression algorithm from gzip
sys.path.append(folder_path)
from gzip import compress

# vectorize algorithm
comp_vec = np.vectorize(compress)
# compress each row as new variable comp_str
df['comp_str'] = comp_vec(df['row_enc'])
# get length of compressed row *** value used for identifying outliers
df['comp_len'] = df['comp_str'].astype(str).str.len()

# VISUALIZE
# create boxplot of compression lengths
comp_box = sns.boxplot(df['comp_len'])
comp_box.figure.savefig(folder_path + 'comp_length_boxplot.png')
plt.clf()
'''
sns.boxplot(df['comp_len']) 
plt.show()
'''



# print update and seconds elapsed
print('compress - {}'.format(str(dt.timedelta(seconds=(dt.datetime.now() - start).seconds))))

# #############################################################################
## IDENTIFY OUTLIERS

# get z_scores for lengths of compressed rows 
df['z'] = stats.zscore(df['comp_len'])
# get absolute values of z_scores
df['z_abs'] = np.abs(df['z'])

# VISUALIZE
# create plot of distribution of z_scores
z_dist = sns.distplot(df['z']) 
z_dist.figure.savefig(folder_path + 'z_score_distribution.png')
plt.clf()
'''
sns.distplot(df['z']) 
plt.show()
'''


# z_score to use as min for outliers
z_score = 2.5
# create outlier column (1 = outlier, 0 = not) 
df['outlier'] = (df['z'] > z_score).astype('int')
# print number of outliers
df['outlier'].sum()

# commented results for different z_scores
#14,628 - number of anomolies from isolation forest 
# z > 2.5 => 5,189
# z > 2.43 => 10,576
# z > 2.425 => 15,668
# z > 2.4 => 19,257
# z > 2 => 18,112


# print update and seconds elapsed
print('identify outliers - {}'.format(str(dt.timedelta(seconds=(dt.datetime.now() - start).seconds))))


# output full dataframe to csv if uncommented
df.to_csv(folder_path + output_data_csv, index=True)
# print update and seconds elapsed
print('print csv - {}'.format(str(dt.timedelta(seconds=(dt.datetime.now() - start).seconds))))


'''
# #############################################################################
## COMPARE TO ISOLATION FOREST 

# import results from isolation forest
df_IF = pd.read_csv(folder_path + 'gsa_sample_results_not_encoded.csv', usecols=['TRANSACTIONID', 'Anomaly'], dtype=str)[['TRANSACTIONID', 'Anomaly']]
# reformat transaction id so that it matches format used in compression (string with .0 removed)
df_IF['TRANSACTIONID'] = df_IF['TRANSACTIONID'].str.replace('\.0', '', regex=True)
# set transaction id as index for dataframe
df_IF.set_index('TRANSACTIONID', drop=True, inplace=True)

# create list of transaction IDs that are outliers from: 
comp_list = df.index[df['outlier']==1].tolist() #compression
IF_list = df_IF.index[df_IF['Anomaly']=='-1'].tolist() #isolation forest

# listr of transaction IDs both methods identified as outliers
same_list = list(set(comp_list).intersection(IF_list))
# print number of transaction IDs both methods identified as outliers
len(same_list)

# commented results for different z_scores and percent of comp results that are anomolies in isolation forest
# 2.4 => 3,283
# 2.5 => 2,948 => 57%
# 3 => 1,715 => 53% 
# 2 => 4,331 => 24% 


# print update and seconds elapsed
print('compared to isolation forest - {}'.format(str(dt.timedelta(seconds=(dt.datetime.now() - start).seconds))))



## VISUALIZAIONS
# reformat isolation anomoly column (1 = anomoly, 0 = not)
df_IF['IF_Anomaly'] = (df_IF['Anomaly']=='-1').astype('int')
# add isolation forest anomolies to compression df
df = df.join(df_IF['IF_Anomaly'])


df2 = df.head(1000)
df2['IF_Anomaly'].sum()

df3 = df[df['IF_Anomaly']==1]


# Visualize each transaction compression length z_score with color showing if anomoly using isolation forest 
sns.swarmplot(y='z', x='outlier', hue='IF_Anomaly', data=df, dodge=False)
plt.legend()
plt.show()

sns.boxplot(x=df['IF_Anomaly'], y=df['z_abs'], showmeans=True)
plt.show()


sns.distplot(df3['z']) 
plt.show()


# output full dataframe to csv if uncommented
#df.to_csv(folder_path + 'comp.csv', index=True)

#df_old = df.copy(deep=True)



# import results from isolation forest
df_IF2 = pd.read_csv(folder_path + 'gsa_sample_results_not_encoded_5percent.csv', usecols=['TRANSACTIONID', 'Anomaly', 'Anomaly Score'], dtype=str)[['TRANSACTIONID', 'Anomaly', 'Anomaly Score']]
df_IF2['Anomaly Score'] = df_IF2['Anomaly Score'].astype(float)
# reformat transaction id so that it matches format used in compression (string with .0 removed)
df_IF2['TRANSACTIONID'] = df_IF2['TRANSACTIONID'].str.replace('\.0', '', regex=True)
# set transaction id as index for dataframe
df_IF2.set_index('TRANSACTIONID', drop=True, inplace=True)

# create list of transaction IDs that are outliers from: 
comp_list = df.index[df['outlier']==1].tolist() #compression
IF_list2 = df_IF2.index[df_IF2['Anomaly']=='-1'].tolist() #isolation forest

# listr of transaction IDs both methods identified as outliers
same_list2 = list(set(comp_list).intersection(IF_list2))
# print number of transaction IDs both methods identified as outliers
len(same_list2)

# reformat isolation anomoly column (1 = anomoly, 0 = not)
df_IF2['IF_Anomaly'] = (df_IF2['Anomaly']=='-1').astype('int')
# add isolation forest anomolies to compression df
df = df.join(df_IF2[['IF_Anomaly', 'Anomaly Score']])

df5 = df[df['IF_Anomaly2']==1]

sns.distplot(df5['z'], groupby=df5['IF_Anomaly2']) 
plt.show()

sns.distplot(df['z'][df['IF_Anomaly']==1]) 
plt.show()


# TOP 100
# biggest z-scores
comp_top = df.nlargest(1000, columns='z').index.tolist() #compression
IF5_top = df.nsmallest(1000, columns='Anomaly Score').index.tolist() #IF5

same_top = list(set(comp_top).intersection(IF5_top))
len(same_top)


df.drop(columns=['IF_Anomaly', 'Anomaly', 'Anomaly Score'], inplace=True)


sns.distplot(df['z'][IF5_top]) 
plt.show()

sns.distplot(df['Anomaly Score'][comp_top]) 
plt.show()

'''