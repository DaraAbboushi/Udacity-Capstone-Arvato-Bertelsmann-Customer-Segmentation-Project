#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries here; add more as necessary

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

                        
def check_value(x):
    '''
    Input: x-is the value in the column
    
    Purpose: converting string to float type or 
    replace string 'X' and 'XX' with NaN or 
    if it's already float return it the same
    
    Output: x- is either float or nan value
    
    '''
    if type(x) == float:
        return x
    elif x == 'X' or (x == 'XX'):
        return np.nan
    else:
        return float(x)


def clean_data(df):
    '''
    Cleaning and performing feature extracting and engineering to the dataframe
    
    Input: df (DataFrame)
    
    Output:
        df1 (DataFrame): cleaned df DataFrame
    '''
    #for cleaning the data there are many steps:
    #drops columns with more than 20% of missing values
    print('A) The First Step: Dropping the unnecessary columns. \n')
    print('The columns that were successfully dropped are: \n 1) Columns that have more than 20% of missing values. \n 2) Columns that have no description in the files. \n 3) Columns that are noisy because it contains many different items. \n 4) The ID Column.')

    
    df.drop(['ALTER_KIND1','ALTER_KIND2','ALTER_KIND3','ALTER_KIND4','EXTSEL992','KK_KUNDENTYP',
                     'RT_KEIN_ANREIZ','CJT_TYP_6','D19_VERSI_ONLINE_QUOTE_12','CJT_TYP_2','EINGEZOGENAM_HH_JAHR',
                     'D19_LOTTO','CJT_KATALOGNUTZER','VK_ZG11','UMFELD_ALT','RT_SCHNAEPPCHEN','AGER_TYP', 'ALTER_HH', 
                     'D19_BANKEN_ONLINE_QUOTE_12','D19_GESAMT_ONLINE_QUOTE_12', 'D19_KONSUMTYP','D19_VERSAND_ONLINE_QUOTE_12',
                     'GEBURTSJAHR','KBA05_BAUMAX','TITEL_KZ','D19_BANKEN_DATUM', 'D19_BANKEN_OFFLINE_DATUM',                                        'D19_BANKEN_ONLINE_DATUM', 
                     'D19_GESAMT_DATUM',  'D19_GESAMT_OFFLINE_DATUM', 'D19_GESAMT_ONLINE_DATUM','D19_TELKO_DATUM', 
                     'D19_TELKO_OFFLINE_DATUM','D19_TELKO_ONLINE_DATUM',  'D19_VERSAND_DATUM', 'D19_VERSAND_OFFLINE_DATUM', 
                     'D19_VERSAND_ONLINE_DATUM', 'D19_VERSI_DATUM', 'D19_VERSI_OFFLINE_DATUM','D19_VERSI_ONLINE_DATUM',
                     'CAMEO_DEU_2015', 'CAMEO_INTL_2015', 'LP_FAMILIE_FEIN', 'LP_STATUS_FEIN',  'ANREDE_KZ', 
                     'GREEN_AVANTGARDE',  'SOHO_KZ', 'VERS_TYP',  'LP_LEBENSPHASE_GROB',
                     'LP_LEBENSPHASE_FEIN','EINGEFUEGT_AM','D19_LETZTER_KAUF_BRANCHE',
                     'PRAEGENDE_JUGENDJAHRE','PLZ8_BAUMAX'], axis=1,inplace=True)
    
    
    df1 = df.copy()
    print("Creating a copy of the dataframe after dropping columns. \n")
    
    try:
        df1.drop(['PRODUCT_GROUP','CUSTOMER_GROUP','ONLINE_PURCHASE'], axis=1,inplace=True)
    except:
        pass
    
    
    #replace O with 0 and W with 1
    print("B) The Second Step: Replace O with 0 and W with 1 in OST_WEST_KZ attribute. \n")
    
    df1['OST_WEST_KZ'].replace(['O','W'], [0, 1], inplace=True)
    
    print('C) The Third Step: Rename the WOHNLAGE column to TYPE_RESIDENTIAL_AREA. \n')
    df1['TYPE_RESIDENTIAL_AREA'] = df1['WOHNLAGE']
    print('D) The Fourth Step: apply feature engineering on TYPE_RESIDENTIAL_AREA values. It is better to reduce the noise. \n')
    
    df1['TYPE_RESIDENTIAL_AREA'].replace([-1,0,1,2,3,4,5,7,8], [np.nan,np.nan,0,0,2,2,0,1,1], inplace=True)
    
    print('E) The Fifth Step: Dropping the original column: WOHNLAGE. \n')
    
    df1.drop(['WOHNLAGE'], axis=1, inplace=True)
    
    
    
    #change object type of CAMEO_DEUG_2015 to numeric type
    print("Feature extracting CAMEO_DEUG_2015. \n")
    print('F) The Sixth Step: Checking the values of the object type column: CAMEO_DEUG_2015 and transforming to a numeric type. \n')
    
    df1['CAMEO_DEUG_2015'] = df1['CAMEO_DEUG_2015'].apply(lambda x: check_value(x))
    
    print("G) The Seventh Step: removing columns with start with KBA05. \n ")
    
    KBA05_list = []
    for f in df1.columns:
        if 'KBA05' in f:
            KBA05_list.append(f)
       
    df1.drop(KBA05_list, axis=1, inplace=True)

    
    #name of column of that contains XX string, to check for any, we didn't put it above with other function because that column had numbers and we wanted to convert it to a numeric type
    for i in df1.columns:
        df1[i].astype('str').apply(lambda x: print(df1[i].name) if x.startswith('XX') else 'pass')

    #imputing nan values 
    print("H) The Eighth Step: Imputing Nan values using the most frequent number.")
    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    df1[df1.columns] = imputer.fit_transform(df1)

          
    return df1
    
  




