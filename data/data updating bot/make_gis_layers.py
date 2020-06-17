import pandas as pd
import numpy as np
import sklearn as skl
from sklearn import preprocessing
import matplotlib.pyplot as plt

data=pd.read_csv('../merged_data.csv', dtype={'FIPS': object})

list(data.columns)

scores=pd.read_csv('../scores.csv', dtype={'FIPS': object})
#scores.head()


# ### Severe COVID Complications
severe=scores[['FIPS','State','County','severe_cases']].copy()
severe=severe.merge(data[['covid_cases','Years of Potential Life Lost Rate','% Fair or Poor Health',
                               '% Smokers','% Adults with Obesity','% 65 and over','% Adults with Diabetes'
                               ,'FIPS']], on='FIPS')
severe=severe.rename(columns={'severe_cases':'Severe COVID Case Complications'})
severe.to_csv('../severe_cases_score_data.csv')


# ### Economic Harm
economic=scores[['FIPS','State','County','economic']].copy()
economic=economic.merge(data[['% Uninsured','% Children in Poverty','Income Ratio','% Single-Parent Households',
                             '% Severe Housing Cost Burden','% Severe Housing Problems',
                              '% Enrolled in Free or Reduced Lunch','% Unemployed','FIPS',
                             'High School Graduation Rate']], on='FIPS')
economic=economic.rename(columns={'economic':'Risk of Severe Economic Harm'})
economic.to_csv('../economic_score_data.csv')


# ### Mobile Health
mobile=scores[['FIPS','State','County','mobile_health']].copy()
mobile=mobile.merge(data[['% Smokers','% Uninsured','% Adults with Obesity','% Adults with Diabetes',
                         '% 65 and over','Primary Care Physicians Rate','% Fair or Poor Health',
                          'internet_percent','FIPS']], on='FIPS')
mobile=mobile.rename(columns={'mobile_health':'Need for Mobile Health Resources','% Uninsured_y':'% Uninsured',
                             'internet_percent':'% Home Internet Access'})
#mobile.head()
mobile.to_csv('../mobile_health_score_data.csv')
