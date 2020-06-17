import pandas as pd
import numpy as np
import sklearn as skl
from sklearn import preprocessing
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)

#read in data
data=pd.read_csv('../merged_data.csv', dtype={'FIPS': object})
data=data.drop(data.columns[0], axis=1)
#data.head()

# make a function to create quantiles of columns

colname = lambda col, suffix: '{}_{}'.format(suffix, col)
def add_quantiles(data, columns, suffix, quantiles=4, labels=None):
    """ 
    For each column name in columns, create a new categorical column with
        the same name as colum, with the suffix specified added, that
        specifies the quantile of the row in the original column using
        pandas.qcut().

        Input
        _____
        data:
            pandas dataframe
            columns:
            list of column names in `data` for which this function is to create
            quantiles
        suffix:
        string suffix for new column names ({`suffix`}_{collumn_name})
        labels:
            list of labels for each quantile (should have length equal to `quantiles`)
        Output
          ______
        pandas dataframe containing original columns, plus new columns with quantile
        information for specified columns.
    """
    d = data.copy()
    quantile_labels = labels or [
        '{:.0f}%'.format(i*100/quantiles) for i in range(1, quantiles+1)
    ]
    for column in columns:
        d[colname(column, suffix)] = d[column].rank(pct=True)
        #pd.qcut(d[column], quantiles, labels=quantile_labels)
    return d


cols=['Years of Potential Life Lost Rate', '% Fair or Poor Health', 'Average Number of Physically Unhealthy Days',
      'Average Number of Mentally Unhealthy Days', '% Low Birthweight', '% Smokers', '% Adults with Obesity',
      'Food Environment Index', '% Physically Inactive', '% With Access to Exercise Opportunities',
      '% Excessive Drinking', '% Driving Deaths with Alcohol Involvement', 'Chlamydia Rate', 'Teen Birth Rate',
      'Primary Care Physicians Rate','Dentist Rate','% With Annual Mammogram','% Vaccinated',
      'High School Graduation Rate', '% Some College','% Unemployed','% Children in Poverty','Income Ratio',
      '% Single-Parent Households','Social Association Rate','Violent Crime Rate','Injury Death Rate',
      'Average Daily PM2.5','% Severe Housing Problems','% Drive Alone to Work','% Long Commute - Drives Alone',
      'Life Expectancy','Age-Adjusted Death Rate','Child Mortality Rate','Infant Mortality Rate',
      '% Frequent Mental Distress','% Adults with Diabetes','HIV Prevalence Rate','% Food Insecure',
      '% Limited Access to Healthy Foods','Drug Overdose Mortality Rate','Motor Vehicle Mortality Rate',
      '% Insufficient Sleep','% Uninsured','% Children Uninsured','% Disconnected Youth','Average Reading Performance', 
      'Average Math Performance','Median Household Income','% Enrolled in Free or Reduced Lunch',
      'Black/White Segregation Index','non-White/White Segregation Index','Homicide Rate','Suicide Rate (Age-Adjusted)',
      'Firearm Fatalities Rate','Juvenile Arrest Rate','% Homeowners','% Severe Housing Cost Burden',
      'Population','% less than 18 years of age',
      '% 65 and over','% Not Proficient in English',
      'internet_consumer','internet_all','internet_hhs','internet_ratio',
      'covid_cases']
data_q=add_quantiles(data, cols, 'q', 10)


# **notes:** need to add functionality to include covid deaths (look at nan values)

# ## Create scores
# All scores are on a scale of 1-100 based on county's percentile ranking for a variety of factors. Below, we define several sample scores for different use cases. The user could also build their own score calculator. 
# make a new dataframe to store our scores
scores=data[['FIPS','State','County']].copy()
#scores.head()


# **COVID vulnerability severe cases**: cases, life lost rate, fair or poor health, smokers, obesity, diabetes, %65 and older, 

# describes likelihood that constituents develop severe complications from covid
scores['severe_cases']=((3*data_q['q_covid_cases']+data_q['q_Years of Potential Life Lost Rate']+
                        data_q['q_% Fair or Poor Health']+data_q['q_% Smokers']+
                         data_q['q_% Adults with Obesity']+data_q['q_% Adults with Diabetes']+
                        data_q['q_% 65 and over'])/9)*100

#scores.head()

# **overall poor physical health**: 
# **COVID vulnerability: health system**: uninsured, provider rates

# **Health system access**: mamomgrams, uninsured, provider rates, vaccinations, 

# **Food services**: free lunch, food insecure, food environment index, limited access to health foods, enrolled in free lunch

# describes existing need for food-based community efforts and non-profits
scores['food_services']=((data_q['q_Food Environment Index']+2*data_q['q_% Food Insecure']+
                         data_q['q_% Limited Access to Healthy Foods']+
                          2*data_q['q_% Enrolled in Free or Reduced Lunch'])/6)*100

#scores.head()


# **disparate economic harm**: unemployed, %children in poverty, income ratio, single parents, severe housing problems,
# disconnected youth, hs grad rate, enrolled in free lunch, severe housing cost burden, covid cases, uninsured

# describes the likelihood that a community will experience disparate economic harm due to COVID complications
scores['economic']=((3*data_q['q_% Unemployed']+3*data_q['q_% Children in Poverty']+
                     data_q['q_Income Ratio']+2*data_q['q_% Single-Parent Households']+
                    data_q['q_% Severe Housing Problems']+
                    data_q['q_High School Graduation Rate']+2*data_q['q_% Enrolled in Free or Reduced Lunch']
                    +3*data_q['q_% Severe Housing Cost Burden']+data_q['q_covid_cases']+
                    data_q['q_% Uninsured'])/19)*100

#scores.head()


# **community connectedness**: single parents, social association, drive alone to work, disconnected youth, segregation index


#describes the likelihood that an area could benefit from community connecting services
scores['community']=((data_q['q_% Single-Parent Households']+3*(1-data_q['q_Social Association Rate'])+
                      data_q['q_% Long Commute - Drives Alone']+data_q['q_% Drive Alone to Work']+
                     data_q['q_Black/White Segregation Index'])/7)*100

#scores.head()


# **mental health apps:** frequent mental distress, disconnected youth, average mentally unhealthy days, suicide rate

#describes existing need for additional mental health support 
scores['mental_health']=((2*data_q['q_Average Number of Mentally Unhealthy Days']+
                         data_q['q_% Excessive Drinking']+2*(1-data_q['q_Social Association Rate'])+
                          2*data_q['q_% Frequent Mental Distress']+2*data_q['q_Suicide Rate (Age-Adjusted)'])/9)*100

# **mobile health needs:** uninsured, smokers, obesity, fair or poor health, PCP rate, internet ratio, smokers, obesity, diabetes, %65 and older

#describes the likelihood that a community could benefit from mobile health services
scores['mobile_health']=((3*data_q['q_% Uninsured']+data_q['q_% Smokers']+
                          data_q['q_% Adults with Obesity']+2*data_q['q_% Adults with Diabetes']+
                        2*data_q['q_% 65 and over']+3*(1-data_q['q_Primary Care Physicians Rate'])+
                         2*data_q['q_% Fair or Poor Health']+3*(1-data_q['q_internet_ratio']))/17)*100

#scores.head()


# **overwhelming health system:** low birthweight, smokers, obesity, fair or poor health, excessive drinking, driving deaths with alcohol, PCP rate, violent crime rate, injury death rate, hiv prevalence, drug overdose, motor vehicle mortality, suicide rate

# In[20]:


#describes the likelihood that the existing health infrastructure will be overwhelmed by a covid outbreak
scores['overwhelm']=((data_q['q_% Smokers']+data_q['q_% Adults with Obesity']+2*data_q['q_% Adults with Diabetes']
                        +2*data_q['q_% 65 and over']+data_q['q_% Low Birthweight']
                        +2*data_q['q_% Fair or Poor Health']+data_q['q_% Excessive Drinking']
                        +3*(1-data_q['q_Primary Care Physicians Rate'])+data_q['q_Violent Crime Rate']
                        +data_q['q_Injury Death Rate']+data_q['q_HIV Prevalence Rate']
                         +data_q['q_Motor Vehicle Mortality Rate']
                     )/18)*100


#scores.head()


# **information desert:** not profiient in english, community association, internet ratio

# desribes the likelihood that constituents have difficulty accessing reliable covid-19 data
scores['info']=((data_q['q_% Not Proficient in English']+(1-data_q['q_Social Association Rate'])
                +2*data_q['q_internet_ratio'])/4)*100

#scores.head()


# **Platform extensions:**
# - users can add their own layers (data or calculated)
# - we can create better scores with ML
# - add functionality to incorporate things that don't exist for every county


#save the scores
scores.to_csv('../scores.csv', index=False)



