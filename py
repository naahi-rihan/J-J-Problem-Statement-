import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
%matplotlib inline
train_data = pd.read_csv('train.csv')
print(train_data.shape)
train_data.head()
(10147, 27)
customer_id	destination	passanger	weather	temperature	time	coupon	expiration	gender	age	...	CoffeeHouse	CarryAway	RestaurantLessThan20	Restaurant20To50	toCoupon_GEQ5min	toCoupon_GEQ15min	toCoupon_GEQ25min	direction_same	direction_opp	Y
0	258868	No Urgent Place	Friend(s)	Sunny	80	6PM	Restaurant(<20)	1d	Male	21	...	1~3	4~8	4~8	never	1	1	0	0	1	1
1	318369	Work	Alone	Sunny	80	7AM	Restaurant(<20)	2h	Male	21	...	1~3	4~8	1~3	less1	1	0	0	1	0	0
2	320906	No Urgent Place	Alone	Sunny	80	10AM	Coffee House	2h	Female	21	...	gt8	4~8	1~3	1~3	1	1	0	0	1	0
3	412393	Work	Alone	Rainy	55	7AM	Restaurant(<20)	2h	Female	26	...	less1	4~8	1~3	never	1	1	1	0	1	0
4	290854	Home	Alone	Snowy	30	6PM	Coffee House	1d	Male	31	...	less1	4~8	less1	never	1	1	0	0	1	0
5 rows × 27 columns

test_data = pd.read_csv('test.csv')
print(test_data.shape)
test_data.head()
(2537, 26)
customer_id	destination	passanger	weather	temperature	time	coupon	expiration	gender	age	...	Bar	CoffeeHouse	CarryAway	RestaurantLessThan20	Restaurant20To50	toCoupon_GEQ5min	toCoupon_GEQ15min	toCoupon_GEQ25min	direction_same	direction_opp
0	374679	No Urgent Place	Friend(s)	Sunny	80	6PM	Coffee House	1d	Female	below21	...	never	1~3	less1	4~8	less1	1	0	0	0	1
1	469678	Home	Alone	Sunny	80	6PM	Carry out & Take away	2h	Male	21	...	1~3	never	gt8	4~8	1~3	1	1	0	1	0
2	216140	No Urgent Place	Alone	Rainy	55	10AM	Coffee House	1d	Female	26	...	never	never	1~3	less1	never	1	1	0	0	1
3	184301	No Urgent Place	Partner	Sunny	80	6PM	Bar	1d	Male	50plus	...	never	4~8	4~8	1~3	less1	1	1	0	0	1
4	148720	Work	Alone	Sunny	30	7AM	Carry out & Take away	1d	Female	26	...	never	never	1~3	4~8	less1	1	1	0	0	1
5 rows × 26 columns

#no of Features is 26 and Target Column is Y
target_col = 'Y'
len(train_data.columns.drop(target_col))
26
#Target Feature is Y
train_data.Y.value_counts().plot.pie(autopct = '%.2f%%')
<AxesSubplot:ylabel='Y'>

#Datatype of all features in X
train_data.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10147 entries, 0 to 10146
Data columns (total 27 columns):
 #   Column                Non-Null Count  Dtype 
---  ------                --------------  ----- 
 0   customer_id           10147 non-null  int64 
 1   destination           10147 non-null  object
 2   passanger             10147 non-null  object
 3   weather               10147 non-null  object
 4   temperature           10147 non-null  int64 
 5   time                  10147 non-null  object
 6   coupon                10147 non-null  object
 7   expiration            10147 non-null  object
 8   gender                10147 non-null  object
 9   age                   10147 non-null  object
 10  maritalStatus         10147 non-null  object
 11  has_children          10147 non-null  int64 
 12  education             10147 non-null  object
 13  occupation            10147 non-null  object
 14  income                10147 non-null  object
 15  car                   84 non-null     object
 16  Bar                   10059 non-null  object
 17  CoffeeHouse           9975 non-null   object
 18  CarryAway             10025 non-null  object
 19  RestaurantLessThan20  10050 non-null  object
 20  Restaurant20To50      9999 non-null   object
 21  toCoupon_GEQ5min      10147 non-null  int64 
 22  toCoupon_GEQ15min     10147 non-null  int64 
 23  toCoupon_GEQ25min     10147 non-null  int64 
 24  direction_same        10147 non-null  int64 
 25  direction_opp         10147 non-null  int64 
 26  Y                     10147 non-null  int64 
dtypes: int64(9), object(18)
memory usage: 2.1+ MB
# Time converting to Integer values by using date time format
train_data['time'] = pd.to_datetime(train_data['time'],format = '%I%p').dt.hour
test_data['time'] = pd.to_datetime(test_data['time'],format = '%I%p').dt.hour
train_data.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10147 entries, 0 to 10146
Data columns (total 27 columns):
 #   Column                Non-Null Count  Dtype 
---  ------                --------------  ----- 
 0   customer_id           10147 non-null  int64 
 1   destination           10147 non-null  object
 2   passanger             10147 non-null  object
 3   weather               10147 non-null  object
 4   temperature           10147 non-null  int64 
 5   time                  10147 non-null  int64 
 6   coupon                10147 non-null  object
 7   expiration            10147 non-null  object
 8   gender                10147 non-null  object
 9   age                   10147 non-null  object
 10  maritalStatus         10147 non-null  object
 11  has_children          10147 non-null  int64 
 12  education             10147 non-null  object
 13  occupation            10147 non-null  object
 14  income                10147 non-null  object
 15  car                   84 non-null     object
 16  Bar                   10059 non-null  object
 17  CoffeeHouse           9975 non-null   object
 18  CarryAway             10025 non-null  object
 19  RestaurantLessThan20  10050 non-null  object
 20  Restaurant20To50      9999 non-null   object
 21  toCoupon_GEQ5min      10147 non-null  int64 
 22  toCoupon_GEQ15min     10147 non-null  int64 
 23  toCoupon_GEQ25min     10147 non-null  int64 
 24  direction_same        10147 non-null  int64 
 25  direction_opp         10147 non-null  int64 
 26  Y                     10147 non-null  int64 
dtypes: int64(10), object(17)
memory usage: 2.1+ MB
# drop the Car column because it has less then 1% values only
train_data.drop(columns = 'car', inplace = True)
test_data.drop(columns = 'car', inplace = True)
#Fill Null values using mode method
train_data.update(train_data['Bar'].fillna(train_data['Bar'].mode()[0]))
train_data.update(train_data['CoffeeHouse'].fillna(train_data['CoffeeHouse'].mode()[0]))
train_data.update(train_data['CarryAway'].fillna(train_data['CarryAway'].mode()[0]))
train_data.update(train_data['RestaurantLessThan20'].fillna(train_data['RestaurantLessThan20'].mode()[0]))
train_data.update(train_data['Restaurant20To50'].fillna(train_data['Restaurant20To50'].mode()[0]))
test_data.update(test_data['Bar'].fillna(test_data['Bar'].mode()[0]))
test_data.update(test_data['CoffeeHouse'].fillna(test_data['CoffeeHouse'].mode()[0]))
test_data.update(test_data['CarryAway'].fillna(test_data['CarryAway'].mode()[0]))
test_data.update(test_data['RestaurantLessThan20'].fillna(test_data['RestaurantLessThan20'].mode()[0]))
test_data.update(test_data['Restaurant20To50'].fillna(test_data['Restaurant20To50'].mode()[0]))
train_data.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10147 entries, 0 to 10146
Data columns (total 26 columns):
 #   Column                Non-Null Count  Dtype 
---  ------                --------------  ----- 
 0   customer_id           10147 non-null  int64 
 1   destination           10147 non-null  object
 2   passanger             10147 non-null  object
 3   weather               10147 non-null  object
 4   temperature           10147 non-null  int64 
 5   time                  10147 non-null  int64 
 6   coupon                10147 non-null  object
 7   expiration            10147 non-null  object
 8   gender                10147 non-null  object
 9   age                   10147 non-null  object
 10  maritalStatus         10147 non-null  object
 11  has_children          10147 non-null  int64 
 12  education             10147 non-null  object
 13  occupation            10147 non-null  object
 14  income                10147 non-null  object
 15  Bar                   10147 non-null  object
 16  CoffeeHouse           10147 non-null  object
 17  CarryAway             10147 non-null  object
 18  RestaurantLessThan20  10147 non-null  object
 19  Restaurant20To50      10147 non-null  object
 20  toCoupon_GEQ5min      10147 non-null  int64 
 21  toCoupon_GEQ15min     10147 non-null  int64 
 22  toCoupon_GEQ25min     10147 non-null  int64 
 23  direction_same        10147 non-null  int64 
 24  direction_opp         10147 non-null  int64 
 25  Y                     10147 non-null  int64 
dtypes: int64(10), object(16)
memory usage: 2.0+ MB
train_data._get_numeric_data()
customer_id	temperature	time	has_children	toCoupon_GEQ5min	toCoupon_GEQ15min	toCoupon_GEQ25min	direction_same	direction_opp	Y
0	258868	80	18	0	1	1	0	0	1	1
1	318369	80	7	0	1	0	0	1	0	0
2	320906	80	10	0	1	1	0	0	1	0
3	412393	55	7	0	1	1	1	0	1	0
4	290854	30	18	0	1	1	0	0	1	0
...	...	...	...	...	...	...	...	...	...	...
10142	201838	80	18	0	1	1	0	1	0	0
10143	248838	80	18	0	1	0	0	1	0	1
10144	173367	80	10	0	1	0	0	0	1	1
10145	488688	30	22	1	1	0	0	0	1	0
10146	431162	30	22	1	1	1	0	0	1	0
10147 rows × 10 columns

train_data.describe(include='object')
destination	passanger	weather	coupon	expiration	gender	age	maritalStatus	education	occupation	income	Bar	CoffeeHouse	CarryAway	RestaurantLessThan20	Restaurant20To50
count	10147	10147	10147	10147	10147	10147	10147	10147	10147	10147	10147	10147	10147	10147	10147	10147
unique	3	4	3	5	2	2	8	5	6	25	9	5	5	5	5	5
top	No Urgent Place	Alone	Sunny	Coffee House	1d	Female	21	Married partner	Bachelors degree	Unemployed	
37499	never	less1	1~3	1~3	less1
freq	5045	5802	8015	3191	5643	5204	2133	4086	3511	1485	1622	4238	2868	3863	4393	5037
train_data.age.value_counts()
21         2133
26         2033
31         1636
50plus     1431
36         1065
41          879
46          538
below21     432
Name: age, dtype: int64
# 50 plus and below 21 categories in the data can be converted to 51 and 20 respectively and the Age gets conevrted to Integer
train_data['age'] = train_data['age'].apply(lambda v: 51 if v == '50plus' else (20 if v=='below21' else v))
test_data['age'] = test_data['age'].apply(lambda v: 51 if v == '50plus' else (20 if v=='below21' else v))
# Converting to numerical data type
train_data['age'] = train_data['age'].astype('int64')
test_data['age'] = test_data['age'].astype('int64')
train_data.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10147 entries, 0 to 10146
Data columns (total 26 columns):
 #   Column                Non-Null Count  Dtype 
---  ------                --------------  ----- 
 0   customer_id           10147 non-null  int64 
 1   destination           10147 non-null  object
 2   passanger             10147 non-null  object
 3   weather               10147 non-null  object
 4   temperature           10147 non-null  int64 
 5   time                  10147 non-null  int64 
 6   coupon                10147 non-null  object
 7   expiration            10147 non-null  object
 8   gender                10147 non-null  object
 9   age                   10147 non-null  int64 
 10  maritalStatus         10147 non-null  object
 11  has_children          10147 non-null  int64 
 12  education             10147 non-null  object
 13  occupation            10147 non-null  object
 14  income                10147 non-null  object
 15  Bar                   10147 non-null  object
 16  CoffeeHouse           10147 non-null  object
 17  CarryAway             10147 non-null  object
 18  RestaurantLessThan20  10147 non-null  object
 19  Restaurant20To50      10147 non-null  object
 20  toCoupon_GEQ5min      10147 non-null  int64 
 21  toCoupon_GEQ15min     10147 non-null  int64 
 22  toCoupon_GEQ25min     10147 non-null  int64 
 23  direction_same        10147 non-null  int64 
 24  direction_opp         10147 non-null  int64 
 25  Y                     10147 non-null  int64 
dtypes: int64(11), object(15)
memory usage: 2.0+ MB
# checking feature occupation, it has 25 unique categories

train_data.occupation.value_counts().sort_values(ascending=True).plot.barh(figsize=(14,5))
<AxesSubplot:>

import seaborn as sns
# Check the outliers based on coupon usage in occupation
occu = train_data[train_data['Y']==1]['occupation'].value_counts()
print(occu.sum())
occu
5768
Unemployed                                   806
Student                                      765
Computer & Mathematical                      638
Sales & Related                              498
Education&Training&Library                   391
Management                                   388
Office & Administrative Support              307
Arts Design Entertainment Sports & Media     280
Business & Financial                         249
Retired                                      184
Food Preparation & Serving Related           144
Healthcare Practitioners & Technical         139
Healthcare Support                           130
Transportation & Material Moving             108
Community & Social Services                   98
Architecture & Engineering                    89
Legal                                         84
Construction & Extraction                     82
Protective Service                            82
Personal Care & Service                       78
Life Physical Social Science                  77
Production Occupations                        56
Installation Maintenance & Repair             54
Building & Grounds Cleaning & Maintenance     22
Farming Fishing & Forestry                    19
Name: occupation, dtype: int64
plt.figure(figsize=(5,5))
sns.boxplot(data=occu,y=occu.values)
<AxesSubplot:>

occu.plot.kde()
<AxesSubplot:ylabel='Density'>

occu_outlier = ['Unemployed','Student','Computer & Mathematical','Sales & Related','Education&Training&Library',\
                'Management','Office & Administrative Support','Arts Design Entertainment Sports & Media',\
                'Business & Financial','Retired']
train_data['occupation'] = train_data['occupation'].apply(lambda v: 'others' if v not in occu_outlier else v)
test_data['occupation'] = test_data['occupation'].apply(lambda v: 'others' if v not in occu_outlier else v)
train_data.occupation.value_counts()
others                                      2122
Unemployed                                  1485
Student                                     1245
Computer & Mathematical                     1111
Sales & Related                              896
Education&Training&Library                   753
Management                                   665
Arts Design Entertainment Sports & Media     526
Office & Administrative Support              517
Business & Financial                         433
Retired                                      394
Name: occupation, dtype: int64
# Removed the Skewness and outliers
occu = train_data[train_data['Y']==1]['occupation'].value_counts()
print(occu.sum())
occu.plot.kde()
plt.figure(figsize=(5,5))
sns.boxplot(data=occu,y=occu.values)
5768
<AxesSubplot:>


train_data.describe(include = 'object')
destination	passanger	weather	coupon	expiration	gender	maritalStatus	education	occupation	income	Bar	CoffeeHouse	CarryAway	RestaurantLessThan20	Restaurant20To50
count	10147	10147	10147	10147	10147	10147	10147	10147	10147	10147	10147	10147	10147	10147	10147
unique	3	4	3	5	2	2	5	6	11	9	5	5	5	5	5
top	No Urgent Place	Alone	Sunny	Coffee House	1d	Female	Married partner	Bachelors degree	others	
37499	never	less1	1~3	1~3	less1
freq	5045	5802	8015	3191	5643	5204	4086	3511	2122	1622	4238	2868	3863	4393	5037
# Income column analysis
train_data.income.value_counts().sort_values(ascending=True).plot.barh(figsize=(14,5))
<AxesSubplot:>

# Analyze Income brackets for positive results
income = train_data[train_data['Y']==1]['income'].value_counts()
print(income.sum())
income
5768
$25000 - $37499     966
$12500 - $24999     848
$100000 or More     822
$37500 - $49999     809
$50000 - $62499     787
Less than $12500    480
$87500 - $99999     371
$62500 - $74999     356
$75000 - $87499     329
Name: income, dtype: int64
sns.boxplot(data=income,y=income.values)
<AxesSubplot:>

income.plot.kde()
<AxesSubplot:ylabel='Density'>

# Income bracket does not have outliers, however the data skewness is left skewed
income.skew()
-0.20304477193411624
# Gender Analysis
train_data.gender.value_counts().sort_values(ascending=False)
Female    5204
Male      4943
Name: gender, dtype: int64
gender = train_data[train_data['Y']==1]['gender'].value_counts()
print(gender.sum())
gender
5768
Male      2915
Female    2853
Name: gender, dtype: int64
sns.boxplot(data=gender,y=gender.values)
<AxesSubplot:>

gender.plot.kde()
<AxesSubplot:ylabel='Density'>

# Analyse Coupons
train_data.coupon.value_counts().sort_values(ascending=True).plot.barh(figsize=(14,5))
<AxesSubplot:>

coupon = train_data[train_data['Y']==1]['coupon'].value_counts()
print(coupon.sum())
coupon
5768
Coffee House             1600
Restaurant(<20)          1575
Carry out & Take away    1414
Bar                       664
Restaurant(20-50)         515
Name: coupon, dtype: int64
sns.boxplot(data=coupon,y=coupon.values)
<AxesSubplot:>

coupon.plot.kde()
<AxesSubplot:ylabel='Density'>

coupon.skew()
-0.5734002084393751
# Analyze passanger
train_data.passanger.value_counts().sort_values(ascending=True).plot.barh(figsize=(14,5))
<AxesSubplot:>

passanger = train_data[train_data['Y']==1]['passanger'].value_counts()
print(passanger.sum())
passanger
5768
Alone        3049
Friend(s)    1800
Partner       510
Kid(s)        409
Name: passanger, dtype: int64
sns.boxplot(data=passanger,y=passanger.values)
<AxesSubplot:>

passanger.plot.kde()
<AxesSubplot:ylabel='Density'>

passanger.skew()
0.7899972523633265
#Education Analysis
train_data.education.value_counts().sort_values(ascending=True).plot.barh(figsize=(14,5))
<AxesSubplot:>

education = train_data[train_data['Y']==1]['education'].value_counts()
print(education.sum())
education
5768
Some college - no degree                  2069
Bachelors degree                          1948
Graduate degree (Masters or Doctorate)     781
Associates degree                          505
High School Graduate                       412
Some High School                            53
Name: education, dtype: int64
sns.boxplot(data=education,y=education.values)
<AxesSubplot:>

education.plot.kde()
<AxesSubplot:ylabel='Density'>

education.skew()
0.6486777732125808
# Weather Analysis
train_data.weather.value_counts().sort_values(ascending=True).plot.barh(figsize=(14,5))
<AxesSubplot:>

weather = train_data[train_data['Y']==1]['weather'].value_counts()
print(weather.sum())
weather
5768
Sunny    4757
Snowy     546
Rainy     465
Name: weather, dtype: int64
sns.boxplot(data=weather,y=weather.values)
<AxesSubplot:>

weather.plot.kde()
<AxesSubplot:ylabel='Density'>

weather.skew()
1.7299297492720773
# Analyze maritalStatus
train_data.maritalStatus.value_counts().sort_values(ascending=True).plot.barh(figsize=(14,5))
<AxesSubplot:>

maritalStatus = train_data[train_data['Y']==1]['maritalStatus'].value_counts()
print(maritalStatus.sum())
maritalStatus
5768
Single               2296
Married partner      2224
Unmarried partner     968
Divorced              232
Widowed                48
Name: maritalStatus, dtype: int64
sns.boxplot(data=maritalStatus,y=maritalStatus.values)
<AxesSubplot:>

maritalStatus.plot.kde()
<AxesSubplot:ylabel='Density'>

maritalStatus.skew()
0.1976308791571414
train_data.describe(include = 'object')
destination	passanger	weather	coupon	expiration	gender	maritalStatus	education	occupation	income	Bar	CoffeeHouse	CarryAway	RestaurantLessThan20	Restaurant20To50
count	10147	10147	10147	10147	10147	10147	10147	10147	10147	10147	10147	10147	10147	10147	10147
unique	3	4	3	5	2	2	5	6	11	9	5	5	5	5	5
top	No Urgent Place	Alone	Sunny	Coffee House	1d	Female	Married partner	Bachelors degree	others	
37499	never	less1	1~3	1~3	less1
freq	5045	5802	8015	3191	5643	5204	4086	3511	2122	1622	4238	2868	3863	4393	5037
# Analyze Bar
train_data.Bar.value_counts().sort_values(ascending=True).plot.barh(figsize=(14,5))
<AxesSubplot:>

bar = train_data[train_data['Y']==1]['Bar'].value_counts()
print(bar.sum())
bar
5768
never    2241
less1    1576
1~3      1227
4~8       556
gt8       168
Name: Bar, dtype: int64
sns.boxplot(data=bar,y=bar.values)
<AxesSubplot:>

bar.plot.kde()
<AxesSubplot:ylabel='Density'>

bar.skew()
0.14370130208924572
# Analyze CoffeeHouse
train_data.CoffeeHouse.value_counts().sort_values(ascending=True).plot.barh(figsize=(14,5))
<AxesSubplot:>

CoffeeHouse = train_data[train_data['Y']==1]['CoffeeHouse'].value_counts()
print(CoffeeHouse.sum())
CoffeeHouse
5768
1~3      1691
less1    1594
never    1074
4~8       889
gt8       520
Name: CoffeeHouse, dtype: int64
sns.boxplot(data=CoffeeHouse,y=CoffeeHouse.values)
<AxesSubplot:>

CoffeeHouse.plot.kde()
<AxesSubplot:ylabel='Density'>

CoffeeHouse.skew()
-0.11603470770618135
# Analyze CarryAway
train_data.CarryAway.value_counts().sort_values(ascending=True).plot.barh(figsize=(14,5))
<AxesSubplot:>

CarryAway = train_data[train_data['Y']==1]['CarryAway'].value_counts()
print(CarryAway.sum())
CarryAway
5768
1~3      2244
4~8      1980
less1     748
gt8       735
never      61
Name: CarryAway, dtype: int64
sns.boxplot(data=CarryAway,y=CarryAway.values)
<AxesSubplot:>

CarryAway.plot.kde()
<AxesSubplot:ylabel='Density'>

CarryAway.skew()
0.22088977238793162
train_data.describe()
customer_id	temperature	time	age	has_children	toCoupon_GEQ5min	toCoupon_GEQ15min	toCoupon_GEQ25min	direction_same	direction_opp	Y
count	10147.000000	10147.000000	10147.000000	10147.000000	10147.000000	10147.0	10147.000000	10147.000000	10147.000000	10147.000000	10147.000000
mean	311272.276831	63.172366	13.802700	32.434710	0.412634	1.0	0.563024	0.119838	0.212181	0.787819	0.568444
std	106781.701016	19.232595	5.401365	10.428837	0.492332	0.0	0.496037	0.324788	0.408872	0.408872	0.495318
min	123472.000000	30.000000	7.000000	20.000000	0.000000	1.0	0.000000	0.000000	0.000000	0.000000	0.000000
25%	221439.000000	55.000000	10.000000	21.000000	0.000000	1.0	0.000000	0.000000	0.000000	1.000000	0.000000
50%	310062.000000	80.000000	14.000000	31.000000	0.000000	1.0	1.000000	0.000000	0.000000	1.000000	1.000000
75%	401537.000000	80.000000	18.000000	41.000000	1.000000	1.0	1.000000	0.000000	0.000000	1.000000	1.000000
max	499988.000000	80.000000	22.000000	51.000000	1.000000	1.0	1.000000	1.000000	1.000000	1.000000	1.000000
#Customer ID is a unique column so we will remove the column while training the Model
# Distribution of numerical columns
train_data.temperature.value_counts().sort_values(ascending=True).plot.pie(autopct = '%.2f%%')
<AxesSubplot:ylabel='temperature'>

train_data.time.value_counts().sort_values(ascending=True).plot.pie(autopct = '%.2f%%')
<AxesSubplot:ylabel='time'>

train_data.age.value_counts().sort_values(ascending=True).plot.barh(figsize=(14,5))
<AxesSubplot:>

train_data.has_children.value_counts().sort_values(ascending=True).plot.pie(autopct='%.2f%%')
<AxesSubplot:ylabel='has_children'>

train_data.direction_opp.value_counts().sort_values(ascending=True).plot.pie(autopct='%.2f%%')
<AxesSubplot:ylabel='direction_opp'>

train_data.direction_same.value_counts().sort_values(ascending=True).plot.pie(autopct='%.2f%%')
<AxesSubplot:ylabel='direction_same'>

#Bi-Variate Analysis 
# Destination Vs Coupon Use column Y
destination_Y = pd.crosstab(index = train_data['destination'], columns = train_data['Y'])
destination_Y
Y	0	1
destination		
Home	1273	1299
No Urgent Place	1847	3198
Work	1259	1271
plt.figure(figsize=(14,5))
destination_Y.plot.bar()
plt.ylabel("Count")
plt.legend()
<matplotlib.legend.Legend at 0x1d4fae26ca0>
<Figure size 1008x360 with 0 Axes>

# Has Children Vs Coupon Use column Y
has_children_Y = pd.crosstab(index = train_data['has_children'], columns = train_data['Y'])
has_children_Y
Y	0	1
has_children		
0	2471	3489
1	1908	2279
plt.figure(figsize=(14,5))
has_children_Y.plot.bar()
plt.ylabel("Count")
plt.legend()
<matplotlib.legend.Legend at 0x1d4fafa93a0>
<Figure size 1008x360 with 0 Axes>

# Age Vs Coupon Use column Y
age_Y = pd.crosstab(index = train_data['age'], columns = train_data['Y'])
age_Y
Y	0	1
age		
20	164	268
21	862	1271
26	817	1216
31	751	885
36	495	570
41	363	516
46	235	303
51	692	739
plt.figure(figsize=(14,5))
age_Y.plot()
plt.ylabel("Count")
plt.legend()
<matplotlib.legend.Legend at 0x1d4fb033730>
<Figure size 1008x360 with 0 Axes>

# time Vs Coupon Use column Y
time_Y = pd.crosstab(index = train_data['time'], columns = train_data['Y'])
time_Y
Y	0	1
time		
7	1259	1271
10	704	1129
14	557	1059
18	1076	1500
22	783	809
plt.figure(figsize=(14,5))
time_Y.plot()
plt.title("Time Vs Coupon used")
plt.ylabel("Count")
plt.legend()
<matplotlib.legend.Legend at 0x1d4fb0bdf10>
<Figure size 1008x360 with 0 Axes>

# Multivariate comparison toCoupon_GEQ5min,toCoupon_GEQ15min and toCoupon_GEQ25min
t_Y = pd.crosstab(index = train_data['Y'], columns = train_data['toCoupon_GEQ5min'])
# t1_Y = pd.crosstab(index = train_data['Y'], columns = train_data['toCoupon_GEQ15min'])
# t2_Y = pd.crosstab(index = train_data['Y'], columns = train_data['toCoupon_GEQ25min'])
# t_Y = pd.crosstab(index = train_data['toCoupon_GEQ5min'], columns = train_data['Y'])
t1_Y = pd.crosstab(index = train_data['toCoupon_GEQ15min'], columns = train_data['Y'])
t2_Y = pd.crosstab(index = train_data['toCoupon_GEQ25min'], columns = train_data['Y'])
plt.figure(figsize=(14,5))
t_Y.plot()
t1_Y.plot()
t2_Y.plot()
plt.legend()
<matplotlib.legend.Legend at 0x1d4fb0eab80>
<Figure size 1008x360 with 0 Axes>



# toCoupon_GEQ5min is single value through out hence the column can be dropped
train_data.drop(columns = 'toCoupon_GEQ5min', inplace=True)
test_data.drop(columns = 'toCoupon_GEQ5min', inplace=True)
# Occupation Vs Coupon Use
occu_Y = pd.crosstab(index = train_data['occupation'], columns = train_data['Y'])
occu_Y
Y	0	1
occupation		
Arts Design Entertainment Sports & Media	246	280
Business & Financial	184	249
Computer & Mathematical	473	638
Education&Training&Library	362	391
Management	277	388
Office & Administrative Support	210	307
Retired	210	184
Sales & Related	398	498
Student	480	765
Unemployed	679	806
others	860	1262
occu_Y.plot(figsize=(14,5))
plt.title("Occupation Vs Coupon used")
plt.ylabel("Count")
plt.xticks(rotation=60)
plt.legend()
plt.show()

# Multivariate Analysis
train_data.pivot_table(index = 'occupation', values='customer_id',columns=['age','Y'],aggfunc="count")
age	20	21	26	31	36	41	46	51
Y	0	1	0	1	0	1	0	1	0	1	0	1	0	1	0	1
occupation																
Arts Design Entertainment Sports & Media	NaN	NaN	31.0	64.0	31.0	21.0	70.0	57.0	14.0	23.0	54.0	68.0	16.0	22.0	30.0	25.0
Business & Financial	NaN	NaN	27.0	26.0	21.0	46.0	26.0	31.0	24.0	46.0	55.0	65.0	18.0	15.0	13.0	20.0
Computer & Mathematical	NaN	NaN	34.0	84.0	125.0	187.0	179.0	160.0	79.0	95.0	23.0	59.0	9.0	19.0	24.0	34.0
Education&Training&Library	11.0	7.0	45.0	28.0	30.0	71.0	53.0	69.0	83.0	66.0	50.0	53.0	3.0	13.0	87.0	84.0
Management	NaN	NaN	33.0	57.0	72.0	96.0	42.0	41.0	27.0	36.0	18.0	59.0	52.0	60.0	33.0	39.0
Office & Administrative Support	NaN	NaN	22.0	36.0	79.0	101.0	40.0	41.0	12.0	27.0	21.0	34.0	10.0	22.0	26.0	46.0
Retired	NaN	NaN	NaN	NaN	12.0	5.0	NaN	NaN	3.0	11.0	NaN	NaN	11.0	12.0	184.0	156.0
Sales & Related	NaN	NaN	101.0	134.0	97.0	84.0	49.0	116.0	35.0	37.0	31.0	37.0	33.0	37.0	52.0	53.0
Student	124.0	243.0	267.0	444.0	17.0	33.0	34.0	32.0	27.0	7.0	NaN	NaN	11.0	6.0	NaN	NaN
Unemployed	10.0	6.0	169.0	248.0	168.0	260.0	87.0	96.0	104.0	57.0	37.0	39.0	18.0	26.0	86.0	74.0
others	19.0	12.0	133.0	150.0	165.0	312.0	171.0	242.0	87.0	165.0	74.0	102.0	54.0	71.0	157.0	208.0
# we can see that Age group in student, Unemployed and Retired (it is clubbed in others under age 50+)
# have a good positive coupon use rate
train_data.pivot_table(index = 'coupon', values='customer_id',columns=['Y'],aggfunc="count")
Y	0	1
coupon		
Bar	959	664
Carry out & Take away	509	1414
Coffee House	1591	1600
Restaurant(20-50)	662	515
Restaurant(<20)	658	1575
# converting categorical columns to Numerical column
train_dummies = pd.get_dummies(train_data)
test_dummies = pd.get_dummies(test_data)
train_dummies.shape
(10147, 85)
# Correlation of the current numeric columns
train_dummies.corr().style.background_gradient()
customer_id	temperature	time	age	has_children	toCoupon_GEQ15min	toCoupon_GEQ25min	direction_same	direction_opp	Y	destination_Home	destination_No Urgent Place	destination_Work	passanger_Alone	passanger_Friend(s)	passanger_Kid(s)	passanger_Partner	weather_Rainy	weather_Snowy	weather_Sunny	coupon_Bar	coupon_Carry out & Take away	coupon_Coffee House	coupon_Restaurant(20-50)	coupon_Restaurant(<20)	expiration_1d	expiration_2h	gender_Female	gender_Male	maritalStatus_Divorced	maritalStatus_Married partner	maritalStatus_Single	maritalStatus_Unmarried partner	maritalStatus_Widowed	education_Associates degree	education_Bachelors degree	education_Graduate degree (Masters or Doctorate)	education_High School Graduate	education_Some High School	education_Some college - no degree	occupation_Arts Design Entertainment Sports & Media	occupation_Business & Financial	occupation_Computer & Mathematical	occupation_Education&Training&Library	occupation_Management	occupation_Office & Administrative Support	occupation_Retired	occupation_Sales & Related	occupation_Student	occupation_Unemployed	occupation_others	income_$100000 or More	income_
24999	income_
37499	income_
49999	income_
62499	income_
74999	income_
87499	income_
99999	income_Less than $12500	Bar_1~3	Bar_4~8	Bar_gt8	Bar_less1	Bar_never	CoffeeHouse_1~3	CoffeeHouse_4~8	CoffeeHouse_gt8	CoffeeHouse_less1	CoffeeHouse_never	CarryAway_1~3	CarryAway_4~8	CarryAway_gt8	CarryAway_less1	CarryAway_never	RestaurantLessThan20_1~3	RestaurantLessThan20_4~8	RestaurantLessThan20_gt8	RestaurantLessThan20_less1	RestaurantLessThan20_never	Restaurant20To50_1~3	Restaurant20To50_4~8	Restaurant20To50_gt8	Restaurant20To50_less1	Restaurant20To50_never
customer_id	1.000000	0.009802	-0.002119	0.012083	0.016644	-0.001647	0.001947	-0.012535	0.012535	-0.004595	-0.010415	0.007181	0.002172	-0.002515	-0.011779	0.010024	0.013363	-0.011543	0.001725	0.007069	-0.001004	-0.006585	-0.007504	0.020527	-0.000339	-0.002080	0.002080	-0.002521	0.002521	-0.008541	0.021634	-0.008451	-0.009902	-0.011013	-0.003645	0.001979	0.005969	0.000806	-0.011271	-0.002708	-0.003419	-0.001133	-0.011231	-0.001554	0.007000	0.007754	-0.005217	0.004541	-0.011189	-0.004028	0.015436	-0.001656	-0.022659	-0.001855	-0.007236	-0.012163	0.008208	0.025622	0.016422	0.011502	0.008443	-0.020482	0.016687	-0.000235	-0.000647	-0.006406	0.012283	0.011801	-0.025051	0.015258	-0.007037	0.015738	-0.001263	-0.008512	-0.005501	-0.009209	0.009906	0.003295	-0.006937	0.013077	0.013133	-0.010790	0.002017	-0.004116	-0.003967
temperature	0.009802	1.000000	-0.045365	-0.012911	-0.012415	-0.155627	-0.215181	0.092184	-0.092184	0.053147	-0.056184	0.136521	-0.101287	-0.107612	0.110835	0.024010	-0.007607	-0.139724	-0.614258	0.578290	-0.131952	-0.131282	0.171864	0.033304	0.022587	-0.125488	0.125488	0.027896	-0.027896	0.025611	-0.024996	0.015834	0.005504	-0.025832	-0.030748	-0.003697	0.022950	0.000784	-0.012921	0.007100	0.019100	-0.047250	0.002798	0.017033	0.042768	-0.018646	0.019366	-0.007178	0.018359	-0.030605	-0.008402	-0.015293	0.008177	0.022645	0.019788	0.016444	0.002042	-0.049802	-0.013917	-0.009896	-0.027135	0.013898	-0.009072	-0.000889	0.017825	0.015184	-0.036589	0.006067	-0.004704	0.015318	0.016149	0.020692	-0.024398	-0.022993	-0.013305	-0.014233	0.002124	0.036532	-0.003371	-0.028256	0.027768	-0.044886	-0.002316	-0.033712	0.041428
time	-0.002119	-0.045365	1.000000	0.010268	0.028134	-0.059183	-0.189059	-0.001125	0.001125	0.007258	0.599470	0.106539	-0.725884	-0.227345	0.168613	0.080379	0.058901	-0.022240	-0.001021	0.016991	0.026941	-0.051240	-0.075587	0.063034	0.060628	-0.028632	0.028632	-0.006179	0.006179	-0.009258	0.032411	-0.029324	0.001875	-0.005711	0.022978	0.000641	0.012749	-0.012556	0.007026	-0.018536	-0.010059	0.018275	0.019938	-0.001422	-0.017828	0.001245	-0.016270	-0.000336	-0.013145	0.016054	-0.003200	0.008171	-0.023636	-0.019367	0.009082	-0.006239	0.006789	0.029265	0.004392	0.004806	0.000988	-0.017535	-0.013993	0.008864	0.005814	-0.010417	0.002825	0.003641	-0.010298	0.016918	-0.006344	-0.003013	-0.002919	0.013009	0.008047	0.015420	-0.007016	-0.018090	0.002011	0.001710	0.001972	0.008829	-0.008897	0.000285	-0.004746
age	0.012083	-0.012911	0.010268	1.000000	0.443809	0.029903	-0.007263	-0.028568	0.028568	-0.051105	-0.011168	0.028349	-0.021534	-0.034668	-0.009614	0.113666	-0.034146	-0.012082	0.004712	0.005145	-0.001354	0.000339	-0.016449	0.020374	0.003565	-0.016074	0.016074	0.100864	-0.100864	0.200265	0.275603	-0.251384	-0.188322	0.178505	0.044159	-0.083384	0.220832	-0.062199	-0.002034	-0.073641	0.047330	0.056986	-0.032201	0.105340	0.057699	0.019049	0.321133	-0.011674	-0.367491	-0.085332	0.083528	0.040430	-0.014508	-0.042449	0.060302	-0.012958	0.004943	0.028168	0.023549	-0.089476	-0.071250	-0.090726	-0.100733	0.024515	0.120314	-0.071698	0.057655	-0.050588	0.004665	0.055432	-0.086128	0.043656	-0.050042	0.096747	0.032506	0.058503	-0.019303	-0.006447	-0.018621	-0.088200	0.004127	-0.013840	-0.048116	0.068505	-0.069475
has_children	0.016644	-0.012415	0.028134	0.443809	1.000000	0.065631	-0.017729	-0.029574	0.029574	-0.040851	-0.005198	0.024927	-0.023582	-0.055466	-0.050975	0.341295	-0.154077	-0.023280	-0.003312	0.019526	0.012721	0.010473	-0.036093	0.028337	-0.002615	-0.013898	0.013898	0.155253	-0.155253	0.209681	0.468162	-0.452543	-0.152923	0.055100	0.106964	-0.072279	0.117237	-0.094404	0.058741	-0.038981	-0.021712	0.037962	0.068317	0.092630	0.058726	-0.054012	0.118565	-0.052012	-0.219488	-0.014590	0.044000	-0.011575	-0.032529	-0.023649	0.053377	0.025097	0.029150	0.004136	0.087151	-0.122924	-0.079119	-0.118262	-0.104119	0.008735	0.157595	0.001345	-0.005291	0.035077	-0.004639	-0.015605	-0.038344	0.099735	-0.130471	0.018258	0.074967	0.087387	-0.049096	-0.028531	-0.022860	-0.031523	-0.017801	0.041497	-0.048720	0.024648	-0.019109
toCoupon_GEQ15min	-0.001647	-0.155627	-0.059183	0.029903	0.065631	1.000000	0.325074	-0.300418	0.300418	-0.081642	-0.137527	0.038763	0.093482	-0.139197	0.130917	0.117543	-0.074664	0.073392	0.107522	-0.136868	0.091177	0.070116	-0.059741	0.051699	-0.120007	-0.042046	0.042046	0.000805	-0.000805	0.011229	0.034227	-0.016361	-0.025832	-0.013741	0.008817	0.000511	0.017698	-0.004383	-0.008001	-0.015299	0.001658	0.006105	0.009850	0.001549	0.001276	-0.010918	0.005316	-0.011533	-0.009668	-0.006796	0.011854	0.008197	-0.017115	-0.002832	0.012508	-0.003584	0.012977	0.002909	0.008655	-0.018720	-0.011144	0.011679	-0.013447	-0.002124	0.008825	0.001833	0.005045	-0.000635	-0.013128	0.008362	-0.007758	0.012492	-0.010291	0.003322	0.000803	-0.008967	0.006287	-0.007977	0.007265	0.010125	0.000795	-0.005131	-0.009103	0.003199	0.001493
toCoupon_GEQ25min	0.001947	-0.215181	-0.189059	-0.007263	-0.017729	0.325074	1.000000	-0.191495	0.191495	-0.098778	0.047977	-0.366925	0.375819	0.277003	-0.220836	-0.102270	-0.043108	0.123086	0.181610	-0.230538	0.181710	-0.011962	-0.091107	-0.052167	-0.007031	0.033440	-0.033440	-0.003423	0.003423	0.015328	-0.028251	0.026259	-0.005768	0.002740	0.004503	-0.008135	0.012045	0.013286	0.006884	-0.011857	-0.005523	-0.005840	-0.002080	-0.002591	-0.004528	0.004200	0.001231	0.001738	0.004441	0.009478	-0.003952	-0.010299	0.010844	-0.007765	0.008645	-0.007173	0.002978	-0.001547	-0.011078	0.016403	-0.008547	0.019686	-0.008656	0.002537	-0.003615	-0.006199	0.010897	-0.000273	-0.008556	0.006725	-0.004334	-0.000236	-0.004004	0.012054	-0.006682	0.001562	-0.001143	-0.013931	0.007914	0.007660	-0.010548	-0.005738	-0.008168	0.004475	0.013115
direction_same	-0.012535	0.092184	-0.001125	-0.028568	-0.029574	-0.300418	-0.191495	1.000000	-1.000000	0.014182	0.359776	-0.516060	0.234667	0.376033	-0.310594	-0.114220	-0.065441	0.019448	-0.063542	0.035128	-0.011422	0.135294	-0.036895	-0.035934	-0.048757	-0.029273	0.029273	0.009071	-0.009071	0.004275	-0.022594	0.010674	0.013081	0.001383	-0.000884	-0.008091	0.007934	0.017765	0.007598	-0.008103	-0.018056	-0.003428	-0.008285	0.001129	-0.018604	0.003620	-0.010729	0.008398	0.024123	0.023126	-0.013779	-0.008193	-0.005922	0.000554	0.012150	-0.021262	-0.002815	-0.001399	-0.002900	0.034502	-0.001160	-0.001094	-0.015237	0.008606	-0.001086	-0.023066	0.004211	0.015383	-0.008851	0.019396	-0.013232	0.009788	-0.002179	0.005395	0.005660	0.007730	-0.009922	-0.005188	0.004426	0.004269	-0.004591	0.010545	-0.006398	-0.013381	0.019252
direction_opp	0.012535	-0.092184	0.001125	0.028568	0.029574	0.300418	0.191495	-1.000000	1.000000	-0.014182	-0.359776	0.516060	-0.234667	-0.376033	0.310594	0.114220	0.065441	-0.019448	0.063542	-0.035128	0.011422	-0.135294	0.036895	0.035934	0.048757	0.029273	-0.029273	-0.009071	0.009071	-0.004275	0.022594	-0.010674	-0.013081	-0.001383	0.000884	0.008091	-0.007934	-0.017765	-0.007598	0.008103	0.018056	0.003428	0.008285	-0.001129	0.018604	-0.003620	0.010729	-0.008398	-0.024123	-0.023126	0.013779	0.008193	0.005922	-0.000554	-0.012150	0.021262	0.002815	0.001399	0.002900	-0.034502	0.001160	0.001094	0.015237	-0.008606	0.001086	0.023066	-0.004211	-0.015383	0.008851	-0.019396	0.013232	-0.009788	0.002179	-0.005395	-0.005660	-0.007730	0.009922	0.005188	-0.004426	-0.004269	0.004591	-0.010545	0.006398	0.013381	-0.019252
Y	-0.004595	0.053147	0.007258	-0.051105	-0.040851	-0.081642	-0.098778	0.014182	-0.014182	1.000000	-0.074576	0.131406	-0.076882	-0.100172	0.125912	-0.039346	0.017177	-0.065554	-0.064951	0.098134	-0.140365	0.162912	-0.091667	-0.095728	0.146805	0.133069	-0.133069	-0.041871	0.041871	-0.005053	-0.040028	0.054456	-0.010241	-0.018867	-0.014000	-0.019997	-0.033058	0.007452	0.034986	0.043133	-0.017054	0.002819	0.004116	-0.028117	0.008028	0.011867	-0.041164	-0.007942	0.034743	-0.021470	0.027282	0.012450	0.008946	0.023881	-0.006739	0.022367	-0.017256	-0.052045	-0.026277	0.010130	0.051247	0.048396	0.003763	-0.002160	-0.067808	0.100910	0.045902	0.007888	-0.016039	-0.129728	0.019711	0.016083	0.010280	-0.054121	-0.013277	-0.008101	0.014451	0.027299	-0.027000	-0.005135	0.025700	0.042780	0.029247	-0.022386	-0.038091
destination_Home	-0.010415	-0.056184	0.599470	-0.011168	-0.005198	-0.137527	0.047977	0.359776	-0.359776	-0.074576	1.000000	-0.579435	-0.335825	0.358200	-0.348737	-0.072001	-0.014452	-0.032785	0.027623	0.002450	-0.005802	0.024608	-0.028220	0.146213	-0.099531	0.009415	-0.009415	0.012202	-0.012202	0.003768	-0.010946	0.011361	-0.002577	0.000911	0.004560	0.001930	0.010676	0.011152	0.010451	-0.020440	-0.007488	0.003638	-0.001893	-0.009392	-0.018822	0.005104	-0.009228	0.011089	0.013414	0.019608	-0.012741	-0.001895	-0.006503	-0.009975	0.015553	-0.016901	-0.001197	0.007028	-0.003968	0.023401	-0.004932	-0.008736	-0.010208	0.007156	0.005870	-0.015848	0.009334	0.007302	-0.019604	0.024613	-0.006611	0.005337	-0.008330	0.008712	0.003318	0.014398	-0.008535	-0.016397	0.004341	0.000261	-0.001350	0.001210	-0.010595	-0.009401	0.017542
destination_No Urgent Place	0.007181	0.136521	0.106539	0.028349	0.024927	0.038763	-0.366925	-0.516060	0.516060	0.131406	-0.579435	1.000000	-0.573098	-0.743183	0.601857	0.209914	0.163842	-0.038009	-0.096541	0.102577	0.009173	-0.082027	0.053685	-0.152157	0.126921	-0.065320	0.065320	-0.013164	0.013164	-0.012241	0.043998	-0.048167	0.013280	-0.008372	0.005889	0.003050	-0.001873	-0.028635	-0.011618	0.012149	0.007537	0.003624	0.018698	0.011742	0.021798	-0.004525	0.008271	-0.014230	-0.030641	-0.023605	0.012582	0.009667	-0.011717	-0.001315	-0.011816	0.023864	0.007863	0.010497	0.010031	-0.035941	0.007491	-0.004310	0.005695	-0.007375	0.001160	0.020925	-0.018450	-0.006941	0.018409	-0.021334	0.011508	-0.008094	0.007790	-0.012452	0.000615	-0.006429	0.004839	0.019870	-0.011781	-0.004589	0.008269	0.000957	0.007656	0.008928	-0.025241
destination_Work	0.002172	-0.101287	-0.725884	-0.021534	-0.023582	0.093482	0.375819	0.234667	-0.234667	-0.076882	-0.335825	-0.573098	1.000000	0.498740	-0.344923	-0.170204	-0.174822	0.076892	0.083799	-0.121014	-0.004767	0.070056	-0.033670	0.028834	-0.046608	0.066025	-0.066025	0.002945	-0.002945	0.010359	-0.039844	0.044243	-0.012757	0.008759	-0.011390	-0.005466	-0.008570	0.021880	0.002918	0.006512	-0.001182	-0.007846	-0.019705	-0.004127	-0.006266	0.000097	-0.000281	0.005296	0.021925	0.007565	-0.001730	-0.009267	0.020081	0.011549	-0.001982	-0.010586	-0.007883	-0.019198	-0.007604	0.018008	-0.003699	0.013765	0.003682	0.001328	-0.007243	-0.008249	0.011938	0.000679	-0.001564	-0.000092	-0.006653	0.003988	-0.000628	0.005631	-0.004046	-0.007047	0.002989	-0.006477	0.009251	0.005042	-0.008199	-0.002322	0.001805	-0.000865	0.011533
passanger_Alone	-0.002515	-0.107612	-0.227345	-0.034668	-0.055466	-0.139197	0.277003	0.376033	-0.376033	-0.100172	0.358200	-0.743183	0.498740	1.000000	-0.691588	-0.341268	-0.350528	0.072439	0.110922	-0.138812	0.015201	0.012420	-0.012696	0.063440	-0.060005	0.118401	-0.118401	0.002543	-0.002543	0.030662	-0.106947	0.125779	-0.042410	0.014543	-0.010618	-0.004426	-0.008777	0.042348	0.017376	-0.008335	-0.009670	-0.011418	-0.033974	-0.019423	-0.031583	0.015743	-0.006482	0.011001	0.046206	0.024165	-0.002619	-0.020215	0.031681	0.010625	0.008168	-0.047765	-0.011011	-0.015295	-0.011788	0.054170	-0.006827	0.012756	0.006191	-0.002304	-0.001723	-0.030009	0.028298	0.005790	-0.011459	0.015987	-0.011420	-0.000527	0.002307	0.015713	-0.004818	-0.001565	-0.008805	-0.013938	0.020757	0.009353	-0.017401	0.003375	0.000679	-0.007222	0.027752
passanger_Friend(s)	-0.011779	0.110835	0.168613	-0.009614	-0.050975	0.130917	-0.220836	-0.310594	0.310594	0.125912	-0.348737	0.601857	-0.344923	-0.691588	1.000000	-0.176748	-0.181544	-0.109354	-0.066643	0.131349	-0.092754	0.053562	0.070547	-0.119712	0.044864	-0.106770	0.106770	-0.027033	0.027033	0.002284	-0.078241	0.106834	-0.039240	0.009832	-0.009080	0.010375	-0.018720	-0.001506	-0.016264	0.012636	0.006337	-0.001319	-0.002862	-0.012444	0.008698	0.009820	0.003581	0.002131	0.009315	-0.017482	0.001308	0.000438	-0.001331	0.009301	0.002038	0.017571	0.004420	-0.017078	-0.017311	-0.007593	0.001745	0.029294	0.039624	-0.020765	-0.012543	0.010270	-0.005976	0.003129	0.001809	-0.009689	0.005176	-0.009881	0.025689	-0.016850	-0.003407	-0.009270	-0.009615	0.023514	0.003504	0.004105	-0.008421	-0.011277	0.034144	0.005200	-0.003289
passanger_Kid(s)	0.010024	0.024010	0.080379	0.113666	0.341295	0.117543	-0.102270	-0.114220	0.114220	-0.039346	-0.072001	0.209914	-0.170204	-0.341268	-0.176748	1.000000	-0.089584	-0.016407	-0.045474	0.047228	0.039395	-0.031720	-0.071095	0.062972	0.026156	-0.009994	0.009994	0.046840	-0.046840	-0.006309	0.263517	-0.211565	-0.060049	-0.029612	0.023804	0.012466	0.048881	-0.049553	-0.002023	-0.036344	0.022590	0.032785	0.057950	0.045116	0.031745	-0.045329	-0.006774	-0.042041	-0.078375	0.003975	-0.004664	0.017893	-0.045890	-0.021897	0.000167	0.025112	0.021615	0.024772	0.060910	-0.064936	-0.019922	-0.052016	-0.041946	-0.006584	0.065492	0.030429	-0.017924	0.020728	-0.021814	-0.007304	-0.008886	0.031451	-0.055302	0.017319	0.014678	0.013612	-0.017785	-0.002680	0.015186	-0.027684	0.010493	0.021197	-0.023446	0.004302	-0.022200
passanger_Partner	0.013363	-0.007607	0.058901	-0.034146	-0.154077	-0.074664	-0.043108	-0.065441	0.065441	0.017177	-0.014452	0.163842	-0.174822	-0.350528	-0.181544	-0.089584	1.000000	0.060478	-0.047403	-0.007276	0.081537	-0.076072	-0.019774	0.015317	0.010143	-0.031767	0.031767	-0.007450	0.007450	-0.052074	0.056937	-0.186644	0.196515	-0.012546	0.010041	-0.020765	-0.002469	-0.024589	-0.003173	0.030343	-0.014917	-0.009628	0.008392	0.010221	0.011421	0.000705	0.012490	0.018133	-0.020444	-0.019200	0.007151	0.017816	-0.009449	-0.012268	-0.017947	0.032656	-0.008535	0.030112	-0.011103	-0.020951	0.028874	-0.018328	-0.032867	0.043484	-0.041077	0.007409	-0.023402	-0.035547	0.038874	-0.005967	0.020823	-0.014142	0.009219	-0.018196	-0.000365	0.004182	0.048329	-0.009852	-0.057384	0.003899	0.034093	-0.008852	-0.032446	0.000408	-0.022509
weather_Rainy	-0.011543	-0.139724	-0.022240	-0.012082	-0.023280	0.073392	0.123086	0.019448	-0.019448	-0.065554	-0.032785	-0.038009	0.076892	0.072439	-0.109354	-0.016407	0.060478	1.000000	-0.117093	-0.637528	0.176367	0.059644	-0.089653	-0.065172	-0.061621	-0.039149	0.039149	-0.019756	0.019756	-0.011184	-0.009924	-0.000230	0.012810	0.023908	0.018297	-0.007368	-0.013283	0.006418	0.017771	-0.000336	-0.012464	0.012742	-0.016375	-0.018332	-0.032051	0.014437	-0.014512	0.000680	0.007623	0.025479	0.014672	-0.003240	0.000155	-0.011104	-0.011299	-0.026271	-0.003739	0.030495	0.008495	0.032898	0.016695	-0.010510	-0.004573	0.001988	-0.007734	-0.016238	0.024069	-0.001548	0.008985	-0.011546	-0.017030	-0.014959	0.020901	0.020780	0.010114	-0.005099	0.011935	-0.024252	0.002059	0.028532	-0.009725	0.030798	-0.004773	0.017644	-0.029556
weather_Snowy	0.001725	-0.614258	-0.001021	0.004712	-0.003312	0.107522	0.181610	-0.063542	0.063542	-0.064951	0.027623	-0.096541	0.083799	0.110922	-0.066643	-0.045474	-0.047403	-0.117093	1.000000	-0.690477	0.007944	0.025123	-0.070609	0.003441	0.045678	0.112288	-0.112288	-0.024760	0.024760	-0.014030	0.007718	-0.009886	0.005388	0.017694	0.026022	0.002525	-0.015235	0.007733	0.012792	-0.013352	-0.024191	0.034356	-0.000038	0.003870	-0.032564	0.008245	-0.011853	0.000175	-0.013420	0.021941	0.006271	0.008602	-0.002455	-0.024293	-0.003138	-0.017007	0.000340	0.033062	0.003501	0.015689	0.022251	-0.017561	0.015649	-0.001391	-0.011994	-0.018636	0.025009	0.000299	0.007077	-0.009056	-0.019115	-0.012179	0.017811	0.020177	0.018734	0.007293	-0.005388	-0.015894	0.002531	0.020546	-0.021438	0.041455	0.001491	0.021273	-0.029725
weather_Sunny	0.007069	0.578290	0.016991	0.005145	0.019526	-0.136868	-0.230538	0.035128	-0.035128	0.098134	0.002450	0.102577	-0.121014	-0.138812	0.131349	0.047228	-0.007276	-0.637528	-0.690477	1.000000	-0.134623	-0.062932	0.120076	0.044799	0.009447	-0.058594	0.058594	0.033598	-0.033598	0.019030	0.001241	0.007837	-0.013510	-0.031140	-0.033514	0.003408	0.021494	-0.010674	-0.022868	0.010603	0.027845	-0.035932	0.011957	0.010350	0.048607	-0.016911	0.019765	-0.000631	0.004858	-0.035579	-0.015551	-0.004313	0.001792	0.026933	0.010664	0.032328	0.002459	-0.047860	-0.008903	-0.036133	-0.029422	0.021279	-0.008809	-0.000369	0.014937	0.026284	-0.036932	0.000895	-0.012034	0.015434	0.027232	0.020344	-0.029040	-0.030788	-0.021900	-0.001944	-0.004513	0.029994	-0.003464	-0.036721	0.023714	-0.054591	0.002320	-0.029354	0.044587
coupon_Bar	-0.001004	-0.131952	0.026941	-0.001354	0.012721	0.091177	0.181710	-0.011422	0.011422	-0.140365	-0.005802	0.009173	-0.004767	0.015201	-0.092754	0.039395	0.081537	0.176367	0.007944	-0.134623	1.000000	-0.211002	-0.295543	-0.158063	-0.231784	0.192323	-0.192323	-0.000739	0.000739	-0.014490	0.027108	-0.037076	0.019392	0.002289	0.014211	-0.005980	-0.021532	0.006075	0.003702	0.009493	-0.002587	-0.000343	0.005422	-0.009684	-0.014521	0.002820	-0.008378	0.009178	-0.000931	0.006447	0.003695	0.002172	-0.003952	0.001880	-0.009208	-0.007048	-0.010009	0.018984	0.007491	0.004776	0.010491	-0.015836	-0.013530	0.003124	0.002255	-0.005323	-0.002019	-0.000146	0.008519	-0.001828	-0.004364	-0.000788	0.004874	0.004165	-0.005456	0.012667	0.000159	-0.017797	-0.003605	0.002760	0.004677	0.002770	-0.012198	0.011474	-0.017878
coupon_Carry out & Take away	-0.006585	-0.131282	-0.051240	0.000339	0.010473	0.070116	-0.011962	0.135294	-0.135294	0.162912	0.024608	-0.082027	0.070056	0.012420	0.053562	-0.031720	-0.076072	0.059644	0.025123	-0.062932	-0.211002	1.000000	-0.327515	-0.175162	-0.256859	0.050395	-0.050395	0.003405	-0.003405	0.017699	-0.010436	0.010237	-0.011473	0.009775	-0.000097	0.005083	0.007193	0.014632	0.006513	-0.019379	-0.000776	0.007391	-0.006080	-0.000676	-0.010189	-0.004550	-0.003474	-0.006031	0.012305	0.002541	0.004236	-0.001335	-0.001752	0.001790	0.001685	-0.008063	0.005250	-0.006682	0.002034	0.008856	0.001873	0.000187	0.003080	0.002797	-0.005181	-0.001202	0.007475	0.005661	-0.005879	-0.002433	-0.013513	0.008844	0.004299	0.002541	0.000601	0.001758	-0.001969	-0.002714	0.000068	0.006243	-0.006839	0.010051	0.000113	0.002725	-0.001920
coupon_Coffee House	-0.007504	0.171864	-0.075587	-0.016449	-0.036093	-0.059741	-0.091107	-0.036895	0.036895	-0.091667	-0.028220	0.053685	-0.033670	-0.012696	0.070547	-0.071095	-0.019774	-0.089653	-0.070609	0.120076	-0.295543	-0.327515	1.000000	-0.245344	-0.359774	-0.157040	0.157040	0.005716	-0.005716	-0.009770	-0.025082	0.018895	0.017893	-0.016597	-0.011492	0.002173	0.009339	-0.010539	-0.008805	0.005026	0.009177	-0.022230	0.001778	0.002590	0.020476	0.001366	0.006698	0.002415	0.010012	-0.011409	-0.013215	-0.002937	0.008823	-0.001206	0.009049	0.019900	0.003594	-0.022320	-0.010726	-0.014697	-0.006080	0.015909	0.001021	0.000209	-0.004628	0.007677	-0.005808	-0.000342	0.004751	-0.007962	0.014502	-0.001012	-0.011049	-0.006004	-0.007338	-0.006639	0.006637	0.017370	-0.004467	-0.025257	0.012820	-0.025697	0.003048	-0.015292	0.020261
coupon_Restaurant(20-50)	0.020527	0.033304	0.063034	0.020374	0.028337	0.051699	-0.052167	-0.035934	0.035934	-0.095728	0.146213	-0.152157	0.028834	0.063440	-0.119712	0.062972	0.015317	-0.065172	0.003441	0.044799	-0.158063	-0.175162	-0.245344	1.000000	-0.192415	0.053545	-0.053545	-0.007781	0.007781	-0.003674	0.018226	0.000968	-0.023272	0.000882	0.001948	0.010832	-0.006392	-0.006190	0.000424	-0.004043	-0.005571	0.013360	0.003085	-0.000404	-0.008875	0.004242	-0.001118	-0.004264	-0.017273	0.008487	0.007461	0.009507	-0.007722	-0.006839	0.002205	0.001940	-0.006248	0.004123	0.006520	-0.002389	-0.005091	-0.005130	0.008056	-0.008681	0.012115	-0.009861	0.008809	-0.004141	-0.010029	0.016363	-0.007028	-0.013181	0.010720	0.013712	0.011617	0.000270	-0.012763	-0.006020	0.013389	0.018859	-0.005019	0.002440	0.002015	0.010300	-0.010219
coupon_Restaurant(<20)	-0.000339	0.022587	0.060628	0.003565	-0.002615	-0.120007	-0.007031	-0.048757	0.048757	0.146805	-0.099531	0.126921	-0.046608	-0.060005	0.044864	0.026156	0.010143	-0.061621	0.045678	0.009447	-0.231784	-0.256859	-0.359774	-0.192415	1.000000	-0.083226	0.083226	-0.002959	0.002959	0.009866	-0.000090	0.001195	-0.008370	0.006646	-0.001108	-0.010326	0.006721	-0.002620	0.000104	0.007425	-0.002955	0.007899	-0.003422	0.006617	0.006399	-0.003000	0.004056	-0.001826	-0.008687	-0.001882	0.001768	-0.004715	0.001235	0.003281	-0.005293	-0.009939	0.004691	0.011353	-0.001570	0.005714	-0.000305	-0.000031	0.001686	0.001066	-0.001271	0.004865	-0.005585	-0.001643	0.000451	0.000194	0.005824	0.003653	-0.004282	-0.009958	0.003504	-0.005639	0.004149	0.003500	-0.002216	0.005382	-0.008157	0.014955	0.005712	-0.003553	0.002826
expiration_1d	-0.002080	-0.125488	-0.028632	-0.016074	-0.013898	-0.042046	0.033440	-0.029273	0.029273	0.133069	0.009415	-0.065320	0.066025	0.118401	-0.106770	-0.009994	-0.031767	-0.039149	0.112288	-0.058594	0.192323	0.050395	-0.157040	0.053545	-0.083226	1.000000	-1.000000	-0.000426	0.000426	0.012086	-0.023994	0.021873	-0.005260	0.007656	-0.009556	-0.002316	-0.007535	0.007212	0.012602	0.007685	-0.002256	-0.006676	0.001998	-0.024036	-0.018294	0.033811	-0.007304	0.001896	0.013074	0.008505	-0.004437	-0.015506	0.017406	0.011889	-0.014219	-0.025355	0.013421	-0.005895	0.004690	0.019634	-0.009297	0.008756	-0.018186	0.008190	0.001264	-0.020546	0.008360	-0.007506	0.012349	0.006185	-0.012382	0.007672	0.000369	0.008505	-0.006853	0.001979	-0.002188	-0.017791	0.012923	0.004276	0.000381	-0.002562	-0.007040	-0.000477	0.004512
expiration_2h	0.002080	0.125488	0.028632	0.016074	0.013898	0.042046	-0.033440	0.029273	-0.029273	-0.133069	-0.009415	0.065320	-0.066025	-0.118401	0.106770	0.009994	0.031767	0.039149	-0.112288	0.058594	-0.192323	-0.050395	0.157040	-0.053545	0.083226	-1.000000	1.000000	0.000426	-0.000426	-0.012086	0.023994	-0.021873	0.005260	-0.007656	0.009556	0.002316	0.007535	-0.007212	-0.012602	-0.007685	0.002256	0.006676	-0.001998	0.024036	0.018294	-0.033811	0.007304	-0.001896	-0.013074	-0.008505	0.004437	0.015506	-0.017406	-0.011889	0.014219	0.025355	-0.013421	0.005895	-0.004690	-0.019634	0.009297	-0.008756	0.018186	-0.008190	-0.001264	0.020546	-0.008360	0.007506	-0.012349	-0.006185	0.012382	-0.007672	-0.000369	-0.008505	0.006853	-0.001979	0.002188	0.017791	-0.012923	-0.004276	-0.000381	0.002562	0.007040	0.000477	-0.004512
gender_Female	-0.002521	0.027896	-0.006179	0.100864	0.155253	0.000805	-0.003423	0.009071	-0.009071	-0.041871	0.012202	-0.013164	0.002945	0.002543	-0.027033	0.046840	-0.007450	-0.019756	-0.024760	0.033598	-0.000739	0.003405	0.005716	-0.007781	-0.002959	-0.000426	0.000426	1.000000	-1.000000	0.100452	0.060485	-0.145774	0.046671	0.034164	0.034346	-0.024311	0.020968	0.008431	-0.004530	-0.015801	0.131832	-0.048842	-0.111630	0.060040	-0.094853	0.095806	-0.004151	-0.060820	-0.084442	0.138008	-0.002080	-0.065082	0.029275	0.102300	0.038618	-0.061552	-0.051457	0.016050	-0.049523	0.011903	-0.095020	-0.124340	-0.020980	0.041434	0.116138	0.038794	0.008062	-0.008350	0.044713	-0.088498	-0.024844	0.005869	0.020460	0.023095	-0.052060	0.018704	0.028074	-0.012878	-0.046967	-0.004150	0.006431	-0.009305	-0.027017	-0.002871	0.012547
gender_Male	0.002521	-0.027896	0.006179	-0.100864	-0.155253	-0.000805	0.003423	-0.009071	0.009071	0.041871	-0.012202	0.013164	-0.002945	-0.002543	0.027033	-0.046840	0.007450	0.019756	0.024760	-0.033598	0.000739	-0.003405	-0.005716	0.007781	0.002959	0.000426	-0.000426	-1.000000	1.000000	-0.100452	-0.060485	0.145774	-0.046671	-0.034164	-0.034346	0.024311	-0.020968	-0.008431	0.004530	0.015801	-0.131832	0.048842	0.111630	-0.060040	0.094853	-0.095806	0.004151	0.060820	0.084442	-0.138008	0.002080	0.065082	-0.029275	-0.102300	-0.038618	0.061552	0.051457	-0.016050	0.049523	-0.011903	0.095020	0.124340	0.020980	-0.041434	-0.116138	-0.038794	-0.008062	0.008350	-0.044713	0.088498	0.024844	-0.005869	-0.020460	-0.023095	0.052060	-0.018704	-0.028074	0.012878	0.046967	0.004150	-0.006431	0.009305	0.027017	0.002871	-0.012547
maritalStatus_Divorced	-0.008541	0.025611	-0.009258	0.200265	0.209681	0.011229	0.015328	0.004275	-0.004275	-0.005053	0.003768	-0.012241	0.010359	0.030662	0.002284	-0.006309	-0.052074	-0.011184	-0.014030	0.019030	-0.014490	0.017699	-0.009770	-0.003674	0.009866	0.012086	-0.012086	0.100452	-0.100452	1.000000	-0.169976	-0.160386	-0.094084	-0.020758	-0.005130	-0.063961	0.069402	0.015402	-0.017004	0.010316	0.025491	0.000505	-0.039203	-0.058612	-0.042788	0.024277	0.107438	0.056302	-0.025972	-0.063244	0.052240	-0.083096	0.186269	0.032977	0.016565	-0.047498	-0.010615	-0.056136	-0.056870	-0.026774	-0.009183	-0.030856	-0.035509	-0.024806	0.059226	-0.064109	0.045066	0.054652	0.021099	-0.030019	-0.028376	0.041619	0.001648	-0.011275	-0.022647	0.021508	-0.067103	0.073931	0.002173	-0.027265	-0.043949	0.064647	0.027940	0.002979	-0.003429
maritalStatus_Married partner	0.021634	-0.024996	0.032411	0.275603	0.468162	0.034227	-0.028251	-0.022594	0.022594	-0.040028	-0.010946	0.043998	-0.039844	-0.106947	-0.078241	0.263517	0.056937	-0.009924	0.007718	0.001241	0.027108	-0.010436	-0.025082	0.018226	-0.000090	-0.023994	0.023994	0.060485	-0.060485	-0.169976	1.000000	-0.636110	-0.373146	-0.082327	0.044648	-0.011748	0.156260	-0.141263	-0.025571	-0.051241	0.066338	0.006601	0.177372	0.120196	0.077316	-0.076930	0.007639	-0.079180	-0.236622	0.016498	-0.030382	0.211600	-0.193026	-0.099329	-0.014650	0.097589	0.020818	0.082050	0.126100	-0.200689	-0.060810	-0.134197	-0.116713	0.169005	0.011180	0.018753	-0.054985	-0.010203	0.041999	-0.012039	0.010944	0.029104	-0.041140	-0.028983	0.044014	-0.007290	0.063419	-0.001165	-0.037084	-0.083299	0.058835	0.003333	-0.068187	0.057351	-0.121664
maritalStatus_Single	-0.008451	0.015834	-0.029324	-0.251384	-0.452543	-0.016361	0.026259	0.010674	-0.010674	0.054456	0.011361	-0.048167	0.044243	0.125779	0.106834	-0.211565	-0.186644	-0.000230	-0.009886	0.007837	-0.037076	0.010237	0.018895	0.000968	0.001195	0.021873	-0.021873	-0.145774	0.145774	-0.160386	-0.636110	1.000000	-0.352093	-0.077682	-0.059136	0.077055	-0.128716	0.101024	0.023687	-0.004047	0.009829	0.000592	-0.110641	-0.111397	-0.079321	0.086164	-0.103033	0.052317	0.179317	0.053558	-0.025994	-0.081919	0.037583	0.045887	0.029351	-0.077724	-0.059312	-0.030370	-0.056630	0.187039	0.011150	0.175355	0.133437	-0.092694	-0.069183	0.006552	0.058770	0.010263	-0.073120	0.015947	-0.053643	-0.032433	0.076625	0.043767	0.005629	-0.043857	-0.036955	0.000362	0.081912	0.059925	-0.073227	0.017398	0.091641	-0.037989	0.090619
maritalStatus_Unmarried partner	-0.009902	0.005504	0.001875	-0.188322	-0.152923	-0.025832	-0.005768	0.013081	-0.013081	-0.010241	-0.002577	0.013280	-0.012757	-0.042410	-0.039240	-0.060049	0.196515	0.012810	0.005388	-0.013510	0.019392	-0.011473	0.017893	-0.023272	-0.008370	-0.005260	0.005260	0.046671	-0.046671	-0.094084	-0.373146	-0.352093	1.000000	-0.045569	0.015302	-0.030815	-0.063683	0.014174	0.013980	0.058992	-0.106264	-0.004042	-0.058808	0.025054	0.030833	-0.017255	0.019710	0.014403	0.101179	-0.046048	0.018495	-0.115818	0.076239	0.064498	-0.016617	0.008078	0.061717	-0.031046	-0.054168	-0.003236	0.082691	-0.026361	0.003701	-0.080397	0.019907	0.016402	-0.017025	-0.020490	0.026174	-0.017031	0.063431	-0.019247	-0.035828	-0.012738	-0.049717	0.050688	-0.002298	-0.029054	-0.058284	0.049284	0.056276	-0.054272	-0.039834	-0.054031	0.055569
maritalStatus_Widowed	-0.011013	-0.025832	-0.005711	0.178505	0.055100	-0.013741	0.002740	0.001383	-0.001383	-0.018867	0.000911	-0.008372	0.008759	0.014543	0.009832	-0.029612	-0.012546	0.023908	0.017694	-0.031140	0.002289	0.009775	-0.016597	0.000882	0.006646	0.007656	-0.007656	0.034164	-0.034164	-0.020758	-0.082327	-0.077682	-0.045569	1.000000	0.020024	-0.072933	-0.041385	0.120615	-0.008236	0.028421	-0.023445	-0.021169	-0.035159	-0.028388	-0.026554	-0.023233	0.175122	-0.031205	-0.037498	-0.041516	0.102229	-0.040247	0.108573	-0.043736	-0.040843	-0.038741	-0.026532	-0.027189	-0.027544	0.145111	-0.049355	-0.030493	-0.017199	-0.028224	0.088202	-0.058639	-0.040496	-0.031186	0.007613	0.106354	0.023612	0.004198	-0.037806	0.000615	-0.010969	0.014573	0.009817	-0.033611	0.000528	-0.013205	-0.059187	-0.024666	-0.014787	0.100992	-0.044871
education_Associates degree	-0.003645	-0.030748	0.022978	0.044159	0.106964	0.008817	0.004503	-0.000884	0.000884	-0.014000	0.004560	0.005889	-0.011390	-0.010618	-0.009080	0.023804	0.010041	0.018297	0.026022	-0.033514	0.014211	-0.000097	-0.011492	0.001948	-0.001108	-0.009556	0.009556	0.034346	-0.034346	-0.005130	0.044648	-0.059136	0.015302	0.020024	1.000000	-0.230230	-0.130641	-0.086687	-0.025998	-0.227629	0.044966	-0.066826	-0.034194	-0.049099	0.060136	0.009224	0.026813	-0.053835	-0.068254	0.122863	-0.001880	-0.040840	0.021142	-0.041785	0.034815	0.014246	-0.032503	0.080980	-0.040012	0.010469	-0.081449	0.039106	-0.054291	-0.004794	0.066043	-0.008242	-0.007478	-0.098445	0.081275	-0.005882	0.005100	0.014450	-0.042555	0.007534	0.019243	0.084322	-0.076960	-0.066315	0.034300	0.000652	0.016618	-0.023222	-0.006335	-0.002519	0.000765
education_Bachelors degree	0.001979	-0.003697	0.000641	-0.083384	-0.072279	0.000511	-0.008135	-0.008091	0.008091	-0.019997	0.001930	0.003050	-0.005466	-0.004426	0.010375	0.012466	-0.020765	-0.007368	0.002525	0.003408	-0.005980	0.005083	0.002173	0.010832	-0.010326	-0.002316	0.002316	-0.024311	0.024311	-0.063961	-0.011748	0.077055	-0.030815	-0.072933	-0.230230	1.000000	-0.300222	-0.199212	-0.059746	-0.523106	0.028031	0.085256	0.034221	0.077026	0.027543	-0.027217	-0.012149	0.041599	-0.085743	-0.048550	-0.047499	0.093377	-0.073444	-0.017657	-0.053751	0.055434	0.072253	0.052919	0.022417	-0.135100	0.067052	0.090629	-0.009131	-0.046105	-0.060242	-0.025573	0.051085	-0.075743	0.001672	0.033379	0.058602	-0.018844	-0.070298	0.025303	-0.048910	0.035943	-0.062741	0.057292	-0.013391	-0.014180	0.086016	-0.037817	-0.003931	-0.008648	-0.064346
education_Graduate degree (Masters or Doctorate)	0.005969	0.022950	0.012749	0.220832	0.117237	0.017698	0.012045	0.007934	-0.007934	-0.033058	0.010676	-0.001873	-0.008570	-0.008777	-0.018720	0.048881	-0.002469	-0.013283	-0.015235	0.021494	-0.021532	0.007193	0.009339	-0.006392	0.006721	-0.007535	0.007535	0.020968	-0.020968	0.069402	0.156260	-0.128716	-0.063683	-0.041385	-0.130641	-0.300222	1.000000	-0.113041	-0.033902	-0.296830	-0.049870	0.038674	0.035155	0.140084	0.078144	-0.095634	0.079051	-0.060494	-0.078550	-0.083929	0.035814	0.038034	-0.069271	-0.051166	-0.016149	0.019244	0.071584	0.050709	0.083541	-0.083420	0.007008	-0.036176	-0.040607	0.112267	-0.073030	0.037566	0.102101	0.123887	-0.105171	-0.093525	-0.047365	0.037116	-0.011718	0.040197	-0.045153	-0.065675	0.095283	0.035981	-0.048555	-0.024138	0.005963	0.055081	-0.060871	0.009956	-0.031055
education_High School Graduate	0.000806	0.000784	-0.012556	-0.062199	-0.094404	-0.004383	0.013286	0.017765	-0.017765	0.007452	0.011152	-0.028635	0.021880	0.042348	-0.001506	-0.049553	-0.024589	0.006418	0.007733	-0.010674	0.006075	0.014632	-0.010539	-0.006190	-0.002620	0.007212	-0.007212	0.008431	-0.008431	0.015402	-0.141263	0.101024	0.014174	0.120615	-0.086687	-0.199212	-0.113041	1.000000	-0.022496	-0.196961	-0.029141	0.007249	-0.057630	-0.077540	-0.025635	-0.028275	0.079111	0.011564	0.010766	0.100014	-0.003863	-0.090909	0.093413	0.005094	0.058956	-0.050586	-0.072471	-0.025258	-0.025260	0.087693	-0.047919	-0.059666	0.164289	-0.029523	0.043373	-0.040333	-0.056023	0.056679	-0.004392	0.054164	0.001961	-0.028117	0.039695	0.006988	-0.029961	-0.005870	-0.025869	0.026291	0.012695	0.014729	-0.075950	0.052699	0.160609	-0.065340	0.081832
education_Some High School	-0.011271	-0.012921	0.007026	-0.002034	0.058741	-0.008001	0.006884	0.007598	-0.007598	0.034986	0.010451	-0.011618	0.002918	0.017376	-0.016264	-0.002023	-0.003173	0.017771	0.012792	-0.022868	0.003702	0.006513	-0.008805	0.000424	0.000104	0.012602	-0.012602	-0.004530	0.004530	-0.017004	-0.025571	0.023687	0.013980	-0.008236	-0.025998	-0.059746	-0.033902	-0.022496	1.000000	-0.059071	-0.019206	-0.017342	0.033093	-0.023255	-0.021752	-0.019032	-0.016509	-0.025563	-0.030718	0.085601	0.008255	-0.032970	0.138158	-0.035828	-0.033458	-0.031736	-0.021735	-0.022273	-0.022564	0.055421	0.063220	0.048788	-0.014089	-0.050470	-0.027924	0.043435	-0.033173	0.051122	-0.051558	0.003092	-0.022110	0.028392	-0.030970	0.024087	-0.008986	-0.032764	0.042331	-0.027534	0.018449	-0.010818	-0.048485	0.162055	-0.012114	-0.042895	0.018319
education_Some college - no degree	-0.002708	0.007100	-0.018536	-0.073641	-0.038981	-0.015299	-0.011857	-0.008103	0.008103	0.043133	-0.020440	0.012149	0.006512	-0.008335	0.012636	-0.036344	0.030343	-0.000336	-0.013352	0.010603	0.009493	-0.019379	0.005026	-0.004043	0.007425	0.007685	-0.007685	-0.015801	0.015801	0.010316	-0.051241	-0.004047	0.058992	0.028421	-0.227629	-0.523106	-0.296830	-0.196961	-0.059071	1.000000	0.000649	-0.074686	-0.014467	-0.106045	-0.104758	0.111341	-0.102574	0.034120	0.185417	-0.031893	0.022818	-0.042684	0.038425	0.084578	0.018898	-0.045945	-0.063349	-0.122576	-0.042906	0.134620	-0.008196	-0.064101	-0.013548	-0.009794	0.056202	0.016919	-0.086871	0.004329	0.038476	0.010012	-0.023916	-0.007249	0.089091	-0.067775	0.088649	-0.029593	0.045402	-0.053409	0.018747	0.025739	-0.051679	-0.045152	-0.031159	0.045302	0.040080
occupation_Arts Design Entertainment Sports & Media	-0.003419	0.019100	-0.010059	0.047330	-0.021712	0.001658	-0.005523	-0.018056	0.018056	-0.017054	-0.007488	0.007537	-0.001182	-0.009670	0.006337	0.022590	-0.014917	-0.012464	-0.024191	0.027845	-0.002587	-0.000776	0.009177	-0.005571	-0.002955	-0.002256	0.002256	0.131832	-0.131832	0.025491	0.066338	0.009829	-0.106264	-0.023445	0.044966	0.028031	-0.049870	-0.029141	-0.019206	0.000649	1.000000	-0.049366	-0.081988	-0.066200	-0.061922	-0.054177	-0.046996	-0.072768	-0.087443	-0.096814	-0.120236	0.016732	0.063394	-0.014655	-0.002359	-0.021583	-0.024124	0.077386	-0.036387	-0.069330	-0.076948	0.018318	0.117298	-0.018100	0.028222	-0.003114	0.000300	-0.019429	0.035864	-0.022168	-0.016708	-0.046167	0.089585	-0.015066	0.052550	-0.005136	0.067455	-0.078379	-0.019699	0.024147	0.031561	-0.057519	-0.034484	-0.020545	0.039629
occupation_Business & Financial	-0.001133	-0.047250	0.018275	0.056986	0.037962	0.006105	-0.005840	-0.003428	0.003428	0.002819	0.003638	0.003624	-0.007846	-0.011418	-0.001319	0.032785	-0.009628	0.012742	0.034356	-0.035932	-0.000343	0.007391	-0.022230	0.013360	0.007899	-0.006676	0.006676	-0.048842	0.048842	0.000505	0.006601	0.000592	-0.004042	-0.021169	-0.066826	0.085256	0.038674	0.007249	-0.017342	-0.074686	-0.049366	1.000000	-0.074031	-0.059775	-0.055912	-0.048919	-0.042435	-0.065706	-0.078956	-0.087418	-0.108566	0.156444	-0.039511	-0.068143	-0.038547	-0.056917	0.074264	0.085597	-0.021729	-0.062601	0.017906	0.056654	-0.036214	0.045177	-0.074989	0.052195	0.028473	-0.010646	0.024488	-0.095982	-0.027961	-0.003890	0.022310	0.029840	-0.023097	0.054654	-0.039400	0.039254	-0.047177	-0.027806	0.007922	0.082595	-0.031137	-0.034076	-0.002936
occupation_Computer & Mathematical	-0.011231	0.002798	0.019938	-0.032201	0.068317	0.009850	-0.002080	-0.008285	0.008285	0.004116	-0.001893	0.018698	-0.019705	-0.033974	-0.002862	0.057950	0.008392	-0.016375	-0.000038	0.011957	0.005422	-0.006080	0.001778	0.003085	-0.003422	0.001998	-0.001998	-0.111630	0.111630	-0.039203	0.177372	-0.110641	-0.058808	-0.035159	-0.034194	0.034221	0.035155	-0.057630	0.033093	-0.014467	-0.081988	-0.074031	1.000000	-0.099275	-0.092860	-0.081246	-0.070477	-0.109126	-0.131132	-0.145185	-0.180310	0.120365	-0.074817	-0.071132	-0.017259	0.055100	0.069298	0.049860	-0.032073	-0.085441	0.024947	-0.071488	-0.060144	0.085976	-0.037131	0.002794	-0.053450	0.052314	0.042743	-0.039536	0.016925	-0.017378	0.054222	-0.056785	0.020031	-0.041399	0.021715	0.043750	0.009475	-0.046180	0.009312	0.032118	-0.051713	0.009782	-0.024013
occupation_Education&Training&Library	-0.001554	0.017033	-0.001422	0.105340	0.092630	0.001549	-0.002591	0.001129	-0.001129	-0.028117	-0.009392	0.011742	-0.004127	-0.019423	-0.012444	0.045116	0.010221	-0.018332	0.003870	0.010350	-0.009684	-0.000676	0.002590	-0.000404	0.006617	-0.024036	0.024036	0.060040	-0.060040	-0.058612	0.120196	-0.111397	0.025054	-0.028388	-0.049099	0.077026	0.140084	-0.077540	-0.023255	-0.106045	-0.066200	-0.059775	-0.099275	1.000000	-0.074978	-0.065600	-0.056905	-0.088111	-0.105880	-0.117227	-0.145587	-0.014668	-0.038139	0.073494	-0.015237	0.098635	0.005664	-0.052955	-0.005648	-0.083948	-0.063445	-0.014514	-0.048562	0.042674	0.036977	0.050987	0.001516	0.006076	-0.056640	0.002445	-0.011360	0.036203	-0.032718	0.007233	-0.030973	0.083472	-0.033260	0.017387	-0.071952	-0.037287	-0.010802	-0.014535	-0.041754	0.073101	-0.060140
occupation_Management	0.007000	0.042768	-0.017828	0.057699	0.058726	0.001276	-0.004528	-0.018604	0.018604	0.008028	-0.018822	0.021798	-0.006266	-0.031583	0.008698	0.031745	0.011421	-0.032051	-0.032564	0.048607	-0.014521	-0.010189	0.020476	-0.008875	0.006399	-0.018294	0.018294	-0.094853	0.094853	-0.042788	0.077316	-0.079321	0.030833	-0.026554	0.060136	0.027543	0.078144	-0.025635	-0.021752	-0.104758	-0.061922	-0.055912	-0.092860	-0.074978	1.000000	-0.061361	-0.053228	-0.082418	-0.099038	-0.109652	-0.136179	0.000835	-0.016939	-0.009020	-0.028081	-0.000447	0.013662	0.019632	0.125253	-0.078523	0.050558	0.096858	-0.007184	0.001556	-0.094270	0.026979	0.005438	0.007505	-0.022074	-0.013778	-0.054268	0.033815	0.055766	-0.031910	0.029970	-0.020819	-0.021189	0.065844	-0.008155	0.023571	0.095669	-0.035960	0.140279	-0.083717	-0.031994
occupation_Office & Administrative Support	0.007754	-0.018646	0.001245	0.019049	-0.054012	-0.010918	0.004200	0.003620	-0.003620	0.011867	0.005104	-0.004525	0.000097	0.015743	0.009820	-0.045329	0.000705	0.014437	0.008245	-0.016911	0.002820	-0.004550	0.001366	0.004242	-0.003000	0.033811	-0.033811	0.095806	-0.095806	0.024277	-0.076930	0.086164	-0.017255	-0.023233	0.009224	-0.027217	-0.095634	-0.028275	-0.019032	0.111341	-0.054177	-0.048919	-0.081246	-0.065600	-0.061361	1.000000	-0.046571	-0.072109	-0.086651	-0.095937	-0.119147	-0.093004	0.043886	0.056693	0.060839	-0.016202	0.002118	0.027660	-0.033823	-0.068702	0.022812	0.096974	0.005982	-0.014765	-0.061731	0.003333	-0.026463	-0.042039	-0.020033	0.067733	0.015854	-0.032291	0.006311	0.023252	-0.025348	0.016437	0.017788	-0.053874	-0.014448	0.042186	0.002464	-0.026085	-0.034171	-0.005056	0.033345
occupation_Retired	-0.005217	0.019366	-0.016270	0.321133	0.118565	0.005316	0.001231	-0.010729	0.010729	-0.041164	-0.009228	0.008271	-0.000281	-0.006482	0.003581	-0.006774	0.012490	-0.014512	-0.011853	0.019765	-0.008378	-0.003474	0.006698	-0.001118	0.004056	-0.007304	0.007304	-0.004151	0.004151	0.107438	0.007639	-0.103033	0.019710	0.175122	0.026813	-0.012149	0.079051	0.079111	-0.016509	-0.102574	-0.046996	-0.042435	-0.070477	-0.056905	-0.053228	-0.046571	1.000000	-0.062552	-0.075166	-0.083221	-0.103354	-0.052639	0.000224	-0.087671	0.049544	0.084707	-0.013991	-0.030266	0.054629	0.005915	-0.056447	-0.061124	-0.034475	-0.073177	0.157684	-0.089450	0.046639	-0.062513	-0.015138	0.111784	-0.079843	0.027375	-0.046423	0.123167	-0.021988	0.025145	0.002988	0.003724	-0.052799	0.036579	-0.019590	-0.049443	-0.015505	0.036136	0.011305
occupation_Sales & Related	0.004541	-0.007178	-0.000336	-0.011674	-0.052012	-0.011533	0.001738	0.008398	-0.008398	-0.007942	0.011089	-0.014230	0.005296	0.011001	0.002131	-0.042041	0.018133	0.000680	0.000175	-0.000631	0.009178	-0.006031	0.002415	-0.004264	-0.001826	0.001896	-0.001896	-0.060820	0.060820	0.056302	-0.079180	0.052317	0.014403	-0.031205	-0.053835	0.041599	-0.060494	0.011564	-0.025563	0.034120	-0.072768	-0.065706	-0.109126	-0.088111	-0.082418	-0.072109	-0.062552	1.000000	-0.116386	-0.128859	-0.160033	-0.056597	0.048166	0.035802	0.009437	-0.007627	-0.006507	0.072372	-0.085493	-0.022185	0.031778	-0.004806	-0.017943	-0.052615	0.030831	-0.020204	-0.002654	-0.035556	-0.020249	0.068320	-0.014386	0.050245	-0.043691	-0.013884	0.023788	0.042824	0.067688	-0.104322	-0.040156	-0.040987	0.016244	0.010292	-0.045898	-0.035274	0.039597
occupation_Student	-0.011189	0.018359	-0.013145	-0.367491	-0.219488	-0.009668	0.004441	0.024123	-0.024123	0.034743	0.013414	-0.030641	0.021925	0.046206	0.009315	-0.078375	-0.020444	0.007623	-0.013420	0.004858	-0.000931	0.012305	0.010012	-0.017273	-0.008687	0.013074	-0.013074	-0.084442	0.084442	-0.025972	-0.236622	0.179317	0.101179	-0.037498	-0.068254	-0.085743	-0.078550	0.010766	-0.030718	0.185417	-0.087443	-0.078956	-0.131132	-0.105880	-0.099038	-0.086651	-0.075166	-0.116386	1.000000	-0.154844	-0.192305	-0.011084	0.053321	0.010645	-0.084408	-0.044419	0.003073	-0.082381	-0.018063	0.184484	-0.018813	-0.016614	0.038612	-0.029778	0.038380	0.025991	0.056578	0.101881	-0.067305	-0.069811	0.023523	0.015172	-0.018167	-0.048614	0.042448	-0.030315	-0.002597	0.009207	0.012009	0.069085	-0.005290	-0.031133	-0.013532	0.004793	0.024391
occupation_Unemployed	-0.004028	-0.030605	0.016054	-0.085332	-0.014590	-0.006796	0.009478	0.023126	-0.023126	-0.021470	0.019608	-0.023605	0.007565	0.024165	-0.017482	0.003975	-0.019200	0.025479	0.021941	-0.035579	0.006447	0.002541	-0.011409	0.008487	-0.001882	0.008505	-0.008505	0.138008	-0.138008	-0.063244	0.016498	0.053558	-0.046048	-0.041516	0.122863	-0.048550	-0.083929	0.100014	0.085601	-0.031893	-0.096814	-0.087418	-0.145185	-0.117227	-0.109652	-0.095937	-0.083221	-0.128859	-0.154844	1.000000	-0.212914	-0.051667	-0.063684	-0.036047	0.033258	-0.044693	-0.013728	-0.054875	0.059818	0.215853	-0.033513	-0.023752	0.021016	-0.047505	0.076196	-0.029725	-0.006628	-0.006865	0.030510	0.008180	0.072155	-0.055419	-0.006620	-0.004992	-0.045296	-0.055097	-0.010977	-0.019437	0.108537	-0.017916	-0.040590	0.025560	0.045180	-0.030202	0.054765
occupation_others	0.015436	-0.008402	-0.003200	0.083528	0.044000	0.011854	-0.003952	-0.013779	0.013779	0.027282	-0.012741	0.012582	-0.001730	-0.002619	0.001308	-0.004664	0.007151	0.014672	0.006271	-0.015551	0.003695	0.004236	-0.013215	0.007461	0.001768	-0.004437	0.004437	-0.002080	0.002080	0.052240	-0.030382	-0.025994	0.018495	0.102229	-0.001880	-0.047499	0.035814	-0.003863	0.008255	0.022818	-0.120236	-0.108566	-0.180310	-0.145587	-0.136179	-0.119147	-0.103354	-0.160033	-0.192305	-0.212914	1.000000	-0.001717	0.032304	0.063347	0.036778	-0.017032	-0.069436	-0.037744	-0.002749	-0.050221	0.060633	-0.018838	0.012149	0.037715	-0.076297	-0.016015	-0.023439	-0.031760	0.045328	0.008772	0.008058	0.001859	-0.044871	0.033902	-0.018146	-0.012075	-0.046160	0.047964	0.035136	-0.005950	-0.043360	0.030209	0.045042	0.057013	-0.061778
income_$100000 or More	-0.001656	-0.015293	0.008171	0.040430	-0.011575	0.008197	-0.010299	-0.008193	0.008193	0.012450	-0.001895	0.009667	-0.009267	-0.020215	0.000438	0.017893	0.017816	-0.003240	0.008602	-0.004313	0.002172	-0.001335	-0.002937	0.009507	-0.004715	-0.015506	0.015506	-0.065082	0.065082	-0.083096	0.211600	-0.081919	-0.115818	-0.040247	-0.040840	0.093377	0.038034	-0.090909	-0.032970	-0.042684	0.016732	0.156444	0.120365	-0.014668	0.000835	-0.093004	-0.052639	-0.056597	-0.011084	-0.051667	-0.001717	1.000000	-0.164819	-0.175085	-0.163501	-0.155086	-0.106214	-0.108843	-0.110265	-0.119016	0.008198	0.039741	-0.039763	0.128534	-0.131834	-0.003162	-0.017615	0.010866	0.037376	-0.029326	-0.006476	-0.007242	0.098228	-0.059732	-0.043911	-0.032550	0.089291	0.066950	-0.113472	-0.015427	0.108867	0.068404	-0.059197	-0.027902	-0.110045
income_
24999	-0.022659	0.008177	-0.023636	-0.014508	-0.032529	-0.017115	0.010844	-0.005922	0.005922	0.008946	-0.006503	-0.011717	0.020081	0.031681	-0.001331	-0.045890	-0.009449	0.000155	-0.002455	0.001792	-0.003952	-0.001752	0.008823	-0.007722	0.001235	0.017406	-0.017406	0.029275	-0.029275	0.186269	-0.193026	0.037583	0.076239	0.108573	0.021142	-0.073444	-0.069271	0.093413	0.138158	0.038425	0.063394	-0.039511	-0.074817	-0.038139	-0.016939	0.043886	0.000224	0.048166	0.053321	-0.063684	0.032304	-0.164819	1.000000	-0.179108	-0.167257	-0.158649	-0.108654	-0.111344	-0.112799	-0.121751	0.026525	0.015178	-0.036765	-0.054857	0.032158	-0.014871	0.032060	-0.025843	0.047468	-0.044159	-0.036015	0.010174	0.036343	0.015669	-0.044920	0.021047	-0.032261	-0.014786	0.041768	-0.054078	-0.111035	-0.036925	0.044373	0.005198	0.129191
income_
37499	-0.001855	0.022645	-0.019367	-0.042449	-0.023649	-0.002832	-0.007765	0.000554	-0.000554	0.023881	-0.009975	-0.001315	0.011549	0.010625	0.009301	-0.021897	-0.012268	-0.011104	-0.024293	0.026933	0.001880	0.001790	-0.001206	-0.006839	0.003281	0.011889	-0.011889	0.102300	-0.102300	0.032977	-0.099329	0.045887	0.064498	-0.043736	-0.041785	-0.017657	-0.051166	0.005094	-0.035828	0.084578	-0.014655	-0.068143	-0.071132	0.073494	-0.009020	0.056693	-0.087671	0.035802	0.010645	-0.036047	0.063347	-0.175085	-0.179108	1.000000	-0.177675	-0.168531	-0.115422	-0.118279	-0.119825	-0.129334	-0.033490	0.019987	-0.045766	0.009922	0.022114	0.006558	0.008158	-0.046528	-0.039089	0.059294	0.089998	-0.033034	-0.067525	-0.014743	-0.005428	0.010192	-0.076714	-0.051622	0.052119	0.198066	-0.004993	-0.064406	-0.064329	-0.025906	0.105577
income_
49999	-0.007236	0.019788	0.009082	0.060302	0.053377	0.012508	0.008645	0.012150	-0.012150	-0.006739	0.015553	-0.011816	-0.001982	0.008168	0.002038	0.000167	-0.017947	-0.011299	-0.003138	0.010664	-0.009208	0.001685	0.009049	0.002205	-0.005293	-0.014219	0.014219	0.038618	-0.038618	0.016565	-0.014650	0.029351	-0.016617	-0.040843	0.034815	-0.053751	-0.016149	0.058956	-0.033458	0.018898	-0.002359	-0.038547	-0.017259	-0.015237	-0.028081	0.060839	0.049544	0.009437	-0.084408	0.033258	0.036778	-0.163501	-0.167257	-0.177675	1.000000	-0.157381	-0.107785	-0.110454	-0.111897	-0.120777	-0.043163	-0.030647	0.050333	0.005212	0.030257	-0.049849	-0.034530	0.061307	0.078220	-0.044625	-0.021342	0.003507	0.022443	-0.010636	0.046769	0.079614	-0.054590	0.062843	-0.072125	-0.053646	0.003781	-0.060071	0.154902	0.008008	-0.037775
income_
62499	-0.012163	0.016444	-0.006239	-0.012958	0.025097	-0.003584	-0.007173	-0.021262	0.021262	0.022367	-0.016901	0.023864	-0.010586	-0.047765	0.017571	0.025112	0.032656	-0.026271	-0.017007	0.032328	-0.007048	-0.008063	0.019900	0.001940	-0.009939	-0.025355	0.025355	-0.061552	0.061552	-0.047498	0.097589	-0.077724	0.008078	-0.038741	0.014246	0.055434	0.019244	-0.050586	-0.031736	-0.045945	-0.021583	-0.056917	0.055100	0.098635	-0.000447	-0.016202	0.084707	-0.007627	-0.044419	-0.044693	-0.017032	-0.155086	-0.158649	-0.168531	-0.157381	1.000000	-0.102238	-0.104769	-0.106138	-0.114561	-0.024455	-0.006925	-0.002935	-0.025123	0.047270	0.042449	-0.027720	0.002841	-0.092788	0.075819	0.004367	0.002043	-0.071975	0.029951	0.093319	0.043421	-0.004900	-0.045855	0.002948	-0.050885	0.050510	-0.010371	-0.016362	-0.007774	-0.036093
income_
74999	0.008208	0.002042	0.006789	0.004943	0.029150	0.012977	0.002978	-0.002815	0.002815	-0.017256	-0.001197	0.007863	-0.007883	-0.011011	0.004420	0.021615	-0.008535	-0.003739	0.000340	0.002459	-0.010009	0.005250	0.003594	-0.006248	0.004691	0.013421	-0.013421	-0.051457	0.051457	-0.010615	0.020818	-0.059312	0.061717	-0.026532	-0.032503	0.072253	0.071584	-0.072471	-0.021735	-0.063349	-0.024124	0.074264	0.069298	0.005664	0.013662	0.002118	-0.013991	-0.006507	0.003073	-0.013728	-0.069436	-0.106214	-0.108654	-0.115422	-0.107785	-0.102238	1.000000	-0.071753	-0.072691	-0.078460	0.017599	0.088457	-0.004729	-0.080394	0.010241	-0.004776	0.023965	-0.010634	0.041883	-0.052157	-0.017061	0.047616	-0.008026	-0.023875	-0.028948	-0.049440	0.036572	0.046187	-0.021894	0.017483	0.027694	0.013936	0.002389	-0.007661	-0.031838
income_
87499	0.025622	-0.049802	0.029265	0.028168	0.004136	0.002909	-0.001547	-0.001399	0.001399	-0.052045	0.007028	0.010497	-0.019198	-0.015295	-0.017078	0.024772	0.030112	0.030495	0.033062	-0.047860	0.018984	-0.006682	-0.022320	0.004123	0.011353	-0.005895	0.005895	0.016050	-0.016050	-0.056136	0.082050	-0.030370	-0.031046	-0.027189	0.080980	0.052919	0.050709	-0.025258	-0.022273	-0.122576	0.077386	0.085597	0.049860	-0.052955	0.019632	0.027660	-0.030266	0.072372	-0.082381	-0.054875	-0.037744	-0.108843	-0.111344	-0.118279	-0.110454	-0.104769	-0.071753	1.000000	-0.074490	-0.080402	0.022108	-0.082464	0.086960	0.020574	-0.019204	-0.056516	0.006218	-0.060950	-0.027241	0.122860	-0.002080	-0.010711	0.062041	-0.031694	-0.029664	-0.010150	0.061140	-0.020982	-0.031029	-0.035712	0.008388	0.024124	-0.039991	0.018728	-0.034490
income_
99999	0.016422	-0.013917	0.004392	0.023549	0.087151	0.008655	-0.011078	-0.002900	0.002900	-0.026277	-0.003968	0.010031	-0.007604	-0.011788	-0.017311	0.060910	-0.011103	0.008495	0.003501	-0.008903	0.007491	0.002034	-0.010726	0.006520	-0.001570	0.004690	-0.004690	-0.049523	0.049523	-0.056870	0.126100	-0.056630	-0.054168	-0.027544	-0.040012	0.022417	0.083541	-0.025260	-0.022564	-0.042906	-0.036387	-0.021729	-0.032073	-0.005648	0.125253	-0.033823	0.054629	-0.085493	-0.018063	0.059818	-0.002749	-0.110265	-0.112799	-0.119825	-0.111897	-0.106138	-0.072691	-0.074490	1.000000	-0.081453	0.055636	-0.083542	-0.007755	0.017180	-0.010463	0.056260	0.006834	0.007073	0.016072	-0.085337	0.055573	-0.035675	-0.103578	0.077283	-0.030052	-0.042244	-0.013061	-0.048553	0.123866	-0.036179	-0.018493	0.015589	0.004928	0.059080	-0.069121
income_Less than $12500	0.011502	-0.009896	0.004806	-0.089476	-0.122924	-0.018720	0.016403	0.034502	-0.034502	0.010130	0.023401	-0.035941	0.018008	0.054170	-0.007593	-0.064936	-0.020951	0.032898	0.015689	-0.036133	0.004776	0.008856	-0.014697	-0.002389	0.005714	0.019634	-0.019634	0.011903	-0.011903	-0.026774	-0.200689	0.187039	-0.003236	0.145111	0.010469	-0.135100	-0.083420	0.087693	0.055421	0.134620	-0.069330	-0.062601	-0.085441	-0.083948	-0.078523	-0.068702	0.005915	-0.022185	0.184484	0.215853	-0.050221	-0.119016	-0.121751	-0.129334	-0.120777	-0.114561	-0.078460	-0.080402	-0.081453	1.000000	-0.002671	0.025437	0.029437	-0.043564	0.017244	0.029888	0.014584	0.059511	-0.069679	-0.008479	-0.079070	0.033626	0.026217	0.030685	0.027771	-0.061307	0.029898	0.002601	0.041354	0.005641	-0.070946	0.087624	-0.043729	-0.002929	0.049641
Bar_1~3	0.008443	-0.027135	0.000988	-0.071250	-0.079119	-0.011144	-0.008547	-0.001160	0.001160	0.051247	-0.004932	0.007491	-0.003699	-0.006827	0.001745	-0.019922	0.028874	0.016695	0.022251	-0.029422	0.010491	0.001873	-0.006080	-0.005091	-0.000305	-0.009297	0.009297	-0.095020	0.095020	-0.009183	-0.060810	0.011150	0.082691	-0.049355	-0.081449	0.067052	0.007008	-0.047919	0.063220	-0.008196	-0.076948	0.017906	0.024947	-0.063445	0.050558	0.022812	-0.056447	0.031778	-0.018813	-0.033513	0.060633	0.008198	0.026525	-0.033490	-0.043163	-0.024455	0.017599	0.022108	0.055636	-0.002671	1.000000	-0.149693	-0.084429	-0.302447	-0.416858	-0.028170	0.168666	0.031965	-0.062614	-0.064150	-0.050411	0.149882	-0.077097	-0.042661	-0.053848	-0.008423	0.083765	-0.022210	-0.074949	0.008183	0.035013	0.175935	0.017013	-0.039987	-0.103529
Bar_4~8	-0.020482	0.013898	-0.017535	-0.090726	-0.118262	0.011679	0.019686	-0.001094	0.001094	0.048396	-0.008736	-0.004310	0.013765	0.012756	0.029294	-0.052016	-0.018328	-0.010510	-0.017561	0.021279	-0.015836	0.000187	0.015909	-0.005130	-0.000031	0.008756	-0.008756	-0.124340	0.124340	-0.030856	-0.134197	0.175355	-0.026361	-0.030493	0.039106	0.090629	-0.036176	-0.059666	0.048788	-0.064101	0.018318	0.056654	-0.071488	-0.014514	0.096858	0.096974	-0.061124	-0.004806	-0.016614	-0.023752	-0.018838	0.039741	0.015178	0.019987	-0.030647	-0.006925	0.088457	-0.082464	-0.083542	0.025437	-0.149693	1.000000	-0.052163	-0.186862	-0.257548	0.085369	0.041329	0.051475	-0.083178	-0.067775	-0.021891	0.066539	0.046201	-0.091863	-0.033269	0.017940	0.077215	-0.003257	-0.100676	-0.040052	0.076049	0.004555	-0.003153	-0.027906	-0.053481
Bar_gt8	0.016687	-0.009072	-0.013993	-0.100733	-0.104119	-0.013447	-0.008656	-0.015237	0.015237	0.003763	-0.010208	0.005695	0.003682	0.006191	0.039624	-0.041946	-0.032867	-0.004573	0.015649	-0.008809	-0.013530	0.003080	0.001021	0.008056	0.001686	-0.018186	0.018186	-0.020980	0.020980	-0.035509	-0.116713	0.133437	0.003701	-0.017199	-0.054291	-0.009131	-0.040607	0.164289	-0.014089	-0.013548	0.117298	-0.036214	-0.060144	-0.048562	-0.007184	0.005982	-0.034475	-0.017943	0.038612	0.021016	0.012149	-0.039763	-0.036765	-0.045766	0.050333	-0.002935	-0.004729	0.086960	-0.007755	0.029437	-0.084429	-0.052163	1.000000	-0.105393	-0.145261	-0.022948	0.019300	0.171900	-0.060379	-0.043133	0.001944	-0.079638	0.157495	-0.037552	-0.018764	-0.129579	0.082796	0.113260	-0.011503	-0.022590	-0.033692	0.003703	0.413160	-0.081573	-0.013303
Bar_less1	-0.000235	-0.000889	0.008864	0.024515	0.008735	-0.002124	0.002537	0.008606	-0.008606	-0.002160	0.007156	-0.007375	0.001328	-0.002304	-0.020765	-0.006584	0.043484	0.001988	-0.001391	-0.000369	0.003124	0.002797	0.000209	-0.008681	0.001066	0.008190	-0.008190	0.041434	-0.041434	-0.024806	0.169005	-0.092694	-0.080397	-0.028224	-0.004794	-0.046105	0.112267	-0.029523	-0.050470	-0.009794	-0.018100	0.045177	0.085976	0.042674	0.001556	-0.014765	-0.073177	-0.052615	-0.029778	-0.047505	0.037715	0.128534	-0.054857	0.009922	0.005212	-0.025123	-0.080394	0.020574	0.017180	-0.043564	-0.302447	-0.186862	-0.105393	1.000000	-0.520365	0.063504	-0.049633	-0.057103	0.111857	-0.105366	0.016956	-0.015068	0.043402	-0.023126	-0.067219	0.030325	-0.105023	0.033783	0.079127	-0.057029	0.010289	-0.046375	-0.064596	0.053693	-0.030224
Bar_never	-0.000647	0.017825	0.005814	0.120314	0.157595	0.008825	-0.003615	-0.001086	0.001086	-0.067808	0.005870	0.001160	-0.007243	-0.001723	-0.012543	0.065492	-0.041077	-0.007734	-0.011994	0.014937	0.002255	-0.005181	-0.004628	0.012115	-0.001271	0.001264	-0.001264	0.116138	-0.116138	0.059226	0.011180	-0.069183	0.019907	0.088202	0.066043	-0.060242	-0.073030	0.043373	-0.027924	0.056202	0.028222	-0.074989	-0.037131	0.036977	-0.094270	-0.061731	0.157684	0.030831	0.038380	0.076196	-0.076297	-0.131834	0.032158	0.022114	0.030257	0.047270	0.010241	-0.019204	-0.010463	0.017244	-0.416858	-0.257548	-0.145261	-0.520365	1.000000	-0.075236	-0.120468	-0.061168	0.016487	0.199664	0.036866	-0.117441	-0.056602	0.119729	0.129176	0.012994	-0.043868	-0.049139	0.049362	0.075244	-0.068978	-0.103228	-0.093055	0.026876	0.145195
CoffeeHouse_1~3	-0.006406	0.015184	-0.010417	-0.071698	0.001345	0.001833	-0.006199	-0.023066	0.023066	0.100910	-0.015848	0.020925	-0.008249	-0.030009	0.010270	0.030429	0.007409	-0.016238	-0.018636	0.026284	-0.005323	-0.001202	0.007677	-0.009861	0.004865	-0.020546	0.020546	0.038794	-0.038794	-0.064109	0.018753	0.006552	0.016402	-0.058639	-0.008242	-0.025573	0.037566	-0.040333	0.043435	0.016919	-0.003114	0.052195	0.002794	0.050987	0.026979	0.003333	-0.089450	-0.020204	0.025991	-0.029725	-0.016015	-0.003162	-0.014871	0.006558	-0.049849	0.042449	-0.004776	-0.056516	0.056260	0.029888	-0.028170	0.085369	-0.022948	0.063504	-0.075236	1.000000	-0.236194	-0.181894	-0.367095	-0.323288	0.055657	0.017908	-0.028690	-0.066195	-0.024230	0.090111	0.049259	-0.070712	-0.105784	-0.047324	0.088740	0.029677	-0.017310	-0.021122	-0.087628
CoffeeHouse_4~8	0.012283	-0.036589	0.002825	0.057655	-0.005291	0.005045	0.010897	0.004211	-0.004211	0.045902	0.009334	-0.018450	0.011938	0.028298	-0.005976	-0.017924	-0.023402	0.024069	0.025009	-0.036932	-0.002019	0.007475	-0.005808	0.008809	-0.005585	0.008360	-0.008360	0.008062	-0.008062	0.045066	-0.054985	0.058770	-0.017025	-0.040496	-0.007478	0.051085	0.102101	-0.056023	-0.033173	-0.086871	0.000300	0.028473	-0.053450	0.001516	0.005438	-0.026463	0.046639	-0.002654	0.056578	-0.006628	-0.023439	-0.017615	0.032060	0.008158	-0.034530	-0.027720	0.023965	0.006218	0.006834	0.014584	0.168666	0.041329	0.019300	-0.049633	-0.120468	-0.236194	1.000000	-0.125614	-0.253512	-0.223260	0.007165	0.090056	-0.031055	-0.087729	-0.044182	-0.058466	0.120890	0.043565	-0.099628	-0.011535	-0.032226	0.092749	0.007292	0.017948	-0.046758
CoffeeHouse_gt8	0.011801	0.006067	0.003641	-0.050588	0.035077	-0.000635	-0.000273	0.015383	-0.015383	0.007888	0.007302	-0.006941	0.000679	0.005790	0.003129	0.020728	-0.035547	-0.001548	0.000299	0.000895	-0.000146	0.005661	-0.000342	-0.004141	-0.001643	-0.007506	0.007506	-0.008350	0.008350	0.054652	-0.010203	0.010263	-0.020490	-0.031186	-0.098445	-0.075743	0.123887	0.056679	0.051122	0.004329	-0.019429	-0.010646	0.052314	0.006076	0.007505	-0.042039	-0.062513	-0.035556	0.101881	-0.006865	-0.031760	0.010866	-0.025843	-0.046528	0.061307	0.002841	-0.010634	-0.060950	0.007073	0.059511	0.031965	0.051475	0.171900	-0.057103	-0.061168	-0.181894	-0.125614	1.000000	-0.195230	-0.171933	-0.156544	0.091699	0.174325	-0.079620	0.030268	-0.078888	-0.051683	0.327035	-0.082947	-0.040962	0.000580	0.071807	0.199706	-0.024522	-0.089785
CoffeeHouse_less1	-0.025051	-0.004704	-0.010298	0.004665	-0.004639	-0.013128	-0.008556	-0.008851	0.008851	-0.016039	-0.019604	0.018409	-0.001564	-0.011459	0.001809	-0.021814	0.038874	0.008985	0.007077	-0.012034	0.008519	-0.005879	0.004751	-0.010029	0.000451	0.012349	-0.012349	0.044713	-0.044713	0.021099	0.041999	-0.073120	0.026174	0.007613	0.081275	0.001672	-0.105171	-0.004392	-0.051558	0.038476	0.035864	0.024488	0.042743	-0.056640	-0.022074	-0.020033	-0.015138	-0.020249	-0.067305	0.030510	0.045328	0.037376	0.047468	-0.039089	0.078220	-0.092788	0.041883	-0.027241	0.016072	-0.069679	-0.062614	-0.083178	-0.060379	0.111857	0.016487	-0.367095	-0.253512	-0.195230	1.000000	-0.346992	0.046942	-0.130845	0.031836	0.101719	-0.068669	0.068175	-0.057659	-0.112364	0.090446	-0.059000	-0.051544	-0.050623	-0.037986	0.067988	0.015545
CoffeeHouse_never	0.015258	0.015318	0.016918	0.055432	-0.015605	0.008362	0.006725	0.019396	-0.019396	-0.129728	0.024613	-0.021334	-0.000092	0.015987	-0.009689	-0.007304	-0.005967	-0.011546	-0.009056	0.015434	-0.001828	-0.002433	-0.007962	0.016363	0.000194	0.006185	-0.006185	-0.088498	0.088498	-0.030019	-0.012039	0.015947	-0.017031	0.106354	-0.005882	0.033379	-0.093525	0.054164	0.003092	0.010012	-0.022168	-0.095982	-0.039536	0.002445	-0.013778	0.067733	0.111784	0.068320	-0.069811	0.008180	0.008772	-0.029326	-0.044159	0.059294	-0.044625	0.075819	-0.052157	0.122860	-0.085337	-0.008479	-0.064150	-0.067775	-0.043133	-0.105366	0.199664	-0.323288	-0.223260	-0.171933	-0.346992	1.000000	-0.008231	-0.014553	-0.095620	0.085227	0.113926	-0.064461	-0.053904	-0.062494	0.149947	0.148346	-0.010476	-0.100866	-0.081526	-0.048860	0.172138
CarryAway_1~3	-0.007037	0.016149	-0.006344	-0.086128	-0.038344	-0.007758	-0.004334	-0.013232	0.013232	0.019711	-0.006611	0.011508	-0.006653	-0.011420	0.005176	-0.008886	0.020823	-0.017030	-0.019115	0.027232	-0.004364	-0.013513	0.014502	-0.007028	0.005824	-0.012382	0.012382	-0.024844	0.024844	-0.028376	0.010944	-0.053643	0.063431	0.023612	0.005100	0.058602	-0.047365	0.001961	-0.022110	-0.023916	-0.016708	-0.027961	0.016925	-0.011360	-0.054268	0.015854	-0.079843	-0.014386	0.023523	0.072155	0.008058	-0.006476	-0.036015	0.089998	-0.021342	0.004367	-0.017061	-0.002080	0.055573	-0.079070	-0.050411	-0.021891	0.001944	0.016956	0.036866	0.055657	0.007165	-0.156544	0.046942	-0.008231	1.000000	-0.558552	-0.295625	-0.324637	-0.085773	0.060039	-0.065294	-0.159773	0.127192	0.003352	0.093098	-0.051999	-0.065009	-0.024196	-0.019342
CarryAway_4~8	0.015738	0.020692	-0.003013	0.043656	0.099735	0.012492	-0.000236	0.009788	-0.009788	0.016083	0.005337	-0.008094	0.003988	-0.000527	-0.009881	0.031451	-0.014142	-0.014959	-0.012179	0.020344	-0.000788	0.008844	-0.001012	-0.013181	0.003653	0.007672	-0.007672	0.005869	-0.005869	0.041619	0.029104	-0.032433	-0.019247	0.004198	0.014450	-0.018844	0.037116	-0.028117	0.028392	-0.007249	-0.046167	-0.003890	-0.017378	0.036203	0.033815	-0.032291	0.027375	0.050245	0.015172	-0.055419	0.001859	-0.007242	0.010174	-0.033034	0.003507	0.002043	0.047616	-0.010711	-0.035675	0.033626	0.149882	0.066539	-0.079638	-0.015068	-0.117441	0.017908	0.090056	0.091699	-0.130845	-0.014553	-0.558552	1.000000	-0.268607	-0.294967	-0.077934	0.065277	0.053022	0.035939	-0.158072	-0.063215	-0.042256	0.107958	-0.050160	0.032237	-0.041364
CarryAway_gt8	-0.001263	-0.024398	-0.002919	-0.050042	-0.130471	-0.010291	-0.004004	-0.002179	0.002179	0.010280	-0.008330	0.007790	-0.000628	0.002307	0.025689	-0.055302	0.009219	0.020901	0.017811	-0.029040	0.004874	0.004299	-0.011049	0.010720	-0.004282	0.000369	-0.000369	0.020460	-0.020460	0.001648	-0.041140	0.076625	-0.035828	-0.037806	-0.042555	-0.070298	-0.011718	0.039695	-0.030970	0.089091	0.089585	0.022310	0.054222	-0.032718	0.055766	0.006311	-0.046423	-0.043691	-0.018167	-0.006620	-0.044871	0.098228	0.036343	-0.067525	0.022443	-0.071975	-0.008026	0.062041	-0.103578	0.026217	-0.077097	0.046201	0.157495	0.043402	-0.056602	-0.028690	-0.031055	0.174325	0.031836	-0.095620	-0.295625	-0.268607	1.000000	-0.156117	-0.041248	-0.193285	0.156478	0.295631	-0.154002	-0.049658	0.048173	-0.006523	0.196723	-0.059683	-0.048625
CarryAway_less1	-0.008512	-0.022993	0.013009	0.096747	0.018258	0.003322	0.012054	0.005395	-0.005395	-0.054121	0.008712	-0.012452	0.005631	0.015713	-0.016850	0.017319	-0.018196	0.020780	0.020177	-0.030788	0.004165	0.002541	-0.006004	0.013712	-0.009958	0.008505	-0.008505	0.023095	-0.023095	-0.011275	-0.028983	0.043767	-0.012738	0.000615	0.007534	0.025303	0.040197	0.006988	0.024087	-0.067775	-0.015066	0.029840	-0.056785	0.007233	-0.031910	0.023252	0.123167	-0.013884	-0.048614	-0.004992	0.033902	-0.059732	0.015669	-0.014743	-0.010636	0.029951	-0.023875	-0.031694	0.077283	0.030685	-0.042661	-0.091863	-0.037552	-0.023126	0.119729	-0.066195	-0.087729	-0.079620	0.101719	0.085227	-0.324637	-0.294967	-0.156117	1.000000	-0.045296	0.009617	-0.118720	-0.106410	0.171430	0.130697	-0.096640	-0.058582	-0.022430	0.031698	0.116088
CarryAway_never	-0.005501	-0.013305	0.008047	0.032506	0.074967	0.000803	-0.006682	0.005660	-0.005660	-0.013277	0.003318	0.000615	-0.004046	-0.004818	-0.003407	0.014678	-0.000365	0.010114	0.018734	-0.021900	-0.005456	0.000601	-0.007338	0.011617	0.003504	-0.006853	0.006853	-0.052060	0.052060	-0.022647	0.044014	0.005629	-0.049717	-0.010969	0.019243	-0.048910	-0.045153	-0.029961	-0.008986	0.088649	0.052550	-0.023097	0.020031	-0.030973	0.029970	-0.025348	-0.021988	0.023788	0.042448	-0.045296	-0.018146	-0.043911	-0.044920	-0.005428	0.046769	0.093319	-0.028948	-0.029664	-0.030052	0.027771	-0.053848	-0.033269	-0.018764	-0.067219	0.129176	-0.024230	-0.044182	0.030268	-0.068669	0.113926	-0.085773	-0.077934	-0.041248	-0.045296	1.000000	0.003768	-0.028177	0.005682	0.029468	-0.014408	-0.064576	-0.026911	-0.016134	0.046370	0.036624
RestaurantLessThan20_1~3	-0.009209	-0.014233	0.015420	0.058503	0.087387	-0.008967	0.001562	0.007730	-0.007730	-0.008101	0.014398	-0.006429	-0.007047	-0.001565	-0.009270	0.013612	0.004182	-0.005099	0.007293	-0.001944	0.012667	0.001758	-0.006639	0.000270	-0.005639	0.001979	-0.001979	0.018704	-0.018704	0.021508	-0.007290	-0.043857	0.050688	0.014573	0.084322	0.035943	-0.065675	-0.005870	-0.032764	-0.029593	-0.005136	0.054654	-0.041399	0.083472	-0.020819	0.016437	0.025145	0.042824	-0.030315	-0.055097	-0.012075	-0.032550	0.021047	0.010192	0.079614	0.043421	-0.049440	-0.010150	-0.042244	-0.061307	-0.008423	0.017940	-0.129579	0.030325	0.012994	0.090111	-0.058466	-0.078888	0.068175	-0.064461	0.060039	0.065277	-0.193285	0.009617	0.003768	1.000000	-0.548466	-0.292895	-0.390321	-0.115076	-0.001889	-0.123194	-0.128862	0.129011	-0.044257
RestaurantLessThan20_4~8	0.009906	0.002124	-0.007016	-0.019303	-0.049096	0.006287	-0.001143	-0.009922	0.009922	0.014451	-0.008535	0.004839	0.002989	-0.008805	-0.009615	-0.017785	0.048329	0.011935	-0.005388	-0.004513	0.000159	-0.001969	0.006637	-0.012763	0.004149	-0.002188	0.002188	0.028074	-0.028074	-0.067103	0.063419	-0.036955	-0.002298	0.009817	-0.076960	-0.062741	0.095283	-0.025869	0.042331	0.045402	0.067455	-0.039400	0.021715	-0.033260	-0.021189	0.017788	0.002988	0.067688	-0.002597	-0.010977	-0.046160	0.089291	-0.032261	-0.076714	-0.054590	-0.004900	0.036572	0.061140	-0.013061	0.029898	0.083765	0.077215	0.082796	-0.105023	-0.043868	0.049259	0.120890	-0.051683	-0.057659	-0.053904	-0.065294	0.053022	0.156478	-0.118720	-0.028177	-0.548466	1.000000	-0.210412	-0.280402	-0.082669	0.111445	0.113554	-0.062247	-0.067273	-0.087183
RestaurantLessThan20_gt8	0.003295	0.036532	-0.018090	-0.006447	-0.028531	-0.007977	-0.013931	-0.005188	0.005188	0.027299	-0.016397	0.019870	-0.006477	-0.013938	0.023514	-0.002680	-0.009852	-0.024252	-0.015894	0.029994	-0.017797	-0.002714	0.017370	-0.006020	0.003500	-0.017791	0.017791	-0.012878	0.012878	0.073931	-0.001165	0.000362	-0.029054	-0.033611	-0.066315	0.057292	0.035981	0.026291	-0.027534	-0.053409	-0.078379	0.039254	0.043750	0.017387	0.065844	-0.053874	0.003724	-0.104322	0.009207	-0.019437	0.047964	0.066950	-0.014786	-0.051622	0.062843	-0.045855	0.046187	-0.020982	-0.048553	0.002601	-0.022210	-0.003257	0.113260	0.033783	-0.049139	-0.070712	0.043565	0.327035	-0.112364	-0.062494	-0.159773	0.035939	0.295631	-0.106410	0.005682	-0.292895	-0.210412	1.000000	-0.149742	-0.044147	0.058375	0.127611	0.394646	-0.147047	-0.103519
RestaurantLessThan20_less1	-0.006937	-0.003371	0.002011	-0.018621	-0.022860	0.007265	0.007914	0.004426	-0.004426	-0.027000	0.004341	-0.011781	0.009251	0.020757	0.003504	0.015186	-0.057384	0.002059	0.002531	-0.003464	-0.003605	0.000068	-0.004467	0.013389	-0.002216	0.012923	-0.012923	-0.046967	0.046967	0.002173	-0.037084	0.081912	-0.058284	0.000528	0.034300	-0.013391	-0.048555	0.012695	0.018449	0.018747	-0.019699	-0.047177	0.009475	-0.071952	-0.008155	-0.014448	-0.052799	-0.040156	0.012009	0.108537	0.035136	-0.113472	0.041768	0.052119	-0.072125	0.002948	-0.021894	-0.031029	0.123866	0.041354	-0.074949	-0.100676	-0.011503	0.079127	0.049362	-0.105784	-0.099628	-0.082947	0.090446	0.149947	0.127192	-0.158072	-0.154002	0.171430	0.029468	-0.390321	-0.280402	-0.149742	1.000000	-0.058832	-0.174216	-0.065395	-0.065881	0.064612	0.184093
RestaurantLessThan20_never	0.013077	-0.028256	0.001710	-0.088200	-0.031523	0.010125	0.007660	0.004269	-0.004269	-0.005135	0.000261	-0.004589	0.005042	0.009353	0.004105	-0.027684	0.003899	0.028532	0.020546	-0.036721	0.002760	0.006243	-0.025257	0.018859	0.005382	0.004276	-0.004276	-0.004150	0.004150	-0.027265	-0.083299	0.059925	0.049284	-0.013205	0.000652	-0.014180	-0.024138	0.014729	-0.010818	0.025739	0.024147	-0.027806	-0.046180	-0.037287	0.023571	0.042186	0.036579	-0.040987	0.069085	-0.017916	-0.005950	-0.015427	-0.054078	0.198066	-0.053646	-0.050885	0.017483	-0.035712	-0.036179	0.005641	0.008183	-0.040052	-0.022590	-0.057029	0.075244	-0.047324	-0.011535	-0.040962	-0.059000	0.148346	0.003352	-0.063215	-0.049658	0.130697	-0.014408	-0.115076	-0.082669	-0.044147	-0.058832	1.000000	-0.015136	-0.032398	-0.019423	-0.103350	0.184041
Restaurant20To50_1~3	0.013133	0.027768	0.001972	0.004127	-0.017801	0.000795	-0.010548	-0.004591	0.004591	0.025700	-0.001350	0.008269	-0.008199	-0.017401	-0.008421	0.010493	0.034093	-0.009725	-0.021438	0.023714	0.004677	-0.006839	0.012820	-0.005019	-0.008157	0.000381	-0.000381	0.006431	-0.006431	-0.043949	0.058835	-0.073227	0.056276	-0.059187	0.016618	0.086016	0.005963	-0.075950	-0.048485	-0.051679	0.031561	0.007922	0.009312	-0.010802	0.095669	0.002464	-0.019590	0.016244	-0.005290	-0.040590	-0.043360	0.108867	-0.111035	-0.004993	0.003781	0.050510	0.027694	0.008388	-0.018493	-0.070946	0.035013	0.076049	-0.033692	0.010289	-0.068978	0.088740	-0.032226	0.000580	-0.051544	-0.010476	0.093098	-0.042256	0.048173	-0.096640	-0.064576	-0.001889	0.111445	0.058375	-0.174216	-0.015136	1.000000	-0.145208	-0.087055	-0.586055	-0.264156
Restaurant20To50_4~8	-0.010790	-0.044886	0.008829	-0.013840	0.041497	-0.005131	-0.005738	0.010545	-0.010545	0.042780	0.001210	0.000957	-0.002322	0.003375	-0.011277	0.021197	-0.008852	0.030798	0.041455	-0.054591	0.002770	0.010051	-0.025697	0.002440	0.014955	-0.002562	0.002562	-0.009305	0.009305	0.064647	0.003333	0.017398	-0.054272	-0.024666	-0.023222	-0.037817	0.055081	0.052699	0.162055	-0.045152	-0.057519	0.082595	0.032118	-0.014535	-0.035960	-0.026085	-0.049443	0.010292	-0.031133	0.025560	0.030209	0.068404	-0.036925	-0.064406	-0.060071	-0.010371	0.013936	0.024124	0.015589	0.087624	0.175935	0.004555	0.003703	-0.046375	-0.103228	0.029677	0.092749	0.071807	-0.050623	-0.100866	-0.051999	0.107958	-0.006523	-0.058582	-0.026911	-0.123194	0.113554	0.127611	-0.065395	-0.032398	-0.145208	1.000000	-0.036279	-0.244233	-0.110084
Restaurant20To50_gt8	0.002017	-0.002316	-0.008897	-0.048116	-0.048720	-0.009103	-0.008168	-0.006398	0.006398	0.029247	-0.010595	0.007656	0.001805	0.000679	0.034144	-0.023446	-0.032446	-0.004773	0.001491	0.002320	-0.012198	0.000113	0.003048	0.002015	0.005712	-0.007040	0.007040	-0.027017	0.027017	0.027940	-0.068187	0.091641	-0.039834	-0.014787	-0.006335	-0.003931	-0.060871	0.160609	-0.012114	-0.031159	-0.034484	-0.031137	-0.051713	-0.041754	0.140279	-0.034171	-0.015505	-0.045898	-0.013532	0.045180	0.045042	-0.059197	0.044373	-0.064329	0.154902	-0.016362	0.002389	-0.039991	0.004928	-0.043729	0.017013	-0.003153	0.413160	-0.064596	-0.093055	-0.017310	0.007292	0.199706	-0.037986	-0.081526	-0.065009	-0.050160	0.196723	-0.022430	-0.016134	-0.128862	-0.062247	0.394646	-0.065881	-0.019423	-0.087055	-0.036279	1.000000	-0.146422	-0.065998
Restaurant20To50_less1	-0.004116	-0.033712	0.000285	0.068505	0.024648	0.003199	0.004475	-0.013381	0.013381	-0.022386	-0.009401	0.008928	-0.000865	-0.007222	0.005200	0.004302	0.000408	0.017644	0.021273	-0.029354	0.011474	0.002725	-0.015292	0.010300	-0.003553	-0.000477	0.000477	-0.002871	0.002871	0.002979	0.057351	-0.037989	-0.054031	0.100992	-0.002519	-0.008648	0.009956	-0.065340	-0.042895	0.045302	-0.020545	-0.034076	0.009782	0.073101	-0.083717	-0.005056	0.036136	-0.035274	0.004793	-0.030202	0.057013	-0.027902	0.005198	-0.025906	0.008008	-0.007774	-0.007661	0.018728	0.059080	-0.002929	-0.039987	-0.027906	-0.081573	0.053693	0.026876	-0.021122	0.017948	-0.024522	0.067988	-0.048860	-0.024196	0.032237	-0.059683	0.031698	0.046370	0.129011	-0.067273	-0.147047	0.064612	-0.103350	-0.586055	-0.244233	-0.146422	1.000000	-0.444296
Restaurant20To50_never	-0.003967	0.041428	-0.004746	-0.069475	-0.019109	0.001493	0.013115	0.019252	-0.019252	-0.038091	0.017542	-0.025241	0.011533	0.027752	-0.003289	-0.022200	-0.022509	-0.029556	-0.029725	0.044587	-0.017878	-0.001920	0.020261	-0.010219	0.002826	0.004512	-0.004512	0.012547	-0.012547	-0.003429	-0.121664	0.090619	0.055569	-0.044871	0.000765	-0.064346	-0.031055	0.081832	0.018319	0.040080	0.039629	-0.002936	-0.024013	-0.060140	-0.031994	0.033345	0.011305	0.039597	0.024391	0.054765	-0.061778	-0.110045	0.129191	0.105577	-0.037775	-0.036093	-0.031838	-0.034490	-0.069121	0.049641	-0.103529	-0.053481	-0.013303	-0.030224	0.145195	-0.087628	-0.046758	-0.089785	0.015545	0.172138	-0.019342	-0.041364	-0.048625	0.116088	0.036624	-0.044257	-0.087183	-0.103519	0.184093	0.184041	-0.264156	-0.110084	-0.065998	-0.444296	1.000000
# from the above correlation Age, has_children, oCoupon_GEQ15min, toCoupon_GEQ25min, direction_same and direction_opp 
# have significance in both positive and negative side of correlation.
# Customer ID, temperature and time columns can be removed from the Model
# Check Multicollinearity relations ship between variables using pearson method
corr = train_dummies.corr(method='pearson')
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize = (50,30))
sns.heatmap(corr, cmap = 'RdYlGn_r', vmax = 1.0, vmin = -1.0, mask=mask,linewidths=2.5)
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.savefig('Multicollinearity.png')

# In the above analysis we have used the pearson method hence the values which are in between -1 and 1, hence all the
# columns with 0 values can be removed based on the collinear relationship
# Based on the analysis, we can remove the columns 'customer_id', 'temperature', 'time',
#        'weather_Rainy','weather_Snowy', 'weather_Sunny'
#Dropping the columns as indentified during Correlation analysis
train_dummies.drop(columns = ['customer_id','temperature','time','weather_Rainy','weather_Snowy','weather_Sunny'],inplace=True)
test_dummies.drop(columns = ['customer_id','temperature','time','weather_Rainy','weather_Snowy','weather_Sunny'],inplace=True)
# Splitting the data in to feature and Target column
X = train_dummies.drop(columns = 'Y').copy()
y = train_dummies['Y']
print(X.shape,y.shape)
(10147, 78) (10147,)
Logistic Regression Model
#import Logistic regression and cross validation library
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_validate
# Instantiate and use Cross Validation score to get the scores for training data
model = LogisticRegression(max_iter=1000)
cross_score = cross_val_score(model,X,y,cv=5,scoring='accuracy')
cross_score
array([0.68275862, 0.67142857, 0.68408083, 0.67028093, 0.69048793])
avg_score = np.mean(cross_score)
print("The accuracy score of the model using logistic regression is: ",avg_score)
The accuracy score of the model using logistic regression is:  0.6798073743526744
X_cv_lr_results = cross_validate(model,X,y,cv=5,return_train_score=True)
X_cv_lr_results
{'fit_time': array([0.23836184, 0.3141613 , 0.35006452, 0.31814957, 0.27925396]),
 'score_time': array([0.00498557, 0.00499845, 0.00498533, 0.00399494, 0.0039885 ]),
 'test_score': array([0.68275862, 0.67142857, 0.68408083, 0.67028093, 0.69048793]),
 'train_score': array([0.68547493, 0.68683011, 0.68440503, 0.68883962, 0.68415866])}
model.fit(X,y)
LogisticRegression(max_iter=1000)
y_test = model.predict(test_dummies)
y_test
array([1, 1, 1, ..., 0, 0, 0], dtype=int64)
columnid = test_data[['customer_id']]
submit = pd.DataFrame(np.hstack((columnid,y_test.reshape(-1,1))),columns = ['customer_id','Y'])
submit
customer_id	Y
0	374679	1
1	469678	1
2	216140	1
3	184301	1
4	148720	1
...	...	...
2532	356045	0
2533	498759	1
2534	356159	0
2535	218541	0
2536	467387	0
2537 rows × 2 columns

submit.to_csv('sub12_lr.csv',index=False)
Decision Tree Model
#instantiate the model
dt=DecisionTreeClassifier(random_state=80)
#Hyper tuning the parameters to find the relevant parameter
params = {'max_depth':list(range(1,21)), 'criterion':['gini', 'entropy']}
cv_model = GridSearchCV(dt, param_grid = params,cv=5,return_train_score=True)
cv_model.fit(X,y)
X_cv_dt_results = pd.DataFrame(cv_model.cv_results_)
print(cv_model.best_params_)
{'criterion': 'gini', 'max_depth': 11}
X_cv_dt_results[['params','mean_test_score']].sort_values(by='mean_test_score', ascending=False).head()
params	mean_test_score
10	{'criterion': 'gini', 'max_depth': 11}	0.703457
11	{'criterion': 'gini', 'max_depth': 12}	0.699910
33	{'criterion': 'entropy', 'max_depth': 14}	0.699714
8	{'criterion': 'gini', 'max_depth': 9}	0.698334
9	{'criterion': 'gini', 'max_depth': 10}	0.698236
model_dt = DecisionTreeClassifier(max_depth = 11, random_state=80)
cross_score_dt = cross_val_score(model_dt,X,y,cv=5,scoring='accuracy')
cross_score_dt
array([0.72167488, 0.70837438, 0.69886644, 0.70576639, 0.68260227])
avg_score_dt = np.mean(cross_score_dt)
print("The Accuracy score of Decision Tree is ",avg_score_dt)
The Accuracy score of Decision Tree is  0.7034568704523328
#create the prediction file and save it
model_dt.fit(X,y)
y_test_dt = model_dt.predict(test_dummies)
submit1 = pd.DataFrame(np.hstack((columnid,y_test_dt.reshape(-1,1))),columns=['customer_id','Y'])
submit1.to_csv('sub12_dt.csv',index=False)
y_test_dt
array([0, 0, 0, ..., 0, 1, 1], dtype=int64)
Random Forest Tree Model
#instantiate the model
ft=RandomForestClassifier()
#Hyper tuning the parameters to find the relevant parameter
params = {'n_estimators':list(range(1,150))}
rs_ft = RandomizedSearchCV(estimator=ft,param_distributions=params,cv=5, return_train_score=True)
rs_ft.fit(X,y)
X_rscv_results = pd.DataFrame(rs_ft.cv_results_)
X_rscv_results = X_rscv_results.sort_values(by = 'param_n_estimators' )
plt.plot(X_rscv_results.param_n_estimators,X_rscv_results.mean_train_score,label='Train Score')
plt.plot(X_rscv_results.param_n_estimators,X_rscv_results.mean_test_score, label='Test Score')
plt.legend()
plt.ylim(0,1.1)
plt.show()

X_rscv_results.sort_values(by='mean_test_score', ascending=False)[['param_n_estimators','mean_test_score']].head()
param_n_estimators	mean_test_score
0	104	0.761605
3	125	0.760816
7	123	0.759929
8	141	0.757958
1	148	0.757169
# from the above we can see that 104 estimators is optimum
params = {'n_estimators':list(range(2,120))}
ft = RandomForestClassifier()
gs_ft = GridSearchCV(estimator=ft,param_grid=params,cv=5, return_train_score=True)
gs_ft.fit(X,y)
X_cv_results = pd.DataFrame(gs_ft.cv_results_)
X_cv_results = X_cv_results.sort_values(by = 'param_n_estimators' )
plt.plot(X_cv_results.param_n_estimators,X_cv_results.mean_train_score,label='Train Score')
plt.plot(X_cv_results.param_n_estimators,X_cv_results.mean_test_score, label='Test Score')
plt.legend()
plt.ylim(0,1.1)
plt.show()

X_cv_results.sort_values(by='mean_test_score', ascending=False)[['param_n_estimators','mean_test_score']].head()
param_n_estimators	mean_test_score
115	117	0.765940
101	103	0.761802
91	93	0.761702
106	108	0.761604
109	111	0.761210
gs_ft.best_params_
{'n_estimators': 117}
rf_model = RandomForestClassifier(n_estimators=137,random_state=80) 
cross_score_rf = cross_val_score(rf_model,X,y,cv=5,scoring='accuracy')
cross_score_rf.mean()
#When we execute with estimators as 117 we get accuracy score of 0.7598, hence we tried to execute using 137 which is 0.7617
0.761702700012382
rf_model.fit(X,y)
y_test_rf = rf_model.predict(test_dummies)
y_test_rf
array([1, 1, 0, ..., 0, 0, 1], dtype=int64)
# acc_score_rf=X_cv_results[X_cv_results['param_n_estimators']==60]['mean_test_score'].values[0]
# X_cv_results.mean_test_score.max()
print("The accuracy score of the model using Random Forest is: ",cross_score_rf.mean())
The accuracy score of the model using Random Forest is:  0.761702700012382
# Save to file the result of the Random Forest
submit2 = pd.DataFrame(np.hstack((columnid,y_test_rf.reshape(-1,1))),columns=['customer_id','Y'])
submit2.to_csv('sub12_rf.csv',index=False)
Random Search

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, RandomizedSearchCV, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import seaborn as sns
import matplotlib.pyplot as plt 
%matplotlib inline
C:\Users\deepak\Softwares\Anaconda\lib\site-packages\sklearn\ensemble\weight_boosting.py:29: DeprecationWarning: numpy.core.umath_tests is an internal NumPy module and should not be imported. It will be removed in a future NumPy release.
  from numpy.core.umath_tests import inner1d
df = pd.read_csv('train.csv')
df = df.apply(lambda x: x.fillna(x.value_counts().index[0]))
dummies = pd.get_dummies(df).copy()
dummies.drop(columns=['expiration_2h', 'maritalStatus_Married partner', 'customer_id', 'direction_opp', 'age_below21', 'age_21', 'occupation_Retired', 'weather_Snowy', 'destination_No Urgent Place', 'time_7AM'], inplace = True)
target_cols = 'Y'
input_cols = dummies.columns.drop(target_cols)
X = dummies[input_cols]
y_train = dummies['Y']
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X)
params = {'n_estimators':list(range(3,400))}
estimator = RandomForestClassifier()
random_search = RandomizedSearchCV(estimator = estimator, param_distributions=params, cv=5, return_train_score=True).fit(X_train, y_train)
#df_cv_results=pd.DataFrame(grid_search.cv_results_)
df_cv_results=pd.DataFrame(random_search.cv_results_)
df_cv_results = df_cv_results.sort_values(by='param_n_estimators')
random_search.best_estimator_
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=179, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
df_cv_results[df_cv_results['param_n_estimators']==179]['mean_test_score']
4    0.766532
Name: mean_test_score, dtype: float64
test_data = pd.read_csv("test.csv")
test_data = test_data.apply(lambda x: x.fillna(x.value_counts().index[0]))
test_dummies = pd.get_dummies(test_data).copy()
cust_id = test_data[['customer_id']]
test_dummies.drop(columns=['expiration_2h', 'maritalStatus_Married partner', 'customer_id', 'direction_opp', 'age_below21', 'age_21', 'occupation_Retired', 'weather_Snowy', 'destination_No Urgent Place', 'time_7AM'], inplace = True)
y_pred = random_search.best_estimator_.predict(test_dummies)
prod_squad = pd.DataFrame(np.hstack((cust_id, y_pred.reshape(-1,1))), columns=['customer_id','Y'])
prod_squad.to_csv('Pro_Squad_sub9_rf.csv',index=False)
Conclusion
# Accuracy score of Logistic Regeression Model
print("The Accuracy score of the model using logistic regression is: %.2f%%" % (avg_score*100))
# Accuracy score of Decision Tree Model
print("The Accuracy score of Decision Tree is: %.2f%%" % (avg_score_dt*100))
# Accuracy score of Random Forest Model
print("The Accuracy score of the model using Random Forest is: %.2f%%" % (cross_score_rf.mean()*100))
The Accuracy score of the model using logistic regression is: 67.98%
The Accuracy score of Decision Tree is: 70.35%
The Accuracy score of the model using Random Forest is: 76.17%
print("The Accuracy score of the model using Random Forest is: %.2f%%" % (df_cv_results[df_cv_results['param_n_estimators']==179]['mean_test_score']*100))
The Accuracy score of the model using Random Forest is: 76.65%
