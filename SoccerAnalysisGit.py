#!/usr/bin/env python
# coding: utf-8

# In[69]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns


# In[70]:


#Serie A Data
seriea = pd.read_csv('Soccer Value Analysis - 2020_2021 Season Serie A.csv')
seriea


# In[71]:


#Premier League Data
prem = pd.read_csv('Soccer Value Analysis - 2020_2021 Season Premier League.csv')
prem


# In[72]:


#La Liga Data
laliga = pd.read_csv('Soccer Value Analysis - 2020_2021 Season La Liga.csv')
laliga


# In[73]:


#Bundesliga Data
buli = pd.read_csv('Soccer Value Analysis - 2020_2021 Season Bundesliga.csv')
buli


# In[74]:


#Ligue 1 Data
ligue1 = pd.read_csv('Soccer Value Analysis - 2020_2021 Season Ligue 1.csv')
ligue1


# In[75]:


#Make League Dummy Variables
seriea['SerieA'] =  np.ones(len(seriea))
seriea['PremierLeague'] = np.zeros(len(seriea))
seriea['LaLiga'] = np.zeros(len(seriea))
seriea['Bundesliga'] = np.zeros(len(seriea))
seriea['Ligue1'] = np.zeros(len(seriea))
prem['SerieA'] = np.zeros(len(prem))
prem['PremierLeague'] =  np.ones(len(prem))
prem['LaLiga'] = np.zeros(len(prem))
prem['Bundesliga'] = np.zeros(len(prem))
prem['Ligue1'] = np.zeros(len(prem))
laliga['SerieA'] = np.zeros(len(laliga))
laliga['PremierLeague'] = np.zeros(len(laliga))
laliga['LaLiga'] = np.ones(len(laliga))
laliga['Bundesliga'] = np.zeros(len(laliga))
laliga['Ligue1'] = np.zeros(len(laliga))
buli['SerieA'] = np.zeros(len(buli))
buli['PremierLeague'] = np.zeros(len(buli))
buli['LaLiga'] = np.zeros(len(buli))
buli['Bundesliga'] = np.ones(len(buli))
buli['Ligue1'] = np.zeros(len(buli))
ligue1['SerieA'] = np.zeros(len(ligue1))
ligue1['PremierLeague'] = np.zeros(len(ligue1))
ligue1['LaLiga'] = np.zeros(len(ligue1))
ligue1['Bundesliga'] = np.zeros(len(ligue1))
ligue1['Ligue1'] = np.ones(len(ligue1))
ligue1


# In[76]:


#Combine all league dataframes
data = seriea.append(prem)
data = data.append(laliga)
data = data.append(buli)
data = data.append(ligue1)
data = data.reset_index()
data = data.drop('index',axis=1)
data


# In[77]:


from datetime import datetime

def days_between(d1):
    """Returns how old a player was (in days) by the end of the 2020-2021 season (23/05/2021)"""
    d1 = datetime.strptime(d1, "%d/%m/%Y")
    d2 = datetime.strptime('23/05/2021', "%d/%m/%Y")
    return abs((d2 - d1).days)
days_between(data['Birthday(DD/MM/YYYY)'].tolist()[0])


# In[78]:


#Apply days_between function on all players
days_old = []
for index,i in enumerate(data['Birthday(DD/MM/YYYY)'].tolist()):
    print(data['Name'].tolist()[index])
    days_old.append(days_between(i))
data['DaysOld'] = days_old


# In[79]:


#Generate xG differentials 
xg = []
xgd = []
for i in data['xG'].tolist():
    if '+' in i:
        index = i.index('+')
        xg.append(float(i[:index]))
        xgd.append(float(i[index+1:]))
    elif '-' in i:
        index = i.index('-')
        xg.append(float(i[:index]))
        xgd.append(float(i[index:]))
    else:
        xg.append(float(i))
        xgd.append(0)
data = data.drop('xG',axis=1)
data['xG'] = xg
data['xGd'] = xgd
data


# In[80]:


#Generate xA differentials
xa = []
xad = []
for i in data['xA'].tolist():
    if '+' in i:
        index = i.index('+')
        xa.append(float(i[:index]))
        xad.append(float(i[index+1:]))
    elif '-' in i:
        index = i.index('-')
        xa.append(float(i[:index]))
        xad.append(float(i[index:]))
    else:
        xa.append(float(i))
        xad.append(0)
data = data.drop('xA',axis=1)
data['xA'] = xa
data['xAd'] = xad
data


# In[81]:


#Generate NPxG differentials
npxg = []
npxgd = []
for i in data['NPxG'].tolist():
    if '+' in i:
        index = i.index('+')
        npxg.append(float(i[:index]))
        npxgd.append(float(i[index+1:]))
    elif '-' in i:
        index = i.index('-')
        npxg.append(float(i[:index]))
        npxgd.append(float(i[index:]))
    else:
        npxg.append(float(i))
        npxgd.append(0)
data = data.drop('NPxG',axis=1)
data['NPxG'] = npxg
data['NPxGd'] = npxgd
data


# In[82]:


#Players that played for multiple teams in the 2020-2021 season have a row for each team they played for
#This cell combines their cumulative stats for the season such that the data frame has 1 row per unique player
name = []
bday = []
mv = []
team = []
pos = []
apps = []
mins = []
g = []
npg = []
a = []
s90 = []
kp90 = []
xgc = []
xgb = []
xg = []
xa = []
npxg = []
xg90 = []
npxg90 = []
xa90 = []
xg90xa90 = []
npxg90xa90 = []
xgc90 = []
xgb90 = []
y = []
r = []
do = []
xgd = []
npxgd = []
xad = []
sa = []
epl = []
ll = []
bl = []
l1 = []
for index,i in enumerate(data['Name'].tolist()):
    for india,x in enumerate(data['Name'].tolist()):
        if india > index and i == x and data['DaysOld'].tolist()[index] == data['DaysOld'].tolist()[india] and data['MarketValue'].tolist()[index] == data['MarketValue'].tolist()[india]:
            print(i)
            if i not in name:
                name.append(i)
                bday.append(data['Birthday(DD/MM/YYYY)'].tolist()[index])
                mv.append(data['MarketValue'].tolist()[index])
                team.append(data['Team'].tolist()[index] + ' ' + data['Team'].tolist()[india])
                pos.append(data['Position'].tolist()[index] + ' ' + data['Position'].tolist()[india])
                apps.append(data['Appearances'].tolist()[index] + data['Appearances'].tolist()[india])
                mins.append(data['Minutes'].tolist()[index] + data['Minutes'].tolist()[india])
                g.append(data['G'].tolist()[index] + data['G'].tolist()[india])
                npg.append(data['NPG'].tolist()[index] + data['NPG'].tolist()[india])
                a.append(data['A'].tolist()[index] + data['A'].tolist()[india])
                s1 = data['Sh90'].tolist()[index]/90*data['Minutes'].tolist()[index]
                s2 = data['Sh90'].tolist()[india]/90*data['Minutes'].tolist()[india]
                s3 = s1 + s2
                s90.append(s3/mins[-1]*90)
                s1 = data['KP90'].tolist()[index]/90*data['Minutes'].tolist()[index]
                s2 = data['KP90'].tolist()[india]/90*data['Minutes'].tolist()[india]
                s3 = s1 + s2
                kp90.append(s3/mins[-1]*90)
                xgc.append(data['xGChain'].tolist()[index] + data['xGChain'].tolist()[india])
                xgb.append(data['xGBuildup'].tolist()[index] + data['xGBuildup'].tolist()[india])
                xg.append(data['xG'].tolist()[index] + data['xG'].tolist()[india])
                xa.append(data['xA'].tolist()[index] + data['xA'].tolist()[india])
                npxg.append(data['NPxG'].tolist()[index] + data['NPxG'].tolist()[india])
                xg90.append(xg[-1]/mins[-1]*90)
                npxg90.append(npxg[-1]/mins[-1]*90)
                xa90.append(xa[-1]/mins[-1]*90)
                xg90xa90.append(xg90[-1] + xa90[-1])
                npxg90xa90.append(npxg90[-1] + xa90[-1])
                xgc90.append(xgc[-1]/mins[-1]*90)
                xgb90.append(xgb[-1]/mins[-1]*90)
                y.append(data['Y'].tolist()[index] + data['Y'].tolist()[india])
                r.append(data['R'].tolist()[index] + data['R'].tolist()[india])
                do.append(data['DaysOld'].tolist()[index])
                xgd.append(xg[-1] - g[-1])
                npxgd.append(npxg[-1] - npg[-1])
                xad.append(xa[-1] - a[-1])
                sa.append(max(data['SerieA'].tolist()[index],data['SerieA'].tolist()[india]))
                epl.append(max(data['PremierLeague'].tolist()[index],data['PremierLeague'].tolist()[india]))
                ll.append(max(data['LaLiga'].tolist()[index],data['LaLiga'].tolist()[india]))
                bl.append(max(data['Bundesliga'].tolist()[index],data['Bundesliga'].tolist()[india]))
                l1.append(max(data['Ligue1'].tolist()[index],data['Ligue1'].tolist()[india]))
            elif i in name:
                value = name.index(i)
                if bday[value] == data['Birthday(DD/MM/YYYY)'].tolist()[india] and mv[value] == data['MarketValue'].tolist()[index]:
                    team[value] += ' ' + data['Team'].tolist()[india]
                    pos[value] += ' ' + data['Position'].tolist()[india]
                    apps[value] += data['Appearances'].tolist()[india]
                    mins[value] += data['Minutes'].tolist()[india]
                    g[value] += data['G'].tolist()[india]
                    npg[value] += data['NPG'].tolist()[india]
                    a[value] += data['A'].tolist()[india]
                    s1 = s90[value]/90*mins[value]
                    s2 = data['Sh90'].tolist()[india]/90*data['Minutes'].tolist()[india]
                    s3 = s1 + s2
                    s90[value] = s3/mins[-1]*90
                    s1 = kp90[value]/90*mins[value]
                    s2 = data['KP90'].tolist()[india]/90*data['Minutes'].tolist()[india]
                    s3 = s1 + s2
                    kp90[value] = s3/mins[-1]*90
                    xgc[value] += data['xGChain'].tolist()[india]
                    xgb[value] += data['xGBuildup'].tolist()[india]
                    xg[value] += data['xG'].tolist()[india]
                    xa[value] += data['xA'].tolist()[india]
                    npxg[value] += data['NPxG'].tolist()[india]
                    xg90[value] = xg[value]/mins[value]*90
                    npxg90[value] = npxg[value]/mins[value]*90
                    xa90[value] = xa[value]/mins[value]*90
                    xg90xa90[value] = xg90[value] + xa90[value]
                    npxg90xa90[value] = npxg90[value] + xa90[value]
                    xgc90[value] = xgc[value]/mins[value]*90
                    xgb90[value] = xgb[value]/mins[value]*90
                    y[value] += data['Y'].tolist()[india]
                    r[value] += data['R'].tolist()[india]
                    xgd[value] = xg[value] - g[value]
                    npxgd[value] = npxg[value] - npg[value]
                    xad[value] = xa[value] - a[value]
                    sa[value] = max(sa[value],data['SerieA'].tolist()[india])
                    epl[value] = max(epl[value],data['PremierLeague'].tolist()[india])
                    ll[value] = max(ll[value],data['LaLiga'].tolist()[india])
                    bl[value] = max(bl[value],data['Bundesliga'].tolist()[india])
                    l1[value] = max(bl[value],data['Ligue1'].tolist()[india])
for index,i in enumerate(data['Name'].tolist()):
    if i in name:
        value = name.index(i)
        if bday[value] == data['Birthday(DD/MM/YYYY)'].tolist()[index] and mv[value] == data['MarketValue'].tolist()[index]:
            data['Team'][index] = team[value]
            data['Position'][index] = pos[value]
            data['Appearances'][index] = apps[value]
            data['Minutes'][index] = mins[value]
            data['G'][index] = g[value]     
            data['NPG'][index] = npg[value]  
            data['A'][index] = a[value]  
            data['Sh90'][index] = s90[value]
            data['KP90'][index] = kp90[value]
            data['xGChain'][index] = xgc[value]  
            data['xGBuildup'][index] = xgb[value]  
            data['xG'][index] = xg[value]
            data['xA'][index] = xa[value]
            data['NPxG'][index] = npxg[value]
            data['xG90'][index] = xg90[value]
            data['NPxG90'][index] = npxg90[value]
            data['xA90'][index] = xa90[value]
            data['xG90+xA90'][index] = xg90xa90[value]
            data['NPxG90+xA90'][index] = npxg90xa90[value]
            data['xGChain90'][index] = xgc90[value]
            data['xGBuildup90'][index] = xgb90[value]
            data['Y'][index] = y[value]
            data['R'][index] = r[value]
            data['xGd'][index] = xgd[value]
            data['NPxGd'][index] = npxgd[value]
            data['xAd'][index] = xad[value]
            data['SerieA'][index] = sa[value]
            data['PremierLeague'][index] = epl[value]
            data['LaLiga'][index] = ll[value]
            data['Bundesliga'][index] = bl[value]
            data['Ligue1'][index] = l1[value]
data = data.drop_duplicates()
data


# In[83]:


#Create dummy variables for position: Forward, Midfielder, Defender, Goalkeeper, and Substitute
d = []
m = []
f = []
s = []
gk = []
for i in data['Position'].tolist():
    if 'F' in i:
        f.append(1)
    if 'M' in i:
        m.append(1)
    if 'D' in i:
        d.append(1)
    if 'S' in i:
        s.append(1)
    if 'GK' in i:
        gk.append(1)
    ml = max([len(f),len(m),len(d),len(s),len(gk)])
    if len(f) < ml:
        f.append(0)
    if len(m) < ml:
        m.append(0)
    if len(d) < ml:
        d.append(0)
    if len(s) < ml:
        s.append(0)
    if len(gk) < ml:
        gk.append(0)
data['Forward'] = f
data['Midfielder'] = m
data['Defender'] = d
data['Substitute'] = s
data['GoalKeeper'] = gk
data


# In[84]:


#Renaming some columns
d = data['xG90+xA90'].tolist()
data['xG90plusxA90'] = d
data = data.drop('xG90+xA90',axis=1)
d = data['NPxG90+xA90'].tolist()
data['NPxG90plusxA90'] = d
data = data.drop('NPxG90+xA90',axis=1)
data


# In[85]:


#Get penalty goals (PG), penalty xG (PxG), PxG differential, PG per 90, PxG per 90, Goals per 90, NPG per 90,
#Assists per 90, Shots, Key Passes, Goals per Shot, NPG per Shot, age group dummy variables, penalties taken, 
#penalty accuracy, and xA per Key Pass
pg = []
pxg = []
pxgd = []
pg90 = []
pxg90 = []
g90 = []
npg90 = []
a90 = []
shots = []
kps = []
gpsh = []
npgpsh = []
teen = []
twenty_to_25 = []
twenty_five_to_30 = []
thirty_to_35 = []
thirty_five_plus = []
penalties = []
penalty_accuracy = []
xapkp = []
for index,i in enumerate(data['G'].tolist()):
    pg.append(i-data['NPG'].tolist()[index])
    pxg.append(data['xG'].tolist()[index] - data['NPxG'].tolist()[index])
    pxgd.append(pxg[-1]-pg[-1])
    pg90.append(pg[-1]/data['Minutes'].tolist()[index]*90)
    pxg90.append(pxg[-1]/data['Minutes'].tolist()[index]*90)
    g90.append(i/data['Minutes'].tolist()[index]*90)
    npg90.append(data['NPG'].tolist()[index]/data['Minutes'].tolist()[index]*90)
    a90.append(data['A'].tolist()[index]/data['Minutes'].tolist()[index]*90)
    shots.append(data['Sh90'].tolist()[index]/90*data['Minutes'].tolist()[index])
    kps.append(data['KP90'].tolist()[index]/90*data['Minutes'].tolist()[index])
    if kps[-1] == 0:
        xapkp.append(0)
    else:
        xapkp.append(data['xA'].tolist()[index]/kps[-1])
    age = data['DaysOld'].tolist()[index]/365.25
    penalties.append(round(pxg[-1]/0.765))
    if penalties[-1] != 0:
        penalty_accuracy.append(pg[-1]/penalties[-1])
    else:
        penalty_accuracy.append(0)
    if age < 20:
        teen.append(1)
        twenty_to_25.append(0)
        twenty_five_to_30.append(0)
        thirty_to_35.append(0)
        thirty_five_plus.append(0)
    elif 20 <= age < 25:
        teen.append(0)
        twenty_to_25.append(1)
        twenty_five_to_30.append(0)
        thirty_to_35.append(0)
        thirty_five_plus.append(0)
    elif 25 <= age < 30:
        teen.append(0)
        twenty_to_25.append(0)
        twenty_five_to_30.append(1)
        thirty_to_35.append(0)
        thirty_five_plus.append(0)
    elif 30 <= age < 35:
        teen.append(0)
        twenty_to_25.append(0)
        twenty_five_to_30.append(0)
        thirty_to_35.append(1)
        thirty_five_plus.append(0)
    else:
        teen.append(0)
        twenty_to_25.append(0)
        twenty_five_to_30.append(0)
        thirty_to_35.append(0)
        thirty_five_plus.append(1)
    if shots[-1] == 0:
        gpsh.append(0)
        npgpsh.append(0)
    else:   
        gpsh.append(i/shots[-1])
        npgpsh.append(data['NPG'].tolist()[index]/(shots[-1]-penalties[-1]))
data['PG'] = pg
data['PxG'] = pxg
data['PxGd'] = pxgd
data['PenaltiesAwarded'] = penalties
data['PenaltyAccuracy'] = penalty_accuracy
data['G90'] = g90
data['NPG90'] = npg90
data['A90'] = a90
data['Sh'] = shots
data['KP'] = kps
data['GperShot'] = gpsh
data['NPGperShot'] = npgpsh
data['Teenager'] = teen
data['TwentyToTwentyFive'] = twenty_to_25
data['TwentyFiveToThirty'] = twenty_five_to_30
data['ThirtyToThirtyFive'] = thirty_to_35
data['ThirtyFivePlus'] = thirty_five_plus
data['xAperKP'] = xapkp
data


# In[86]:


#Calculate xG per Shot and NPxG per Shot
d = []
for index,i in enumerate(data['Sh90'].tolist()):
    if i == 0:
        d.append(0)
    else:
        d.append(data['xG90'].tolist()[index]/i)
data['xGperShot'] = d
d = []
for index,i in enumerate(data['Sh'].tolist()):
    if i == 0:
        d.append(0)
    else:
        d.append(data['NPxG'].tolist()[index]/(i-data['PenaltiesAwarded'].tolist()[index]))
data['NPxGperShot'] = d
data


# In[87]:


#Get table for top 10 NPxGd overperformers
data[['Name','Team','NPxGd']].sort_values('NPxGd',ascending=True).head(10) 


# In[88]:


#Get table for top 10 NPxG per Shot players
s = [round(i) for i in data['Sh'].tolist()]
k =[round(i) for i in data['KP'].tolist()]
data['Sh'] = s
data['KP'] = k
data[data['Sh'] >= 76][['Name','Team','NPxGperShot','Sh']].sort_values('NPxGperShot',ascending=False).head(10) 


# In[89]:


#Get table for top 10 xA per KP players
data[data['KP'] >= 76][['Name','Team','xAperKP','KP']].sort_values('xAperKP',ascending=False).head(10) 


# In[90]:


#Top 10 penalty takers
data[data['PenaltiesAwarded'] >= 7][['Name','Team','PenaltyAccuracy','PenaltiesAwarded']].sort_values('PenaltyAccuracy',ascending=False).head(10) 


# In[91]:


#Bottom 5 penalty takers
#Top 10 penalty takers
data[data['PenaltiesAwarded'] >= 7][['Name','Team','PenaltyAccuracy','PenaltiesAwarded']].sort_values('PenaltyAccuracy',ascending=True).head(5) 


# In[92]:


import sklearn
import sklearn.linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


# In[93]:


#Select which data you want to build your model with as well as any features you do not want right off the bat
m = data[data['Midfielder'] == 1]
f = data[data['Forward'] == 1]
extra = data[data['Name'] == 'Federico Chiesa']
ng = data[data['GoalKeeper']==0]
data = data[data.isin(m)]
data = data[data['Minutes'] >= 900]
columns = []
for i in list(data.columns):
    if i != 'Name' and i != 'MarketValue' and i != 'Birthday(DD/MM/YYYY)' and i != 'Team' and i != 'Position' and i != 'GoalKeeper' and i != 'Substitute' and i != 'Midfielder' and i != 'DaysOld':
        columns.append(i)


# In[94]:


#Get 70/30 train/test split
X = data
for i in list(data.columns):
    if i not in columns:
        X = X.drop(i,axis=1)
y = data['MarketValue'].tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=88)


# In[95]:


#Lasso regression model with default settings
clf = sklearn.linear_model.Lasso(random_state=88)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_train)
print('Training R-Squared:', sklearn.metrics.r2_score(y_train,y_pred))
print('Training MSE:', sklearn.metrics.mean_squared_error(y_train,y_pred))
print('Training MAE:', sklearn.metrics.mean_absolute_error(y_train,y_pred))
y_pred = clf.predict(X_test)
print('Testing R-Squared:', sklearn.metrics.r2_score(y_test,y_pred))
print('Testing MSE:', sklearn.metrics.mean_squared_error(y_test,y_pred))
print('Testing MAE:', sklearn.metrics.mean_absolute_error(y_test,y_pred))


# In[96]:


#OPTIMIZED LASSO REGRESSION (MSE)


# In[97]:


#Grid search CV to find optimal lambda value with MSE as evaluation metric
grid_values = {'alpha': np.arange(0.01,5,0.01)} 
lasso = sklearn.linear_model.Lasso(random_state=88)
cv = KFold(n_splits=5,random_state=333,shuffle=True) 
lasso_cv1 = GridSearchCV(lasso, param_grid=grid_values, scoring='neg_mean_squared_error', cv=cv,verbose=2)
lasso_cv1.fit(X_train, y_train)


# In[98]:


print(lasso_cv1.best_params_)


# In[99]:


#Plot CV evaluation metrics
ccp_alpha = lasso_cv1.cv_results_['param_alpha'].data
mse_scores = lasso_cv1.cv_results_['mean_test_score']

plt.figure(figsize=(8, 6))
plt.xlabel('Lambda', fontsize=16)
plt.ylabel('CV MSE', fontsize=16)
plt.scatter(ccp_alpha, mse_scores, s=30)
plt.plot(ccp_alpha, mse_scores, linewidth=3)
plt.grid(True, which='both')

plt.tight_layout()
plt.show()


# In[100]:


#Train and test optimized model
clf = sklearn.linear_model.Lasso(lasso_cv1.best_params_['alpha'],random_state=88)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_train)
print('Training R-Squared:', sklearn.metrics.r2_score(y_train,y_pred))
print('Training MSE:', sklearn.metrics.mean_squared_error(y_train,y_pred))
print('Training MAE:', sklearn.metrics.mean_absolute_error(y_train,y_pred))
y_pred = clf.predict(X_test)
print('Testing R-Squared:', sklearn.metrics.r2_score(y_test,y_pred))
print('Testing MSE:', sklearn.metrics.mean_squared_error(y_test,y_pred))
print('Testing MAE:', sklearn.metrics.mean_absolute_error(y_test,y_pred))


# In[101]:


#OPTIMIZED LASSO REGRESSION (R2)


# In[102]:


#Grid search CV to find optimal lambda value with R2 as evaluation metric
grid_values = {'alpha': np.arange(0.01,5,0.01)} 
lasso = sklearn.linear_model.Lasso(random_state=88)
cv = KFold(n_splits=5,random_state=333,shuffle=True) 
lasso_cv = GridSearchCV(lasso, param_grid=grid_values, scoring='r2', cv=cv,verbose=2)
lasso_cv.fit(X_train, y_train)


# In[103]:


print(lasso_cv.best_params_)


# In[104]:


#Plot CV evaluation metrics
ccp_alpha = lasso_cv.cv_results_['param_alpha'].data
R2_scores = lasso_cv.cv_results_['mean_test_score']

plt.figure(figsize=(8, 6))
plt.xlabel('Lambda', fontsize=16)
plt.ylabel('CV R2', fontsize=16)
plt.scatter(ccp_alpha, R2_scores, s=30)
plt.plot(ccp_alpha, R2_scores, linewidth=3)
plt.grid(True, which='both')

plt.tight_layout()
plt.show()


# In[105]:


#Train and test optimized model on data
clf = sklearn.linear_model.Lasso(lasso_cv.best_params_['alpha'],random_state=88)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_train)
print('Training R-Squared:', sklearn.metrics.r2_score(y_train,y_pred))
print('Training MSE:', sklearn.metrics.mean_squared_error(y_train,y_pred))
print('Training MAE:', sklearn.metrics.mean_absolute_error(y_train,y_pred))
y_pred = clf.predict(X_test)
print('Testing R-Squared:', sklearn.metrics.r2_score(y_test,y_pred))
print('Testing MSE:', sklearn.metrics.mean_squared_error(y_test,y_pred))
print('Testing MAE:', sklearn.metrics.mean_absolute_error(y_test,y_pred))


# In[106]:


#Ridge regression model with default settings
clf = sklearn.linear_model.Ridge(random_state=88)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_train)
print('Training R-Squared:', sklearn.metrics.r2_score(y_train,y_pred))
print('Training MSE:', sklearn.metrics.mean_squared_error(y_train,y_pred))
print('Training MAE:', sklearn.metrics.mean_absolute_error(y_train,y_pred))
y_pred = clf.predict(X_test)
print('Testing R-Squared:', sklearn.metrics.r2_score(y_test,y_pred))
print('Testing MSE:', sklearn.metrics.mean_squared_error(y_test,y_pred))
print('Testing MAE:', sklearn.metrics.mean_absolute_error(y_test,y_pred))


# In[107]:


#RIDGE REGRESSION OPTIMIZED (MSE)


# In[108]:


#Grid search CV using MSE as evaluation metric
grid_values = {'alpha':  np.arange(0.1,100,0.1)} 
ridge = sklearn.linear_model.Ridge(random_state=88)
cv = KFold(n_splits=5,random_state=333,shuffle=True) 
ridge_cv1 = GridSearchCV(ridge, param_grid=grid_values, scoring='neg_mean_squared_error', cv=cv,verbose=2)
ridge_cv1.fit(X_train, y_train)


# In[109]:


print(ridge_cv1.best_params_)


# In[110]:


#Plot CV evaluation metrics
ccp_alpha = ridge_cv1.cv_results_['param_alpha'].data
mse_scores = ridge_cv1.cv_results_['mean_test_score']

plt.figure(figsize=(8, 6))
plt.xlabel('Lambda', fontsize=16)
plt.ylabel('CV MSE', fontsize=16)
plt.scatter(ccp_alpha, mse_scores, s=30)
plt.plot(ccp_alpha, mse_scores, linewidth=3)
plt.grid(True, which='both')

plt.tight_layout()
plt.show()


# In[111]:


#Train and test optimized model on the data
clf = sklearn.linear_model.Ridge(ridge_cv1.best_params_['alpha'],random_state=88)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_train)
print('Training R-Squared:', sklearn.metrics.r2_score(y_train,y_pred))
print('Training MSE:', sklearn.metrics.mean_squared_error(y_train,y_pred))
print('Training MAE:', sklearn.metrics.mean_absolute_error(y_train,y_pred))
y_pred = clf.predict(X_test)
print('Testing R-Squared:', sklearn.metrics.r2_score(y_test,y_pred))
print('Testing MSE:', sklearn.metrics.mean_squared_error(y_test,y_pred))
print('Testing MAE:', sklearn.metrics.mean_absolute_error(y_test,y_pred))


# In[112]:


#RIDGE REGRESSION OPTIMIZED (R2)


# In[113]:


#Grid search CV to find optimal lambda value with R2 as evaluation metric
grid_values = {'alpha': np.arange(0.1,100,0.1)} 
ridge = sklearn.linear_model.Ridge(random_state=88)
cv = KFold(n_splits=5,random_state=333,shuffle=True) 
ridge_cv = GridSearchCV(ridge, param_grid=grid_values, scoring='r2', cv=cv,verbose=2)
ridge_cv.fit(X_train, y_train)


# In[114]:


print(ridge_cv.best_params_)


# In[115]:


#Plot CV evaluation metrics
ccp_alpha = ridge_cv.cv_results_['param_alpha'].data
R2_scores = ridge_cv.cv_results_['mean_test_score']

plt.figure(figsize=(8, 6))
plt.xlabel('Lambda', fontsize=16)
plt.ylabel('CV R2', fontsize=16)
plt.scatter(ccp_alpha, R2_scores, s=30)
plt.plot(ccp_alpha, R2_scores, linewidth=3)
plt.grid(True, which='both')

plt.tight_layout()
plt.show()


# In[116]:


#Train and test optimized model on the data
clf = sklearn.linear_model.Ridge(ridge_cv.best_params_['alpha'],random_state=88)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_train)
print('Training R-Squared:', sklearn.metrics.r2_score(y_train,y_pred))
print('Training MSE:', sklearn.metrics.mean_squared_error(y_train,y_pred))
print('Training MAE:', sklearn.metrics.mean_absolute_error(y_train,y_pred))
y_pred = clf.predict(X_test)
print('Testing R-Squared:', sklearn.metrics.r2_score(y_test,y_pred))
print('Testing MSE:', sklearn.metrics.mean_squared_error(y_test,y_pred))
print('Testing MAE:', sklearn.metrics.mean_absolute_error(y_test,y_pred))


# In[117]:


#Elastic Net regression with default settings
clf = sklearn.linear_model.ElasticNet(random_state=88)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_train)
print('Training R-Squared:', sklearn.metrics.r2_score(y_train,y_pred))
print('Training MSE:', sklearn.metrics.mean_squared_error(y_train,y_pred))
print('Training MAE:', sklearn.metrics.mean_absolute_error(y_train,y_pred))
y_pred = clf.predict(X_test)
print('Testing R-Squared:', sklearn.metrics.r2_score(y_test,y_pred))
print('Testing MSE:', sklearn.metrics.mean_squared_error(y_test,y_pred))
print('Testing MAE:', sklearn.metrics.mean_absolute_error(y_test,y_pred))


# In[118]:


#GridSearchCV to find optimal parameters with MSE as evaluation metric
grid_values = {'alpha':  np.arange(0,100,10),
              'l1_ratio': np.arange(0,1,0.1)
              } 
en = sklearn.linear_model.ElasticNet(random_state=88)
cv = KFold(n_splits=5,random_state=333,shuffle=True) 
en_cv1 = GridSearchCV(en, param_grid=grid_values, scoring='neg_mean_squared_error', cv=cv,verbose=2)
en_cv1.fit(X_train, y_train)


# In[119]:


print(en_cv1.best_params_)


# In[120]:


#Plot CV evaluation metrics
ccp_alpha = en_cv1.cv_results_['param_alpha'].data
mse_scores = en_cv1.cv_results_['mean_test_score']

plt.figure(figsize=(8, 6))
plt.xlabel('Lambda', fontsize=16)
plt.ylabel('CV MSE', fontsize=16)
plt.scatter(ccp_alpha, mse_scores, s=30)
plt.plot(ccp_alpha, mse_scores, linewidth=3)
plt.grid(True, which='both')

plt.tight_layout()
plt.show()


# In[121]:


#Train and test model on data
clf = sklearn.linear_model.ElasticNet(alpha=en_cv1.best_params_['alpha'],l1_ratio=en_cv1.best_params_['l1_ratio'],random_state=88)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_train)
print('Training R-Squared:', sklearn.metrics.r2_score(y_train,y_pred))
print('Training MSE:', sklearn.metrics.mean_squared_error(y_train,y_pred))
print('Training MAE:', sklearn.metrics.mean_absolute_error(y_train,y_pred))
y_pred = clf.predict(X_test)
print('Testing R-Squared:', sklearn.metrics.r2_score(y_test,y_pred))
print('Testing MSE:', sklearn.metrics.mean_squared_error(y_test,y_pred))
print('Testing MAE:', sklearn.metrics.mean_absolute_error(y_test,y_pred))


# In[122]:


#GridSearchCV to find optimal parameters with R2 as evaluation metric
grid_values = {'alpha':  np.arange(0,100,10),
              'l1_ratio': np.arange(0,1.01,0.1)
              } 
en = sklearn.linear_model.ElasticNet(random_state=88)
cv = KFold(n_splits=5,random_state=333,shuffle=True) 
en_cv = GridSearchCV(en, param_grid=grid_values, scoring='r2', cv=cv,verbose=2)
en_cv.fit(X_train, y_train)


# In[123]:


print(en_cv.best_params_)


# In[124]:


#Plot CV evaluation metrics
ccp_alpha = en_cv.cv_results_['param_alpha'].data
R2_scores = en_cv.cv_results_['mean_test_score']

plt.figure(figsize=(8, 6))
plt.xlabel('Lambda', fontsize=16)
plt.ylabel('CV R2', fontsize=16)
plt.scatter(ccp_alpha, R2_scores, s=30)
plt.plot(ccp_alpha, R2_scores, linewidth=3)
plt.grid(True, which='both')

plt.tight_layout()
plt.show()


# In[125]:


#Train and test model on data
clf = sklearn.linear_model.ElasticNet(alpha=en_cv.best_params_['alpha'],l1_ratio=en_cv.best_params_['l1_ratio'],random_state=88)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_train)
print('Training R-Squared:', sklearn.metrics.r2_score(y_train,y_pred))
print('Training MSE:', sklearn.metrics.mean_squared_error(y_train,y_pred))
print('Training MAE:', sklearn.metrics.mean_absolute_error(y_train,y_pred))
y_pred = clf.predict(X_test)
print('Testing R-Squared:', sklearn.metrics.r2_score(y_test,y_pred))
print('Testing MSE:', sklearn.metrics.mean_squared_error(y_test,y_pred))
print('Testing MAE:', sklearn.metrics.mean_absolute_error(y_test,y_pred))


# In[126]:


#Find correlations between each feature and Market Value
correlations = []
for i in columns:
    correlations.append(np.corrcoef(data[i].tolist(),data['MarketValue'].tolist())[0][1]**2)
d = pd.DataFrame()
d['Feature'] = columns
d['R-Squared With Market Value'] = correlations
d = d.sort_values('R-Squared With Market Value',ascending=False)
d


# In[127]:


#Plot scatter plot of each market value vs. each feature
fig, ax = plt.subplots(8,8,figsize=(15,15), sharey=False,sharex=False)
row = 0
column = -1
    
for i in d['Feature'].tolist():
    
    if column < 6:
        column += 1
    else:
        column = 0
        row += 1
    ax[row][column].scatter(data[i].tolist(),data['MarketValue'].tolist())
    ax[row][column].set_title(i + ' ' + str(round(np.corrcoef(data[i].tolist(),data['MarketValue'].tolist())[0][1]**2,2)))
    print(columns.index(i))
    print(i)
    print(np.corrcoef(data[i].tolist(),data['MarketValue'].tolist())[0][1]**2)
    print()
plt.tight_layout()


# In[128]:


#Plot predicted market value according to model vs. actual market value
#Also print the name and value information of each player where predicted market value >= actual market value + 10
figure(figsize=(10, 8), dpi=80)
plt.title('Predicted Market Value vs. Actual Market Value')
plt.xlabel('Actual Market Value')
plt.ylabel('Predicted Market Value')
x = data['MarketValue'].tolist()
#pred = model1.predict(data).tolist()
pred = clf.predict(X).tolist()
predicted_minus_actual = []
for index, i in enumerate(pred):
    predicted_minus_actual.append(i-x[index])
    if i >= x[index] + 10:
        print(data['Name'].tolist()[index])
        print('Predicted: ' + str(i))
        print('Actual: ' + str(x[index]))
        print()
data['PredictedMinusActual'] = predicted_minus_actual
data['PredictedMarketValue'] = pred
plt.scatter(x,pred);
for i, txt in enumerate(data['Name'].tolist()):
    if data['MarketValue'].tolist()[i] >= 75 or data['PredictedMarketValue'].tolist()[i] >= 60:
        plt.annotate(txt, (x[i], pred[i]))


# In[129]:


pd.set_option('display.max_rows', 600)


# In[130]:


#Show top 10 most overrated players
data.sort_values('PredictedMinusActual')[['Name','MarketValue','PredictedMarketValue','PredictedMinusActual']].head(10)


# In[131]:


#Show top 10 midfielders according to model
data.sort_values('PredictedMarketValue',ascending=False)[['Name','MarketValue','PredictedMarketValue','PredictedMinusActual']].head(10)


# In[132]:


#Show top 10 most overrated midfielders
data.sort_values('PredictedMinusActual',ascending=False)[['Name','MarketValue','PredictedMarketValue','PredictedMinusActual']].head(10)


# In[133]:


#Plot model feature coefficients
coefs = pd.DataFrame()
coefs['Features'] = X_train.columns
coefs['Coefficients'] = clf.coef_
coefs.sort_values('Coefficients',ascending=True)


# In[134]:


#Plot model feature coefficients in other order
coefs.sort_values('Coefficients',ascending=False)


# In[ ]:





# In[ ]:




