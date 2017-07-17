
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
# from matplotlib.colors import ListedColormap


# get_ipython().magic('matplotlib inline')

sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 32})
sns.set_context("poster", rc={"font.size": 34, "axes.titlesize": 34, "axes.labelsize": 34, "lines.linewidth": 2})
plt.rcParams['figure.figsize'] = 15, 12

pd.options.display.max_columns = 999
pd.options.display.max_rows = 100


# In[2]:

import DataContainer as DataContainer


# ### Loading Data:

# In[3]:

dc = DataContainer.DataContainer()


# In[4]:

df = dc.df
sites = [a for a in df.site.unique()]
sites.sort()


# In[5]:

sites


# In[6]:

df.head()


# In[7]:

df.shape


# In[8]:

save_fig = False


# In[9]:

import datetime
for s, c in zip(sites, sns.color_palette()):
    plt.plot_date(df[df['site'] == s]['date'], df[df['site'] == s]['FC'], label=s, color=c)
    plt.xlabel('Date')
    plt.ylabel('Flux CO2, umol/m.sq/s')
#     plt.xlim([-20,50])
    plt.ylim([-60, 60])
plt.legend(frameon=1)
plt.xlim([datetime.date(1996, 1, 1), datetime.date(2017, 1, 1)])
if save_fig:
    plt.savefig('img/FvsTime/all.png', dpi=150)
# plt.show()


# In[10]:

for s, c in zip(sites, sns.color_palette()):
    plt.plot_date(df[df['site'] == s]['date'], df[df['site'] == s]['FC'], label=s, color=c)

    result = DataContainer.linear_fit(df[df['site'] == s], 'FC', 'index')
    plt.plot(df[df['site'] == s]['date'].values, result.fittedvalues.values, color=sns.xkcd_rgb["red"], lw=3)
    lin_fit_y0 = r'$y_0$ = %.2e' % (result.params[0])
    lin_fit_k = r'$k$ = %.2e' % (result.params[1] * 365)
    plt.annotate(lin_fit_y0, xy=(0.81, 0.91), xycoords='axes fraction', color='r', fontsize=20)
    plt.annotate(lin_fit_k, xy=(0.81, 0.87), xycoords='axes fraction', color='r', fontsize=20)
    plt.xlabel('Date')
    plt.ylabel('Flux CO2, umol/m.sq/s')
#     plt.xlim([-20,50])
    plt.ylim([-60, 60])
    plt.xlim([datetime.date(1996, 1, 1), datetime.date(2017, 1, 1)])
    plt.legend(frameon=1)
    if save_fig:
        plt.savefig('img/FvsTime/{}.png'.format(s), dpi=150)
    # plt.show()


# In[11]:

for s, c in zip(sites, sns.color_palette()):
    #     plt.scatter(df[df['site']==s]['TS_2'],df[df['site']==s]['FC'], label=s, color=c)
    g = sns.jointplot(df[df['site'] == s]['index'], df[df['site'] == s]['FC'], label=s, color=c, size=15, kind='reg', dropna=1, ylim=(-40, 40))
    g.set_axis_labels('Index', 'Flux CO2, umol/m.sq/s')
#     ax = plt.gca()
#     xticks = ax.get_xticks()
#     xticks_dates = [datetime.datetime.fromtimestamp(x).strftime('%Y') for x in xticks]
#     ax.set_xticklabels(xticks_dates)
#     plt.xlim([datetime.date(1996, 1, 1), datetime.date(2017, 1, 1)])
    if save_fig:
        plt.savefig('img/FvsTimeDen/{}.png'.format(s), dpi=150)
    # plt.show()


# In[12]:

for s, c in zip(sites, sns.color_palette()):
    plt.scatter(df[df['site'] == s]['TS_2'], df[df['site'] == s]['FC'], label=s, color=c)
    plt.xlabel('Soil Temperature, C')
    plt.ylabel('Flux CO2, umol/m.sq/s')
    plt.xlim([-20, 50])
    plt.ylim([-60, 60])
plt.legend(frameon=1)
if save_fig:
    plt.savefig('img/FvsT/all.png', dpi=150)
# plt.show()


# In[13]:

for s, c in zip(sites, sns.color_palette()):
    #     plt.scatter(df[df['site']==s]['TS_2'],df[df['site']==s]['FC'], label=s, color=c)
    g = sns.jointplot(df[df['site'] == s]['TS_2'], df[df['site'] == s]['FC'], label=s, color=c, size=15, kind='reg', dropna=1, xlim=(-20, 50), ylim=(-40, 40))
    g.set_axis_labels('Soil Temperature, C', 'Flux CO2, umol/m.sq/s')
    if save_fig:
        plt.savefig('img/FvsT/{}.png'.format(s), dpi=150)
    # plt.show()


# In[14]:

for s, c in zip(sites, sns.color_palette()):
    plt.scatter(df[df['site'] == s]['P'], df[df['site'] == s]['FC'], label=s, color=c)

plt.xlim([0, 25])
plt.ylim([-60, 60])
plt.xlabel('Precipitation, mm')
plt.ylabel('Flux CO2, umol/m.sq/s')
plt.legend(frameon=1)
if save_fig:
    plt.savefig('img/FvsP/all.png'.format(s), dpi=150)

# plt.show()


# In[15]:

for s, c in zip(sites, sns.color_palette()):
    #     plt.scatter(df[df['site']==s]['P'],df[df['site']==s]['FC'], label=s, color=c)
    g = sns.jointplot(df[df['site'] == s]['P'], df[df['site'] == s]['FC'], label=s, color=c, size=15, kind='reg', dropna=1, xlim=(0, 25), ylim=(-40, 40))
    g.set_axis_labels('Precipitation, mm', 'Flux CO2, umol/m.sq/s')
    if save_fig:
        plt.savefig('img/FvsP/{}.png'.format(s), dpi=150)
    # plt.show()


# In[16]:

for s, c in zip(sites, sns.color_palette()):
    plt.scatter(df[df['site'] == s]['SWC_1'], df[df['site'] == s]['FC'], label=s, color=c)
plt.xlabel('Soil water content, -')
plt.ylabel('Flux CO2, umol/m.sq/s')
plt.xlim([0, 100])
plt.ylim([-50, 50])
plt.legend(frameon=1)
if save_fig:
    plt.savefig('img/FvsSWC/all.png'.format(s), dpi=150)
# plt.show()


# In[17]:

for s, c in zip(sites, sns.color_palette()):
    #     plt.scatter(df[df['site']==s]['SWC_1'],df[df['site']==s]['FC'], label=s, color=c)
    try:
        g = sns.jointplot(df[df['site'] == s]['SWC_1'], df[df['site'] == s]['FC'], label=s, color=c, size=15, kind='reg', dropna=1, xlim=(0, 100), ylim=(-40, 40))
    except:
        pass
    g.set_axis_labels('Soil Water Content, -', 'Flux CO2, umol/m.sq/s')
    if save_fig:
        plt.savefig('img/FvsSWC/{}.png'.format(s), dpi=150)
    # plt.show()


# In[18]:

df["dayofyear"] = df.index.map(lambda x: x.dayofyear)


# In[19]:

for s, c in zip(sites, sns.color_palette()):
    plt.scatter(df[df['site'] == s]['dayofyear'], df[df['site'] == s]['FC'], label=s, color=c)
plt.xlabel('Day of Year, #')
plt.ylabel('Flux CO2, umol/m.sq/s')
plt.xlim([0, 365])
plt.ylim([-50, 50])
plt.legend(frameon=1)
if save_fig:
    plt.savefig('img/FvsY/all.png'.format(s), dpi=150)
# plt.show()


# In[20]:

import datetime
plt.scatter(df[df['site'] == 'US-FPe'].index.to_pydatetime(), df[df['site'] == 'US-FPe']['FC'].values)
plt.xlim([datetime.date(2004, 6, 26), datetime.date(2004, 6, 27)])


# In[21]:

for s, c in zip(sites, sns.color_palette()):
    #     plt.scatter(df[df['site']==s]['SWC_1'],df[df['site']==s]['FC'], label=s, color=c)
    g = sns.jointplot(df[df['site'] == s]['dayofyear'], df[df['site'] == s]['FC'], label=s, color=c, size=15, kind='reg', dropna=1, xlim=(1, 365), ylim=(-40, 40))
    g.set_axis_labels('Day of Year, #', 'Flux CO2, umol/m.sq/s')
    if save_fig:
        plt.savefig('img/FvsY/{}.png'.format(s), dpi=150)
    # plt.show()


# In[22]:

df['week_n'] = df.index.to_series().dt.week


# In[23]:

for s, c in zip(sites, sns.color_palette()):
    #     plt.scatter(df[df['site']==s]['SWC_1'],df[df['site']==s]['FC'], label=s, color=c)
    g = sns.boxplot(df[df['site'] == s]['week_n'], df[df['site'] == s]['FC'])
    plt.ylim([-15, 15])
    g.set_xticks(np.arange(1, 53, 5))
    g.set_xticklabels(np.arange(1, 53, 5))
    plt.ylabel('Flux CO2, umol/m.sq/s')
    plt.xlabel('Week number, #')
    plt.legend(frameon=1)
    if save_fig:
        plt.savefig('img/FvsW/{}.png'.format(s), dpi=150)
    # plt.show()


# In[24]:

import ephem


def sunrise_and_sunset(lat, lon, d):
    fred = ephem.Observer()
    d = pd.to_datetime(d)
    fred.date = d.strftime('%Y/%-m/%-d %H:%M:%S')
    fred.lon = str(lon)  # Note that lon should be in string format
    fred.lat = str(lat)  # Note that lat should be in string format
    sunrise = fred.previous_rising(ephem.Sun())  # Sunrise
    sunset = fred.next_setting(ephem.Sun())  # Sunset
    return ephem.localtime(sunrise).time(), ephem.localtime(sunset).time()

sunrise_and_sunset = np.vectorize(sunrise_and_sunset)


# In[25]:

# for site in df.site.unique():


# In[26]:

df['sunrise'], df['sunset'] = sunrise_and_sunset(df['LOCATION_LAT'], df['LOCATION_LONG'], df.index)  # 5min


# In[27]:

df['day_light_bool'] = (df.index.time >= df['sunrise']) & (df.index.time <= df['sunset'])


# In[28]:

df[['sunrise', 'sunset', 'day_light_bool']].tail()


# In[29]:

for s, c in zip(sites, sns.color_palette()):
    #     plt.scatter(df[df['site']==s]['SWC_1'],df[df['site']==s]['FC'], label=s, color=c)
    #     sns.boxplot(df[(df['site']==s) & (df['day_light_bool'])]['week_n'],df[df['site']==s]['FC'])
    g = sns.boxplot(df[(df['site'] == s) & (df['day_light_bool'] == False)]['week_n'], df[df['site'] == s]['FC'])
    g.set_xticks(np.arange(1, 53, 5))
    g.set_xticklabels(np.arange(1, 53, 5))
    plt.ylim([-15, 15])
    plt.ylabel('Flux CO2, umol/m.sq/s')
    plt.xlabel('Week number, #')
    # plt.show()


# In[30]:

# for s,c in zip(sites, sns.color_palette()):
#     plt.scatter(df[df['site']==s]['SWC_1'],df[df['site']==s]['FC'], label=s, color=c)
g = sns.pointplot(x='week_n', y='FC', data=df, hue='site')
g.set_xticks(np.arange(1, 53, 5))
g.set_xticklabels(np.arange(1, 53, 5))
plt.ylim([-6, 4])
plt.legend(frameon=1)
plt.ylabel('Flux CO2, umol/m.sq/s')
plt.xlabel('Week number, #')

plt.legend(frameon=1)
if save_fig:
    plt.savefig('img/FvsW-mean/all.png'.format(s), dpi=150)


# plt.show()


# In[31]:

from pvlib import clearsky, atmosphere
from pvlib.location import Location


def solar_rad(lat, long, d):
    d = pd.to_datetime(d)
    tus = Location(lat, long)
    cs = tus.get_clearsky(d, model='haurwitz')
    return cs

solar_rad = np.vectorize(solar_rad)


# In[ ]:


# In[32]:

# year/month/day optionally followed by hours:minutes:seconds
# df.index[-1]


# In[33]:

#sunrise_and_sunset(df['LOCATION_LAT'][-1], df['LOCATION_LONG'][-1], df.index[-1])


# In[34]:

sns.pointplot(x='week_n', y='FC', data=df[df["day_light_bool"] == False], linestyles=[':' for _ in range(len(sites))], hue='site')
g = sns.pointplot(x='week_n', y='FC', data=df[df["day_light_bool"] == True], hue='site')
plt.ylim([-10, 4])
plt.ylabel('Flux CO2, umol/m.sq/s')
plt.xlabel('Week number, #')
g.set_xticks(np.arange(1, 53, 5))
g.set_xticklabels(np.arange(1, 53, 5))
plt.legend(frameon=1)
if save_fig:
    plt.savefig('img/FvsW_day/all.png'.format(s), dpi=150)

# plt.show()


# In[35]:

for s, c in zip(sites, sns.color_palette()):
    sns.pointplot(x='week_n', y='FC', data=df[(df["day_light_bool"] == False) & (df["site"] == s)], linestyles=[':'], label=s, color=c, size=15)
    g = sns.pointplot(x='week_n', y='FC', data=df[(df["day_light_bool"] == True) & (df["site"] == s)], color=c)
    g.set_xticks(np.arange(1, 53, 5))
    g.set_xticklabels(np.arange(1, 53, 5))
    plt.ylim([-10, 4])
    plt.ylabel('Flux CO2, umol/m.sq/s')
    plt.xlabel('Week number, #')
    if save_fig:
        plt.savefig('img/FvsW_day/{}.png'.format(s), dpi=150)
    # plt.show()


# In[36]:

# sns.pointplot(x='index', y='FC', data=df[df["day_light_bool"] == True], hue='site')


# In[37]:

# %timeit df['clear_sky_solar_rad'] = solar_rad(df['LOCATION_LAT'], df['LOCATION_LONG'], df.index)
# %timeit df['clear_sky_solar_rad'] = df.apply(lambda x: solar_rad(df['LOCATION_LAT'].values, df['LOCATION_LONG'].values, df['TIMESTAMP_START'].values), axis=1)


# In[38]:

# sns.lmplot(x='index', y='FC',data=df, size=15, hue='day_light_bool')


# In[39]:

import statsmodels.formula.api as sm
result = sm.ols(formula="FC ~ index", data=df[df['day_light_bool']]).fit()
result.summary()


# In[ ]:


# In[40]:

result = sm.ols(formula="FC ~ index", data=df[df['day_light_bool'] == False]).fit()
result.summary()


# In[41]:

for s, c in zip(sites, sns.color_palette()):
    plt.scatter(df[(df["day_light_bool"] == False) & (df["site"] == s)]['TS_2'], df[(df["day_light_bool"] == False) & (df["site"] == s)]['FC'], label=s, color=c)
    plt.xlabel('Soil Temperature, C')
    plt.ylabel('Flux CO2, umol/m.sq/s')
    plt.xlim([-20, 50])
    plt.ylim([-60, 60])
plt.legend(frameon=1)
if save_fig:
    plt.savefig('img/FvsT_night/all.png', dpi=150)
# plt.show()


# In[42]:

for s, c in zip(sites, sns.color_palette()):
    #     plt.scatter(df[df['site']==s]['TS_2'],df[df['site']==s]['FC'], label=s, color=c)
    g = sns.jointplot(x='TS_2', y='FC', data=df[(df["day_light_bool"] == False) & (df["site"] == s)],
                      label=s, color=c, size=15, kind='reg', dropna=1, xlim=(-20, 50), ylim=(-40, 40))
    g.set_axis_labels('Soil Temperature, C', 'Flux CO2, umol/m.sq/s')
    if save_fig:
        plt.savefig('img/FvsT_night/{}.png'.format(s), dpi=150)
    # plt.show()


# In[43]:

for s, c in zip(sites, sns.color_palette()):
    plt.scatter(df[(df["day_light_bool"] == True) & (df["site"] == s)]['TS_2'], df[(df["day_light_bool"] == True) & (df["site"] == s)]['FC'], label=s, color=c)
    plt.xlabel('Soil Temperature, C')
    plt.ylabel('Flux CO2, umol/m.sq/s')
    plt.xlim([-20, 50])
    plt.ylim([-60, 60])
plt.legend(frameon=1)
if save_fig:
    plt.savefig('img/FvsT_day/all.png', dpi=150)
# plt.show()


# In[44]:

for s, c in zip(sites, sns.color_palette()):
    #     plt.scatter(df[df['site']==s]['TS_2'],df[df['site']==s]['FC'], label=s, color=c)
    g = sns.jointplot(x='TS_2', y='FC', data=df[(df["day_light_bool"] == True) & (df["site"] == s)],
                      label=s, color=c, size=15, kind='reg', dropna=1, xlim=(-20, 50), ylim=(-40, 40))
    g.set_axis_labels('Soil Temperature, C', 'Flux CO2, umol/m.sq/s')
    if save_fig:
        plt.savefig('img/FvsT_day/{}.png'.format(s), dpi=150)
    # plt.show()


# In[45]:

for s, c in zip(sites, sns.color_palette()):
    plt.scatter(df[(df["day_light_bool"] == False) & (df["site"] == s)]['SWC_1'], df[(df["day_light_bool"] == False) & (df["site"] == s)]['FC'], label=s, color=c)
    plt.xlabel('Soil Water Content, -')
    plt.ylabel('Flux CO2, umol/m.sq/s')
    plt.xlim([0, 100])
    plt.ylim([-40, 40])
plt.legend(frameon=1)
if save_fig:
    plt.savefig('img/FvsSWC_night/all.png', dpi=150)
# plt.show()


# In[46]:

for s, c in zip(sites, sns.color_palette()):
    #     plt.scatter(df[df['site']==s]['SWC_1'],df[df['site']==s]['FC'], label=s, color=c)
    try:
        g = sns.jointplot(x='SWC_1', y='FC', data=df[(df["day_light_bool"] == False) & (df["site"] == s)],
                          label=s, color=c, size=15, kind='reg', dropna=1, xlim=(0, 100), ylim=(-40, 40))
    except:
        pass
    g.set_axis_labels('Soil Water Content, -', 'Flux CO2, umol/m.sq/s')
    if save_fig:
        plt.savefig('img/FvsSWC_night/{}.png'.format(s), dpi=150)
    # plt.show()


# In[47]:

def theta_fun(SWC):
    return 3.11 * SWC - 2.42 * SWC * SWC


# In[48]:

def max_flux(df_column):
    return np.mean(df_column) + 3 * np.std(df_column)


# In[49]:

def min_flux(df_column):
    return np.mean(df_column) - 3 * np.std(df_column)


# In[50]:

def temp_dep_fun(Tk, Fmax, Ea, dS, Hd):
    R = 8.314
    opt_T = 290
    a = np.exp(Ea * (Tk - opt_T) / opt_T / R / Tk)
    b = 1 + np.exp((opt_T * dS - Hd) / opt_T / R)
    c = 1 + np.exp((Tk * dS - Hd) / Tk / R)
    return Fmax * a * b / c


# In[51]:

def temp_dep_fun_2(Tk, Fmax, Ea, dS, Hd):
    R = 8.314
    opt_T = 290
    a = np.exp(-Ea / R / Tk)
    c = 1 + np.exp((Tk * dS - Hd) / Tk / R)
    return Fmax * a / c


# In[52]:

def temp_dep_fun_3(Tk, T0=284.05718055, Ea=-67549.55123981, dS=543.64843782, Hd=155944.4165113):
    R = 8.314
    a = np.exp((-Ea / R / T0) * (1 - T0 / Tk))
    b = 1 + np.exp((Tk * dS - Hd) / Tk / R)
    return a / b


# In[53]:

from sklearn.ensemble import RandomForestRegressor


# In[54]:


model = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=1)


# In[55]:

df['site_n'] = -9999
for n, s in enumerate(sites):
    df.loc[df['site'] == s, 'site_n'] = n


# In[56]:

columns = ['SWC_1', 'TS_2', 'LOCATION_LONG', 'LOCATION_LAT', 'dayofyear', 'site_n', 'site', 'IGBP', 'CLIMATE_KOEPPEN']
target = ["FC"]

df_train = df[(df["day_light_bool"] == False) & (df['IGBP'] == 'ENF')][columns + target]
df_train.shape


# In[57]:

df_train['FC_scaled'] = np.nan


# In[58]:

df_train['site'].unique()


# In[59]:

for site in df_train['site'].unique():
    df_train.loc[df_train['site'] == site, 'FC_scaled'] = (df_train[df_train['site'] == site]['FC'] - min_flux(df_train[df_train['site'] == site]['FC'])) / \
        (max_flux(df_train[df_train['site'] == site]['FC']) - min_flux(df_train[df_train['site'] == site]['FC']))


# In[60]:

df_train.head()


# In[61]:

target = ["FC_scaled"]


# In[62]:

df_train.dropna(inplace=True)
df_train.shape


# In[63]:

sns.jointplot(x='SWC_1', y='FC_scaled', data=df_train, size=15, kind='reg', dropna=1, xlim=(0, 100), ylim=(-1, 2))
sns.jointplot(x='TS_2', y='FC_scaled', data=df_train, size=15, kind='reg', dropna=1, xlim=(-10, 30), ylim=(-1, 2))


# In[64]:

# Import a convenience function to split the sets.
from sklearn.cross_validation import train_test_split

# Generate the training set.  Set random_state to be able to replicate results.
train = df_train.sample(frac=0.5, random_state=1)
# Select anything not in the training set and put it in the testing set.
test = df_train.loc[~df_train.index.isin(train.index)]
# Print the shapes of both sets.
print(train.shape)
print(test.shape)


# In[65]:

from sklearn.metrics import mean_squared_error

# Initialize the model with some parameters.
model = RandomForestRegressor(n_estimators=200, min_samples_leaf=10, random_state=1, n_jobs=6)
# Fit the model to the data.
model.fit(train[['SWC_1', 'TS_2']], train[target])
# Make predictions.
predictions = model.predict(test[['SWC_1', 'TS_2']])
# Compute the error.
mean_squared_error(predictions, test[target])


# In[66]:

for feature in zip(columns, model.feature_importances_):
    print(feature)


# In[67]:

train.head()


# In[68]:

test['predicted_FC'] = predictions


# In[69]:

site_n = len(df.site.unique())
site_n


# In[70]:

test.head()


# In[71]:

test['week_n'] = test.index.to_series().dt.week
train['week_n'] = train.index.to_series().dt.week


# In[72]:

for s in np.arange(0, site_n):
    #     plt.plot(test.index, test['FC'], label=s, color=c)
    try:
        g = sns.pointplot(x='week_n', y='FC_scaled', data=train[train['site_n'] == s], ci=0, hue='site', label=s)
        g = sns.pointplot(x='week_n', y='predicted_FC', linestyles=':', data=test[test['site_n'] == s], ci=0)
        g.set_xticks(np.arange(1, 53, 5))
        g.set_xticklabels(np.arange(1, 53, 5))
    except:
        pass
    plt.ylabel('Flux CO2, umol/m.sq/s')
    plt.xlabel('Week number, #')
    plt.legend()

    if save_fig:
        #     if True:
        plt.savefig('img/F_ML_train_all/{}.png'.format(s), dpi=150)

    plt.ylim(0, 1)
    # plt.show()


# In[73]:

train[['SWC_1', 'TS_2']].corr()


# In[74]:

train.columns


# In[75]:

col_ren = ['SWC', 'TS']


# In[76]:

from matplotlib.colors import ListedColormap
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(train[['SWC_1', 'TS_2']].corr(), vmin=-1, vmax=1, cmap=ListedColormap(sns.color_palette(
    "RdBu_r", 31)))
fig.colorbar(cax)
ticks = np.arange(0, 2, 1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(col_ren)
ax.set_yticklabels(col_ren)
if save_fig:
    plt.savefig('img/corr_2.png'.format(s), dpi=150)
# plt.show()


# In[77]:


features = {f: v for f, v in zip(col_ren, model.feature_importances_)}
D = features
# import operator
# D = sorted(D.items(), key=operator.itemgetter(1))


# In[78]:

plt.bar(range(len(D)), D.values(), align='center')
plt.xticks(range(len(D)), D.keys())
plt.ylabel('% importance')
if save_fig:
    plt.savefig('img/importance_3.png'.format(s), dpi=150)
# plt.show()
# plt.show()


# In[79]:

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from scipy.cluster.vq import kmeans, vq
# from statsmodels.sandbox.tools.tools_pca import pcasvd


# data = df_train
# data = (data - data.mean()) / data.std()
# pca = pcasvd(data, keepdim=0, demean=False)

# values = data.values
# centroids, _ = kmeans(values, 3)
# idx, _ = vq(values, centroids)

# colors = ['gby'[i] for i in idx]

# plt.figure(1)
# biplot(plt, pca, labels=data.index, colors=colors,
#        xpc=1, ypc=2)
plt.show()


# In[80]:

df.head()


# In[81]:

columns


# In[82]:

sites


# In[83]:

T_test = np.linspace(0., 25.0, num=2501)


# In[84]:

SWC_test = np.linspace(0.0, 100.0, num=1001)


# In[85]:

t_arr = []  # np.zeros((1,1))
s_arr = []  # np.zeros((1,1))
for i, t in enumerate(T_test):
    for j, s in enumerate(SWC_test):
        t_arr.append(t)
        s_arr.append(s)


# In[86]:

fun_arr = pd.DataFrame({"TS_2": t_arr, "SWC_1": s_arr})


# In[87]:

fun_arr['FC'] = model.predict(fun_arr[['SWC_1', 'TS_2']])


# In[88]:

# %matplotlib inline

# import matplotlib.pyplot as p
# plt.switch_backend('Qt4Agg')
import scipy as sp

cond = 10
swc = []
ts = []
for i, c in zip(np.arange(0, 11), sns.color_palette("Blues", 11)):
    cond = i * 10
    p0 = [2.90000000e+02,  -1.09357077e+05,  7.00896111e+02,   2.03652714e+05]
    xdata = fun_arr[fun_arr['SWC_1'] == cond]['TS_2'] + 273.15
    ydata = fun_arr[fun_arr['SWC_1'] == cond]['FC']
    try:
        popt, pcov = sp.optimize.curve_fit(temp_dep_fun_3, xdata, ydata, p0)
        plt.plot(xdata, temp_dep_fun_3(xdata, *popt), label=r'$\theta$ = ' + str(cond) + ', $T_k$ = ' + str(popt[0])[:7], lw=6, alpha=1, color=c)
    except:
        pass
#     plt.scatter(xdata, ydata, label=None, alpha=0.03, color=c)
    swc.append(cond)
    ts.append(popt[0])
popt, pcov = sp.optimize.curve_fit(temp_dep_fun_3, fun_arr['TS_2'] + 273.15, fun_arr['FC'], p0)
plt.plot(fun_arr['TS_2'] + 273.15, temp_dep_fun_3(fun_arr['TS_2'] + 273.15, *popt), lw=6, alpha=1, color='k', label='Average')
plt.legend(frameon=1)
plt.xlabel('Temperature, [K]')
plt.ylabel(r'$\frac{F}{F_{max}}$, [-]')


plt.ylim(0, 1)
if save_fig:
    # if True:
    plt.savefig('img/T_dependence_all_ENF_lines.png', dpi=150)
# plt.show()


# In[89]:

# matplotlib.rcsetup.interactive_bk
# matplotlib.rcsetup.non_interactive_bk
# matplotlib.rcsetup.all_backends


# In[90]:

T_test = np.linspace(0., 25.0, num=25)
SWC_test = np.linspace(0.0, 100.0, num=11)
t_arr = []  # np.zeros((1,1))
s_arr = []  # np.zeros((1,1))
for i, t in enumerate(T_test):
    for j, s in enumerate(SWC_test):
        t_arr.append(t)
        s_arr.append(s)
d3_arr = pd.DataFrame({"TS_2": t_arr, "SWC_1": s_arr})
d3_arr['FC'] = model.predict(d3_arr[['SWC_1', 'TS_2']])


# In[91]:

# DataFrame from 2D-arrays
# get_ipython().magic('matplotlib inline')
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
ax = Axes3D(fig)

ax.plot_trisurf(d3_arr.TS_2, d3_arr.SWC_1, d3_arr.FC, cmap=cm.jet, linewidth=0.2)
# plt.show()


# In[92]:

plt.scatter(swc, ts)
# plt.show()


# In[93]:

cond = 10
for i in np.arange(0, 11):
    cond = (10 - i) * 10
    p0 = [2.90000000e+02,  -1.09357077e+05,  7.00896111e+02,   2.03652714e+05]
    xdata = fun_arr[fun_arr['SWC_1'] == cond]['TS_2'] + 273.15
    ydata = fun_arr[fun_arr['SWC_1'] == cond]['FC']
    try:
        popt, pcov = sp.optimize.curve_fit(temp_dep_fun_3, xdata, ydata, p0)
#         if False:
        if cond == 70:
            a = 1.
            c = 'k'
            res_T = popt
        else:
            a = 0.5
            c = None
        plt.plot(xdata, temp_dep_fun_3(xdata, *popt), label=r'$\theta$ = ' + str(cond), lw=6, alpha=0.5, color=c)
    except:
        pass
    plt.scatter(xdata, ydata, label=None, alpha=0.03, color=c)
popt, pcov = sp.optimize.curve_fit(temp_dep_fun_3, fun_arr['TS_2'] + 273.15, fun_arr['FC'], p0)
plt.plot(fun_arr['TS_2'] + 273.15, temp_dep_fun_3(fun_arr['TS_2'] + 273.15, *popt), lw=6, alpha=1, color='k', label='Average')
plt.legend(frameon=1)
plt.xlabel('Temperature, [K]')
plt.ylabel(r'$\frac{F}{F_{max}}$, [-]')


plt.ylim(0, 1)
if save_fig:
    # if True:
    plt.savefig('img/T_dependence_ENF_data.png', dpi=150)


# In[94]:

for cond in np.linspace(0, 25, 6):
    xdata = fun_arr[fun_arr['TS_2'] == cond]['SWC_1']
    ydata = fun_arr[fun_arr['TS_2'] == cond]['FC']
    plt.scatter(xdata, ydata, label=cond)

plt.ylim(0, 1)
plt.xlim(5, 60)
plt.legend(frameon=1)
plt.xlabel('Soil Water Content, [-]')
plt.ylabel(r'$\frac{F}{F_{max}}$, [-]')


# In[95]:

train.site.unique()


# In[96]:

#     plt.plot(test.index, test['FC'], label=s, color=c)
for site in ['CA-TP4', 'CA-Qfo', 'CA-Qcu', 'CA-TP2', 'CA-TP3', 'CA-TP1', 'CA-Obs', 'CA-Ojp', 'CA-NS3', 'CA-NS1']:
    g = sns.pointplot(x='week_n', y='FC_scaled', data=train[train['site'] == site], ci=0, label=site)
    g = sns.pointplot(x='week_n', y='predicted_FC', linestyles=':', data=test[test['site'] == site], ci=0)
    train.loc[train['site'] == site, 'estimated_FC'] = temp_dep_fun_3(train[train['site'] == site]['TS_2'] + 273.15, *res_T)
    g = sns.pointplot(x='week_n', y='estimated_FC', linestyles=':', data=train[train['site'] == site], ci=0, color='g', label='eq')
    g.set_xticks(np.arange(1, 53, 5))
    g.set_xticklabels(np.arange(1, 53, 5))
    plt.ylabel(r'$\frac{F}{F_{max}}$, [-]')
    plt.xlabel('Week number, #')
    plt.legend()

    if save_fig:
        #     if True:
        plt.savefig('img/F_scaled_ML_ENF/{}.png'.format(s), dpi=150)

    plt.title(site)
    plt.ylim(0, 1)
    # plt.show()
# data = train[train['site_n']==s]['FC_scaled']
# Fx_range = max_flux(data) - min_flux(data)
# train.loc[train['site_n']==s, 'estimated_FC'] = 0.75*Fx_range*temp_dep_fun_3(train[train['site_n']==s]['TS_2']+273.15, *trial) + 0.25*Fx_range*theta_dep(train[train['site_n']==s]['SWC_1']/100) + min_flux(data) # *theta_dep(train[train['site_n']==s]['SWC_1']/100, **params)
# train.loc[train['site_n']==s, 'estimated_FC_T'] =
# Fx_range*temp_dep_fun_3(train[train['site_n']==s]['TS_2']+273.15,
# *trial)  + min_flux(data) #
# *theta_dep(train[train['site_n']==s]['SWC_1']/100, **params)


# In[97]:

cond = 90
for i in np.arange(0, 14):
    cond = (i) * 5 + 10
    plt.scatter(fun_arr[fun_arr['SWC_1'] == cond]['TS_2'] + 273.15, np.log10(fun_arr[fun_arr['SWC_1'] == cond]['FC']), label=r'$\theta$ = ' + str(cond))
    plt.xlabel('Temperature, K')
    plt.ylabel(r'Flux $CO_2$, umol/m.sq/s')
plt.legend()
if save_fig:
    # if True:
    plt.savefig('img/T_dependence.png', dpi=150)


# In[98]:

cond = 10
plt.scatter(fun_arr[fun_arr['TS_2'] == cond]['SWC_1'], (fun_arr[fun_arr['TS_2'] == cond]['FC']))


# In[99]:

import scipy as sp
cond = 10

xdata = fun_arr[fun_arr['SWC_1'] == cond]['TS_2'] + 273.15
ydata = fun_arr[fun_arr['SWC_1'] == cond]['FC']
popt, pcov = sp.optimize.curve_fit(temp_dep_fun_3, xdata, ydata, p0=[290,  -1.09357077e+05, 7.00896111e+02,   2.03652714e+05])


# In[100]:

popt


# In[101]:

cond = 10
for i in np.arange(2, 3):
    cond = 30
    p0 = [2.90000000e+02,  -1.09357077e+05,  7.00896111e+02,   2.03652714e+05]
    xdata = fun_arr[fun_arr['SWC_1'] == cond]['TS_2'] + 273.15
    ydata = fun_arr[fun_arr['SWC_1'] == cond]['FC']
    try:
        popt, pcov = sp.optimize.curve_fit(temp_dep_fun_3, xdata, ydata, p0)
        plt.plot(xdata, temp_dep_fun_3(xdata, *popt), label=cond, lw=6)
        print(popt)
    except:
        pass
    plt.scatter(xdata, ydata, label=None)
    plt.legend()
    plt.xlabel('Temperature, K')
    plt.ylabel('Flux CO2, umol/m.sq/s')
# plt.ylim(-1,1)


# In[102]:

res_T


# In[103]:

for i in np.arange(0, 10):
    cond = 1 * i + 10
    xdata = fun_arr[fun_arr['TS_2'] == cond]['SWC_1']
    ydata = fun_arr[fun_arr['TS_2'] == cond]['FC']
#     try:
#         popt, pcov = sp.optimize.curve_fit(temp_dep_fun_3, xdata, ydata)
#         plt.plot(xdata, temp_dep_fun_3(xdata, *popt ), label=cond, lw=6)
#     except:
#         pass
    plt.scatter(xdata, ydata / max(ydata), label=cond)
    plt.legend()
    plt.xlabel('Soil Water Content, [-]')
    plt.ylabel(r'$\frac{F}{F_{max}}$, [-]')
    plt.ylim(0, 1)
    plt.xlim(5, 80)


# In[104]:

import math


def norm_dist(x, mu=0, sig=0.2):
    return 1 / np.sqrt(2 * math.pi * sig**2) * np.exp(-(x - mu)**2 / (2 * sig**2))


# In[105]:

cond = 20
xdata = fun_arr[fun_arr['TS_2'] == cond]['SWC_1']
ydata = fun_arr[fun_arr['TS_2'] == cond]['FC']
#     try:
#         popt, pcov = sp.optimize.curve_fit(temp_dep_fun_3, xdata, ydata)
#         plt.plot(xdata, temp_dep_fun_3(xdata, *popt ), label=cond, lw=6)
#     except:
#         pass
plt.scatter(xdata, ydata / max(ydata), label=cond)
plt.legend()
plt.xlabel('Soil Water Content, -')
plt.ylabel('Flux Scale Factor, [0-1]')

x = SWC_test / 100
mu = 0.11
sig = 0.04
plt.plot(x * 100, norm_dist(x, mu, sig) / max(norm_dist(x, mu, sig)))
mu = 0.37
sig = 0.1
plt.plot(x * 100, 0.65 * norm_dist(x, mu, sig) / max(norm_dist(x, mu, sig)))
mu = 0.71
sig = 0.1
plt.plot(x * 100, 0.55 * norm_dist(x, mu, sig) / max(norm_dist(x, mu, sig)))


# In[106]:

def sin_dumped(x, A, B, k, omega, phi):
    return A * np.exp(-k * x) * (np.cos(omega * x + phi) + np.sin(omega * x + phi)) + B


# In[107]:

# from scipy.fftpack import dct, idct
# import matplotlib.pyplot as plt
# N = 301
# t = SWC_test
# x = ydata/max(ydata)
# y = dct(x, norm='ortho')
# window = np.zeros(N)
# window[:30] = 1
# yr = idct(y*window, norm='ortho')
# sum(abs(x-yr)**2) / sum(abs(x)**2)

# plt.scatter(t, x)
# plt.plot(t, yr, lw=5)

# plt.legend(['x', '$x_{20}$', '$x_{15}$'])
plt.show()


# In[108]:

def theta_dep(x, mu1=0.11, sig1=0.04, mu2=0.37, sig2=0.1, mu3=0.71, sig3=0.1, mu4=1.1, sig4=0.2):
    a = 0.85 * norm_dist(x, mu1, sig1) / max(norm_dist(x, mu1, sig1))
    b = 0.5 * norm_dist(x, mu2, sig2) / max(norm_dist(x, mu2, sig2))
    c = 0.3 * norm_dist(x, mu3, sig3) / max(norm_dist(x, mu3, sig3))
    d = 0.3 * norm_dist(x, mu4, sig4) / max(norm_dist(x, mu4, sig4))
    return a + b + c + d


# In[109]:

cond = 30
# for cond in [30,20,10, 0]:
xdata = fun_arr[fun_arr['TS_2'] == cond]['SWC_1']
ydata = fun_arr[fun_arr['TS_2'] == cond]['FC']
Fx_range = max_flux(fun_arr[fun_arr['TS_2'] == cond]['FC']) - min_flux(fun_arr[fun_arr['TS_2'] == cond]['FC'])
plt.scatter(xdata / 100, (ydata - min_flux(fun_arr[fun_arr['TS_2'] == cond]['FC'])) / Fx_range, label='T = ' + str(cond) + 'C', alpha=1.0)


cond = 20
# for cond in [30,20,10]:
xdata = fun_arr[fun_arr['TS_2'] == cond]['SWC_1']
ydata = fun_arr[fun_arr['TS_2'] == cond]['FC']
Fx_range = max_flux(fun_arr[fun_arr['TS_2'] == cond]['FC']) - min_flux(fun_arr[fun_arr['TS_2'] == cond]['FC'])
plt.scatter(xdata / 100, (ydata - min_flux(fun_arr[fun_arr['TS_2'] == cond]['FC'])) / Fx_range, label='T = ' + str(cond) + 'C', alpha=1.0)

cond = 10
# for cond in [30,20,10]:
xdata = fun_arr[fun_arr['TS_2'] == cond]['SWC_1']
ydata = fun_arr[fun_arr['TS_2'] == cond]['FC']
Fx_range = max_flux(fun_arr[fun_arr['TS_2'] == cond]['FC']) - min_flux(fun_arr[fun_arr['TS_2'] == cond]['FC'])
plt.scatter(xdata / 100, (ydata - min_flux(fun_arr[fun_arr['TS_2'] == cond]['FC'])) / Fx_range, label='T = ' + str(cond) + 'C', alpha=1.0)

cond = 0
# for cond in [30,20,10]:
xdata = fun_arr[fun_arr['TS_2'] == cond]['SWC_1']
ydata = fun_arr[fun_arr['TS_2'] == cond]['FC']
Fx_range = max_flux(fun_arr[fun_arr['TS_2'] == cond]['FC']) - min_flux(fun_arr[fun_arr['TS_2'] == cond]['FC'])
plt.scatter(xdata / 100, (ydata - min_flux(fun_arr[fun_arr['TS_2'] == cond]['FC'])) / Fx_range, label='T = ' + str(cond) + 'C', alpha=.03)

plt.ylabel(r'$\frac{F}{F_{max}}$, [-]')
plt.legend()
plt.xlabel(r'$\theta$, [-]')

plt.ylim(0, 1)
plt.xlim(0, 1)
if save_fig:
    # if True:
    plt.savefig('img/Theta_dependence_0.png', dpi=150)


# In[110]:

for cond in [30, 20, 10, 0]:
    xdata = fun_arr[fun_arr['TS_2'] == cond]['SWC_1']
    ydata = fun_arr[fun_arr['TS_2'] == cond]['FC']
    Fx_range = max_flux(fun_arr[fun_arr['TS_2'] == cond]['FC']) - min_flux(fun_arr[fun_arr['TS_2'] == cond]['FC'])

    #     try:
    #         popt, pcov = sp.optimize.curve_fit(temp_dep_fun_3, xdata, ydata)
    #         plt.plot(xdata, temp_dep_fun_3(xdata, *popt ), label=cond, lw=6)
    #     except:
    #         pass
    plt.scatter(xdata / 100, (ydata - min_flux(fun_arr[fun_arr['TS_2'] == cond]['FC'])) / Fx_range, label='T = ' + str(cond) + 'C', alpha=.05)
plt.legend()
plt.xlabel(r'$\theta$, [-]')

x = SWC_test / 100
mu = 0.11
sig = 0.04
a = 0.85 * norm_dist(x, mu, sig) / max(norm_dist(x, mu, sig))
plt.plot(x, a, label='a', lw=2)
mu = 0.37
sig = 0.1
b = 0.5 * norm_dist(x, mu, sig) / max(norm_dist(x, mu, sig))
plt.plot(x, b, label='b', lw=2)
mu = 0.71
sig = 0.1
c = 0.3 * norm_dist(x, mu, sig) / max(norm_dist(x, mu, sig))
plt.plot(x, c, label='c', lw=2)
mu = 1.1
sig = 0.2
d = 0.3 * norm_dist(x, mu, sig) / max(norm_dist(x, mu, sig))
plt.plot(x, d, label='d', lw=2)
plt.plot(x, theta_dep(x, mu1=0.11, sig1=0.04, mu2=0.37, sig2=0.1, mu3=0.71, sig3=0.1, mu4=1.1, sig4=0.2), lw=4, label='a + b + c + d', color='k')
plt.ylabel(r'$\frac{F}{F_{max}}$, [-]')
plt.legend()
plt.ylim(0, 1)
plt.xlim(0, 1)
if save_fig:
    # if True:
    plt.savefig('img/Theta_dependence_fit.png', dpi=150)


# In[111]:


for cond in [20, 10, 0]:
    xdata = fun_arr[fun_arr['TS_2'] == cond]['SWC_1']
    ydata = fun_arr[fun_arr['TS_2'] == cond]['FC']
    Fx_range = max_flux(fun_arr[fun_arr['TS_2'] == cond]['FC']) - min_flux(fun_arr[fun_arr['TS_2'] == cond]['FC'])

    #     try:
    #         popt, pcov = sp.optimize.curve_fit(temp_dep_fun_3, xdata, ydata)
    #         plt.plot(xdata, temp_dep_fun_3(xdata, *popt ), label=cond, lw=6)
    #     except:
    #         pass
    plt.scatter(xdata / 100, (ydata - min_flux(fun_arr[fun_arr['TS_2'] == cond]['FC'])) / Fx_range / temp_dep_fun_3(fun_arr[fun_arr['TS_2'] == cond]['TS_2'] + 273.15, *res_T) / max(
        (ydata - min_flux(fun_arr[fun_arr['TS_2'] == cond]['FC'])) / Fx_range / temp_dep_fun_3(fun_arr[fun_arr['TS_2'] == cond]['TS_2'] + 273.15, *res_T)), label='T = ' + str(cond) + 'C', alpha=.5)
plt.legend()
plt.xlabel(r'$\theta$, [-]')

x = SWC_test / 100
mu = 0.11
sig = 0.04
a = 0.85 * norm_dist(x, mu, sig) / max(norm_dist(x, mu, sig))
plt.plot(x, a, label='a', lw=2)
mu = 0.37
sig = 0.1
b = 0.5 * norm_dist(x, mu, sig) / max(norm_dist(x, mu, sig))
plt.plot(x, b, label='b', lw=2)
mu = 0.71
sig = 0.1
c = 0.3 * norm_dist(x, mu, sig) / max(norm_dist(x, mu, sig))
plt.plot(x, c, label='c', lw=2)
mu = 1.1
sig = 0.2
d = 0.3 * norm_dist(x, mu, sig) / max(norm_dist(x, mu, sig))
plt.plot(x, d, label='d', lw=2)
plt.plot(x, theta_dep(x, mu1=0.11, sig1=0.04, mu2=0.37, sig2=0.1, mu3=0.71, sig3=0.1, mu4=1.1, sig4=0.2), lw=4, label='a + b + c + d', color='k')
plt.ylabel(r'$\frac{F}{F_{max}}$, [-]')
plt.legend()
plt.ylim(0, 1)
plt.xlim(0, 1)
if save_fig:
    # if True:
    plt.savefig('img/Theta_dependence_fit.png', dpi=150)


# In[112]:

swc = fun_arr[fun_arr['TS_2'] == cond]['SWC_1']
ts = fun_arr[fun_arr['TS_2'] == cond]['TS_2']


# In[ ]:


# In[113]:

cond = 10
for i in np.arange(1, 3):
    cond = i * 10
    xdata = df[(df['SWC_1'] > cond - 1) & (df['SWC_1'] < cond + 1) & (df['day_light_bool'] == False)]['TS_2'] + 273.15
    ydata = df[(df['SWC_1'] > cond - 1) & (df['SWC_1'] < cond + 1) & (df['day_light_bool'] == False)]['FC']
#     popt, pcov = sp.optimize.curve_fit(temp_dep_fun_3, xdata, ydata)
#     plt.plot(xdata, temp_dep_fun_3(xdata, *popt ), label=cond, lw=6)
    plt.scatter(xdata, ydata, label=None)
    plt.legend()
    plt.xlabel('Temperature, K')
    plt.ylabel('Flux CO2, umol/m.sq/s')


# In[114]:

train.columns


# In[115]:

train['estimated_FC'] = 0
train['estimated_FC_T'] = 0


# In[116]:

train[train['site_n'] == s].head()


# In[117]:

params = {'mu1': 0.11, 'sig1': 0.1, 'mu2': 0.37, 'sig2': 0.1, 'mu3': 0.71, 'sig3': 0.2, 'mu4': 1.1, 'sig4': 0.2}
trial = res_T
# max_Fx=[12,7,7,6,11,9] # replaced with: Fx = mean + 3*SD
for i, s in enumerate([10, 14, 20, 23, 22, 0]):
    #     train.loc[train['site_n']==s, 'max_Fx'] = max(train[train['site_n']==s]['FC'])
    data = train[train['site_n'] == s]['FC_scaled']
    Fx_range = max_flux(data) - min_flux(data)
    train.loc[train['site_n'] == s, 'estimated_FC'] = 0.75 * Fx_range * temp_dep_fun_3(train[train['site_n'] == s]['TS_2'] + 273.15, *trial) + 0.25 * Fx_range * theta_dep(
        train[train['site_n'] == s]['SWC_1'] / 100) + min_flux(data)  # *theta_dep(train[train['site_n']==s]['SWC_1']/100, **params)
    # *theta_dep(train[train['site_n']==s]['SWC_1']/100, **params)
    train.loc[train['site_n'] == s, 'estimated_FC_T'] = Fx_range * temp_dep_fun_3(train[train['site_n'] == s]['TS_2'] + 273.15, *trial) + min_flux(data)
for i, s in enumerate([10, 14, 20, 23, 22, 0]):  # 10,14,20,23,22,0]):
    #     plt.plot(test.index, test['FC'], label=s, color=c)
    try:
        g = sns.pointplot(x='week_n', y='FC_scaled', data=train[train['site_n'] == s], ci=0, hue='site', label=s)
        g = sns.pointplot(x='week_n', y='estimated_FC', linestyles=':', data=train[train['site_n'] == s], ci=0, color='g')
        g = sns.pointplot(x='week_n', y='estimated_FC_T', linestyles=':', data=train[train['site_n'] == s], ci=0, color='r')
        g.set_xticks(np.arange(1, 53, 5))
        g.set_xticklabels(np.arange(1, 53, 5))
    except:
        pass
    plt.ylabel('Flux CO2, umol/m.sq/s')
    plt.xlabel('Week number, #')
    plt.legend()

    if save_fig:
        #     if True:
        plt.savefig('img/Reverse_engin/{}.png'.format(s), dpi=150)

    # plt.show()


# In[ ]:

trial = res_T
# max_Fx=[6,5,6,3,6,5] # replaced with: Fx = mean + 3*SD
for i, s in enumerate([10, 14, 20, 23, 22, 0]):  # [10,14,20,23,22,0]
    #     train.loc[train['site_n']==s, 'max_Fx'] = max(train[train['site_n']==s]['FC'])
    Fx_range = max_flux(train[train['site_n'] == s]['FC']) - min_flux(train[train['site_n'] == s]['FC'])
    # *theta_dep(train[train['site_n']==s]['SWC_1']/100)
    train.loc[train['site_n'] == s, 'estimated_FC'] = Fx_range * temp_dep_fun_3(train[train['site_n'] == s]['TS_2'] + 273.15, *trial) + min_flux(train[train['site_n'] == s]['FC'])
for i, s in enumerate([10, 14, 20, 23, 22, 0]):  # 10,14,20,23,22,0]):
    #     plt.plot(test.index, test['FC'], label=s, color=c)
    try:
        g = sns.pointplot(x='week_n', y='FC', data=train[train['site_n'] == s], ci=0, hue='site', label=s)
        g = sns.pointplot(x='week_n', y='estimated_FC', linestyles=':', data=train[train['site_n'] == s], ci=0)
        g.set_xticks(np.arange(1, 53, 5))
        g.set_xticklabels(np.arange(1, 53, 5))
    except:
        pass
    plt.ylabel('Flux CO2, umol/m.sq/s')
    plt.xlabel('Week number, #')
    plt.legend()

    if save_fig:
        #     if True:
        plt.savefig('img/Reverse_engin_only_T/{}.png'.format(s), dpi=150)

    # plt.show()


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:

res_T


# In[ ]:

trial = res_T
# max_Fx=[12,7,7,6,11,9] # replaced with: Fx = mean + 3*SD
for i, s in enumerate([10, 14, 20, 23, 22, 0]):
    #     train.loc[train['site_n']==s, 'max_Fx'] = max(train[train['site_n']==s]['FC'])
    #     max_Fx = max_flux(train[train['site_n']==s]['FC'] + 1)
    Fx_range = max_flux(train[train['site_n'] == s]['FC']) - min_flux(train[train['site_n'] == s]['FC'])
    train.loc[train['site_n'] == s, 'estimated_FC'] = 0.75 * Fx_range * temp_dep_fun_3(train[train['site_n'] == s]['TS_2'] + 273.15, *trial) + 0.25 * Fx_range * theta_fun(
        train[train['site_n'] == s]['SWC_1'] / 100) + min_flux(train[train['site_n'] == s]['FC'])  # *theta_fun(train[train['site_n']==s]['SWC_1']/100)
#     train.loc[train['site_n']==s, 'estimated_FC'] = max_Fx*temp_dep_fun_3(train[train['site_n']==s]['TS_2']+273.15, *trial)*theta_fun(train[train['site_n']==s]['SWC_1']/100) - 1.0 #
#     train[train['site_n']==s]['estimated_FC'] = max_Fx*theta_dep(train[train['site_n']==s]['SWC_1']/100)*temp_dep_fun_3(train[train['site_n']==s]['TS_2']+273.15)
for i, s in enumerate([10, 14, 20, 23, 22, 0]):  # 10,14,20,23,22,0]):
    #     plt.plot(test.index, test['FC'], label=s, color=c)
    try:
        g = sns.pointplot(x='week_n', y='FC', data=train[train['site_n'] == s], ci=0, hue='site', label=s)
        g = sns.pointplot(x='week_n', y='estimated_FC', linestyles=':', data=train[train['site_n'] == s], ci=0)
        g.set_xticks(np.arange(1, 53, 5))
        g.set_xticklabels(np.arange(1, 53, 5))
    except:
        pass
    plt.ylabel('Flux CO2, umol/m.sq/s')
    plt.xlabel('Week number, #')
    plt.legend()

    if save_fig:
        #     if True:
        plt.savefig('img/Reverse_engin_book/{}.png'.format(s), dpi=150)

    # plt.show()


# In[ ]:


# In[ ]:


# In[ ]:
