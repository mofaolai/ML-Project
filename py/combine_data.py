#import necessary libraries and files 
import pandas as pd
import numpy as np
import warnings
import time
import datetime as dt
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import folium
import sklearn
import seaborn as sns

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

import os  # for reading in files
import glob
##
old_dir = os.getcwd()
#os.chdir(old_dir)
# dublinbikes folder only has dublinbikes*.csv files
##
os.chdir(old_dir)
# dublinbikes folder only has dublinbikes*.csv files
#path = r"dublinbikes/"
#os.chdir(path)
print(os.getcwd())
os.chdir('C:/Users/imeld/work/ML_CSP/ML-Project')
#os.chdir('~/work/ML_CSP/ML-Project')
folder_path = 'dublinbikes'
file_list = glob.glob(folder_path + "/dublinbikes*.csv")
print(file_list)
##
# constant value switches
MAKE_FILES=True
READ_FILES = False
FILE_MASK = "dublinbikes*.csv"
#FILE_MASK = "dublinbikes_2021*.csv"
##
#https://www.geeksforgeeks.org/how-to-read-multiple-data-files-into-pandas/
#https://realpython.com/read-write-files-python/#text-file-types

print(file_list)
full_data = pd.DataFrame(pd.read_csv(file_list[0]))

for i in range(1, len(file_list)):
    data = pd.read_csv(file_list[i])
    #print(data.head())
    df = pd.DataFrame(data)
    full_data = pd.concat([full_data, df])
os.chdir(old_dir)
##
print(full_data.sample(10))
# write out full data to csv file
if MAKE_FILES : full_data.to_csv("data/full_data.csv")
if READ_FILES : full_data = pd.read_csv("data/full_data.csv")

##

print((full_data['NAME']).unique())

##
full_data['usage'] = full_data['AVAILABLE BIKES'].diff()
full_data.head()

##
#cluster stations
data = full_data
data = data[data['STATUS'] == 'Open']
data = data[(data['LAST UPDATED'] >= '2019-07-01') & (data['LAST UPDATED'] < '2020-04-01')]
data = data[(data['LAST UPDATED'] < '2019-12-01') | (data['LAST UPDATED'] >= '2020-02-01')]

#remove rows where no update actually occurs
data = data.drop(['TIME'], axis = 1)
data.drop_duplicates(keep= 'first',inplace=True)


#get date and time columns
data['DATETIME'] = [dt.datetime.strptime(d, "%Y-%m-%d %H:%M:%S") for d in data["LAST UPDATED"]]
data['LAST UPDATED'] = [dt.datetime.time(d) for d in data['DATETIME']] 
data['DATE'] = [dt.datetime.date(d) for d in data['DATETIME']] 
data['date_for_merge'] = data['DATETIME'].dt.round('H')

#create important features
data['OCCUPANCY_PCT'] =  data['AVAILABLE BIKES'] / data['BIKE STANDS']
data['FULL'] = np.where(data['OCCUPANCY_PCT'] == 0, 1,0 )
data['EMPTY'] = np.where(data['OCCUPANCY_PCT'] == 1, 1,0 )

### create time aggregates needed for clustering
# weekday/saturday/sunday
data['DAY_NUMBER'] = data.DATETIME.dt.dayofweek
data['DAY_TYPE'] = np.where(data['DAY_NUMBER'] <= 4, 'Weekday', (np.where(data['DAY_NUMBER'] == 5, 'Saturday', 'Sunday')))

def bin_time(x):
    if x.time() < dt.time(6):
        return "Overnight "
    elif x.time() < dt.time(11):
        return "6AM-10AM "
    elif x.time() < dt.time(16):
        return "11AM-3PM "
    elif x.time() < dt.time(20):
        return "4PM-7PM "
    elif x.time() <= dt.time(23):
        return "8PM-11PM "
    else:
        return "Overnight "


data["TIME_TYPE"] = data['DATETIME'].apply(bin_time)
data['HOUR'] = data['DATETIME'].dt.hour
data['MONTH'] = data['DATETIME'].dt.month
data['CLUSTER_GROUP'] = data['TIME_TYPE'] + data['DAY_TYPE']

data.sample(5)

##
data.describe()
data.head()

##
#group data into clusters
clustering_df = data[['STATION ID', 'NAME', 'LATITUDE', 'LONGITUDE', 'DAY_TYPE', 'TIME_TYPE', 'OCCUPANCY_PCT','CLUSTER_GROUP']]
clustering_df = clustering_df.groupby(['STATION ID', 'NAME', 'LATITUDE', 'LONGITUDE', 'CLUSTER_GROUP'],as_index=False)['OCCUPANCY_PCT'].mean()
clustering_df  = clustering_df.set_index('STATION ID')

#pivot dataframe for clustering
clustering_df = clustering_df.pivot_table(index= ['NAME', 'STATION ID','LATITUDE', 'LONGITUDE'] , columns=['CLUSTER_GROUP'], values='OCCUPANCY_PCT')
clustering_df  = clustering_df.reset_index()
clustering_df  = clustering_df .set_index('NAME')
clustering_df = clustering_df.dropna()

clustering_df.sample(5)

##
distortions = []
K = range(1,10)
X = np.array(clustering_df.drop(['STATION ID', 'LATITUDE', 'LONGITUDE'], 1).astype(float))
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(X)
    distortions.append(kmeanModel.inertia_)
    
plt.figure(figsize=(10,7))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

##

#clustering algo
X = np.array(clustering_df.drop(['STATION ID', 'LATITUDE', 'LONGITUDE'], 1).astype(float))
KM = KMeans(n_clusters=5) 
#KM = KMeans(n_clusters=3) 
KM.fit(X)
clusters = KM.predict(X)

locations = clustering_df
locations['Cluster'] = clusters
locations = locations.reset_index()
locations.head(5)

##
colordict = {0: 'blue', 1: 'red', 2: 'orange', 3: 'green', 4: 'purple'}
dublin_map = folium.Map([53.345, -6.2650], zoom_start=13.5)
for LATITUDE, LONGITUDE, Cluster in zip(locations['LATITUDE'],locations['LONGITUDE'], locations['Cluster']):
    folium.CircleMarker(
        [LATITUDE, LONGITUDE],
        color = 'b',
        radius = 8,
        fill_color=colordict[Cluster],
        fill=True,
        fill_opacity=0.9
        ).add_to(dublin_map)
dublin_map

##

fig,ax = plt.subplots(1, 1, figsize=(40, 10))
ax.plot(range(0,len(data['AVAILABLE BIKES'])),data["TIME_TYPE"])
plt.show()

##
#'HANOVER QUAY' in Grand Canal Dock
#'FITZWILLIAM SQUARE EAST' in south Dublin
#'ST JAMES HOSPITAL (LUAS)'- 
# 

## can you add cluster as a column to the full data?
# maybe add 1 to each cluster - so add mater hospital to st james
# pick two on dart line & two further out?
###CHANGE station_names HERE

station_names = ['HANOVER QUAY','FITZWILLIAM SQUARE EAST', 'ST JAMES HOSPITAL (LUAS)']


##

station_names_mask = full_data['NAME'].isin(station_names)
data = full_data[station_names_mask]
##

# write out full data to csv file

if MAKE_FILES : data.to_csv("data/station_data.csv")
#data = pd.read_csv("data/station_data.csv")

##
#df.drop(columns=["ADDRESS","LATITUDE","LONGITUDE","LAST UPDATED","NAME","STATION ID","AVAILABLE BIKE STANDS","STATUS"],axis=1,inplace=True)
#df["TIME"]=pd.to_datetime(df["TIME"])  
#df.sort_values(by=['TIME'],inplace=True)
#df["GAP AMOUNT"]=df["TIME"].diff().dt.seconds/60





