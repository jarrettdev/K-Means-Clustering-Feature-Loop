#================================================================
# K-Means Looper
# DataRanch.info X Jarrett Devereaux
#================================================================

#%%
#sklearn imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
#%%
#Update csv here
csv_name = 'combined_cars.csv'
drop_col = 'id'

#%%
#read in data + drop duplicates
df = pd.read_csv(f'data/{csv_name}', low_memory=False)
df = df.drop_duplicates(drop_col)
#%%
def complete_kmeans(optimal_clusters, X, col1, col2, csv_name):
    kmeans = KMeans(n_clusters=optimal_clusters)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.xlabel(col1)
    plt.ylabel(col2)
    #plt.show()
    #make a directory with csv_name if one doesn't exist
    csv_name = csv_name.split('.')[0]
    try:
        os.mkdir(f'./data_pics')
    except Exception as e:
        print(e)
    #output plt to file
    plt.savefig(f'data_pics/{csv_name}_{col1}_{col2}_kmeans.png')
    #clear plots for the next iteration in the loop
    plt.clf()
    plt.cla()
    plt.close()
#%%
#function that takes in two column names and processes the columns for k-means
def produce_kmeans(df, col1, col2, csv_name):
    df[col1] = df[col1].astype(float)
    df[col2] = df[col2].astype(float)
    df[col1] = df[col1].fillna(0)
    df[col2] = df[col2].fillna(0)
    x_list = df[col1]
    y_list = df[col2]
    zipped_list = list(zip(x_list, y_list))
    X = np.array(zipped_list)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    #Uncomment to use the elbow method
    #K-Means elbow method to determine the optimal number of clusters
    #================================================================
    #dist_df = []
    #for num_clusters in range(1, 11):
        #kmeans = KMeans(n_clusters=num_clusters)
        #kmeans.fit(X)
        #wcss = kmeans.inertia_
        #dist_df.append([num_clusters, wcss])
    #dist_df = pd.DataFrame(dist_df, columns=['num_clusters', 'wcss'])
    #dist_df.plot(x='num_clusters', y='wcss', kind='line')
    #plt.show()
    #plt.scatter(x_list, y_list, s=50)
    #plt.show()
    #================================================================
    
    try:
        #uncomment to edit the number of clusters you want to use
        #================================================================
        #optimal_clusters = int(input('How many clusters would you like?'))
        #================================================================
        optimal_clusters = 3
    except:
        optimal_clusters = 3
    complete_kmeans(optimal_clusters, X, col1, col2, csv_name)

#%%
# Driver (edit the column name here)
#====================================================
#run produce_kmeans on every column in df vs price
for col in df.columns:
    try:
        produce_kmeans(df, 'price', col, csv_name)
    except:
        continue
#====================================================
# %%
