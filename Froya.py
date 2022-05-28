import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time
import datetime
from sklearn.decomposition import PCA
from sklearn import preprocessing
from scipy.cluster import hierarchy
from sklearn.cluster import KMeans

start_time = time.time()

# =============================================================================
# base
# =============================================================================
base = pd.read_csv("C:/Users/FerGo/OneDrive/ACIT/2021/Statistical Learning/Data Bases/Froya/Froya_10min.txt")


# =============================================================================
# decomposing time in the dataframe
# =============================================================================
base["Date"] = pd.to_datetime(base["Timestamp"]).dt.date
base["Hour"] = pd.to_datetime(base["Timestamp"]).dt.time
base["Year"] = pd.to_datetime(base["Timestamp"]).dt.year


# =============================================================================
# creating functional dataframes 
# =============================================================================
yearly = base.drop(base[base["Year"] != 2010].index)


base = base.set_index("Timestamp")
not_sula = pd.DataFrame(index = base.index)
not_sula = not_sula.join(base.iloc[:,0:31])
not_sula = not_sula.dropna()
not_sula.name = "not_sula"



sula = pd.DataFrame(index = base.index)
sula = sula.join(base.iloc[:,31:36])
sula = sula.dropna()
sula.name = sula.columns[0][:-1]
sula.name = "sula"

# =============================================================================
# Graphs used in report
# =============================================================================


# =============================================================================
# defining colors
# =============================================================================
temp = ["LightPink"]
spd = ["Moccasin"]
direc = ["CadetBlue"]
press = ["DarkSeaGreen"]
on_s = ["BurlyWood"]
of_s = ["DarkSlateGrey"]


# =============================================================================
# scatter for wind speed trough time
# =============================================================================
fig, axs = plt.subplots(6,2, sharex="col", figsize=(7,7))
fig.subplots_adjust(hspace = 1)
fig.suptitle("Speed", fontsize=18)
axs = axs.ravel()

for i in range(12):
    axs[i].scatter(base["Date"],base["WS"+str(i+1)], c=spd)
    axs[i].set_title("Position"+str(i+1))


# =============================================================================
# scatter for wind speed trough time
# =============================================================================
fig, axs = plt.subplots(3,1, sharex="col", figsize=(17,10))
fig.subplots_adjust(hspace = 0.5)
fig.suptitle("Wind", fontsize=18)
axs = axs.ravel()


axs[0].plot(yearly["WS1"])
axs[0].set_title("Speed")
axs[1].plot(yearly["WD1"])
axs[1].set_title("Direction")
axs[2].plot(yearly["AT0"])
axs[2].set_title("Temperature")


plt.plot(yearly["AT1"])

# =============================================================================
# creating a scatterplot matrix
# =============================================================================
matrix = pd.DataFrame()
matrix["Date"] = yearly["Date"]
matrix["WS1"] = yearly["WS1"]
matrix["WD1"] = yearly["WD1"]
matrix["AT1"] = yearly["AT1"]

sns.pairplot(matrix, corner=True, diag_kind= "hist", kind="kde")


# =============================================================================
# plot Speed vs Dir through a year
# =============================================================================
c_dict = yearly["Timestamp"].map(pd.Series(data=np.arange(len(yearly)), index=yearly["Timestamp"].values).to_dict())

plt.figure()
plt.scatter(yearly["WD12"], yearly["WS12"], s=0.05 , c=c_dict, cmap=plt.cm.rainbow)
#plt.colorbar()
plt.show()

print ("this took "+ str(datetime.timedelta(seconds=round((time.time() - start_time)))) + " to run")



fig, axs = plt.subplots(12,1, sharex="col", figsize=(7, 8))
fig.subplots_adjust(hspace = .5)
fig.suptitle("Wind Directions", fontsize=18)
axs = axs.ravel()

for i in range(12):
    axs[i].scatter(range(1, len(not_sula)+1), not_sula["WD"+str(i+1)], s=0.01, c=direc)
    axs[i].set_title("WD"+str(i+1))
    
plt.scatter(range(1, len(not_sula)+1), not_sula["WD1"])


fig, axs = plt.subplots(12,1, sharex="col", figsize=(7, 7))
fig.subplots_adjust(hspace = 0.5)
fig.suptitle("Wind Speeds", fontsize=18)
axs = axs.ravel()

for i in range(12):
    axs[i].scatter(range(1, len(not_sula)+1), not_sula["WS"+str(i+1)], s=0.01, c=spd)
    axs[i].set_title("WS"+str(i+1), fontsize = "small", pad = 0) 

fig, axs = plt.subplots(7,1, sharex="col", figsize=(7, 7))
fig.subplots_adjust(hspace = 0.5)
fig.suptitle("Temperatures", fontsize=18)
axs = axs.ravel()

for i in range(7):
    axs[i].scatter(range(1, len(not_sula)+1), not_sula["AT"+str(i)], s=0.01, c=temp)
    axs[i].set_title("AT"+str(i+1), fontsize = "small", pad = 0) 



# =============================================================================
#
#  PCA by variable
#
# =============================================================================


def df_build(kind):
    # creates smaller df
    df = pd.DataFrame(index = not_sula.index)
    for i in range(*kind):
        df = df.join(not_sula.iloc[:,i])
    
    df.name = df.columns[0][:-1]
    
    return df

def scaling(df):
    # scales the df
    o_indexes = [*df.index]
    o_columns = [*df.columns]
    name = df.name
    
    df = preprocessing.scale(df)
    df =  pd.DataFrame(df, index = o_indexes, columns = o_columns)
    df.name = name
    
    return df 

def pca_model(df):
    #graphs scree plot models pca, graphs pca and returns a df with PCs
    #modeling
    model = PCA()
    model.fit(df)
    model_df = model.transform(df)
    o_indexes = df.index
    o_columns = df.columns
    
    #scree plot info
    percent_var = np.round(model.explained_variance_ratio_ * 100, 2)
    labels = ["PC"+ str(x) for x in range(1,len(percent_var)+1)] 
    
    #scree plot graphing
    plt.figure()
    plt.bar(x=range(1,len(percent_var)+1), height=percent_var, tick_label = labels)
    for i, v in enumerate(percent_var):
        plt.text(i + 0.5, v + 0.5, str(v)+"%", fontsize="small")
    plt.title("{} scree plot".format(df.name))
    plt.show()
    
    #pca plot
    
    pca_df = pd.DataFrame(model_df, columns = labels, index=o_indexes)

    plt.figure()
    plt.scatter(pca_df["PC1"], pca_df["PC2"], s=0.1)
    plt.xlabel("PC1 - {0}%" . format(percent_var[0]))
    plt.ylabel("PC2 - {0}%" . format(percent_var[1]))
    plt.title("{} PCA".format(df.name))
    plt.show()
    
    # vectors
    
    comp = pd.DataFrame(model.components_, columns = labels, index = o_columns)

    return pca_df, comp

# sepatating dfs
speed = (0,12)
direction = (12,24)
temperature = (24,31)

df_speed = df_build(speed)    
df_direction = df_build(direction)
df_temperature = df_build(temperature)


# modeling dfs
pca_speed , _ = pca_model(df_speed)
pca_direction , _  = pca_model(df_direction)
pca_temperature , _ = pca_model(df_temperature)

del(speed, direction, temperature)


# =============================================================================
# Scatter plot of PCA's
# =============================================================================

pca_k = pd.DataFrame(index = pca_speed.index)
pca_k["PCD"] = pca_direction["PC1"]
pca_k["PCS"] = pca_speed["PC1"]
pca_k["PCT"] = pca_temperature["PC1"]

plt.figure()
plt.scatter(pca_k["PCD"], pca_k["PCS"], s=0.1)
plt.xlabel("PC1 Direction")
plt.ylabel("PC1 Speed")
plt.title("")
plt.show()

fig = plt.figure(figsize = (6,6))
ax = plt.axes(projection="3d")
ax.set_xlabel("WD")
ax.set_ylabel("WS")
ax.set_zlabel("AT")
ax.scatter(pca_k["PCD"], pca_k["PCS"], pca_k["PCT"], s=0.01)
plt.title("PCA")
plt.show()



# =============================================================================
# 
# clustering
# 
# =============================================================================

# =============================================================================
# k means
# =============================================================================


model = KMeans(n_clusters=6)
model.fit(pca_k)

cluster_df = pca_k.copy()
cluster_df["class"] = model.labels_ 
cluster_df["color"] = "DarkSlateGrey"
cluster_df["color"] = cluster_df["color"].where(cluster_df["class"]==1, "BurlyWood")
print("Inertia = {0:.2f}" . format(model.inertia_))


# =============================================================================
# plots
# =============================================================================

fig = plt.figure(figsize = (6,6))
ax = plt.axes(projection="3d")
ax.set_xlabel("WD")
ax.set_ylabel("WS")
ax.set_zlabel("AT")
ax.scatter(cluster_df["PCD"], cluster_df["PCS"], cluster_df["PCT"], color = cluster_df["color"], s=0.01)
plt.title("K-Means clustering")
ax.legend()
plt.show()

fig, axs = plt.subplots(3,1, sharex="col", figsize=(7,4.5))
fig.subplots_adjust(hspace = .5)
fig.suptitle("PCA Clustering", fontsize=18)
axs = axs.ravel()

for i in range(3):
    axs[i].scatter(range(1, len(cluster_df)+1), cluster_df.iloc[:,i], c = cluster_df["color"], s=0.01)
    axs[i].set_title(cluster_df.columns[i])


on_shore = not_sula.copy()
on_shore["class"] = cluster_df["class"]
on_shore = on_shore.drop(on_shore[on_shore["class"] != 0].index)

off_shore = not_sula.copy()
off_shore["class"] = cluster_df["class"]
off_shore = off_shore.drop(off_shore[off_shore["class"] != 1].index)


plt.figure()
plt.hist(on_shore["WD1"],130, histtype="step", color = on_s, label = "On-Shore")
plt.hist(off_shore["WD1"],130, histtype="step", color = of_s, label = "Off-Shore")
plt.title("Wind Direction")
plt.legend()
plt.show()

plt.figure()
plt.hist(on_shore["WS1"],40, histtype="step", color = on_s, label = "On-Shore")
plt.hist(off_shore["WS1"],40, histtype="step", color = of_s, label = "Off-Shore")
plt.title("Wind Speed")
plt.legend()
plt.show()


plt.figure()
plt.hist(on_shore["AT1"],30, histtype="step", color = on_s, label = "On-Shore")
plt.hist(off_shore["AT1"],30, histtype="step", color = of_s, label = "Off-Shore")
plt.title("Temperature")
plt.legend()
plt.show()

# =============================================================================
# comparison table
# =============================================================================

comparisons = pd.DataFrame()

comparisons["On_Shore_Mean"] = on_shore.mean()
comparisons["Off_Shore_Mean"] = off_shore.mean()
comparisons["Difference_Mean"] = comparisons["Off_Shore_Mean"] - comparisons["On_Shore_Mean"]

comparisons["On_Shore_Var"] = on_shore.var()
comparisons["Off_Shore_Var"] = off_shore.var()
comparisons["Difference_Var"] = comparisons["Off_Shore_Var"] - comparisons["On_Shore_Var"]

comparisons["On_Shore_Skew"] = on_shore.skew()
comparisons["Off_Shore_Skew"] = off_shore.skew()
comparisons["Difference_Skew"] = comparisons["Off_Shore_Skew"] - comparisons["On_Shore_Skew"]

comparisons["On_Shore_Kurt"] = on_shore.kurtosis()
comparisons["Off_Shore_Kurt"] = off_shore.kurtosis()
comparisons["Difference_Kurt"] = comparisons["Off_Shore_Kurt"] - comparisons["On_Shore_Kurt"]

# comparisons.to_csv("C:/Users/FerGo/OneDrive/ACIT/2021/Statistical Learning/Data Bases/Comparisons.csv")


# =============================================================================
# hierarchical
# =============================================================================
"""

#comp = hierarchy.linkage(pca_k, "single")
#results_linkage = pd.DataFrame(comp, columns = ["P1", "P2", "Dist", "Lvl"])
#results_linkage.to_csv("C:/Users/FerGo/OneDrive/ACIT/2021/Statistical Learning/Data Bases/Single.csv")

#hierarchy.dendrogram(comp)

average_clus = pd.read_csv("C:/Users/FerGo/OneDrive/ACIT/2021/Statistical Learning/Data Bases/Average.csv")
average_clus = average_clus.drop(columns = {"Unnamed: 0"})

hierarchy.dendrogram(average_clus)

complete_clus = pd.read_csv("C:/Users/FerGo/OneDrive/ACIT/2021/Statistical Learning/Data Bases/Complete.csv")
complete_clus = complete_clus.drop(columns = {"Unnamed: 0"})

hierarchy.dendrogram(complete_clus)

single_clus = pd.read_csv("C:/Users/FerGo/OneDrive/ACIT/2021/Statistical Learning/Data Bases/Single.csv")
single_clus = single_clus.drop(columns = {"Unnamed: 0"})

hierarchy.dendrogram(single_clus)


"""
average_clus = pd.read_csv("C:/Users/FerGo/OneDrive/ACIT/2021/Statistical Learning/Data Bases/Average.csv")
average_clus = average_clus.drop(columns = {"Unnamed: 0"})

clus = hierarchy.fcluster(average_clus, t= 170, criterion = "distance")


hierarchical_df = pca_k.copy()
hierarchical_df["cluster"] = clus

# =============================================================================
# plots
# =============================================================================

fig = plt.figure(figsize = (6,6))
ax = plt.axes(projection="3d")
ax.set_xlabel("WD")
ax.set_ylabel("WS")
ax.set_zlabel("AT")
ax.scatter(hierarchical_df["PCD"], hierarchical_df["PCS"], hierarchical_df["PCT"], c = hierarchical_df["cluster"], cmap= "tab20", s=0.01)
plt.title("Hierarchical clustering")
ax.legend()
plt.show()

fig, axs = plt.subplots(3,1, sharex="col", figsize=(7,4.5))
fig.subplots_adjust(hspace = .5)
fig.suptitle("PCA Clustering", fontsize=18)
axs = axs.ravel()

for i in range(3):
    axs[i].scatter(range(1, len(hierarchical_df)+1), hierarchical_df.iloc[:,i], c = hierarchical_df["cluster"], cmap= "tab20",  s=0.01)
    axs[i].set_title(cluster_df.columns[i])


def df_build_3(cluster):
    # creates smaller df
    df = not_sula.copy()
    df["cluster"] = clus
    df = df.drop(df[df["cluster"] != cluster].index)
    df = df.drop(columns = {"cluster"})
    return df

for i in range(1,max(clus)+1):
    globals()["cluster_{}".format(i)] = df_build_3(i)

plt.figure()
for i in range(1,max(clus)+1):
    plt.hist( globals()["cluster_{}".format(i)]["WD1"],130, histtype="step" ,label = "cluster {}".format(i))
plt.title("Wind Direction")
plt.legend()
plt.show()


plt.figure()
for i in range(1,max(clus)+1):
    plt.hist( globals()["cluster_{}".format(i)]["WS1"],40, histtype="step" ,label = "cluster {}".format(i))
plt.title("Wind Speed")
plt.legend()
plt.show()

plt.figure()
for i in range(1,max(clus)+1):
    plt.hist( globals()["cluster_{}".format(i)]["AT1"],30, histtype="step" ,label = "cluster {}".format(i))
plt.title("Temperature")
plt.legend()
plt.show()

hc_means = pd.DataFrame()
for i in range(1, max(clus)+1):
    hc_means["cluster_{}".format(i)] = globals()["cluster_{}".format(i)].mean()
hc_means.to_csv("C:/Users/FerGo/OneDrive/ACIT/2021/Statistical Learning/Data Bases/Froya/Stats/means.csv")

hc_variation = pd.DataFrame()
for i in range(1, max(clus)+1):
    hc_variation["cluster_{}".format(i)] = globals()["cluster_{}".format(i)].var()
hc_variation.to_csv("C:/Users/FerGo/OneDrive/ACIT/2021/Statistical Learning/Data Bases/Froya/Stats/variation.csv")
    
hc_skewness = pd.DataFrame()
for i in range(1, max(clus)+1):
    hc_skewness["cluster_{}".format(i)] = globals()["cluster_{}".format(i)].skew()
hc_skewness.to_csv("C:/Users/FerGo/OneDrive/ACIT/2021/Statistical Learning/Data Bases/Froya/Stats/skewness.csv")

print ("this took {} to run".format((datetime.timedelta(seconds=round((time.time() - start_time))))))




# =============================================================================
#
#  PCA by location
#
# =============================================================================

# =============================================================================
# creating df for each 
# =============================================================================
places = []
j = 25
for i in range (0, 12):
    if i % 2 != 0:    
        places.append( ( (i), (i+12), (j) ) )
        j += 1
    else:
        places.append( ( (i), (i+12), (j) ) )    

def df_build_2(place):
    # creates smaller df
    df = pd.DataFrame(index = not_sula.index)
    for i in place:
        df = df.join(not_sula.iloc[:,i])
    
    if len(df.columns[0]) == 4:
        df.name = "Location " + str(df.columns[0][-2:])
    else:
        df.name = "Location " + str(df.columns[0][-1])
    return df

# =============================================================================
# modeling not sula
# =============================================================================

for i in range(0,12):
    globals()["df_place_{}".format(i+1)] = df_build_2(places[i])

for i in range(0,12):
    globals()["df_place_{}".format(i+1)] = scaling(globals()["df_place_{}".format(i+1)])

for i in range(0,12):
    _ , globals()["comp_place_{}".format(i+1)] = pca_model(globals()["df_place_{}".format(i+1)])

for i in range(0,12):
    del (globals()["df_place_{}".format(i+1)])
    
del(places)


# =============================================================================
# Modeling sula
# =============================================================================

sula_pca = pd.DataFrame(index = sula.index)
sula_pca["WSS"] = sula["WSSula"]
sula_pca["WDS"] = sula["WDSula"]
sula_pca["ATS"] = sula["TempSula"]
sula_pca.name = "Sula"
sula_pca = scaling(sula_pca)
sula_pca , comp_sula = pca_model(sula_pca)

# =============================================================================
# Plotting
# =============================================================================

"""
# all together
orig = [0, 0, 0]

color = iter(plt.cm.rainbow(np.linspace(0,1,12)))
fig = plt.figure(figsize = (15,10))
ax = plt.axes( projection = "3d")
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
for i in range(0, 12):
    c = next(color)
    for j in range(0, len(globals()["comp_place_{}".format(i+1)].index)):
        ax.quiver(orig[0], orig[1], orig[2],
                  globals()["comp_place_{}".format(i+1)].iloc[0, j],
                  globals()["comp_place_{}".format(i+1)].iloc[1, j], 
                  globals()["comp_place_{}".format(i+1)].iloc[2, j],
                  color = c)
        ax.text(globals()["comp_place_{}".format(i+1)].iloc[0, j],
                globals()["comp_place_{}".format(i+1)].iloc[1, j], 
                globals()["comp_place_{}".format(i+1)].iloc[2, j],
                "PCA {},{}".format(i+1, j+1))
plt.show()    

"""

#grouping pca's
orig = [0, 0, 0]

for i in range(0, len(globals()["comp_place_{}".format(i+1)].index)):
    color = iter(plt.cm.rainbow(np.linspace(0,1,12)))
    fig = plt.figure(figsize = (6,6))
    ax = plt.axes( projection = "3d")
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel(globals()["comp_place_{}".format(i+1)].index[0][:2])
    ax.set_ylabel(globals()["comp_place_{}".format(i+1)].index[1][:2])
    ax.set_zlabel(globals()["comp_place_{}".format(i+1)].index[2][:2])
    plt.title("PCA {}".format(i+1))
    for j in range(0, 12):
        c = next(color)
        ax.quiver(orig[0], orig[1], orig[2],
              globals()["comp_place_{}".format(j+1)].iloc[0, i],
              globals()["comp_place_{}".format(j+1)].iloc[1, i],
              globals()["comp_place_{}".format(j+1)].iloc[2, i],
              color = c)
        ax.text(globals()["comp_place_{}".format(j+1)].iloc[0, i],
                globals()["comp_place_{}".format(j+1)].iloc[1, i],
                globals()["comp_place_{}".format(j+1)].iloc[2, i],
                "PCA {},{}".format(i+1, j+1),
                color = c)
plt.show()  


#each location per plot
orig = [0, 0, 0]

color = iter(plt.cm.rainbow(np.linspace(0,1,13)))
for i in range(0, 12):
    fig = plt.figure(figsize = (6,6))
    ax = plt.axes( projection = "3d")
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel(globals()["comp_place_{}".format(i+1)].index[0][:2])
    ax.set_ylabel(globals()["comp_place_{}".format(i+1)].index[1][:2])
    ax.set_zlabel(globals()["comp_place_{}".format(i+1)].index[2][:2])
    plt.title("Location {}".format(i+1))
    c = next(color)
    for j in range(0, len(globals()["comp_place_{}".format(i+1)].index)):
        ax.quiver(orig[0], orig[1], orig[2],
                  globals()["comp_place_{}".format(i+1)].iloc[0, j],
                  globals()["comp_place_{}".format(i+1)].iloc[1, j], 
                  globals()["comp_place_{}".format(i+1)].iloc[2, j],
                  color = c)
        ax.text(globals()["comp_place_{}".format(i+1)].iloc[0, j],
                globals()["comp_place_{}".format(i+1)].iloc[1, j], 
                globals()["comp_place_{}".format(i+1)].iloc[2, j],
                "PCA {},{}".format(i+1, j+1))
plt.show()  


fig = plt.figure(figsize = (6,6))
ax = plt.axes( projection = "3d")
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_xlabel(comp_sula.index[0][:2])
ax.set_ylabel(comp_sula.index[1][:2])
ax.set_zlabel(comp_sula.index[2][:2])
plt.title("Sula")
c = next(color)
for i in range(0,3):
    ax.quiver(orig[0], orig[1], orig[2],
              comp_sula.iloc[0,i],
              comp_sula.iloc[1,i],
              comp_sula.iloc[2,i],
              color = c)
    ax.text(comp_sula.iloc[0,i],
            comp_sula.iloc[1,i],
            comp_sula.iloc[2,i],
            "PCA {}". format(i+1))

print ("this took "+ str(datetime.timedelta(seconds=round((time.time() - start_time)))) + " to run")

# =============================================================================
# plotting Sula
# =============================================================================

sula_pca["Hum"] = sula["RelHumSula"] 
sula_pca = sula_pca.dropna()
"""
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

y = sula_pca["Hum"]
X = sula_pca.drop(columns = {"Hum"})

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

parameters = {"n_estimators": range(1, 600),
              "max_depth": range(1,10)}

model = GridSearchCV(ensemble.GradientBoostingRegressor(), parameters)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("mean squared error: {}".format(metrics.mean_squared_error(y_test, y_pred)))
print("R^2: {}".format(metrics.r2_score(y_test, y_pred)))
"""
print ("this took "+ str(datetime.timedelta(seconds=round((time.time() - start_time)))) + " to run")


# =============================================================================
# 
# =============================================================================

model = KMeans(n_clusters=6)
model.fit(pca_k)

recluster_df = pca_k.copy()
recluster_df["class"] = model.labels_ 

print("Inertia = {0:.2f}" . format(model.inertia_))


# =============================================================================
# plots
# =============================================================================

fig = plt.figure(figsize = (6,6))
ax = plt.axes(projection="3d")
ax.set_xlabel("WD")
ax.set_ylabel("WS")
ax.set_zlabel("AT")
ax.scatter(recluster_df["PCD"], recluster_df["PCS"], recluster_df["PCT"], c = recluster_df["class"], s=0.01)
plt.title("K-Means clustering with 6 centroids")
ax.legend()
plt.show()

comparison = pd.DataFrame()
comparison["kmeans"] = recluster_df["class"]
comparison["hier"] = clus



