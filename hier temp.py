import pandas as pd
from scipy.cluster import hierarchy

#comp = hierarchy.linkage(pca_k, "average")
#results_linkage = pd.DataFrame(comp, columns = ["P1", "P2", "Dist", "Lvl"])
#results_linkage.to_csv("C:/Users/FerGo/OneDrive/ACIT/2021/Statistical Learning/Data Bases/Average.csv")

#hierarchy.dendrogram(comp)

average_clus = pd.read_csv("C:/Users/FerGo/OneDrive/ACIT/2021/Statistical Learning/Data Bases/Average.csv")
average_clus = average_clus.drop(columns = {"Unnamed: 0"})

hierarchy.dendrogram(average_clus)

complete_clus = pd.read_csv("C:/Users/FerGo/OneDrive/ACIT/2021/Statistical Learning/Data Bases/Complete.csv")
complete_clus = complete_clus.drop(columns = {"Unnamed: 0"})

hierarchy.dendrogram(complete_clus)

single_clus = pd.read_csv("C:/Users/FerGo/OneDrive/ACIT/2021/Statistical Learning/Data Bases/Froya/Single.csv")
single_clus = single_clus.drop(columns = {"Unnamed: 0"})

hierarchy.dendrogram(single_clus)


clus = hierarchy.fcluster(average_clus, t= 80, criterion = "distance")
clus.max()
