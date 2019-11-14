import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
'exec(%matplotlib inline)'
from sklearn.cluster import KMeans

#datos
'''df = pd.DataFrame({
    'x': [18,28,57,75,39,89,16,56,43,65,47,28,54,83,68,56,4,8,14,38,4,59,84,73,7,80,52,31,42,25],
    'y': [40,23,3,50,84,86,70,30,46,1,48,68,83,16,86,9,39,79,66,73,1,21,35,63,69,25,1,8,35,81]
})'''

'''df = pd.DataFrame({
    'x': [1.08,2.18,0.7,1.4,0.85,0.6,1.54],
    'y': [2.45,1.45,2.47,2.09,2.41,2.38,1.88]
})'''


df = pd.DataFrame({
    'x': [1,2,4,7,5],
    'y': [1,1,5,7,7]
})


#sklearn
kmeans = KMeans(n_clusters=2)
kmeans.fit(df)
labels = kmeans.predict(df)
centroids = kmeans.cluster_centers_

print(centroids)

#tamano inicial de la ventana
fig = plt.figure(figsize=(5, 5))


#colores
colmap = {1: 'r', 2: 'g', 3: 'b'}
#colorear los puntos dadas las agrupaciones hechas por sklearn
colors = map(lambda x: colmap[x+1], labels)


#dibujar los puntos
plt.scatter(df['x'], df['y'], color= [x for x in colors ],alpha=0.3)
#dibujar los centroides finales
for idx, centroid in enumerate(centroids):
    plt.scatter(*centroid, color=colmap[idx+1])


#dimensiones y mostrar
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.show()
