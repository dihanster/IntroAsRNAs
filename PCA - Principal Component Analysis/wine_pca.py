import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_wine

wine = load_wine()
wine.data.shape

pca = PCA(2)  
projected = pca.fit_transform(wine.data)
print(wine.data.shape)
print(projected.shape)

#Mostra as componentes principais
print(pca.components_)

print(pca.explained_variance_)

#Plota com duas componentes
plt.scatter(projected[:, 0], projected[:, 1],
            c=wine.target, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('nipy_spectral', 3))
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.colorbar()
plt.show()

pca = PCA().fit(wine.data)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Numero de Componentes')
plt.ylabel('Variacao Explicada');
plt.show()
