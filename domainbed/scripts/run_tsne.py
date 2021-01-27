from MulticoreTSNE import MulticoreTSNE as TSNE
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    X = []
    labels = []
    for i in range(4):
        x = np.load(f'/scratch/wz727/chestXR/DomainBed/tsne_pretrained/embeddings {i}.npy')
        x = np.concatenate(x, 0)
        print(x.shape)
        X.append(x)
        ls = np.load(f'/scratch/wz727/chestXR/DomainBed/tsne_pretrained/labels {i}.npy')
        labels.extend(ls.flatten())
        # labels.extend([i] * x.shape[0])
    # labels = [np.ones_like(x) * i for i, x in enumerate(X)]
    X = np.concatenate(X, 0)
    # labels = np.concatenate(labels, 0)
    print(X.shape)
    X_embedded = TSNE(n_jobs=16, n_components=2).fit_transform(X)
    plt.scatter(X_embedded[:,0], X_embedded[:,1], c=labels, cmap='viridis')
    plt.savefig('tsne.png')
