from multiprocessing import Pool
from mpi4py.futures import MPIPoolExecutor
from mpi4py import MPI

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time

def f(data):
    r = []
    for d in data:
        
        l1 = d['mlp_layer1']
        l2 = d['mlp_layer2']
        l3 = d['mlp_layer3']
        
        m = MLPClassifier(hidden_layer_sizes=(l1, l2, l3), random_state=35)
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        ac = accuracy_score(y_pred, y_test)
        
        r.append((d, ac))    
    return r

X, y = make_classification(n_samples=10000, random_state=42, n_features=2, n_informative=2, n_redundant=0, class_sep=0.8)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

params = [{'mlp_layer1': [16, 32],
           'mlp_layer2': [16, 32],
           'mlp_layer3': [16, 32]}]

if __name__ == "__main__":
    
    com = MPI.COMM_WORLD
    size = com.Get_size()
    rank = com.Get_rank()
    
    pg = list(ParameterGrid(params))
    chunk_size = len(pg) // size
    if chunk_size == 0: chunk_size = 1    
    
    sublists = [[] for _ in range(size)]
    
    j = 0
    for i in range(size):
        sublists[i].append( pg[j : j + chunk_size])
        
        j += chunk_size
        
    sublists[i].append(pg[j : ])
    
    com.Barrier()

    start = time.time()
    with MPIPoolExecutor(max_workers=size) as executor:
        results = executor.map(f, sublists[rank])
    
    com.Barrier()
    end = time.time()
        
    acc = [item for sublist in results for item in sublist]
    
    for s in acc:
        print(s)
    
    print(f'Execution time: {end-start} ms')

