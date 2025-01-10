from mpi4py import MPI

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0: # master
    

    X, y = make_classification(n_samples=10000, random_state=42, n_features=2, n_informative=2, n_redundant=0, class_sep=0.8)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    params = [{'mlp_layer1': [16, 32],
            'mlp_layer2': [16, 32],
            'mlp_layer3': [16, 32]}]

    pg = ParameterGrid(params)
    
    for i, p in enumerate(list(pg)[1:], start=1):
        snd = {'p': p, "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}
        comm.send(snd, dest=i)
    
    l1 = pg[0]['mlp_layer1']
    l2 = pg[0]['mlp_layer2']
    l3 = pg[0]['mlp_layer3']
    m = MLPClassifier(hidden_layer_sizes=(l1, l2, l3), random_state=42)
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)
    ac = accuracy_score(y_pred, y_test)
    
    results = [None for _ in range(size)]
    
    results[0] = (0, pg[0], ac)
    
    for proc in range(1, size):
        d = comm.recv(source=proc)
        
        results[d[0]] = d
    
    for r in results:
        print(r)
    
else:
    data = comm.recv(source=0)

    l1 = data['p']['mlp_layer1']
    l2 = data['p']['mlp_layer2']
    l3 = data['p']['mlp_layer3']
    m = MLPClassifier(hidden_layer_sizes=(l1, l2, l3), random_state=42)
    m.fit(data['X_train'], data['y_train'])
    y_pred = m.predict(data['X_test'])
    ac = accuracy_score(y_pred, data['y_test'])

    comm.send((rank, data['p'], ac), dest=0)