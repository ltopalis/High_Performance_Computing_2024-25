from multiprocessing import Pool
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time

def f(data):
    l1 = data['mlp_layer1']
    l2 = data['mlp_layer2']
    l3 = data['mlp_layer3']
    
    m = MLPClassifier(hidden_layer_sizes=(l1, l2, l3), random_state=35)
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)
    ac = accuracy_score(y_pred, y_test)
    
    return (data, ac)

X, y = make_classification(n_samples=10000, random_state=42, n_features=2, n_informative=2, n_redundant=0, class_sep=0.8)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

params = [{'mlp_layer1': [16, 32],
           'mlp_layer2': [16, 32],
           'mlp_layer3': [16, 32]}]

pg = ParameterGrid(params)

pool = Pool(len(pg))

start = time.time()
results = pool.map(f, pg)
end = time.time()
for r in results:
    print(r)
print(f'Execution time: {end-start} ms')
    
print()

start = time.time()  
results = pool.map_async(f, pg)
while not results.ready():
    time.sleep(1)
end = time.time()

for r in results.get():
    print(r)

print(f'Execution time: {end-start} ms')
