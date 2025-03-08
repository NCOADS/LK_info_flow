from tqdm.auto import tqdm
import matplotlib

CLUSTER = False

def set_cluster(is_cluster):
    global CLUSTER 
    CLUSTER = is_cluster
    if CLUSTER:
        matplotlib.use('Agg')

def is_cluster():
    return CLUSTER

def track(iterator, **kwargs):
    if not CLUSTER:
        return tqdm(iterator, **kwargs)
    else:
        return iterator
