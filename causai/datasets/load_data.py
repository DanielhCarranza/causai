"""
Load Datasets 

"""
from causai.datasets.syntheticdataset import SyntheticDataset
from causai.util import download_url

list_of_datasets = ['lungcancer', 'bloodpressure', 'lbiddd']

def load_datasets(name:str,**kwargs):
   if name == 'bloodpressure':
       return SyntheticDataset().generate_data(**kwargs)  