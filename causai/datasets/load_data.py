"""
Load Datasets 

"""
from causai.datasets.syntheticdataset import SyntheticDataset
from causai.util import download_url, read_list_edges
import pandas as pd 
list_of_datasets = ['lungcancer','sanchs', 'bloodpressure', 'lbiddd']

def load_datasets(name:str,**kwargs):
   if name == 'bloodpressure':
       return SyntheticDataset().generate_data(**kwargs)
   if name =='sanchs':
       data  = pd.read_csv( "cyto_full_data.csv")
       graph = read_list_edges( "cyto_full_target.csv") 
       return data, graph

def load_sachs():
    download_url("https://raw.githubusercontent.com/FenTechSolutions/CausalDiscoveryToolbox/ff403b654cec8a3cddd6c2ce6a84a8e46255c825/cdt/data/resources/cyto_full_data.csv")
    download_url("https://raw.githubusercontent.com/FenTechSolutions/CausalDiscoveryToolbox/ff403b654cec8a3cddd6c2ce6a84a8e46255c825/cdt/data/resources/cyto_full_target.csv")
