from src import *
reso = 0.1
z=0.2

def selection_function(AQ):
    """selection function for hdpgmm"""
    sep_kpc = AQ[:,0]
    q = AQ[:,1]
    return psel(sep_kpc,z,q,reso)