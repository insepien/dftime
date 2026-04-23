from src import *
reso = 0.1


def selection_function(AQ):
    """selection function for hdpgmm"""
    sep_kpc = AQ[:,0]
    q = AQ[:,1]
    return psel(sep_kpc,0.2,q,reso)