



import os
import matplotlib.pyplot as plt
from rdkit import Chem
import random
from rdkit import DataStructs
import matplotlib.pyplot as plt
from rdkit.Chem.Draw import IPythonConsole

from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Avalon import pyAvalonTools
from sklearn.feature_selection import VarianceThreshold
from tqdm import tqdm
import time
import argparse
import pandas as pd
import numpy as np

#all the fps

FPS_DICT = {'fp1': lambda x: np.array(list(AllChem.RDKFingerprint(x, fpSize=3096, minPath=1, maxPath=2, branchedPaths=True))),
            'fp2': lambda x: np.array(list(AllChem.RDKFingerprint(x, fpSize=3096, minPath=1, maxPath=4, branchedPaths=True))),
            'fp3': lambda x: np.array(list(AllChem.RDKFingerprint(x, fpSize=3096, minPath=1, maxPath=6, branchedPaths=True))),
            'fp4': lambda x: np.array(list(AllChem.RDKFingerprint(x, fpSize=3096, minPath=1, maxPath=8, branchedPaths=True))),
            'fp5': lambda x: np.array(list(AllChem.RDKFingerprint(x, fpSize=3096, minPath=1, maxPath=2, branchedPaths=False))),
            'fp6': lambda x: np.array(list(AllChem.RDKFingerprint(x, fpSize=3096, minPath=1, maxPath=4, branchedPaths=False))),
            'fp7': lambda x: np.array(list(AllChem.RDKFingerprint(x, fpSize=3096, minPath=1, maxPath=6, branchedPaths=False))),
            'fp8': lambda x: np.array(list(AllChem.RDKFingerprint(x, fpSize=3096, minPath=1, maxPath=8, branchedPaths=False))),
            'fp9': lambda x: np.array(list(AllChem.GetHashedAtomPairFingerprintAsBitVect(x, nBits=3096, minLength=1, maxLength=30))),
            'fp10': lambda x: np.array(list(rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(x, nBits=3096, targetSize=4))),
            'fp11': lambda x: np.array(list(AllChem.GetMorganFingerprintAsBitVect(x,nBits=3096, radius=0, useFeatures=False))),
            'fp12': lambda x: np.array(list(AllChem.GetMorganFingerprintAsBitVect(x,nBits=3096, radius=2, useFeatures=False))),
            'fp13': lambda x: np.array(list(AllChem.GetMorganFingerprintAsBitVect(x,nBits=3096, radius=4, useFeatures=False))),
            'fp14': lambda x: np.array(list(AllChem.GetMorganFingerprintAsBitVect(x,nBits=3096, radius=6, useFeatures=False))),
            'fp15': lambda x: np.array(list(AllChem.GetMorganFingerprintAsBitVect(x,nBits=3096, radius=0, useFeatures=True))),
            'fp16': lambda x: np.array(list(AllChem.GetMorganFingerprintAsBitVect(x,nBits=3096, radius=2, useFeatures=True))),
            'fp17': lambda x: np.array(list(AllChem.GetMorganFingerprintAsBitVect(x,nBits=3096, radius=4, useFeatures=True))),
            'fp18': lambda x: np.array(list(AllChem.GetMorganFingerprintAsBitVect(x,nBits=3096, radius=6, useFeatures=True))),
            'fp19': lambda x: np.array(list(pyAvalonTools.GetAvalonFP(x, nBits=3096))),
            'fp20': lambda x: np.array(list(Chem.LayeredFingerprint(x, fpSize=3096, minPath=1, maxPath=2, branchedPaths=True))),
            'fp21': lambda x: np.array(list(Chem.LayeredFingerprint(x, fpSize=3096, minPath=1, maxPath=4, branchedPaths=True))),
            'fp22': lambda x: np.array(list(Chem.LayeredFingerprint(x, fpSize=3096, minPath=1, maxPath=6, branchedPaths=True))),
            'fp23': lambda x: np.array(list(Chem.LayeredFingerprint(x, fpSize=3096, minPath=1, maxPath=8, branchedPaths=True))),
           }

def smiles_to_fp(smiles, fp_combination):
    mol = Chem.MolFromSmiles(smiles)
    fps_list = [FPS_DICT[i](mol) for i in fp_combination]
    return np.array(fps_list, dtype=np.float32).reshape(-1)

def featurizer(csv_path, fp_combination):
    df=pd.read_csv(csv_path)
    df['smiles'] = df['Catalyst'].str.cat(df[['Imine', 'Thiol']], sep='.')
    smiles_list=list(df['smiles'])
    X_ = [smiles_to_fp(i, fp_combination) for i in smiles_list]
    X = np.array(X_)
    y = np.array(df['Output'])

    return X, y



