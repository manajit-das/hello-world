
import argparse
from rdkit import Chem
import pandas as pd

def valid_smiles(text_file):
    with open(text_file) as file:
        content=file.readlines()
        smiles=[lines.rstrip() for lines in content]
    print('Total number of molecules generated:', len(smiles))
    mols=[Chem.MolFromSmiles(smile) for smile in smiles]
    valid=[]
    for i in mols:
        if i != None:
            valid.append(Chem.MolToSmiles(i))
    print('Number of valid molecules:', len(valid))
    return valid
        
if __name__ == "__main__":
    parser=argparse.ArgumentParser(description='given a text file of smiles, returns only the valid smiles')
    parser.add_argument('text_file', help='You need to give a text file containing smiles')
    parser.add_argument('output_file_name', help='Name of the output file you want')
    args=parser.parse_args()

    filtered_mols=valid_smiles(args.text_file)
    df=pd.DataFrame(filtered_mols, columns=['smiles'])
    df.to_csv(args.output_file_name)




