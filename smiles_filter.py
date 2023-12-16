
#this code is written with the help of chatgpt, specially the functions, thanks

from rdkit import Chem
import pandas as pd

input_file = 'reaxys_diels_alder_molecular.csv'
output_file = 'filtered_reaxys_diels_alder_molecular.csv'


def split_smiles(smiles_string):
    substrings = smiles_string.split('>>')
    return [atom for substring in substrings for atom in substring.split('.')]


def is_valid_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

# Function to check if SMILES contain only organic elements
def is_organic_only(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        organic_atoms = set([1, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53])  # Atomic numbers for organic elements
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() not in organic_atoms:
                return False  # Contains non-organic element
        return True  # Only organic elements
    return None  # Invalid SMILES

def generate_smiles_without_isotopes(smiles_with_isotopes):
    mol = Chem.MolFromSmiles(smiles_with_isotopes)

    # Check if the molecule is valid
    if mol is not None:
        # Set the isotope of each atom to 0 to exclude isotope information
        for atom in mol.GetAtoms():
            atom.SetIsotope(0)

        # Convert the modified molecule back to SMILES without isotope information
        smiles_without_isotopes = Chem.MolToSmiles(mol, isomericSmiles=True)

        return Chem.MolToSmiles(Chem.MolFromSmiles(smiles_without_isotopes))
    else:
        return None

df = pd.read_csv(input_file)

result_list = [split_smiles(smiles) for smiles in df['smiles']] #splitting the smiles string to get one molecule per smiles-string
flattened_list = [atom for sublist in result_list for atom in sublist] #just flattening
valid_smiles = [smile for smile in flattened_list if is_valid_smiles(smile)] #getting the rdkit valid mols only
smiles_organic_only = [smiles for smiles in valid_smiles if is_organic_only(smiles)] #only organic molecules
smiles_with_non_organic = [smiles for smiles in valid_smiles if not is_organic_only(smiles)] #incase you want metal containing system
smiles_list = [generate_smiles_without_isotopes(i) for i in smiles_organic_only] #normalizing the isotopes e.g. [2H] to H only
can_org_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(i)) for i in smiles_list] #canonicalizing the molecules
unique_can_org_smiles = list(set(can_org_smiles)) #removing the duplicates
sorted_smiles = sorted(unique_can_org_smiles, key=len) #sorting based on the smiles length
filtered_smiles = [smiles for smiles in sorted_smiles if len(smiles) >= 10] #removing very short sequences/smiles

output_df = pd.DataFrame(filtered_smiles, columns=['smiles'])
output_df.to_csv(output_file, index=False)



