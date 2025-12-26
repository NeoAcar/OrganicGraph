import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import re

# =========================================================================================
# GRAPH DATASET
# =========================================================================================

class MeltingPointDataset(Dataset):
    def __init__(self, csv_file, tokenizer=None, mode='graph'):
        """
        Args:
            csv_file (str): Path to the csv file with 'SMILES' and 'Tm' columns.
            tokenizer (SmilesTokenizer, optional): Tokenizer for sequence mode.
            mode (str): One of 'graph', 'sequence', 'fingerprint'.
        """
        self.data = pd.read_csv(csv_file)
        self.mode = mode
        self.tokenizer = tokenizer
        self.smiles = self.data['SMILES'].tolist()
        self.targets = self.data['Tm'].tolist()
        
        # Precompute fingerprints if in fingerprint mode to save time during training
        if self.mode == 'fingerprint':
            self.fingerprints = [self._get_fingerprint(s) for s in tqdm(self.smiles, desc="Generating Fingerprints")]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smile = self.smiles[idx]
        target = float(self.targets[idx])
        target = torch.tensor(target, dtype=torch.float)

        if self.mode == 'graph':
            return self._get_graph(smile, target)
        elif self.mode == 'sequence':
            return self._get_sequence(smile, target)
        elif self.mode == 'fingerprint':
            return torch.tensor(self.fingerprints[idx], dtype=torch.float), target
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _get_fingerprint(self, smile):
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            print(f"Invalid SMILES: {smile}")
            # Handle invalid SMILES by returning distinct zero vector or handling error
            # For simplicity, returning zeros (though in practice should probably filter these out)
            return np.zeros((2048,), dtype=np.float32)
        
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        arr = np.zeros((0,), dtype=np.int8)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    def _get_graph(self, smile, target):
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            print(f"Invalid SMILES: {smile}")
            return Data(x=torch.zeros((1, 1), dtype=torch.float), edge_index=torch.zeros((2, 0), dtype=torch.long), y=target)

        # --- ÖNEMLİ EKLEME ---
        # Erime noktası polariteye bağlıdır, kısmi yükleri hesaplatıyoruz.
        AllChem.ComputeGasteigerCharges(mol)
        
        # Not: Eğer Hidrojenleri ayrı birer node (yuvarlak) olarak görmek istersen:
        # mol = Chem.AddHs(mol) 
        # satırını buraya ekleyebilirsin. Ama genelde implicit (aşağıdaki gibi) yeterlidir.

        # Atom Features
        atom_features = []
        for atom in mol.GetAtoms():
            feats = [
                # 1. Temel Kimlik
                atom.GetAtomicNum(),
                
                # 2. Yapısal Bilgiler
                atom.GetTotalDegree(),
                int(atom.GetIsAromatic()),
                int(atom.GetHybridization()),
                int(atom.IsInRing()), # Atom halkada mı? (Önemli)
                
                # 3. ERİME NOKTASI İÇİN KRİTİK FİZİKSEL ÖZELLİKLER
                # Hidrojen Bağ Potansiyeli (Data'da H yok demiştin, işte buraya ekliyoruz)
                atom.GetTotalNumHs(), 
                
                # Atom Kütlesi (Van der Waals kuvvetleri için - 0.01 ile normalize)
                atom.GetMass() * 0.01,
                
                # Yük Bilgisi (Polarite)
                atom.GetFormalCharge(),
                # Gasteiger Partial Charge (Hesaplanamazsa 0.0 ver)
                float(atom.GetProp('_GasteigerCharge')) if atom.HasProp('_GasteigerCharge') else 0.0
            ]
            atom_features.append(feats)
        
        x = torch.tensor(atom_features, dtype=torch.float)

        # Edge Features & Connectivity
        row, col = [], []
        edge_features = []
        
        bond_type_map = {
            Chem.rdchem.BondType.SINGLE: 1,
            Chem.rdchem.BondType.DOUBLE: 2,
            Chem.rdchem.BondType.TRIPLE: 3,
            Chem.rdchem.BondType.AROMATIC: 4
        }

        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            
            # Add bidirectional edges (Graf yönsüz olduğu için gidiş-dönüş ekliyoruz)
            row += [start, end]
            col += [end, start]
            
            b_type = bond_type_map.get(bond.GetBondType(), 0)
            b_ring = int(bond.IsInRing())
            
            # Ekstra: Bağın konjuge olup olmadığı (Erime noktası için kararlılık göstergesi)
            b_conj = int(bond.GetIsConjugated()) 

            feat = [b_type, b_ring, b_conj]
            edge_features += [feat, feat] 

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(edge_features, dtype=torch.float)

        # PyG Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=target)
        return data

    def _get_sequence(self, smile, target):
        if self.tokenizer is None:
            print("Tokenizer must be provided for sequence mode")
            raise ValueError("Tokenizer must be provided for sequence mode")
        
        token_ids = self.tokenizer.encode(smile)
        return torch.tensor(token_ids, dtype=torch.long), target

# =========================================================================================
# SMILES TOKENIZER
# =========================================================================================

class SmilesTokenizer:
    def __init__(self, max_len=32):
        self.max_len = max_len
        self.vocab = {"<PAD>": 0, "<UNK>": 1, "<CLS>": 2, "<SEP>": 3, "<MASK>": 4}
        self.inverse_vocab = {0: "<PAD>", 1: "<UNK>", 2: "<CLS>", 3: "<SEP>", 4: "<MASK>"}
        
        # Common SMILES characters
        # ##EMIN##
        # chars = [
        #     'C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'B', 'H', # Atoms
        #     'c', 'n', 'o', 's', # Aromatic
        #     '(', ')', '[', ']', '=', '#', '%', '+', '-', '.', '/', '\\', '@', # Symbols
        #     '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' # Numbers
        # ]
        vocab_list = [
            'C', 'c', '(', ')', '1', 'O', '=', '2', 'Cl', 'N', 'F', 'Br', 
            '3', 'S', 'n', '#', '4', '[Si]', 'I', 'P', '5', '6', '7'
        ]
        
        for i, char in enumerate(vocab_list):
            self.vocab[char] = i + 5
            self.inverse_vocab[i + 5] = char
            
    def encode(self, smile):
        # Regex to tokenize SMILES (atoms, brackets, etc.)
        # ##EMIN## simplfy this
        #pattern =  r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        pattern = r"(\[[^\]]+\]|Cl|Br|Si|.)"
        regex = re.compile(pattern)
        tokens = [token for token in regex.findall(smile)]
        
        token_ids = [self.vocab.get(t, self.vocab["<UNK>"]) for t in tokens]
        
        # Truncate or Pad
        token_ids = token_ids[:self.max_len]
        token_ids = token_ids + [self.vocab["<PAD>"]] * (self.max_len - len(token_ids))
        
        return token_ids

    def __len__(self):
        return len(self.vocab)
