from rdkit import Chem
import torch
import rdkit
from rdkit import RDLogger
from inspect import getmembers, isfunction
from rdkit.Chem import Descriptors
import time
from collections import OrderedDict

from torch import Tensor

with open('./property_name.txt', 'r') as f:
    names = [n.strip() for n in f.readlines()][:53]

descriptor_dict = OrderedDict()
for n in names:
    if n == 'QED':
        descriptor_dict[n] = lambda x: Chem.QED.qed(x)
    else:
        descriptor_dict[n] = getattr(Descriptors, n)

#for i in getmembers(Descriptors, isfunction):
#    if i[0] in names:
#        descriptor_dict[i[0]] = i[1]
#        # print(i[0])    #name of properties
#descriptor_dict['QED'] = lambda x: Chem.QED.qed(x)


def calculate_property(smiles):
    RDLogger.DisableLog('rdApp.*')
    mol = Chem.MolFromSmiles(smiles)
    output = []
    for i, descriptor in enumerate(descriptor_dict):
        # print(descriptor)
        # print((descriptor_dict[descriptor](mol)))
        output.append(descriptor_dict[descriptor](mol))
    return torch.tensor(output, dtype=torch.float)


if __name__ == '__main__':
    st = time.time()
    output: Tensor = calculate_property('Cc1cccc(CNNC(=O)C2(Cc3ccccc3CN=[N+]=[N-])N=C(c3ccc(OCCCO)cc3)OC2c2ccc(Br)cc2)c1')
    print(output, output.size())
    print(time.time() - st)
    print(rdkit.__version__)
