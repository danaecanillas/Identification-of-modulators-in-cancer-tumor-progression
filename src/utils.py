import copy

pathways = ['Cell_Cycle',
      'HIPPO', 'MYC', 'NOTCH', 'NRF2', 'PI3K', 'TGF.Beta', 'RTK_RAS', 'TP53',
      'WNT', 'Hypoxia', 'SRC', 'ESR1', 'ERBB2']

immune_cells = ['B cells naive', 'B cells memory', 'Plasma cells', 'T cells CD8',
    'T cells CD4 naive', 'T cells CD4 memory resting',
    'T cells CD4 memory activated', 'T cells follicular helper',
    'T cells regulatory (Tregs)', 'T cells gamma delta', 'NK cells resting',
    'NK cells activated', 'Monocytes', 'Macrophages M0', 'Macrophages M1',
    'Macrophages M2', 'Dendritic cells resting',
    'Dendritic cells activated', 'Mast cells resting',
    'Mast cells activated', 'Eosinophils', 'Neutrophils']

def flatten(A):
    rt = []
    for i in A:
        if isinstance(i,list): rt.extend(flatten(i))
        else: rt.append(i)
    return rt

comb = []

def combinations(target,data):
    for i in range(len(data)):
        new_target = copy.copy(target)
        new_data = copy.copy(data)
        new_target.append(data[i])
        new_data = data[i+1:]  
        if pathways in new_target:            
            el = flatten(new_target)
            comb.append(el)   
        combinations(new_target,new_data)
    return comb