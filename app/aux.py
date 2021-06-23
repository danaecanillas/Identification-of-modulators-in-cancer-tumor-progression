
pathways = ['Cell_Cycle',
      'HIPPO', 'MYC', 'NOTCH', 'NRF2', 'PI3K', 'TGF.Beta', 'RTK_RAS', 'TP53',
      'WNT', 'Hypoxia', 'SRC', 'ESR1', 'ERBB2']

cells = ['B cells naive', 'B cells memory', 'Plasma cells', 'T cells CD8',
'T cells CD4 naive', 'T cells CD4 memory resting',
'T cells CD4 memory activated', 'T cells follicular helper',
'T cells regulatory (Tregs)', 'T cells gamma delta', 'NK cells resting',
'NK cells activated', 'Monocytes', 'Macrophages M0', 'Macrophages M1',
'Macrophages M2', 'Dendritic cells resting',
'Dendritic cells activated', 'Mast cells resting',
'Mast cells activated', 'Eosinophils', 'Neutrophils']

features = ['B cells naive', 'B cells memory', 'Plasma cells', 'T cells CD8',
      'T cells CD4 naive', 'T cells CD4 memory resting',
      'T cells CD4 memory activated', 'T cells follicular helper',
      'T cells regulatory (Tregs)', 'T cells gamma delta', 'NK cells resting',
      'NK cells activated', 'Monocytes', 'Macrophages M0', 'Macrophages M1',
      'Macrophages M2', 'Dendritic cells resting',
      'Dendritic cells activated', 'Mast cells resting',
      'Mast cells activated', 'Eosinophils', 'Neutrophils', 'Cell_Cycle',
      'HIPPO', 'MYC', 'NOTCH', 'NRF2', 'PI3K', 'TGF.Beta', 'RTK_RAS', 'TP53',
      'WNT', 'Hypoxia', 'SRC', 'ESR1', 'ERBB2', 'PROLIF','stage','grade']
      
params = {
    'k_folds': 5,  
    'lr':0.01,
    'epochs':120,
    'batch_size':200,
    'fc1':200,
    'fc2':120,
    'fc3':84,
    'dropout':0.2
}