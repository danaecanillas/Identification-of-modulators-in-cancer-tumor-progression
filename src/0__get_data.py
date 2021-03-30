import sys
import pandas as pd
from functools import reduce

DATA_FOLDER = "data/IDIBELL/"
GENERATED_FOLDER = "data/generated/"

def get_data():
    '''
    Function that collects all the data provided by IDIBELL.
    Generates several files that contain subsets of variables, 
    a file with the entire dataset, 
    the data split in train and test for training 
    and a descriptive file with the variables used. 

    [OUTPUT]:
    - data/generated/aux/immune_cells.csv
    - data/generated/aux/pathways.csv
    - data/generated/aux/METABRIC.csv
    - data/generated/aux/S.csv
    - data/generated/aux/data_clinical_patient.csv
    - data/generated/aux/mutations.csv
    - data/generated/entire_data.csv
    - data/generated/test.csv
    - data/generated/train.csv
    - data/generated/variables.txt
    '''

    # Immune cell data
    f = GENERATED_FOLDER + "aux/immune_cells.csv"
    immune_cells = pd.read_csv(DATA_FOLDER + "CIBERSORTx_metabric.txt", sep='\t', decimal=",")
    immune_cells = immune_cells.rename(columns={'ID': 'submitter'}).sort_values(by=['submitter'])
    print("[INFO]: Saving " + f)
    immune_cells.to_csv(f,index=False)

    # Pathways data
    f = GENERATED_FOLDER + "aux/pathways.csv"
    pathways = pd.read_csv(DATA_FOLDER + "pathwayscores_clinicalinformation_METABRIC.tsv", sep='\t')
    pathways = pathways[['submitter','Cell_Cycle', 'HIPPO', 'MYC', 'NOTCH', 'NRF2', 'PI3K', 'TGF.Beta', 'RTK_RAS', 'TP53', 'WNT', 'Hypoxia', 'SRC', 'ESR1', 'ERBB2']]
    print("[INFO]: Saving " + f)
    pathways.to_csv(f,index=False)

    # METABRIC complementary variables
    f = GENERATED_FOLDER + "aux/METABRIC.csv"
    METABRIC = pd.read_csv(DATA_FOLDER + "METABRIC_table_S23_revised.txt", sep='\t')
    METABRIC = METABRIC.rename(columns={'METABRIC_ID': 'submitter'}).sort_values(by=['submitter'])
    METABRIC = METABRIC[['submitter','PAM50','age_at_diagnosis','PROLIF','RFS','RFSE','DSSE10','DSS10']]
    print("[INFO]: Saving " + f)
    METABRIC.to_csv(f,index=False)
    
    # S complementary variables
    f = GENERATED_FOLDER + "aux/S.csv"
    S2 = pd.read_csv(DATA_FOLDER + "table_S2_revised.txt", sep='\t', decimal=",")
    S3 = pd.read_csv(DATA_FOLDER + "table_S3_revised.txt", sep='\t', decimal=",")
    S = pd.concat([S2, S3]).rename(columns={'METABRIC_ID': 'submitter'}).sort_values(by=['submitter'])
    S = S[['submitter','grade','stage','lymph_nodes_positive','Treatment']]
    print("[INFO]: Saving " + f)
    S.to_csv(f,index=False)
    
    # Data_clinical_patient complementary variables
    f = GENERATED_FOLDER + "aux/data_clinical_patient.csv"
    data_clinical_patient = pd.read_csv(DATA_FOLDER + "data_clinical_patient.txt", sep='\t')
    data_clinical_patient = data_clinical_patient.rename(columns={'PATIENT_ID': 'submitter'}).sort_values(by=['submitter'])
    data_clinical_patient = data_clinical_patient[['submitter','NPI','CELLULARITY','INTCLUST']]
    print("[INFO]: Saving " + f)
    data_clinical_patient.to_csv(f,index=False)

    # Mutation complementary variables
    f = GENERATED_FOLDER + "aux/mutations.csv"
    mutations = pd.read_csv(DATA_FOLDER + "Mutation_status_METABRIC_adjusted.txt", sep='\t')
    mutations = mutations.rename(columns={'METABRIC_ID': 'submitter'}).sort_values(by=['submitter'])
    mutations = mutations[['submitter','TP53.mut','PIK3CA.mut']]
    print("[INFO]: Saving " + f)
    mutations.to_csv(f,index=False)

    # -------------------------------------------------------------------------------------------

    # Save the entire data collection of all patients
    f = GENERATED_FOLDER + "entire_data.csv"
    data_frames = [immune_cells, pathways, METABRIC, S, mutations, data_clinical_patient]
    entire_data = reduce(lambda  left,right: pd.merge(left,right,on=['submitter'],how='inner'), data_frames)
    print("[INFO]: Saving " + f)
    entire_data.to_csv(f,index=False)
    
    # Split into train/test
    df = entire_data.replace({'PAM50': {"Normal": "NC"}})
    test = df[df["PAM50"] == "NC"]
    train = df[df["PAM50"] != "NC"]
    print("[INFO]: Saving " + GENERATED_FOLDER + "test.csv")
    test.to_csv(GENERATED_FOLDER + "test.csv",index=False)
    print("[INFO]: Saving " + GENERATED_FOLDER + "train.csv")
    train.to_csv(GENERATED_FOLDER + "train.csv",index=False)

    # -------------------------------------------------------------------------------------------

    # Generate a file with the obtained variables 
    with open(GENERATED_FOLDER + "info/variables.txt", 'w') as f:
        for idx, item in enumerate(list(entire_data.columns)):
            if (idx == 0):
                f.write("\nIMMUNE CELLS:\n" )

            if (idx == 23):
                f.write("\nPATHWAYS:\n" )
            
            if (idx == 37):
                f.write("\n------------\n" )
            f.write("- %s\n" % item)
    f.close()
    print("[INFO]: Saving " + GENERATED_FOLDER + "info/variables.txt")
    print("[INFO]: Completed")

if __name__== "__main__":
    get_data()