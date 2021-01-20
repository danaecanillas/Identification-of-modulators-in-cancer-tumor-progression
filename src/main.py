import pandas as pd

def read_data():
    print("Reading data from the tsv file")
    path_file = "data/pathwayscores_clinicalinformation_METABRIC.tsv"
    data = pd.read_csv(path_file, sep='\t')
    return data

def main():
    data = read_data()
    print("I have de data!")

if __name__ == "__main__":
    main()
