import pandas as pd

def loader(opt, path):
    dataset = pd.read_csv(path)
    dataset['ID'] = dataset['ID'].astype('str') # Make id as a string

    if opt.sub_data:
        sub_n = opt.sub_n
    else:
        sub_n = len(dataset)

    vis_n = dataset[dataset['ID']==sub_n].index[-1]+1 # number of visits (rows)
    dataset = dataset.iloc[:vis_n]

    return dataset