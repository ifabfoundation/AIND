import pandas as pd
from utils import prepare_dataset, convert_cofactors_to_dummy, normalize_dataset

def loader(opt, path):
    dataset = pd.read_csv(path)
    dataset = prepare_dataset(dataset, 'RID', 'AGE', exclude=['AGE', 'DX/CN', 'DX/Dementia', 'DX/MCI'])

    # Convert cofactors to dummy variables
    dataset, new_cofactor_names = convert_cofactors_to_dummy(dataset, opt.dummy)

    # Normalize dataset
    dataset = normalize_dataset(dataset)

    if opt.sub_data:
        sub_n = opt.sub_n

        vis_n = dataset[dataset['ID']==sub_n].index[-1]+1 # number of visits (rows)
        dataset = dataset.iloc[:vis_n]

    return dataset