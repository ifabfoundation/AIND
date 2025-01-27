import pandas as pd

def loader(opt, path):
    
    print('--DATA READING.--')
    data_orig = pd.read_csv(path)

    # Converting categorical variables having more than 2 cathegories into onehot (ONLY FOR THIS SPECIFIC DATASET)
    data_dummies = pd.get_dummies(data_orig['Severity'], prefix='Severity', columns=['Severity'])
    data_dummies.rename(columns={'Severity_0' : 'NC', 'Severity_1': 'MCI', 'Severity_2' : 'AD'}, inplace=True)

    dataset = pd.concat([data_orig, data_dummies], axis=1)

    # Split data in a small part (based on number of patients selected)
    if opt.sub_data:
        sub_n = opt.sub_n

        vis_n = dataset[dataset['ID']==sub_n].index[-1]+1 # number of visits (rows)
        dataset = dataset.iloc[:vis_n].copy()

    print('--CREATION OF MONTHS FROM AGE--')
    dataset['month_bl'] = dataset.groupby('ID')['TIME'].transform(lambda x: (x - x.min()) * 12)

    return dataset