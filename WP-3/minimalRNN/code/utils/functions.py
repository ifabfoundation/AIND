from utils.dependencies import *

# Function to create a dictionary that collects subject IDs by number of visits
def split_sub_per_visits(dataset):
    visit_distribution = dataset.groupby('ID').count()
    visit_distribution = visit_distribution[['TIME']].rename(columns={'TIME':'N visits'})
    sub_per_visits = []
    for i in range(visit_distribution['N visits'].max()+1):
        sub_per_visits.append([])

    # Populate sub_per_visits based on the number of visits for each subject
    for id in dataset['ID'].unique():
        i = int(id)
        n = visit_distribution.iloc[i][0]
        sub_per_visits[n].append(id)

    return sub_per_visits

# Function to split the dataset into train, validation
def split_dataset_in_train_val(dataset, percentage):
    # Create a dictionary that collects subject IDs by number of visits
    sub_per_visits = split_sub_per_visits(dataset)

    # Set seed for reproducibility and select IDs for training set.
    np.random.seed(0)

    train_id = []
    for n in range(len(sub_per_visits)):
        if sub_per_visits[n]:
            sublist_id = np.random.choice(sub_per_visits[n], size=int(percentage*len(sub_per_visits[n])), replace=False).tolist()
            train_id += sublist_id

    df_train = dataset[dataset['ID'].isin(train_id)]
    df_test = dataset[~dataset['ID'].isin(train_id)]

    # Get from the test dataset a dataset to predict 
    df_pers = df_test.groupby('ID').head(1) # Get last visit of every patient

    list_index = df_pers.index.tolist()
    df_to_pred = df_test[~df_test.index.isin(list_index)] # Evaluation dataset

    return df_train, df_test, df_pers, df_to_pred


# Function that applies reshape and mask to fit the input values to the shape needed to calculate the mAUC
def mAUC_metric_util_fun(tens_pred, tens_true, tens_mask):

    n_classes = tens_pred.shape[-1]

    tens_true_r = tens_true.reshape(-1, 1) # (10,21) -> (10*21,1) -> (n_pati * n_visits, 1)
    tens_pred_r = tens_pred.reshape(-1, n_classes) # (10,21,3) -> (10*21,3)
    tens_mask_r = tens_mask.reshape(-1, 1) # (10,21) -> (10*21,1) -> (n_pati * n_visits, 1)

    tens_true_r = tens_true_r[tens_mask_r]
    tens_pred_r = tens_pred_r[tens_mask_r.squeeze(-1)]

    # Calculate probability using softmax for multi-class classification
    pred_s_probs = F.softmax(tens_pred_r, dim=-1)

    return tens_true_r, pred_s_probs