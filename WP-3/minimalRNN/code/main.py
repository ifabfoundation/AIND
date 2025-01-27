from utils.dependencies import *
from models.rnn import MinimalRNN
from models.custom_dataset import customDataset
from utils.functions import *
from loss.custom_cross_entropy import customCrossEntropy
from loss.custom_mae import customMAE
from loader import loader

# Python argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, help="Manual Seed. Set to 0 for reproducibility")
parser.add_argument('--sub_data', type=bool, default=False, help='True if you want to use a subset of the entire dataset')
parser.add_argument('--sub_n', type=str, default='100000', help='If sub_data is true, you can choose the number of patients')
parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda', help='Cuda if you want to use GPU otherwise use the value cpu. Requires that you have installed a version of torch with cuda compatibility ')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--gpu_ids', type=int, default=0, help='ids of GPUs to use')
parser.add_argument('--month_step', type=int, default=3, help='defines the time step of the rnn, expressed in months')
parser.add_argument('--batch_size', type=int, default=10, help='size of a batch')
parser.add_argument('--in_drop_rate', type=float, default=0.2, help='dropout ratio for the input variables')
parser.add_argument('--h_drop_rate', type=float, default=0.2, help='dropout ratio for the hidden state')
parser.add_argument('--n_layers', type=int, default=1, help='number of layers (cell) in the RNN')
parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs in train')
parser.add_argument('--weight_dc', type=float, default=0.000001, help='weight decay value')

# Save all arguments in a variable
opt = parser.parse_args()

# Seed
if opt.seed is None:
    opt.seed = np.random.randint(1, 10000) # Set the seed randomly

np.random.seed(opt.seed) # Set numpy seed
torch.manual_seed(opt.seed) # Set torch seed

# Check cuda
if opt.device == "cuda" and torch.cuda.is_available():
    torch.cuda.manual_seed_all(opt.seed) # Set CUDA seed
    opt.cuda_bool = True # Set a bool variable for cuda
else:
    opt.cuda_bool = False # If CPU is selected or CUDA is not available

# PATH
base_path = '../'
data_path = base_path + 'datasets/' # Data path
cognitive_scores_data = data_path + 'cognitive_scores.csv' # Data file

# Used to create different folders in case of the same runs on the same day
now = datetime.now()
data = str(now.month) + '_' + str(now.day) + '_' + str(now.hour) + '_' + str(now.minute)

result_path = base_path + f'results/experiment_{opt.sub_n}_{opt.n_epochs}_{data}/'
image_path = result_path + 'images/' # Path of image results
model_path = result_path + f'weights/' # Folder to save models

# Create folder if they don't exist
os.makedirs(image_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)

# Evaluation metric for continuous values
maeLoss = customMAE()
# Loss Cross Entroy function for categorical values
crossEntropyLoss = customCrossEntropy()

# Define labels
s_labels = [ 'NC', 'MCI', 'AD'] # Categorical
s_true_labels = ['Severity'] # Categorical
g_labels = ['MMSE', 'Memory', 'Language', 'Concentration', 'Praxis'] # Contineous
static_labels = ['TIME', 'sex', 'APOE'] # Static

opt.h_size = 128 # Hidden state size
opt.s_size = len(s_labels) # Categorical labels size
opt.g_size = len(g_labels) # Contineous labels size
opt.static_size = len(static_labels) # Static labels size

# Read dataset
dataset = loader(opt, cognitive_scores_data)
df_train, df_test, df_pers, df_to_pred = split_dataset_in_train_val(data, 0.8)

print('TRAIN SAMPLES: ' + str(len(df_train)))
print('PERSONALIZATION SAMPLES: ' + str(len(df_pers)))
print('VALIDATION SAMPLES: ' + str(len(df_to_pred)))

# Define the Rnn Model
model = MinimalRNN(
    opt.s_size, 
    opt.g_size, 
    opt.static_size, 
    opt.h_size, 
    opt.h_drop_rate, 
    opt.in_drop_rate, 
    opt.n_layers, 
    opt.cuda_bool
)

# Move the model on GPU
if opt.cuda_bool:
    model = model.cuda()

# Gradient optimiser
updator_model = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_dc)

# Spinner toggle for prompt visualization
stop_spinner = False

# Spinner function
def loading_spinner():
    spinner = ['|', '/', '-', '\\']
    idx = 0
    while not stop_spinner:  # Will continue until stop_spinner is True
        sys.stdout.write(spinner[idx % len(spinner)] + "\r")
        sys.stdout.flush()
        idx += 1
        time.sleep(0.1)

# Load spinner like a system thread
spinner_thread = threading.Thread(target=loading_spinner)
spinner_thread.start()

print('--- CREATE TRAIN DATASET ---')
# Create the dataset with the appropriate step
w_month = np.arange(0, dataset['month_bl'].max(), step=opt.step)
train_set = customDataset(df_train, s_labels, s_true_labels, g_labels, static_labels, opt.step, w_month)

print('--- CREATE VAL DATASET ---')
val_set = customDataset(df_test, s_labels, s_true_labels, g_labels, static_labels, opt.step, w_month)

# Stop spinner
stop_spinner = True
spinner_thread.join()

# Counter variable
batch_index = 0

# Best validation values 
best_val_loss = float('inf')
best_val_mae = 0
best_val_entropy = 0
best_val_auc = 0

# Actually it is not used
epochs_without_improvement = 0 # Count the number of epochs without improvement

# Train loss lists
epoch_train_loss_cat_list = []
epoch_train_loss_cont_list = []
epoch_train_total_loss_list = []

# Validation loss lists
epoch_val_loss_cat_list = []
epoch_val_loss_cont_list = []
epoch_val_total_loss_list = []

print('--- START TRAIN ---')
for epoch in range(opt.n_epochs):
    total_train_loss = 0 # Keeps track of total loss
    total_train_cont_loss = 0 # Keeps track of the loss of continuous variables
    total_train_cat_train = 0 # Keeps track of the loss of categorical variables

    gc.collect() # Forces the execution of the garbage collector to search for and free memory occupied by objects no longer used

    model.train() # Train mode

    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True)

    for ids, s_subs, s_true_subs, s_masks, g_subs, g_masks, static_subs in train_loader:

        if opt.cuda_bool:
            s_subs = s_subs.cuda() # Unsigned 8-bit integer
            s_true_subs = s_true_subs.cuda()
            g_subs = g_subs.cuda() # Float tensor
            s_masks = s_masks.cuda()
            g_masks = g_masks.cuda()
            static_subs = static_subs.cuda()

        # Resetting gradients
        updator_model.zero_grad()
        model.zero_grad()

        # Get variable
        pred_s_seq, pred_g_seq = model(s_subs, g_subs, static_subs)

        pred_g_seq = pred_g_seq.permute(1,0,2)
        pred_s_seq = pred_s_seq.permute(1,0,2)

        # Calculate loss
        mae_loss = maeLoss(pred_g_seq, g_subs[:, 1:, :], g_masks)
        cross_entropy_loss = crossEntropyLoss(pred_s_seq, s_true_subs, s_masks)

        summ_loss = 0.5 * mae_loss + 0.5 * cross_entropy_loss
        summ_loss.backward()
        updator_model.step()

        # weighted total loss calculation
        summ_loss_temp = 0.5 * mae_loss.item() + 0.5 * cross_entropy_loss.item()
        total_train_loss += summ_loss_temp * g_subs.size(0)
        total_train_cont_loss += mae_loss.item() * g_subs.size(0)
        total_train_cat_train += cross_entropy_loss.item() * g_subs.size(0)

        batch_index += 1

        # Printing of loss results at the end of a epoch
        if(batch_index % len(train_loader) == 0):
            print(f'Batch[{batch_index}] loss -- all: {summ_loss_temp} ... mae: {mae_loss.item()} ... cross_entropy: {cross_entropy_loss.item()}')

    # Epochs loss
    epoch_train_loss_cont = total_train_cont_loss/len(train_set)
    epoch_train_loss_cat = total_train_cat_train/len(train_set)
    epoch_train_total_loss = total_train_loss/len(train_set)

    epoch_train_loss_cont_list.append(epoch_train_loss_cont)
    epoch_train_loss_cat_list.append(epoch_train_loss_cat)
    epoch_train_total_loss_list.append(epoch_train_total_loss)

    print(f'--------- Epoch[{epoch}] Train loss -- all: {epoch_train_total_loss} ... mae: {epoch_train_loss_cont} ... cross_entropy: {epoch_train_loss_cat} --------------------')

    # Start Validation
    total_val_loss = 0 # Keeps track of total loss in validation
    total_val_cont_loss = 0 # Keeps track of the validation loss of continuous variables
    total_val_cat_train = 0 # Keeps track of the validation loss of categorical variables

    all_true_labels = []  # Save true labels to calculate AUC
    all_pred_probs = []   # Save the predicted probabilities to calculate AUC

    model.eval()

    with torch.no_grad():
        val_loader = DataLoader(val_set, batch_size=opt.batch_size, shuffle=True)
        for ids, s_subs, s_true_subs, s_masks, g_subs, g_masks, static_subs in val_loader:

            if opt.cuda_bool:
                s_subs = s_subs.cuda() # Unsigned 8-bit integer
                s_true_subs = s_true_subs.cuda()
                g_subs = g_subs.cuda() # Float tensor
                s_masks = s_masks.cuda()
                g_masks = g_masks.cuda()
                static_subs = static_subs.cuda()

            # Get variable
            pred_s_seq, pred_g_seq = model(s_subs, g_subs, static_subs)

            pred_g_seq = pred_g_seq.permute(1,0,2)
            pred_s_seq = pred_s_seq.permute(1,0,2)

            # Extract probabilities and labels
            s_true_masked, pred_s_probs = mAUC_metric_util_fun(pred_s_seq, s_true_subs, s_masks)
            all_pred_probs.append(pred_s_probs.cpu().numpy())
            all_true_labels.append(s_true_masked.cpu().numpy())

            # Calculate loss
            mae_loss = maeLoss(pred_g_seq, g_subs[:, 1:, :], g_masks)
            cross_entropy_loss = crossEntropyLoss(pred_s_seq, s_true_subs, s_masks)

            # weighted total loss calculation
            summ_loss_temp = 0.5 * mae_loss.item() + 0.5 * cross_entropy_loss.item()
            total_val_loss += summ_loss_temp * g_subs.size(0)
            total_val_cont_loss += mae_loss.item() * g_subs.size(0)
            total_val_cat_train += cross_entropy_loss.item() * g_subs.size(0)
    
    epoch_val_loss_cont = total_val_cont_loss/len(val_set)
    epoch_val_loss_cat = total_val_cat_train/len(val_set)
    epoch_val_total_loss = total_val_loss/len(val_set)

    # Convert to numpy array to calculate AUC.
    all_true_labels = np.concatenate(all_true_labels, axis=0)
    all_pred_probs = np.concatenate(all_pred_probs, axis=0)

    # Calculate the AUC for each class and then the average (mAUC).
    auc = roc_auc_score(all_true_labels, all_pred_probs, multi_class='ovr')

    epoch_val_loss_cont_list.append(epoch_val_loss_cont)
    epoch_val_loss_cat_list.append(epoch_val_loss_cat)
    epoch_val_total_loss_list.append(epoch_val_total_loss)

    print(f'--------- Epoch[{epoch}] Validation loss -- all: {epoch_val_total_loss} ... mae: {epoch_val_loss_cont} ... cross_entropy: {epoch_val_loss_cat} ... AUC: {auc} --------------------')

    if epoch_val_total_loss < best_val_loss:
        best_val_loss = epoch_val_total_loss
        best_val_mae = epoch_val_loss_cont
        best_val_entropy = epoch_val_loss_cat
        best_val_auc = auc

        torch.save(model.state_dict(), model_path)

print(f'--------- Best Validation loss -- all: {best_val_loss} ... mae: {best_val_mae} ... cross_entropy: {best_val_entropy} ... AUC: {best_val_auc} --------------------')

# Save training results
plt.plot(list(range(len(epoch_train_loss_cont_list))), epoch_train_loss_cont_list, label='Train MAE Loss', color='blue')
plt.plot(list(range(len(epoch_val_loss_cont_list))), epoch_val_loss_cont_list, label='Val MAE Loss', color='red')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('MAE Loss')
plt.legend()
plt.savefig(image_path + 'mae_loss.png')

plt.plot(list(range(len(epoch_train_loss_cat_list))), epoch_train_loss_cat_list, label='Train Cross Entropy Loss', color='blue')
plt.plot(list(range(len(epoch_val_loss_cat_list))), epoch_val_loss_cat_list, label='Val Cross Entropy Loss', color='red')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Cross Entropy Loss')
plt.legend()
plt.savefig(image_path + 'cross_entropy_loss.png')

plt.plot(list(range(len(epoch_train_total_loss_list))), epoch_train_total_loss_list, label='Train Total Loss', color='blue')
plt.plot(list(range(len(epoch_val_total_loss_list))), epoch_val_total_loss_list, label='Val Total Loss', color='red')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Total Loss')
plt.legend()
plt.savefig(image_path + 'total_loss.png')

