#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Title: Main Execution Script for Speech Emotion Recognition (SigWavNet)
Author: Alaa Nfissi
Date: March 31, 2024
Description: The main script to train, validate, and test the speech emotion recognition model (SigWavNet). 
It includes the configuration and execution of experiments, model evaluation, and result reporting.
"""

from utils import * # Import utility functions, classes, and global variables
from model import * # Import model definition
from custom_layers import * # Import custom layer definitions


# Check and print CUDA availability and details
print(f"Is cuda available ? => {torch.cuda.is_available()}")
print(f"How many devices available ? => {torch.cuda.device_count()}")
print(f"Current device => {torch.cuda.current_device()}")
d = 0 # Device ID to use
print(f"Picked device => {d}")
print(f"Device's name => {torch.cuda.get_device_name(d)}")
device = torch.device(f"cuda:{d}") # Set the device for training


# Setup for data loaders based on CUDA availability
if device == f"cuda{d}":
    num_workers = 10 # Use more workers for faster data loading on CUDA
    pin_memory = True # Helps with faster transfer to CUDA device
else:
    num_workers = 0 # CPU mode requires fewer workers
    pin_memory = False # No pinning needed in CPU mode
    
    
# Define paths for experiments and model checkpoints
iemocap_experiments_folder = "experiments"
checkpoint_dir = "models"


def train_SigWavNet(config, checkpoint_dir=None, data=None, max_num_epochs=None):
    
    """
    Trains the SigWavNet model with given configuration, data, and number of epochs.
    
    Parameters:
    - config: A dictionary with configuration parameters for the model and training.
    - checkpoint_dir: Directory where checkpoints are saved.
    - data: The dataset to use for training and validation.
    - max_num_epochs: The maximum number of epochs to train for.
    
    Returns:
    - None
    """
    
    epoch_count = max_num_epochs
    log_interval = 20 # Interval for logging training progress
    
    # Prepare class weights for focal loss
    a, class_counts = np.unique(data['label'], return_counts=True)
    num_classes = len(class_counts)
    total_samples = len(data['label'])

    class_weights = []
    for count in class_counts:
        weight = 1 / (count / total_samples)
        class_weights.append(weight)
    print(class_weights)

    # Initialize criterion with focal loss
    criterion = FocalLoss(alpha=torch.FloatTensor(class_weights), gamma=2)
    
    
    # Initialize SigWavNet model with given config
    model_SigWavNet = SigWavNet(n_input=config['n_input'], hidden_dim=config['hidden_dim'], n_layers=config['n_layers'] , n_output=config['n_output'], inputSize=None, kernelInit=config['kernelInit'], kernTrainable=config['kernTrainable'], level=config['level'], kernelsConstraint=config['mode'], initHT=config['initHT'], trainHT=config['trainHT'], alpha=config['alpha'])
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0" # Use first CUDA device
        pin_memory = True
        if torch.cuda.device_count() > 1:
            model_SigWavNet = nn.DataParallel(model_SigWavNet) # Use DataParallel for multi-GPU
    model_SigWavNet.to(device) # Move model to the chosen device
    
    # Initialize optimizer and learning rate scheduler
    optimiser = optim.Adam(model_SigWavNet.parameters(), lr=0.00001, weight_decay=0.00001)
    scheduler = optim.lr_scheduler.StepLR(optimiser, step_size=20, gamma=0.1)
    
    # Load checkpoint if directory is provided
    if checkpoint_dir:
        model_state, optimiser_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model_SigWavNet.load_state_dict(model_state)
        optimiser.load_state_dict(optimiser_state)
    
    # Prepare dataloaders for cross-validation
    dataloaders = get_dataloaders(data, batch_size=config["batch_size"], num_splits=config["num_splits"])
    
    # Initialize lists to track training and validation metrics
    losses_train = []
    losses_validation = []
    accuracy_train = []
    accuracy_validation = []
    total_train_acc = 0.0
    total_val_acc = 0.0
    
    pbar_update_1 = 1 # Progress bar update size for folds
    
    with tqdm(total=config["num_splits"]) as pbar_1:
        for fold, (train_loader, val_loader) in enumerate(dataloaders, 0):
            pbar_update = 1 / (len(train_loader) + len(val_loader)) # Progress bar update size for batches
            print(f'Fold {fold+1}/{config["num_splits"]}')
            optimiser = optim.Adam(model_SigWavNet.parameters(), lr=0.00001, weight_decay=0.00001)
            scheduler = optim.lr_scheduler.StepLR(optimiser, step_size=20, gamma=0.1)
            with tqdm(total=epoch_count) as pbar:
                
                for epoch in range(1, epoch_count + 1):
        
                    model_SigWavNet.train() # Set model to training mode
                    right = 0 # Track number of correct predictions
    
                    h = model_SigWavNet.init_hidden(config["batch_size"]) # Initialize hidden states
    
    
                    for batch_index, (data, target) in enumerate(train_loader):
        
                        data = data.to(device)
                        target = target.to(device)
        
                        h = [i.data for i in h] # Detach hidden states

                        output, h = model_SigWavNet(data, h) # Forward pass

                        pred = get_probable_idx(output) # Get predicted classes
                        right += nr_of_right(pred, target) # Update correct predictions count

                        loss = criterion(output.squeeze(), target) # Compute loss
        
                        optimiser.zero_grad() # Zero gradients
                        loss.backward() # Backpropagation
                        optimiser.step() # Update weights
                        
                        # Log training progress
                        if batch_index % log_interval == 0:
                            print(f"Train Epoch: {epoch} [{batch_index * len(data)}/{len(train_loader.dataset)} ({100. * batch_index / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}\tAccuracy: {right}/{len(train_loader.dataset)} ({100. * right / len(train_loader.dataset):.0f}%)")
            
                        pbar.update(pbar_update) # Update progress bar
        
                        losses_train.append(loss.item()) # Track training loss
        
                    free_memory([data, target, output]+h) # Free up memory
        ###################################################################################################
                    
                    # Validation loop
                    model_SigWavNet.eval() # Set model to evaluation mode
                    right = 0 # Reset correct predictions count
                    val_loss = 0.0 # Reset validation loss
                    val_steps = 0 # Reset validation steps count
                    h = model_SigWavNet.init_hidden(config["batch_size"]) # Reinitialize hidden states
    
                    for batch_index, (data, target) in enumerate(val_loader):

                        data = data.to(device)
                        target = target.to(device)
        
                        h = [i.data for i in h] # Detach hidden states
        
                        output, h = model_SigWavNet(data, h) # Forward pass

                        pred = get_probable_idx(output) # Get predicted classes

                        right += nr_of_right(pred, target) # Update correct predictions count

                        loss = criterion(output.squeeze(), target) # Compute loss
                        
                        val_loss += loss.item() # Accumulate validation loss
                        val_steps += 1 # Increment validation steps
                        
                        pbar.update(pbar_update) # Update progress bar

                        free_memory([data, target, output]+h) # Free up memory
                        
        
                    # Log validation progress
                    print(f"\nValidation Epoch: {epoch} \tLoss: {loss.item():.6f}\tAccuracy: {right}/{len(val_loader.dataset)} ({100. * right / len(val_loader.dataset):.0f}%)\n")
                    
                    # Calculate validation accuracy
                    acc = 100. * right / len(val_loader.dataset)
                    accuracy_validation.append(acc) # Track validation accuracy
        
                    losses_validation.append(loss.item()) # Track validation loss
        
                    scheduler.step() # Update learning rate
                    
                    # Save checkpoint
                    with tune.checkpoint_dir(epoch) as checkpoint_dir:
                        path = os.path.join(checkpoint_dir, "checkpoint") # Define checkpoint path
                        torch.save((model_SigWavNet.state_dict(), optimiser.state_dict()), path) # Save model and optimizer states
                    
                    tune.report(loss=(val_loss / val_steps), accuracy=acc) # Report metrics to Ray Tune
            pbar_1.update(pbar_update_1) # Update outer progress bar
    print("Finished Training !") # Indicate training completion


    
def test(model, batch_size, data):
    
    """
    Tests the given model on a test dataset.
    
    Parameters:
    - model: The trained model to be tested.
    - batch_size: The batch size to use for testing.
    - data: The test dataset.
    
    Returns:
    - Test set accuracy, predicted labels, and true labels.
    """
    
    model.eval() # Set model to evaluation mode
    right = 0 # Reset correct predictions count
    
    # Compute mean and standard deviation for normalization
    train_mean, train_std = compute_precise_mean_std(data['path'])
    test_transform = MyTransformPipeline(train_mean=train_mean, train_std=train_std) # Initialize transformation pipeline
    test_set = MyDataset(test_ds['path'], test_ds['label'], test_transform) # Create test dataset
    
    # Setup device for testing
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0" # Use first CUDA device
        pin_memory = True # Enable pinning for faster transfers to CUDA device
    
    # Initialize test loader
    test_loader = torch.utils.data.DataLoader(
                test_set,
                batch_size=batch_size,
                shuffle=False,
                drop_last=True,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
    
    h = model.init_hidden(batch_size) # Initialize hidden states
    
    # Lists to store true and predicted labels
    y_true = []
    y_pred = []
    with torch.no_grad(): # Disable gradient computation
        for data, target in test_loader:
        
            data = data.to(device)
            target = target.to(device)
        
            targets = target.data.cpu().numpy() # Get true labels
            y_true.extend(targets)

            h = [i.data for i in h] # Detach hidden states
        
            output, h = model(data, h) # Forward pass
        
        
            pred = get_probable_idx(output) # Get predicted classes
            right += nr_of_right(pred, target) # Update correct predictions count
        
            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy() # Get predicted labels
            y_pred.extend(output)
    
    # Print test set accuracy
    print(f"\nTest set accuracy: {right}/{len(test_loader.dataset)} ({100. * right / len(test_loader.dataset):.0f}%)\n")

    return (100. * right / len(test_loader.dataset)), y_pred, y_true # Return accuracy, predicted labels, and true labels



        
def main(num_samples=10, max_num_epochs=10, gpus_per_trial=1):
    
    """
    Main function to setup and run the model training and testing.
    
    Parameters:
    - num_samples: Number of hyperparameter samples to try.
    - max_num_epochs: Maximum number of epochs to train.
    - gpus_per_trial: Number of GPUs to use per trial.
    
    Returns:
    - None
    """
    
    # Load data and set wavelet
    data, emotionclasses = load_data('IEMOCAP_dataset.csv')
    wt = pywt.Wavelet('db10') # Define wavelet

    # Print model details
    print('SigWavNet model')
    print('Model n_input', 1)
    print('Model n_output', len(emotionclasses))
    
    # Define hyperparameter configuration
    config = {
        "n_input": tune.choice([1]),
        "hidden_dim": tune.choice([32, 64, 128]),
        "n_layers": tune.choice([3, 6, 9]),
        "n_output": tune.choice([len(emotionclasses)]),
        "weight_decay": tune.loguniform(1e-6, 1e-2),
        'level': tune.choice([4, 5, 6, 7, 8, 9, 10]),
        'trainHT': tune.choice([True, False]),
        'initHT': tune.choice([0.5, 0.6, 0.7, 0.8, 0.9, 1]),
        'kernTrainable': tune.choice([True, False]),
        'kernelInit': np.array(wt.filter_bank[0]),
        'alpha': tune.choice([10, 11, 12, 13, 14, 15]),
        'mode': tune.choice(['CQF', 'PerLayer', 'PerFilter']),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.grid_search([2, 4, 8, 16, 32]),
        "num_splits": tune.choice([10])
    }
    
    # Partition data for cross-validation
    data, test_ds, val_ds  = get_dataset_partitions_pd(data,train_split=0.9, val_split=0, test_split=0.1, target_variable='label', data_source='source')
    
    # Set up Ray Tune scheduler and reporter
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    
    reporter = CLIReporter(
        metric_columns=["loss", "accuracy", "training_iteration"])
    
    # Start Ray Tune run
    result = tune.run(
        tune.with_parameters(train_SigWavNet, data=data, max_num_epochs=max_num_epochs),
        resources_per_trial={"cpu": 12, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir=os.path.abspath(iemocap_experiments_folder+"/IEMOCAP_SigWavNet"),
        log_to_file=(os.path.abspath(iemocap_experiments_folder+"/IEMOCAP_SigWavNet_stdout.log"), os.path.abspath(iemocap_experiments_folder+"/IEMOCAP_SigWavNet_stderr.log")),
        name="IEMOCAP_SigWavNet",
        resume='AUTO')
    
    # Extract best trial information
    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))
    
    # Initialize best trained model with best trial configuration
    best_trained_model = SigWavNet(n_input=best_trial.config['n_input'], hidden_dim=best_trial.config['hidden_dim'], n_layers=best_trial.config['n_layers'] ,           n_output=best_trial.config['n_output'], inputSize=None, kernelInit=best_trial.config['kernelInit'], kernTrainable=best_trial.config['kernTrainable'], level=best_trial.config['level'], kernelsConstraint=best_trial.config['mode'], initHT=best_trial.config['initHT'], trainHT=best_trial.config['trainHT'], alpha=best_trial.config['alpha'])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0" # Use first CUDA device
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model) # Use DataParallel for multi-GPU
    best_trained_model.to(device) # Move model to the chosen device

    # Load the best model state
    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimiser_state = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)
    
    # Test the best trained model
    SigWavNet_test_acc_result, y_pred, y_true = test(best_trained_model, best_trial.config["batch_size"], data)
    
    # Print test set accuracy
    print("Best trial test set accuracy: {}".format(SigWavNet_test_acc_result))


if __name__ == "__main__":
    # Main function call, can specify number of samples, epochs, and GPUs per trial
    main(num_samples=10, max_num_epochs=5, gpus_per_trial=1)