def train_simple_model(model, criterion, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes):
    '''
    Function to train a simple PyTorch model.
    It returns the model with the best validation accuracy (we might change this metric)
    Returns losses and accuracies too
    '''
    LOSSES = {'train':[], 'val':[]}
    ACCS = {'train':[], 'val':[]}
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in tqdm.notebook.tqdm(range(num_epochs)):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode        
   
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data.
            for inputs, labels in tqdm.notebook.tqdm(dataloaders[phase]):
                inputs = list(map(lambda x: x.to(device), inputs))
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs[0].size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            LOSSES[phase].append(epoch_loss)
            ACCS[phase].append(epoch_acc.cpu().item())

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, LOSSES, ACCS

def evaluate_model(model, dataloaders, dataset_sizes, criterion, phase='test'):
    '''
    Function to evaluate on test set
    '''
    model.eval()   # Set model to evaluate mode        

    running_loss = 0.0
    running_corrects = 0
    running_incorrects = 0
    pred_list = []
    label_list =[]

    # Iterate over data.
    for inputs, labels in tqdm.notebook.tqdm(dataloaders[phase]):
        inputs = list(map(lambda x: x.to(device), inputs))
        labels = labels.to(device)

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        # statistics
        running_loss += loss.item() * inputs[0].size(0)
        running_corrects += torch.sum(preds == labels.data)
        running_incorrects += torch.sum(preds != labels.data)
        pred_list.extend(preds.cpu().tolist())
        label_list.extend(labels.cpu().tolist())
        
    print('Total Correct Predictions: ' + str(running_corrects))
    print('Total Incorrect Predictions: ' + str(running_incorrects))
    
    return pred_list, label_list