import os
import torchaudio
import numpy as np
import librosa
from scipy import stats
import torch
import wandb
from torch import optim
from torch import nn
from tqdm import tqdm
from itertools import groupby
from src.models import Hubert, Wav2Vec, CNN, CRNN
import pickle

def check_folders(buckeye, timit, ntimit):
    
    if len(os.listdir(buckeye)) == 0:
        raise ValueError('Buckeye folder is empty. Please download the data and check the instructions in the README file to organize it')

    elif len(os.listdir(timit)) == 0:
            raise ValueError('Timit folder is empty. Please download the data and check the instructions in the README file to organize it')
    
    elif os.path.exists(ntimit):
            ntimit_exists= True

    else:
        ntimit_exists= False

    return ntimit_exists
    
    
    


def get_timit_data(path, indices_path, sr=16000):

    with open(indices_path, 'r') as f:
        indices= f.readlines()
        indices= [index.replace('\n', '') for index in indices]

    wavs=[]
    bounds=[]

    for root, _, files in os.walk(path):
        dir_name= root.split('/')[-2]+ '/'+ root.split('/')[-1]
        for file in files:
            file_temp= file.split('/')[-1]
            if file.endswith('.wav') and dir_name+'/'+file_temp in indices:
                wav_file= os.path.join(root, file)

                if "WAV.wav" in wav_file:
                    words_file= wav_file.replace('.WAV.wav', '.WRD')
                else:
                    words_file= wav_file.replace('.wav', '.WRD')
                    
                words = open(words_file, 'r').readlines()
                words = [w.strip().split() for w in words]
                
                if len(words) == 0:
                    
                    print(f"Skipping empty file: {words_file.split('/')[-1]}")

                else:
                    wav, old_sr= torchaudio.load(wav_file)

                    if old_sr != sr:
                        wav= torchaudio.transforms.Resample(old_sr, sr)(wav)
                        
                    wavs.append(wav)
                    bound = [(int(w[0]), int(w[1])) for w in words]
                    bounds.append(bound)

    return wavs, bounds

def get_ntimit_data(path, indices_path, sr=16000):

    with open(indices_path, 'r') as f:
        indices= f.readlines()
        indices= [index.replace('\n', '') for index in indices]

    wavs=[]
    bounds=[]

    for root, _, files in os.walk(path):
        dir_name= root.split('/')[-2]+ '/'+ root.split('/')[-1]
        for file in files:
            file_temp= file.split('/')[-1]
            if file.endswith('.flac') and dir_name+'/'+file_temp in indices:
                wav_file= os.path.join(root, file)

                words_file= wav_file.replace('.flac', '.wrd')
                    
                words = open(words_file, 'r').readlines()
                words = [w.strip().split() for w in words]
                
                if len(words) == 0:
                    
                    print(f"Skipping empty file: {words_file.split('/')[-1]}")

                else:
                    wav, old_sr= torchaudio.load(wav_file)

                    if old_sr != sr:
                        wav= torchaudio.transforms.Resample(old_sr, sr)(wav)
                        
                    wavs.append(wav)
                    bound = [(int(w[0]), int(w[1])) for w in words]
                    bounds.append(bound)

    return wavs, bounds

def get_buckeye_data(path, indices_path, sr=16000):

    with open(indices_path, 'r') as f:
        indices= f.readlines()
        indices= [index.replace('\n', '') for index in indices]

    wavs=[]
    bounds=[]
    
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith('.wav') and file in indices:
                wav_file= os.path.join(root, file)

                words_file= wav_file.replace('.wav', '.word')
                words = open(words_file, 'r').readlines()
                words = [w.strip().split() for w in words]
                
                if len(words) == 0:
                    
                    print(f"Skipping empty file: {words_file.split('/')[-1]}")

                else:
                    wav, old_sr= torchaudio.load(wav_file)

                    if old_sr != sr:
                        wav= torchaudio.transforms.Resample(old_sr, sr)(wav)
                        
                    wavs.append(wav)
                    bound = [(int(w[0]), int(w[1])) for w in words]
                    bounds.append(bound)

    return wavs, bounds

def pad_set(datasets, wavs, labels, bounds):

    max_len= compute_max_len(datasets)
    
    wavs, labels, bounds= pad_audio(wavs, max_len[0]), pad_labels(labels, max_len[1]), pad_bounds(bounds, max_len[2])

    return wavs, labels, bounds


def frame_labels(label,frame_size, hop_length, idx, type):

    if np.bincount(label.astype(int)).shape != (3,):
        raise ValueError(f"Label should have 0, 1 and 2. The label has only {np.unique(label.astype(int))}. Check the label for index {idx}")

    frames = librosa.util.frame(label, frame_length=frame_size, hop_length=hop_length)

    contains_one = np.any(frames == 1, axis=0)

    modes = stats.mode(frames, axis=0).mode

    framed_labels = np.where(contains_one, 1, modes.squeeze()).astype(int)

    if type == 'dev':
        #Retrieving the index of the frames that contain 1
        idxs = np.where(contains_one)[0]
    
        #Labeling the three frames before the frame that contains 1 as 1
        for i in idxs:
            if i+2 < len(framed_labels):
                framed_labels[i-1] = 1
                #framed_labels[i-2] = 1
                #framed_labels[i-3] = 1
                framed_labels[i+1] = 1
                #framed_labels[i+2] = 1
            else:
                framed_labels[i-1] = 1
                #framed_labels[i-2] = 1
                #framed_labels[i-3] = 1
 
    return framed_labels

def get_labels(wavs, bounds, sr, frame_size, hop_length, type='dev'):
    labels=[]

    for i in range(len(wavs)):
        label= np.zeros(wavs[i].size(1))

        for idx, b in enumerate(bounds[i]):

            if sr == 8000:
                start= b[0]//2
                end= b[1]//2

            else:
                start= b[0]
                end= b[1]

            try:
                
                label[start] = 1
                label[start+1:end] = 2

            except:

                if start==end:
                    label[start-1]= 1
                
                

        labels.append(label)

    framed_labels=[]

    for idx, label in enumerate(labels):
        label= frame_labels(label, frame_size, hop_length, idx, type)
        framed_labels.append(label)

    return framed_labels

def compute_max_len_per_set(wavs, labels, bounds):

    max_wav_len= max([w.size(1) for w in wavs])
    max_label_len= max([len(l) for l in labels])
    max_bound_len= max([len(b) for b in bounds])

    return max_wav_len, max_label_len, max_bound_len

def compute_max_len(dataset):

    train_wavs, train_labels, train_bounds, val_wavs, val_labels, val_bounds, test_wavs, test_labels, test_bounds= dataset

    max_train_wav_len, max_train_label_len, max_train_bound_len= compute_max_len_per_set(train_wavs, train_labels, train_bounds)
    max_val_wav_len, max_val_label_len, max_val_bound_len= compute_max_len_per_set(val_wavs, val_labels, val_bounds)
    max_test_wav_len, max_test_label_len, max_test_bound_len= compute_max_len_per_set(test_wavs, test_labels, test_bounds)  

    wav_max= max(max_train_wav_len, max_val_wav_len, max_test_wav_len)
    label_max= max(max_train_label_len, max_val_label_len, max_test_label_len)
    bound_max= max(max_train_bound_len, max_val_bound_len, max_test_bound_len)

    return wav_max, label_max, bound_max

def pad_audio(wavs, max_len=0):

    padded_wavs= []

    for w in wavs:
        pad_len= max_len- w.size(1)
        padded_wav= np.pad(w[0].numpy(), (0, pad_len), 'constant', constant_values=0)
        padded_wavs.append(padded_wav)

    padded_wavs= np.stack(padded_wavs)

    return padded_wavs

def pad_labels(labels, max_len=0):

    padded_labels= []

    for l in labels:
        pad_len= max_len- len(l)
        padded_label= np.pad(l, (0, pad_len), 'constant', constant_values=0)
        padded_labels.append(padded_label)

    padded_labels= np.stack(padded_labels)

    return padded_labels

def pad_bounds(bounds, max_len=0):

    padded_bounds = [b + [(-100, -100)] * (max_len - len(b)) for b in bounds]

    padded_bounds= np.stack(padded_bounds)

    return padded_bounds

def get_data(buckeye_corpus_path, train_indices, val_indices, test_indices, timit_testset_path, timit_indices, ntimit_testset_path, ntimit_indices):
    #check if path exists
    ntimit_exists = check_folders(buckeye_corpus_path, timit_testset_path, ntimit_testset_path)

    SR=16000
    FRAME_SIZE=int(0.025*SR)
    HOP_LENGTH=int(0.025*SR)

    print('Extracting Buckeye data...')
    buckeye_train_wavs, buckeye_train_bounds= get_buckeye_data(buckeye_corpus_path, train_indices, SR)
    buckeye_val_wavs, buckeye_val_bounds= get_buckeye_data(buckeye_corpus_path, val_indices, SR)
    buckeye_test_wavs, buckeye_test_bounds= get_buckeye_data(buckeye_corpus_path, test_indices, SR)

    print('Extracting Timit data...')
    timit_wavs, timit_bounds= get_timit_data(timit_testset_path, timit_indices, SR)

    if ntimit_exists:
        print('Extracting NTimit data...')
        ntimit_wavs, ntimit_bounds= get_ntimit_data(ntimit_testset_path, ntimit_indices, SR)

    print('Extracting Buckeye labels...')
    buckeye_train_labels= get_labels(buckeye_train_wavs, 
                             buckeye_train_bounds, 
                             SR, 
                             FRAME_SIZE, 
                             HOP_LENGTH)

    buckeye_val_labels= get_labels(buckeye_val_wavs, 
                           buckeye_val_bounds, 
                           SR, 
                           FRAME_SIZE, 
                           HOP_LENGTH,
                           type='test')

    buckeye_test_labels= get_labels(buckeye_test_wavs, 
                            buckeye_test_bounds, 
                            SR, 
                            FRAME_SIZE, 
                            HOP_LENGTH, 
                            type='test')

    print('Extracting Timit labels...')
    timit_labels= get_labels(timit_wavs,
                             timit_bounds,
                             SR,
                             FRAME_SIZE,
                             HOP_LENGTH,
                             type='test')

    if ntimit_exists:
        print('Extracting NTimit labels...')
        ntimit_labels= get_labels(ntimit_wavs,
                                  ntimit_bounds,
                                  SR,
                                  FRAME_SIZE,
                                  HOP_LENGTH,
                                  type='test')
    print('\n')
    print('Buckeye Train samples:', len(buckeye_train_wavs), len(buckeye_train_labels), len(buckeye_train_bounds))
    print('Buckey Val samples:', len(buckeye_val_wavs), len(buckeye_val_labels), len(buckeye_val_bounds))
    print('Buckeye Test samples:', len(buckeye_test_wavs), len(buckeye_test_labels), len(buckeye_test_bounds))
    print('Timit Test samples:', len(timit_wavs), len(timit_labels), len(timit_bounds))

    if ntimit_exists:
        print('NTimit Test samples:', len(ntimit_wavs), len(ntimit_labels), len(ntimit_bounds))

    datasets= (buckeye_train_wavs, 
               buckeye_train_labels, 
               buckeye_train_bounds, 
               buckeye_val_wavs, 
               buckeye_val_labels, 
               buckeye_val_bounds, 
               buckeye_test_wavs, 
               buckeye_test_labels, 
               buckeye_test_bounds)

    print('\n')
    print('Padding Buckeye testset...')
    buckeye_wavs, buckeye_labels, buckeye_bounds= pad_set(datasets, 
                                                          buckeye_test_wavs, 
                                                          buckeye_test_labels, 
                                                          buckeye_test_bounds)

    print('Padding Timit testset...')
    timit_wavs, timit_labels, timit_bounds= pad_set(datasets, 
                                                    timit_wavs, 
                                                    timit_labels, 
                                                    timit_bounds)

    if ntimit_exists:   
        print('Padding NTimit testset...')
        ntimit_wavs, ntimit_labels, ntimit_bounds= pad_set(datasets, 
                                                           ntimit_wavs, 
                                                           ntimit_labels, 
                                                           ntimit_bounds)


    test_sets= {'buckeye': (buckeye_wavs, buckeye_labels, buckeye_bounds), 'timit': (timit_wavs, timit_labels, timit_bounds)}

    if ntimit_exists:
        print('NTimit testset added to test_sets')
        test_sets['ntimit']= (ntimit_wavs, ntimit_labels, ntimit_bounds)

    return test_sets, ntimit_exists

def calculate_segmentation_metrics(model_output, target):
    # Calculate predicted classes
    predicted_classes = torch.argmax(model_output.softmax(dim=1), dim=-1)

    # Initialize counters
    N_hit, N_ref, N_f = 0, 0, 0

    for output, ground_truth in zip(predicted_classes, target):
        #find model boundaries and reference boundaries
        model_boundaries = torch.where(output == 1)[0]  # Class 1 for 'start'
        ref_boundaries = torch.where(ground_truth == 1)[0]

        # Update counters
        N_ref += len(ref_boundaries)
        N_f += len(model_boundaries)
        N_hit += len(set(model_boundaries.tolist()) & set(ref_boundaries.tolist()))

    # Calculate metrics
    HR = (N_hit / N_ref) * 100 if N_ref > 0 else 0
    OS = ((N_f / N_ref) - 1) * 100 if N_ref > 0 else 0
    Precision = (N_hit / N_f) if N_f > 0 else 0
    Recall = (N_hit / N_ref) if N_ref > 0 else 0
    F_value = (2 * Precision * Recall / (Precision + Recall)) if (Precision + Recall) > 0 else 0

    # Calculate R value
    r1 = torch.sqrt((torch.tensor(100) - HR)**2.0 + OS**2.0)
    r2 = (-OS + HR - torch.tensor(100)) / torch.sqrt(torch.tensor(2.0))
    R_value = 1 - (torch.abs(r1) + torch.abs(r2)) / 200

    return HR, OS, Precision, Recall, F_value, R_value

def process_batch(batch, model, criterion, device, type='train'):
    src, target, _ = batch
    src, target = src.float().to(device), target.long().to(device)

    out = model(src)

    if type == 'test':
        return out
    
    num_classes = out.size(2)

    out = out.view(-1, num_classes)
    target = target.view(-1)

    loss = criterion(out, target)
    
    return out, target, loss

def train(model, train_loader, val_loader, optimizer, criterion, device, tolerance, hop_length, frame_selection):

    model.train()
    train_hr, train_os, train_f_value, train_r_value, train_loss = 0,0,0,0,0

    for batch in tqdm(train_loader, desc='Training'):
        optimizer.zero_grad()
        output, target, loss = process_batch(batch, model, criterion, device)
        loss.backward()
        optimizer.step()

        # Update training loss
        train_loss += loss.item()


        # Calculate segmentation metrics
        hr, os, _, _, f_value, r_value = calculate_segmentation_metrics(output, target)
        train_hr += hr
        train_os += os
        train_f_value += f_value
        train_r_value += r_value

    # Calculate and reset training metrics
    train_loss /= len(train_loader)
    train_hr /= len(train_loader)
    train_os /= len(train_loader)
    train_f_value /= len(train_loader)
    train_r_value /= len(train_loader)
    
    # Validation phase
    val_precision, val_recall, val_f_value, val_os, val_r_value = predict(model=model, loader=val_loader, device=device, tolerance=tolerance, hop_length=hop_length, frame_selection=frame_selection, desc='validation')
    #model.eval()
    #val_hr, val_os, val_f_value, val_r_value, val_loss = 0,0,0,0,0
#
    #with torch.no_grad():
    #    for batch in tqdm(val_loader, desc='Validation'):
    #        output, target, loss = process_batch(batch, model, criterion, device)
#
    #        # Update validation loss
    #        val_loss += loss.item()
#
    #        # Calculate segmentation metrics
    #        hr, os, _, _, f_value, r_value = calculate_segmentation_metrics(output, target)
    #        val_hr += hr
    #        val_os += os
    #        val_f_value += f_value
    #        val_r_value += r_value
#
    ## Calculate validation metrics
    #val_loss /= len(val_loader)
    #val_hr /= len(val_loader)
    #val_os /= len(val_loader)
    #val_f_value /= len(val_loader)
    #val_r_value /= len(val_loader)
#

    return (  train_loss, train_hr, train_os, train_f_value, train_r_value,
             val_precision, val_recall, val_os, val_f_value, val_r_value)

def trainer(device, model, train_loader, val_loader, lr, epochs, patience, checkpoint, tolerance, hop_length, frame_selection, early_stopping=False):

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epochs = epochs
    patience = patience
    epochs_no_improve = 0

    best_r_value= -np.inf

    for epoch in range(epochs):
        print(f'Epoch: {epoch+1}')

        metrics = train(model, 
                        train_loader, 
                        val_loader, 
                        optimizer, 
                        criterion, 
                        device,
                        tolerance,
                        hop_length,
                        frame_selection
                        )
        
        train_loss = metrics[0]
        train_hr = metrics[1]
        train_os = metrics[2]
        train_f_value = metrics[3]
        train_r_value = metrics[4]
        val_precision = metrics[5]
        val_recall = metrics[6]
        val_os = metrics[7]
        val_f_value = metrics[8]
        val_r_value = metrics[9]
  
        print(f'Train loss: {train_loss},  Train HR: {train_hr}, Train OS: {train_os}, Train F value: {train_f_value}, Train R value: {train_r_value}')
        print(f'Val Precision: {val_precision}, Val Recall: {val_recall}, Val OS: {val_os}, Val F value: {val_f_value}, Val R value: {val_r_value}\n')

        wandb.log({'train/loss': train_loss,
                   'val/precision': val_precision,
                   'train/hr': train_hr,
                   'val/recall': val_recall,
                   'train/os': train_os,
                   'val/os': val_os,
                   'train/f_value': train_f_value,
                   'val/f_value': val_f_value,
                   'train/r_value': train_r_value,
                   'val/r_value': val_r_value,
                   })
        
        # Checking segmentation metrics to the best r_value
        if val_r_value > best_r_value:
            best_r_value = val_r_value
            best_epoch = epoch
            best_os = val_os
            best_f_value = val_f_value
            wandb.log({'best/r_value': best_r_value,
                       'best/epoch': best_epoch,
                       'best/os': best_os,
                       'best/f_value': best_f_value})
            
            torch.save(model.state_dict(), checkpoint)
            print('Model saved with R_value: ', best_r_value)
            print('-'*100, '\n')

            epochs_no_improve = 0

        else:
            epochs_no_improve += 1
            print(
                f'No improvement in validation R-value for {epochs_no_improve} epochs')
            print('-'*100, '\n')

            # Early stopping check
            if early_stopping:
                if epochs_no_improve >= patience:
                    print(f'Early stopping triggered after {epoch + 1} epochs')
                    break

                print('-'*100, '\n')

def find_index(sequence, hop_length,  type):

    all_indices = []
    # Itera su ogni gruppo nel tuo array 2D
    for group in sequence:
        # Trova gli indici di tutti gli '1'
        indices = np.where(group == 1)[0]

        '''
        groups=[ indices[i]*hop_length for i in range(0, len(indices))]

        all_indices.append(groups)

    return all_indices
        '''

        # Raggruppa gli indici consecutivi
        groups = [list(g) for _, g in groupby(indices, lambda i, j=iter(range(1, len(indices)+1)): next(j)-i)]

        if type == 'percentile':
            # Calcola il 25° percentile per ogni gruppo di '1' consecutivi
            percentile_indices = [i for g in groups for i in g if  np.percentile(g, 0)<=i]

            percentile_indices = [i * hop_length for i in percentile_indices]

            all_indices.append(percentile_indices)


        elif type == 'mid':
            # Trova l'indice medio per ogni gruppo di '1' consecutivi
            mid_indices = [g[len(g) // 2] for g in groups]

            #Converting index to time-points
            mid_indices= [i* hop_length for i in mid_indices]

            # Aggiungi gli indici medi alla lista principale
            all_indices.append(mid_indices)

        elif type == 'first':

            first_indices = [g[0] for g in groups]

            first_indices= [i* hop_length for i in first_indices]

            all_indices.append(first_indices)

        elif type == 'last':
                
            last_indices = [g[-1] for g in groups]
    
            last_indices= [i* hop_length for i in last_indices]
    
            all_indices.append(last_indices)

        else:
            raise ValueError('Type should be either [mid, first, last, percentile]')


    return all_indices

def post_process(sequence, hop_length, type, prediction=True):

    if prediction:
        # Calculate predicted classes
        sequence = torch.argmax(sequence.softmax(dim=-1), dim=-1)
        sequence= sequence.detach().cpu().numpy()
    
    all_mid_indices= find_index(sequence, hop_length, type)

    return all_mid_indices

def predict(model, loader, device, tolerance, hop_length, frame_selection, desc='validation', plot_bounds=False, batch_plot_id=0):

    if batch_plot_id>=len(loader):
        raise ValueError('Batch plot id should be less than the length of the loader')

    criterion = nn.CrossEntropyLoss()
    model.to(device)
    model.eval()


    with torch.no_grad():
        total_precision=0
        total_recall=0
        total_f_value=0
        total_os=0
        total_r_value=0

        for idx, (w,l,b) in enumerate(tqdm(loader, desc=desc)):
            output = process_batch((w,l,b), model, criterion, device, type='test')

            post_out= post_process(output, hop_length, frame_selection)

            all_boundaries=[]

            for bounds in b:
                starting_bounds = [int(bound[0]) for bound in bounds if bound[0] != -100]
                all_boundaries.append(starting_bounds)

            if plot_bounds and idx == batch_plot_id:
                arr_w= w.detach().cpu().numpy()
                plot(arr_w, post_out, all_boundaries, 16000)

            N_ref=0
            N_seg=0

            for pred, true in zip(post_out, all_boundaries):
                
                N_ref += len(true)
                N_seg += len(pred)

            N_correct=0

            for idx, pred in enumerate(post_out):
                for p in pred:
                    for idx2,t in enumerate(all_boundaries[idx]):
                        if abs(p-t) <= tolerance:
                            N_correct += 1
                            all_boundaries[idx].pop(idx2)
                            break

            recall= N_correct/N_ref
            precision= N_correct/N_seg
            f_value= 2*precision*recall/(precision+recall)
            
            os = recall/precision - 1
            r1 = np.sqrt((1 - recall)**2 + os**2)
            r2 = (-os + recall - 1)/np.sqrt(2)
            r_value = 1 - (np.abs(r1) + np.abs(r2))/2

            total_precision += precision
            total_recall += recall
            total_f_value += f_value
            total_os += os
            total_r_value += r_value

    total_precision /= len(loader)
    total_recall /= len(loader)
    total_f_value /= len(loader)
    total_os /= len(loader)
    total_r_value /= len(loader)

    print(f'{desc} Precision: {total_precision}, {desc} Recall: {total_recall}, {desc} F value: {total_f_value}, {desc} OS: {total_os}, {desc} R value: {total_r_value}\n')


    if desc == 'validation':
        return total_precision, total_recall, total_f_value, total_os, total_r_value



import matplotlib.pyplot as plt
import librosa
import numpy as np

def plot(test_wavs, preds_bounds, true_bounds, SR):

    for wav, p_bound, t_bound in zip(test_wavs, preds_bounds, true_bounds):
        fig, ax = plt.subplots(figsize=(20, 5))
        wav= wav[:SR*4]
        librosa.display.waveshow(wav, sr=SR, color='orange')
        for p in p_bound:
            p= p/SR
            ax.axvline(x=p, color='red', linestyle='--', linewidth=2, ymin=0.5)
        for t in t_bound:
            t= t/SR
            ax.axvline(x=t, color='green', linestyle='-', linewidth=2, ymax=0.5)
        # Set the label size
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.tick_params(axis='both', which='minor', labelsize=20)
        ax.set_xlabel('Time', fontsize=20)
        plt.show()

def select_model_and_dataset(ntimit_exists):

    models={'1': 'HuBERT', '2': 'Wav2Vec 2.0', '3': 'CNN', '4': 'CRNN'}

    print ('Choose a model to test:')
    print ('1 HuBERT')
    print ('2 Wav2Vec 2.0')
    print ('3 CNN')
    print ('4 CRNN\n')

    model_name= input('Enter the model number: ')

    assert model_name in ['1', '2', '3', '4'], 'Invalid model number, please choose a number between 1 and 4'

    if ntimit_exists:
        datasets={'1': 'Buckeye', '2': 'Timit', '3': 'NTimit'}

        print('Choose a dataset to test:')
        print('1 Buckeye')
        print('2 Timit')
        print('3 NTimit\n')

        dataset= input('Enter the dataset number: ')

        assert dataset in ['1', '2', '3'], 'Invalid dataset number, please choose a number between 1 and 3'

        print(f'You chose model {models[model_name]} and dataset {datasets[dataset]}\n')

        return model_name, dataset
    
    else:
        datasets={'1': 'Buckeye', '2': 'Timit'}

        print('Choose a dataset to test:')
        print('1 Buckeye')
        print('2 Timit\n')

        dataset= input('Enter the dataset number: ')

        assert dataset in ['1', '2'], 'Invalid dataset number, please choose a number between 1 and 2'

        print(f'You chose model {models[model_name]} and dataset {datasets[dataset]}\n')

        return model_name, dataset

def choose_dataset(dataset, test_sets, SR):

        if dataset=='1':
            wavs, labels, bounds = test_sets['buckeye']
            
            TOLERANCE=int(0.06*SR)

        elif dataset=='2':

            wavs, labels, bounds = test_sets['timit']

            TOLERANCE=int(0.04*SR)
        
        elif dataset=='3':
             wavs, labels, bounds = test_sets['ntimit']
                
             TOLERANCE=int(0.04*SR)

        return wavs, labels, bounds, TOLERANCE



def retrieve_model(model_name, time, frames_out, NUM_CLASSES, verbose, freeze):
    if model_name=='1':
        model= Hubert(input_time=time, 
                  num_frames=frames_out, 
                  num_classes=NUM_CLASSES, 
                  verbose=verbose, 
                  freeze=freeze)
        try:
            CHECKPOINT="./checkpoint/WBD_HuBERT.pt"    
            model.load_state_dict(torch.load(CHECKPOINT))
        except:
            raise ValueError('HuBERT checkpoint not found. Please download the weights from the link provided in the README file and store it in the ./checkpoint folder')

    elif model_name=='2':

        model= Wav2Vec(input_time=time,
                       num_frames=frames_out,
                       num_classes=NUM_CLASSES,
                       verbose=verbose,
                       freeze=freeze)
        try:
            CHECKPOINT="./checkpoint/WBD_W2V.pt"
            model.load_state_dict(torch.load(CHECKPOINT))
        except:
            raise ValueError('Wav2Vec checkpoint not found. Please download the weights from the link provided in the README file and store it in the ./checkpoint folder')

    elif model_name=='3':
        model= CNN(input_time=time,
                    num_frames=frames_out,
                    num_classes=NUM_CLASSES,
                    verbose=verbose)
        
        try:
            CHECKPOINT="./checkpoint/WBD_CNN.pt"
            model.load_state_dict(torch.load(CHECKPOINT))
        except:
            raise ValueError('CNN checkpoint not found. Please download the weights from the link provided in the README file and store it in the ./checkpoint folder')

    elif model_name=='4':
        model= CRNN(input_time=time,
                     num_frames=frames_out,
                     num_classes=NUM_CLASSES,
                     verbose=verbose)
        try:
            CHECKPOINT="./checkpoint/WBD_CRNN.pt"
            model.load_state_dict(torch.load(CHECKPOINT))
        except:
            raise ValueError('CRNN checkpoint not found. Please download the weights from the link provided in the README file and store it in the ./checkpoint folder')
        
    return model

def continue_testing():

    get_answer= False

    while not get_answer:

        print('Do you want to test another model? (y/n)')
        answer= input().lower()

        if answer != 'y' and answer != 'n':
            print('Invalid answer. Please enter y or n')

        else:
            get_answer= True

    return answer
