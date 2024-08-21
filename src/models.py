import torch
import torch.nn as nn
import torchaudio


class Hubert(nn.Module):

    def __init__(self, input_time, num_frames, num_classes, verbose=False, freeze=True):
        super(Hubert, self).__init__()
        '''
        Hubert model class

        Args:
            input_time (int): number of samples in the audio
            num_frames (int): number of audio frames in the output
            num_classes (int): number of classes in the output
            verbose (bool): verbose mode
            freeze (bool): freeze the model

        Returns:
            torch.Tensor: output tensor of shape (batch_size, num_frames, num_classes)
        '''

        bundle = torchaudio.pipelines.HUBERT_LARGE
        self.hubert = bundle.get_model()
        self.time = input_time
        self.verbose = verbose

        # freeze the model
        if freeze:
            for param in self.hubert.parameters():
                param.requires_grad = False

        self.linear1 = nn.Linear(self.get_hubert_out(), num_frames)
        self.layernorm1 = nn.LayerNorm(num_frames)
        self.linear2 = nn.Linear(1024, num_classes)

    def get_hubert_out(self):
        dummy_input = torch.randn(1, self.time)
        dummy_input = self.hubert(dummy_input)
        return dummy_input[0].shape[1]

    def forward(self, x):

        x = self.hubert(x)  # (batch_size, seq_length, d_model)
        x = x[0]

        if self.verbose:
            print('Hubert out: ', x.shape)

        x= x.permute(0, 2, 1)

        x = self.linear1(x)  # (batch_size, d_model, num_frames)

        if self.verbose:
            print('Linear1 out: ', x.shape)

        x = self.layernorm1(x)  # (batch_size, d_model, num_frames)

        if self.verbose:
            print('LayerNorm1 out: ', x.shape)

        x=x.permute(0, 2, 1)

        x = self.linear2(x)  # (batch_size, num_frames, 3)

        if self.verbose:
            print('Linear2 out: ', x.shape)

        return x
    


class Wav2Vec(nn.Module):

    def __init__(self, input_time, num_frames, num_classes, verbose=False, freeze=True):
        super(Wav2Vec, self).__init__()
        '''
        Wav2Vec model class

        Args:
            input_time (int): number of samples in the audio
            num_frames (int): number of audio frames in the output
            num_classes (int): number of classes in the output
            verbose (bool): verbose mode
            freeze (bool): freeze the model

        Returns:
            torch.Tensor: output tensor of shape (batch_size, num_frames, num_classes)
        '''

        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.wav2vec = bundle.get_model()
        self.time = input_time
        self.verbose = verbose

        # freeze the model
        if freeze:
            for param in self.wav2vec.parameters():
                param.requires_grad = False

        self.linear1 = nn.Linear(self.get_w2v_out(), num_frames)
        self.layernorm1 = nn.LayerNorm(num_frames)
        self.linear2 = nn.Linear(768, num_classes)

    def get_w2v_out(self):
        dummy_input = torch.randn(1, self.time)
        dummy_input = self.wav2vec(dummy_input)
        return dummy_input[0].shape[1]

    def forward(self, x):

        x = self.wav2vec(x)  # (batch_size, seq_length, 768)
        x = x[0]

        if self.verbose:
            print('W2V out: ', x.shape)

        x= x.permute(0, 2, 1)

        x = self.linear1(x)  # (batch_size, 768, num_frames)

        if self.verbose:
            print('Linear1 out: ', x.shape)

        x = self.layernorm1(x)  # (batch_size, 768, num_frames)

        if self.verbose:
            print('LayerNorm1 out: ', x.shape)

        x=x.permute(0, 2, 1)

        x = self.linear2(x)  # (batch_size, num_frames, 3)

        if self.verbose:
            print('Linear2 out: ', x.shape)

        return x

class CNN(nn.Module):
    def __init__(self, input_time, num_frames, num_classes, verbose=False):
        super(CNN, self).__init__()
        
        self.verbose = verbose
        self.time = input_time

        self.convblock1 = nn.Sequential(nn.Conv1d(1, 16, kernel_size=15, stride=1),
                                        nn.BatchNorm1d(16),
                                        nn.ReLU(),
                                        nn.MaxPool1d(kernel_size=3, stride=2),

                                        nn.Conv1d(16, 32, kernel_size=3, stride=1),
                                        nn.BatchNorm1d(32),
                                        nn.ReLU(),
                                        nn.MaxPool1d(kernel_size=3, stride=2),

                                        nn.Conv1d(32, 64, kernel_size=3, stride=1),
                                        nn.BatchNorm1d(64),
                                        nn.ReLU(),
                                        nn.MaxPool1d(kernel_size=3, stride=2),

                                        nn.Conv1d(64, 128, kernel_size=3, stride=1),
                                        nn.BatchNorm1d(128),
                                        nn.ReLU(),
                                        nn.MaxPool1d(kernel_size=3, stride=2)
                                        )
        
        self.fc= nn.Linear(128, num_frames )
        self.fc1 = nn.Linear(self.get_cnn_out(), num_classes)

    def get_cnn_out(self):
        dummy_input = torch.randn(1, 1, self.time)
        dummy_input = self.convblock1(dummy_input)
        return dummy_input.shape[2]
        
    def forward(self, x):
        x = x.unsqueeze(1)

        if self.verbose:
            print('Input shape: ', x.shape)

        x = self.convblock1(x)

        if self.verbose:
            print('CNN out shape: ', x.shape)

        x= x.transpose(1,2)

        if self.verbose:
            print('CNN out transposed shape: ', x.shape)

        x = self.fc(x)

        if self.verbose:
            print('FC out shape: ', x.shape)

        x= x.transpose(1,2)

        if self.verbose:
            print('FC out transposed shape: ', x.shape)

        x = self.fc1(x)

        if self.verbose:
            print('FC1 out shape: ', x.shape)
            
        return x

class CRNN(nn.Module):
    def __init__(self, input_time, num_frames, num_classes, verbose=False):
        super(CRNN, self).__init__()

        self.verbose = verbose
        self.time = input_time
        
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=11, stride=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2),
            nn.ReLU(),
        )

        self.gru = nn.GRU(self.get_cnn_out(), 80, batch_first=True, bidirectional=True, num_layers=1)
        self.gru2 = nn.GRU(160, 40, batch_first=True, bidirectional=True, num_layers=1)

        self.fc1 = nn.Linear(128, num_frames)
        self.fc2 = nn.Linear(80, num_classes) 

    def get_cnn_out(self):
        dummy_input = torch.randn(1, 1, self.time)
        dummy_input = self.cnn(dummy_input)
        return dummy_input.shape[-1]
    
    def forward(self, x):
        x = x.unsqueeze(1)

        if self.verbose:
            print('Input shape: ', x.shape)

        x = self.cnn(x)

        if self.verbose:
            print('CNN out shape: ', x.shape)

        x, hs = self.gru(x)

        if self.verbose:
            print('GRU out shape: ', x.shape)
            print('Hidden state shape: ', hs.shape)

        x, hs = self.gru2(x)

        if self.verbose:
            print('GRU2 out shape: ', x.shape)
            print('Hidden state shape: ', hs.shape)

        x = x.transpose(1,2)

        if self.verbose:
            print('GRU2 out transposed shape: ', x.shape)

        x = self.fc1(x)

        if self.verbose:
            print('FC1 out shape: ', x.shape)

        x = x.transpose(1,2)

        if self.verbose:
            print('FC1 out transposed shape: ', x.shape)

        x = self.fc2(x)

        if self.verbose:
            print('FC2 out shape: ', x.shape)
            
        return x
