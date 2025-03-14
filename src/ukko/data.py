import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from datetime import datetime

class SineWaveDataset(Dataset):
    def __init__(self, n_samples, n_features, sequence_length, prediction_length=5,
                 base_freq=0.1, noise_level=0.1, seed=42):
        """
        Creates sine wave dataset with different frequencies, phases, and amplitudes for each feature

        Args:
            n_samples: Number of samples in dataset
            n_features: Number of features
            sequence_length: Length of input sequence
            prediction_length: Length of sequence to predict
            base_freq: Base frequency of sine waves
            noise_level: Standard deviation of Gaussian noise
        
        Attributes:
            data (torch.Tensor): The whole time-series. 
                A sample is a tuple:
                    x: the first part up to sequence_length
                    y: le last part at prediction_length
                Thus, the dimention is [n_samples, n_features, sequence_length + prediction_length]
            groundtruth (torch.Tensor)
            
        Returns:
            SineWaveDataset
        """
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.n_features = n_features
        self.base_freq = base_freq
        self.noise_level = noise_level
        self.freq_array = []

        # Create time points
        t = np.linspace(0, (sequence_length + prediction_length) * 2 * np.pi,
                       sequence_length + prediction_length)
        
        #Random frequency, phase, and amplitude for each feature
        phase_array = np.random.uniform(0, 2 * np.pi, n_features)
        amplitude_array = np.random.uniform(0.5, 2.0, n_features)
        freq_array = np.random.uniform(0.5, 2.0, n_features)
        self.freq_array = freq_array
        #print(freq_array)

        # Generate data for each sample
        data = []
        groundtruth = []
        for sampleidx in range(n_samples):
            sample = []
            groundtruthsample = []
            #print(f"Sample {sampleidx}:")
            for f in range(n_features):
                phase = phase_array[f]
                amplitude = amplitude_array[f]
                freq = freq_array[f]
                #print(f"  Feature {f}:")
                #print('  ', f, freq)

                if f==0:
                    # make phase and amplitude random for each sample
                    phase = np.random.uniform(0, 2 * np.pi)
                    amplitude = np.random.uniform(0.5, 2.0)
                elif f==1:
                    # make phase random
                    phase = np.random.uniform(0, 2 * np.pi)
                elif f==2:
                    # make amplitude random
                    amplitude = np.random.uniform(0.5, 2.0)
                else:
                    # keep phase, amp, freq constant between samples. 
                    pass

                # Generate sine wave with noise
                sine_wave = amplitude * np.sin(freq * base_freq * t + phase)
                noise = np.random.normal(0, noise_level, len(t))
                feature_data = sine_wave + noise

                sample.append(feature_data)
                groundtruthsample.append(sine_wave)
            data.append(sample)
            groundtruth.append(groundtruthsample)

        # Convert to torch tensors
        self.data = torch.FloatTensor(data)  # [n_samples, n_features, sequence_length + prediction_length]
        self.groundtruth = torch.FloatTensor(groundtruth)

    def __len__(self):
        return len(self.data)

    # data with noise
    def __getitem__(self, idx):
        x = self.data[idx, :, :self.sequence_length]
        #y = self.data[idx, :, self.sequence_length:self.sequence_length + self.prediction_length]
        y = self.data[idx, :, self.sequence_length + self.prediction_length -1]
        return x, y

    # groundtruth data:
    def __getgtitem__(self, idx):
        x = self.groundtruth[idx, :, :self.sequence_length]
        y = self.groundtruth[idx, :, self.sequence_length:self.sequence_length + self.prediction_length]
        return x, y

    def Subset(self, indices):
        """Create a subset of the dataset."""
        
        # Init the object by generating a dataset 
        subset_dataset = SineWaveDataset(
            n_samples=len(indices),
            n_features=self.n_features,
            sequence_length=self.sequence_length,
            prediction_length=self.prediction_length,
            base_freq=self.base_freq,
            noise_level=self.noise_level
        )
        
        # Override the generated data with subset of current data
        subset_dataset.data = self.data[indices]
        subset_dataset.groundtruth = self.groundtruth[indices]
        
        return subset_dataset
        
    def getdim(self):
        """
        Get dimentions of the dataset.

        Returns:
            n_samples, sequence_length, prediction_length
        """
        # Get total samples
        n_samples = len(self)
        sample_x, sample_y = self[0]
        
        return n_samples, self.n_features, self.sequence_length, self.prediction_length

        

def plot_example_dataset(dataset, sample_idx=0, feature_idx=0):
    """Plot an example from the dataset with markers"""
    x, y = dataset[sample_idx]
    xgt, ygt = dataset.__getgtitem__(sample_idx)

    fig = plt.figure(figsize=(15, 2))

    # Plot data sample with noise
    # Plot input sequence with markers
    plt.plot(range(len(x[feature_idx])), x[feature_idx],
             'o', label='Input sequence', color='blue',
             markersize=4, markerfacecolor='white', markeredgewidth=1)

    # Plot target sequence with different markers
    #plt.plot(range(len(x[feature_idx]), len(x[feature_idx]) + len(y[feature_idx])),
    #         y[feature_idx], 's', label='Target sequence', color='red',
    #         markersize=6, markerfacecolor='white', markeredgewidth=1)
    plt.plot(len(x[feature_idx]) + dataset.prediction_length - 1,
             y[feature_idx], 's', label='Target sequence', color='blue',
             markersize=6, markerfacecolor='white', markeredgewidth=1)

    # Plot groundtruth wave
    xygt = torch.cat((xgt, ygt), dim=1)[feature_idx]
    plt.plot(range(len(xygt)), xygt,
             '-', label='Ground truth', color='gray',
             markersize=0, markerfacecolor='white', markeredgewidth=1)


    plt.title(f'Example Sine Wave - Feature {feature_idx}')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    # Add timestamp and user info
    timestamp = "2025-03-01 14:35:23"
    user = "eric-fey-hus"
    plt.text(0.02, 0.02, f'Generated: {timestamp}\nUser: {user}',
             transform=plt.gca().transAxes, fontsize=8,
             bbox=dict(facecolor='white', alpha=0.8))

    #plt.show()
    return fig