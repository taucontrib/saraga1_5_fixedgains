from torch.utils.data import Dataset, DataLoader
import mirdata
import torch
import pandas as pd
from typing import List, Dict, Tuple
import librosa
import numpy as np
EPS = np.finfo(float).eps
import random
import math
import os

class SaragaFixedGainsDataset(Dataset):
    
    def __init__(self, 
                 path_to_saraga: str, 
                 chunk_size: int = 142290,
                 sr: int = 24000,
                 instrumentals_to_mix: str = ['audio_mridangam_left_path', 'audio_mridangam_right_path','audio_violin_path'],
                 consider_vocal_probabilities: bool = False,
                 essentia_vocal_model: str = 'vgg',
                 vocal_threshold: float = 0.8,
                 snr_lower: float = -5.0,       
                 snr_upper: float =  5.0, 
                 total_snr_levels: int = 11,
                ):
        """ Initialize Saraga 1.5 dataset

        Args:
            path_to_saraga (str): path to the dataset
            chunk_size (int, optional): sample size in samples - set to 142290 for MDX model with sr=24000. Defaults to 142290.
            sr (int, optional): Samplerate. Defaults to 24000.
            instrumentals_to_mix (str, optional): List of instrumentals to mix. Defaults to ['audio_mridangam_left_path', 'audio_mridangam_right_path','audio_violin_path'].
            consider_vocal_probabilities (bool, optional): If true, the loader checks the vocal probabilities and mutes parts with prob<vocal_threshold. Defaults to False.
            essentia_vocal_model (str, optional): Model for vocal probability - either 'vgg' or 'yamnet'. Defaults to 'vgg'.
            vocal_threshold (float, optional): Muting threshold. Defaults to 0.8.
            snr_lower (float, optional): Ratio of vocal and instrumental in dB - randomly sampled between snr_lower and snr_upper. Defaults to -5.0.
            snr_upper (float, optional): Ratio of vocal and instrumental in dB - randomly sampled between snr_lower and snr_upper. Defaults to 5.0.
            total_snr_levels (int, optional): Number of possible SNR values. Defaults to 11.
        """
        
        saraga = mirdata.initialize('saraga_carnatic', data_home=path_to_saraga)
        self.instrumentals_to_mix = instrumentals_to_mix
        self.snr_list = []
        first_snr = float(snr_lower)
        for _ in range(total_snr_levels):
            self.snr_list.append(first_snr)
            first_snr = first_snr + float((snr_upper - snr_lower)/(total_snr_levels-1))
        
        self.vocal_threshold = vocal_threshold
        self.chunk_size = chunk_size
        self.sr = sr
        self.consider_vocal_probabilities = consider_vocal_probabilities
        self.essentia_vocal_model = essentia_vocal_model
        
        # Filter dataset - only use samples with the necessary files stored at 'audio_vocal_path', 'audio_violin_path', etc 
        self.track_dict = {}
        instrumentals_to_mix.append('audio_vocal_path')
        index = 0
        for _, track_information in saraga.load_tracks().items():
            paths = [getattr(track_information, track_name) for track_name in instrumentals_to_mix]
            # check if track has the correct paths
            if not None in paths:
                # check if the files exist
                if not False in [os.path.exists(path) for path in paths]:
                    
                    if self.consider_vocal_probabilities:
                        # load the vocal probabilities for the vocal track
                        vocal_path = getattr(track_information, 'audio_vocal_path')
                        vocal_prob_path = vocal_path.split('multitrack-vocal')[0] + 'vocalprobs.csv'
                        df = pd.read_csv(vocal_prob_path)
                        if self.essentia_vocal_model == 'yamnet':
                            setattr(track_information, 'vocal_probabilities', list(df['vocal_prediction_yamnet']))
                        elif self.essentia_vocal_model == 'vgg':
                            setattr(track_information, 'vocal_probabilities', list(df['vocal_prediction_vgg']))
                        else:
                            print('The attribute essentia_vocal_model has to be "vgg" or "yamnet".')
                    
                    self.track_dict[index] = track_information
                    index = index + 1
        
        print("Found {} samples with existing paths for the tracks: {}".format(len(self.track_dict), instrumentals_to_mix))
        
    def __len__(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        return len(self.track_dict)
    
    
    def __getitem__(self, index):
        """
        Overwrite the __getitem__ method for the saraga dataset with fixed gains

        1. Sample a random starting index for the audio
        2. Load vocal audio
        3. Create instrumental by summing up the different instrumenrals
        4. Create Mix with randomly sampled SNR value (see method snr_mixer)
        Args:
            index (_type_): _description_
        """

        # load vocal track
        vocal = self.load_audio(getattr(self.track_dict[index], 'audio_vocal_path'))
        # choose random offset - NOTE to speed things up for now with the vocal probabilities (one prob per second), a random integer offset in seconds is sampled
        # choose random start index
        offset = random.randint(0, int(math.floor((vocal.shape[0] - self.chunk_size)/self.sr)))
        vocal = vocal[offset*self.sr: offset*self.sr + self.chunk_size]
        duration = self.chunk_size / self.sr
        
        # Silence the parts with low vocal probability        
        vocal = self.silence_vocal(index, offset, duration, vocal)
        
        # load instrumental
        instrumental = self.load_instrumental(index, offset, duration)
        snr = self.get_random_snr()
        vocal, instrumental = self.snr_mixer(vocal, instrumental, snr) 
        return {'vocal': vocal, 'mix': np.sum([vocal, instrumental], axis = 0), 'instrumental': instrumental, 'snr': snr}
    
    def get_random_snr(self):
        """ Sample a random snr value from the snr_list

        Returns:
            float: Random SNR value
        """
        snr = random.sample(self.snr_list,1)[0]
        return snr

    def silence_vocal(self, index: int, offset: int, duration: float, vocal: np.ndarray) -> np.ndarray:
        
        if self.consider_vocal_probabilities:
            vocal_probabilities = self.track_dict[index].vocal_probabilities[offset:offset+int(math.ceil(duration))]
            for i, prob in enumerate(vocal_probabilities):
                if prob < self.vocal_threshold:
                    #print('Silenced the vocal from second {} to second {}.'.format(i, i+1))
                    silence_length = min(vocal.shape[0] - i*self.sr , self.sr)
                    vocal[i*self.sr:(i+1)*self.sr] = np.zeros(silence_length)

        return vocal
    
    def load_instrumental(self, index: int, offset: int, duration: float):
        """Load the instrumental tracks and sum them up

        Args:
            index (int): Dataset index in [0, len(self.track_dict()]
            offset (float): Offset in seconds
            duration (float): Duration of the sample in seconds

        Returns:
            np.ndarray: instrumental
        """
        instrumentals = []
        for track in self.instrumentals_to_mix:
            instrumental = self.load_audio(getattr(self.track_dict[index],track), offset = offset, duration = duration)
            instrumentals.append(instrumental)
        
        instrumental = np.sum(instrumentals, axis = 0)
        if self.is_clipped(instrumental):
            instrumental = instrumental / (np.max(instrumental)) 
        
        return np.sum(instrumentals, axis = 0)
          
    
    def load_audio(self, path, offset = 0, duration = None):
        audio, sr = librosa.load(path, sr=self.sr, offset = offset, duration = duration)
        # mix stereo to mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, dim = 0)
        return audio
        
    def snr_mixer(self, clean: np.ndarray, noise: np.ndarray, snr: float, target_level: int=-25, target_level_lower: int = -35, target_level_upper: int = -15) -> Tuple[np.ndarray, np.ndarray]:
        """ Method for mixing to signal with a given Signal to noise ratio
        
        Following the code from the DNS challenge dataset https://github.com/microsoft/DNS-Challenge
        
        For the carnatic dataset, the clean signal is the vocal, the noise is the instrumental.
        The method returns a tuple with the vocal and the instrumental. The mix (vocal+instrumental) loudness in dB is randomly sampled from [target_level_lower, target_level_upper].

        Args:
            clean (np.ndarray): vocal signal
            noise (np.ndarray): noise signal - in this dataloader, it's the instrumental
            snr (float): signal to noise ratio in db - in this case, it's vocal to instrumental ratio
            target_level (int, optional): Loudness in decibel for the initial normalization of both signals. Defaults to -25.
            target_level_lower (int, optional): Lower bound for sampling a db value for the loudness of the returned mix. Defaults to -35.
            target_level_upper (int, optional): Upper bound for sampling a db value for the loudness of the returned mix. Defaults to -15.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Vocal and instrumental. To receive the mx, both signals have to be added up
        """

        clipping_threshold = 0.99
        # Normalizing to -25 dB FS
        clean = clean/max(abs(clean) + EPS)
        clean = self.normalize(clean, target_level)
        rmsclean = (clean**2).mean()**0.5
        
        noise = noise/(max(abs(noise))+EPS)
        noise = self.normalize(noise, target_level)
        rmsnoise = (noise**2).mean()**0.5
        # Set the noise level for a given SNR
        noisescalar = rmsclean / (10**(snr/20)) / (rmsnoise+EPS)
        noisenewlevel = noise * noisescalar
        # Mix noise and clean speech
        noisyspeech = clean + noisenewlevel
        # Randomly select RMS value between -15 dBFS and -35 dBFS and normalize noisyspeech with that value
        # There is a chance of clipping that might happen with very less probability, which is not a major issue. 
        noisy_rms_level = np.random.randint(target_level_lower, target_level_upper)
        rmsnoisy = (noisyspeech**2).mean()**0.5
        scalarnoisy = 10 ** (noisy_rms_level / 20) / (rmsnoisy+EPS)
        noisyspeech = noisyspeech * scalarnoisy
        clean = clean * scalarnoisy
        noisenewlevel = noisenewlevel * scalarnoisy
        # Final check to see if there are any amplitudes exceeding +/- 1. If so, normalize all the signals accordingly
        if self.is_clipped(noisyspeech):
            noisyspeech_maxamplevel = max(max(abs(noisyspeech))/(clipping_threshold), max(abs(noisenewlevel))/(clipping_threshold))
            noisyspeech = noisyspeech/noisyspeech_maxamplevel
            clean = clean/noisyspeech_maxamplevel
            noisenewlevel = noisenewlevel/noisyspeech_maxamplevel
            noisy_rms_level = int(20*np.log10(scalarnoisy/noisyspeech_maxamplevel*(rmsnoisy+EPS)))
        
        # changed the return types here to clean, noise - but, normalization happens on the sum -> also when beeing added in the model, it will not clip
        return clean, noisenewlevel #, noisyspeech
    
    def normalize(self, audio, target_level=-25):
        '''Normalize the signal to the target level'''
        rms = (audio ** 2).mean() ** 0.5
        scalar = 10 ** (target_level / 20) / (rms+EPS)
        audio = audio * scalar
        return audio
    
    def is_clipped(self, audio, clipping_threshold=0.99):
        return any(abs(audio) > clipping_threshold)