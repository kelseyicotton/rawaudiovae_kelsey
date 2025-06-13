<<<<<<< HEAD
# rawaudiovae

Tested with torch 2.0.1.

Check out tutorial.ipynb for testing a trained model. You can download a dataset with trained models here:
    https://drive.google.com/file/d/1e_X2Ir26iypSdSa6pRCJBy2q5t9zXFBb

Please download this folder and unzip it under the ./content folder

train.py script is for training a rawaudiovae, using a dataset folder. The dataset folder is a simple folder including a folder titled "audio", where the wav files should be.

default.ini is a config file, and hyperparameter can be adjusted using the config file.

The tutorial.ipynb uses a trained self-organizing map as well. The training of SOM can be found in another repo: https://github.com/ktatar/mlaudiosalad

This code is built within the following research residency:
https://kivanctatar.com/Coding-Latent-No-1

We have a paper on this work published at the Sound and Music Conference 2023: https://arxiv.org/pdf/2305.15571

This work was partially supported by the Wallenberg AI, Autonomous Systems and Software Program – Humanities and Society (WASP-HS) funded by the Marianne and Marcus Wallenberg Foundation and the Marcus and Amalia Wallenberg Foundation.


## Kelsey Notes

2025-06-12

- Fixed Import Error in train_iterable.py
  - ItearableAudioDataset → IterableAudioDataset (added missing 'r')
- Updated Configuration Values in kelsey_iterable.ini
  - total_num_frames: 3,086,282 frames × 500 epochs × 100 = 154,314,100,000
  - checkpoint_interval: total_num_frames ÷ 500 × 50 = 15,431,410,000
- Resolved ConfigParser Comment Issues
  - Calculation documentation to [notes] section instead
- Commenced training run on erokia dataset:
  - 12hrs: /mimer/NOBACKUP/groups/x_kelco_musai_2024/erokia/20250612_rawaudiovae-Kelsey/run-004
=======
# Kelsey rawVAE_iterabledataloader Repository

## MAKE SURE YOU ARE ON THE 'CLEAN' BRANCH!!!!!!


## To Do List

- [x] Check if Hannes windowing in audio reconstruction is integrated
  - [x] If not, integrate it
- [x] Option 0: Add 5% silence to FSD50K dataset
  - [x] silence_splitter.py for the 200hours audio file
  - [x] migrated silence files onto Alvis
  - [x] test reconstructions on KC audio
    - [ ] completely shit. find out why. 
      - [ ] audio normalization was happening in the IterableDataloader script! 
        - [x] Remove
        - [x] Retrain
  - [ ] re-integrate the interpolations! 
- [x] Finetune trained VAE on Kelsey sound library
  - [ ] attempting now!

### For Later

- [ ] Option 1: Integrate reconstruction and spectrogram loss
- [ ] Option 2: Integrate ResNet layers
- [ ] Option 3: Integrate MelGAN Approach to segment length reduction in decoder layer
- [ ] Option 4: Integrate Neural FFT layer in the network


## What is this repository about?

This VAE integrates the PyTorch Iterable Dataloader class. This streams the audiofiles in on-demandm and utilises circular file iteration with `itertools.cycle` to support training on large datasets and efficient memory usage.

I have also modified the data loading process to keep the tensors on the CPU until needed, so that we don't unecessarily burn GPU memory allocation when we are loading the data.

This model has been trained on FSD50K, with audio files licensed for Commercial Usage only.

To train:

1. Update default.ini to correspond with your file paths
2. python trainiterable.py --config ./default.ini

---------------------------------------------------------------------------------------------------------
## Acknowledgments

This features the architecture proposed by Kıvanç Tatar. This core architecture was previously published at the Sound and Music Conference 2023: https://arxiv.org/pdf/2305.15571

This work was partially supported by the Wallenberg AI, Autonomous Systems and Software Program – Humanities and Society (WASP-HS) funded by the Marianne and Marcus Wallenberg Foundation and the Marcus and Amalia Wallenberg Foundation.
>>>>>>> windowing-repo/main
