# Kelsey rawVAE_iterabledataloader Repository

This is the repository for Kelsey's dev work.

## To Do List

### 20250612

- Complete overhaul of entire codebase, put back in line with ktatar original codebase and using the new train_iterable and dataset scripts with the IterableDataloader. Everything is now on the 'clean' branch
- Implemented windowing in the IterableAudioDataset class as a preprocessing step. This occurs after segment generation
- We also precompute windows and apply windowing in TestDataset
- Everything has been optimized for all computation to occur on GPU
- All prints get logged to console file

### 20250602

- [ ] MY VAE
  - [ ] Standard100epoch: /mimer/NOBACKUP/groups/x_kelco_musai_2024/erokia/2025-06-02_rawVAE_standarddataloader_erokia_workstation/run-000
  - [ ] Stream100epoch: /mimer/NOBACKUP/groups/x_kelco_musai_2024/erokia/2025-06-02_rawVAE_streamdataloader_erokia_workstation/run-000
- [ ] TATAR VAE
  - [ ] Standard100epoch:
  - [ ] Standard100epoch:
    - [ ] lr variations:
      - [ ] lr:0.0001:
      - [ ] lr:0.00002: /mimer/NOBACKUP/groups/x_kelco_musai_2024/erokia/2025-06-02_tatarrawaudiovae_originaldataloader_loggingKLD-recon_lr0.00005_erokia_workstation/run-000
      - [ ] lr:0.00005: /mimer/NOBACKUP/groups/x_kelco_musai_2024/erokia/2025-06-02_tatarrawaudiovae_originaldataloader_loggingKLD-recon_lr0.00005_erokia_workstation/run-001
      - [ ] lr:0.00001: /mimer/NOBACKUP/groups/x_kelco_musai_2024/erokia/2025-06-02_tatarrawaudiovae_originaldataloader_loggingKLD-recon_lr0.00001_erokia_workstation/run-000

### 20250527

- [ ] 200epoch_standard:
  /mimer/NOBACKUP/groups/x_kelco_musai_2024/erokia/2025-05-27_rawVAE_standarddataloader_erokia_workstation/run-000/tensorboard-logging_20250527-145406
- [ ] 200epoch_stream:

I'm looking into the mismatch between the tensorboard logs and the config file: it was showing 60 epochs on tensorboard although I trained for 200. Revised training script so I create 1 SummaryWriter instance, and ensure all logging calls are written to that file. Also integrated flushing every epoch to ensure that the data is being written. Tensorboard file will only close when training is done, or if there is an error.

Also, I removed the batch limiter I had in the training script.

Additional changes:

- Created separate training scripts for standard and streaming implementations:
  - `train_standard.py`: Uses standard PyTorch DataLoader with fixed dataset
  - `train_stream.py`: Uses IterableDataset with on-demand loading
- Implemented segment counting to ensure fair comparison:
  - Both implementations process ~3.086M segments per epoch
  - Standard: Exactly 3,086,282 segments
  - Streaming: 3,086,336 segments (slight difference due to batch completion)
- Created separate test scripts for each implementation:
  - `tests_standard.py`: Handles reconstruction for standard dataset
  - `tests_stream.py`: Handles reconstruction for streaming dataset
- Performance comparison:
  - Standard: Faster training (~27s/epoch) but higher memory usage
  - Streaming: Slower training (~188s/epoch) but more memory efficient
  - Both implementations show similar loss convergence patterns

### 2025-05-23

- [X] retrain rawVAE with IterableDataloader on OG erokia dataset

  - [X] /mimer/NOBACKUP/groups/x_kelco_musai_2024/erokia/2025-05-22_rawVAE_iterable_erokia_workstation/run-000
- [X] retrain rawVAE original architecture on OG erokia dataset (forgot the KLD and recon logging!)

  - [X] /mimer/NOBACKUP/groups/x_kelco_musai_2024/erokia/2025-05-23_tatarrawaudiovae_originaldataloader_loggingKLD-recon_erokia_workstation/run-001
- [X] compare KLD vs recon loss on both version 1 and version 2

### 2025-05-06

SHE INTERPOLATES! YASSSS

- [X] integrate interpolation function at checkpoints

### 2025-05-05

- [X] Kelsey simplify IterableDataloader script
  - [X] remove file logging
  - [X] move all window calculations directly to GPU
  - [X] print recon and KLD loss
    - [X] add to tensorboard logging
    - [X] then retrain
  - [X] reintegrate interpolations calculations at checkpoints using the checkpointed model
- [X] There is something strange going on in the FSD50K model trained with additional silence in training set; without audio normalization
  - [X] Re-integrated audio normalization in pre-processing step and immediate improvement to quality of audio reconstructions
  - [X] Also increased batch_size to 2048
  - [ ] Integrate a validation set? Using the filter non-commercial audio could be an option
  - [ ] Double check the loss: are we still seeing a spike around 100 epochs?

### Previous

- [X] Check if Hannes windowing in audio reconstruction is integrated
  - [X] If not, integrate it
- [X] Option 0: Add 5% silence to FSD50K dataset
  - [X] silence_splitter.py for the 200hours audio file
  - [X] migrated silence files onto Alvis
  - [X] test reconstructions on KC audio
    - [X] completely shit. find out why.
      - [X] audio normalization was happening in the IterableDataloader script!
        - [X] Remove
        - [X] Retrain
        - [X] Sounds shit still- continual drones
          - [X] add normalization back in?
            - [X] done. magically better
  - [X] re-integrate the interpolations!
- [X] Finetune trained VAE on Kelsey sound library
  - [X] attempting now!
    - [X] sounds ewwww: lots of bugs

### For Later

- [ ] Option 1: Integrate reconstruction and spectrogram loss
- [ ] Option 2: Integrate ResNet layers
- [ ] Option 3: Integrate MelGAN Approach to segment length reduction in decoder layer
- [ ] Option 4: Integrate Neural FFT layer in the network

## What is this repository about?

This VAE integrates the PyTorch Iterable Dataloader class. This streams the audiofiles in on-demandm and utilises circular file iteration with `itertools.cycle` to support training on large datasets and efficient memory usage.

I have also modified the data loading process to keep the tensors on the CPU until needed, so that we don't unecessarily burn GPU memory allocation when we are loading the data.

### Dataset

This model has been trained on :

"Erokia - Electronic Samples Misc (CC0)"

This Pack of sounds contains sounds by the following user:

- Erokia ( https://freesound.org/people/Erokia/ )

You can find this pack online at: https://freesound.org/people/Erokia/packs/26717/

Licenses in this Pack (see below for individual sound licenses)

This pack of sounds contains sounds by the following user:

- Erokia ( https://freesound.org/people/Erokia/ )

You can find this pack online at: https://freesound.org/people/Erokia/packs/26656/

This model has been trained on FSD50K, with audio files licensed for Commercial Usage only.

License details
---------------

Attribution: http://creativecommons.org/licenses/by/3.0/

To train:

1. Update default.ini to correspond with your file paths
2. python trainiterable.py --config ./default.ini

---

## Acknowledgments

This features the architecture proposed by Kıvanç Tatar. This core architecture was previously published at the Sound and Music Conference 2023: https://arxiv.org/pdf/2305.15571

This work was partially supported by the Wallenberg AI, Autonomous Systems and Software Program – Humanities and Society (WASP-HS) funded by the Marianne and Marcus Wallenberg Foundation and the Marcus and Amalia Wallenberg Foundation.
