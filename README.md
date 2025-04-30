# Kelsey rawVAE_iterabledataloader Repository


This is the repository for Kelsey's dev work.

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
        - [ ] Retrain
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