# Kelsey rawVAE_iterabledataloader Repository

## MAKE SURE YOU ARE ON THE 'CLEAN' BRANCH!!!!!!


## To Do List

- [x] Rerun Vae-windowing on erokia: print KLD; total; recon loss per batch
- [ ] Rerun VAE with no windowing on training, calculate between train loss and recon loss; windowing after recon (after loss calculation). 
  - [ ] ! We use windowing to generate the audio files only. Make a new branch for this!

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
