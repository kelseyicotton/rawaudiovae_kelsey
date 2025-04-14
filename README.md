# Kelsey rawVAE_iterable Repository


This is the repository for Kelsey's dev work.

This VAE integrates the PyTorch Iterable Dataloader class. This streams the audiofiles in on-demandm and utilises circular file iteration with `itertools.cycle` to support training on large datasets and efficient memory usage.

I have also modified the data loading process to keep the tensors on the CPU until needed, so that we don't unecessarily burn GPU memory allocation when we are loading the data.

This model has been trained on FSD50K, with audio files licensed for Commercial Usage only.

To train:

1. Update default.ini to correspond with your file paths
2. python trainiterable.py --config ./default.ini

---------------------------------------------------------------------------------------------------------
This features the architecture proposed by Kıvanç Tatar. This core architecture was previously published at the Sound and Music Conference 2023: https://arxiv.org/pdf/2305.15571

This work was partially supported by the Wallenberg AI, Autonomous Systems and Software Program – Humanities and Society (WASP-HS) funded by the Marianne and Marcus Wallenberg Foundation and the Marcus and Amalia Wallenberg Foundation.