# convolutional-audio-transformer

#### Compact Convolutional Transformers for Environmental Sound Classification

This project was done for the course "Human Data Analytics" at the University of Padova (winter semester 2023/24). 
The original code was written using Tensorflow/Keras (requirement of the course), this code is a rewrite using Pytorch instead of Tensorflow. 

A [Compact Convolutional Transformer (CCT)](https://arxiv.org/pdf/2104.05704) is trained on the [ESC-50](https://dl.acm.org/doi/10.1145/2733373.2806390) dataset. As audio representations, mel-spectrograms are used. Additionally various augmentation techniques are applied to prevent overfitting as the ESC-50 dataset is relatively small (1600 audio samples for the train folds, 5 seconds each). 
The transformer encoder uses pre-layernorm and learnable positional embeddings. 

Augmentation Techniques used:
- Time shifting (Raw Audio)
- Background Noise (Raw Audio)
- Mixup (Raw Audio)
- Frequency Masking (Mel-spectrograms)
- Time Masking (Mel-spectrograms)



It acchives a average accuracy of about 82% using the default parameters. 