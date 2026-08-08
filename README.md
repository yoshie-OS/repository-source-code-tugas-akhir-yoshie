# This Repository
This repository is meant to contain research data for a Bachelor's thesis titled "Cross-Domain Evaluation of Emotion Classification in Tweets About the 2025 Citra Film Awards" written by Yoshie Pranedya Adhipratama and co-authored/supervised by Bunyamin and Hasmawati.

# The Research
The thesis aims to train and evaluate IndoBERT for a zero-shot cross-domain emotion classification task in Indonesian. There are two defined domain within this research:
## General Domain
This domain involves tweets about various topics. The dataset is a combination of two datasets by Riccosan et al. and Saputri et al. Their datasets can be obtained through these links:
1. Saputri et al. (original repository): https://github.com/meisaputri21/Indonesian-Twitter-Emotion-Dataset.git
2. Saputri et al. (adapted by Wilie et al. for IndoNLU as IndoNLU-Emot): https://github.com/IndoNLP/indonlu.git
3. Riccosan et al. (original repository): https://github.com/Ricco48/Emotion-Dataset-from-Indonesian-Public-Opinion.git
The combined dataset is only used to train and validate the model during the search for optimal learning-rate and epoch before the model is reset and trained with the entire combined dataset. It is done like so because the objective is to evaluate the model's performance when tested using a dataset with different domain.
## Film Domain
The film domain only contains one dataset which was obtained through data mining Twitter. This domain mainly captures tweets about films nominated for the 2025 Citra Film Awards, but it is referred to as "film domain" for simplicity.