[README_self_modeling.md](https://github.com/user-attachments/files/30621545/README_self_modeling.md)
# reproducing-graziano-self-modeling

A from-scratch reproduction of Figure 2A from *Unexpected Benefits of Self-Modeling in Neural Systems* (Premakumar et al., 2024, [arXiv:2407.10188](https://arxiv.org/abs/2407.10188)).

## Goal

The goal of this project was to reproduce a published result from scratch, working only from the paper's description. I picked Figure 2A, set up the model and training sweep myself, and checked whether I could recover the same trend the authors report.

The exercise is as much about the process as the plot: reading a methods section closely enough to rebuild it, catching the small details that matter (like which weights the metric is actually measured on), running a proper multi-seed experiment, and confirming the output matches the original. Figure 2A tracks how a network's weight distribution narrows over training on MNIST as more weight is placed on a self-modeling auxiliary task, and the aim here was to make that curve fall out of my own implementation rather than take it on faith.

## What the notebook does

`self_modeling_fig2a.ipynb` runs top to bottom:

1. Load and preprocess MNIST.  
2. Define a two-layer MLP with an optional self-modeling head that predicts the hidden layer's own activations.  
3. Define the complexity metric: the standard deviation of the classification weights, measured only on the classification sub-block (self-modeling outputs are pruned first, matching the paper).  
4. Train with a combined loss (cross-entropy plus a weighted self-modeling MSE term), tracking the weight standard deviation each epoch.  
5. Run the full experiment: a baseline plus five auxiliary-weight settings (1, 5, 10, 20, 50), ten seeds each, fifty epochs.  
6. Plot the mean weight standard deviation per condition with 95% confidence bands.

## Result

The reproduction recovers the paper's trend: heavier weight on the self-modeling task drives the weight distribution narrower over training, confirming the self-regularizing effect.

## Running it

pip install torch numpy scipy matplotlib datasets transformers accelerate

jupyter notebook self\_modeling\_fig2a.ipynb

Run the cells in order. The full sweep is sixty training runs, so results are cached to disk after the first run and reloaded on later runs.

## Tech stack

Python, PyTorch, NumPy, SciPy, matplotlib, Hugging Face datasets  
