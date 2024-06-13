# Fusion Transfer Learning (FTL)
![png](demo/FTL_logo.png=200x)
FTL [1] model provides a new paradigm to study high-dimensional dynamical behaviors, such as those in fusion plasma systems. The knowledge transfer process leverages a pre-trained neural encoder-decoder network, initially trained on linear simulations, to effectively capture nonlinear dynamics. The low-dimensional embeddings extract the coherent structures of interest, while preserving the inherent dynamics of the complex system. Experimental results highlight FTL's capacity to capture transitional behaviors and dynamical features in plasma dynamics -- a task often challenging for conventional methods. The model developed in this study is generalizable and can be extended broadly through transfer learning to address various magnetohydrodynamics (MHD) modes.

The publication "FTL: Transfer Learning Nonlinear Plasma Dynamic Transitions in Low Dimensional Embeddings via Deep Neural Networks" by Z. Bai, X. Wei, W. Tang, L. Oliker, Z. Lin and S. Williams is available on [arXiv](https://arxiv.org/abs/2404.17466).

## Installation

1. Clone this repository to your local machine.
2. Add path to `FTL/src/` folder to Python search path using `sys.path.append('<path to mds>/FTL/src')`.

## Dependencies

* Numpy, Pytorch.
* Environment: Python or Jupyter notebook.
* Mac OSX, linux and Windows.

## Getting started

See `demo.ipynb` for demonstrating the approach on leveraging trained ML model to efficiently reconstruct kink modes through fine tuning. The execution of this file in Python/Jupyter illustrations the convergence and reconstruction error over training epochs using transfer learning.

## License (modified BSD license)

See the [LICENSE file](LICENSE) for details.

## References

[1] Z. Bai, X. Wei, W. Tang, L. Oliker, Z. Lin, S. Williams, FTL: Transfer Learning Nonlinear Plasma Dynamic Transitions in Low Dimensional Embeddings via Deep Neural Networks, arXiv Preprint, arxiv.org/abs/2404.17466, 2024.<br />  

