# mini-torch-mle

<img src="https://minitorch.github.io/minitorch.svg" width="50%px">

This repository is my personal implementation of MiniTorch, as detailed in the [official MiniTorch documentation](https://minitorch.github.io/).

## Setup

Follow the installation guide at [MiniTorch Installation Guide](https://minitorch.github.io/install/).

## Training

Visualize training using Streamlit with this command:

```bash
streamlit run app.py -- [module number]
```

This project also implemented a version of LeNet on MNIST: a classic convolutional neural network (CNN) for digit recognition, and for a 1D conv for NLP sentiment classification.

You can run NLP and CV training scripts directly from the command line:

- For NLP training:
  ```bash
  python project/run_sentiment.py
  ```

- For CV training:
  ```bash
  python project/run_mnist_multiclass.py
  ```

## Assignment

Please refer to the `README.md` in each module.
