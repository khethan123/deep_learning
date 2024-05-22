# Music Generation with LSTM

This repository contains a Jupyter notebook that demonstrates the process of generating music using LSTM. 
The notebook is inspired by the hands-on lab from MIT's "Introduction to Deep Learning 2021" course.

## Overview

The notebook guides you through the following steps:
- Processing a dataset of Irish folk songs in ABC notation.
- Building and training an RNN model to learn patterns in the music data[^5^][5].
- Generating new pieces of music using the trained model.

## Dataset

The dataset consists of thousands of Irish folk songs represented in [ABC notation](https://en.wikipedia.org/wiki/ABC_notation).
You can either download the dataset from their [github](https://github.com/aamini/introtodeeplearning/blob/master/mitdeeplearning/data/irish.abc) or from this repo.

## Requirements

<pre>
  tensorflow abcmidi timidity numpy 
</pre>

## Usage

To use this notebook:
1. Just open this notebook in colab.
2. Download the dataset from github and upload it onto the colab notebook.
3. Run the notebook cells sequentially to train the model and generate music.

## License
Copyright Information from author(Chanseok Kang)
Copyright 2021 MIT 6.S191 Introduction to Deep Learning. All Rights Reserved.
Licensed under the MIT License. You may not use this file except in compliance with the License. Use and/or modification of this code outside of 6.S191 must reference:
© MIT 6.S191: [Introduction to Deep Learning](http://introtodeeplearning.com)

