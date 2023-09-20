# Transparent Deep Learning - transparentmodel

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.6%2B-blue.svg)](https://www.python.org/)

## Overview

transparentmodel is a Python package that provides a convenient wrapper for tracking memory usage and performance metrics (such as FLOPs) around large language models. It aims to simplify the process of monitoring resource consumption and computational efficiency during inference, helping researchers and developers optimize their models and deployments.

Currently, the package supports tracking memory usage and FLOPs for models based on the Transformers library. However, future updates are planned to extend support to other frameworks and architectures.

## Features

- Measure memory usage during model inference.
- Calculate FLOPs (floating-point operations) for the model.
- Compatible with models based on the Transformers library.
- Easy-to-use API for integrating with existing codebases.
- Extensible design to support additional frameworks and architectures in the future.

## Installation

You can install the package using `pip`:
<pre>git clone https://github.com/dpleus/transparentmodel
pip install .</pre>


## Usage
### Transformer Inference

<pre>
from transparentmodel.huggingface import inference

# Replace original inference function with the wrapped one
<s>output = model.generate(input_tokens)</s>
output = inference.generate_with_memory_tracking(model, realtime=True)
</pre>

### Transformer Training

<pre>
from transparentmodel.huggingface.training import train_with_memory_tracking

# Replace original inference function with the wrapped one
<s>trainer.train()</s>
train_with_memory_tracking(trainer, realtime=True)
</pre>


Metrics
- System memory: For GPUs and RAM
- Model metrics: Parameter memory & dtype (activations and gradients for training)
- Memory Tracking: Per second, for both GPU and RAM (if realtime=True)
- Summary: Minimum free RAM and Peak Utilization (only CPU yet)
- Deep Dive: Compute Time and Memory Usage per sub-operation

## Documentation
For detailed instructions and more advanced usage examples, please refer to the documentation.

## Contributing
Contributions are welcome! If you encounter any issues or have suggestions for improvements, please open an issue on the GitHub repository.

If you would like to contribute code, please follow the contribution guidelines and submit a pull request.

## License
This project is licensed under the MIT License.
