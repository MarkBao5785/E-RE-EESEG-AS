## Introduction

This is the official implementation of the article titled 'Exploration with Evaluation subnetwork based on Second-order Gradients of Outputs of Reward Estimator Subnetwork and Arm-Selection Subnetwork' In this project, we implement the Evaluation-Explorer subnetwork based on SEcond-order Gradients and Arm-Selection subnetwork (E-RE-EESEG-AS), which introduces second-order Neural Tangent Kernel to exploration.

## Prerequisites

torch 1.9.0, torchvision 0.10.0, sklearn 0.24.1, numpy 1.20.1, scipy 1.6.2, pandas 1.2.4


## Usage

Here's a quick start on how to run the algorithms:

```python
python run_script.py --dataset iris --method EESEG
```

## Contributing

Contributions to this project are welcome. Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.