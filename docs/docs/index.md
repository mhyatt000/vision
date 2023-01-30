# Welcome to [MLEX](https://github.com/mhyatt000/vision)

MLEX is a tool for designing and executing machine learning experiments.
It is built on top of PyTorch, and experiments are defined in .yaml config files.
This allots more time to focus on designing and evaluating ML models, rather than spending time on boilerplate code.

Additionally, MLEX simplifies distributed training and visualization of experiment results.
 
Inspired by [GLIPv2](https://github.com/microsoft/GLIP).

## Commands

!!! warning "TODO"

    none yet!

## Project layout

    ├── configs
    │   └── pretrain
    ├── docs
    ├── general
    │   ├── config
    │   │   └── remove
    │   ├── data
    │   │   ├── datasets
    │   │   │   └── evaluation
    │   │   │       ├── ...
    │   │   ├── samplers
    │   │   └── transforms
    │   ├── engine
    │   ├── losses
    │   ├── models
    │   │   ├── backbone
    │   │   ├── discriminator
    │   │   ├── head
    │   │   │   └── box
    │   │   ├── lang
    │   │   ├── layers
    │   │   └── rpn
    │   ├── solver
    │   ├── structures
    │   ├── tools
    │   └── utils
    │       └── logger
