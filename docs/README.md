# DOCS

---
```
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
```
---

here is a description of how all the moving parts interact. much of this is borrowed from reading the GLIP repo.

minimum  viable product (MVP) model

I made some changes as I saw fit

## general.config

__default__
* containts default configurations for all experiments

__paths_catalog__
* contains functions for finding datasets and model weights

__arguments__
* handles the integration of command line arguments with the experiment configurations
* currently only used to specify config file

## general.engine

includes scripts for inference training etc

__inference__
`refactor!`

## general.models

* contains functions for building models
* VLRCNN is the main building block to connect various components
    * vision backbone
    * language model
    * rpn for fusing vision and language features
    * roi heads for attending to various tasks

(vision + language) -> rpn -> roi

__matcher__

__sampler__

__pooler__

## general.models.layers

layers which can be building blocks for the models

__mlp__


## general.models.head

roi heads ... currently only the box head exists

__head__

__feature_extractor__

__predictor__

__inference__

## general.structures

contains operations for manipulating images and boxes

## (other)
## general.util

__comm__
* communications for distributed systems

__amp__
* util for automatic mixed precision



## tools

* for running training scripts

