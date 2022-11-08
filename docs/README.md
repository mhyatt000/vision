# VOLT

here is a description of how all the moving parts interact. I learned this stuff from reading the GLIP repo to borrow its essential components in my experiments.

minimum  viable product (MVP) model

I made some changes as I saw fit

### general.config

__default__
* containts default configurations for all experiments
__paths_catalog__
* contains functions for finding datasets and model weights

### general.engine

includes scripts for inference training etc

__inference__
`refactor!`

### general.models

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

### general.models.layers

layers which can be building blocks for the models

__mlp__


### general.models.head

roi heads ... currently only the box head exists

__head__
__feature_extractor__
__predictor__
__inference__

### general.structures

contains operations for manipulating images and boxes

### (other)
### general.util

__comm__
* communications for distributed systems
__amp__
* util for automatic mixed precision



### tools

* for running training scripts

