# VOLT

here is a description of how all the moving parts interact. I learned this stuff from reading the GLIP repo to borrow its essential components in my experiments.

minimum  viable product (MVP) model

I made some changes as I saw fit

### general.models

* contains functions for building models
* VLRCNN is the main building block to connect various components
    * vision backbone
    * language model
    * rpn for fusing vision and language features
    * roi heads for attending to various tasks

(vision + language) -> rpn -> roi

### general.models.layers

layers which can be building blocks for the models

__mlp__


### (other)
### general.util

__comm__

communications for distributed systems



### tools

* for running training scripts

