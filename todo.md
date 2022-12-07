
### for classification

11k for test ... check if it is balanced
- classwise accuracy
    - for each cls
    - TP TN FP FN
    - also f1, precision recall 
    - confusion matrix

### for arcface embeddings
- 5k positive pairs (1k per cls)
- 5k negative pairs (1k per cls)
- measure d' population distance from pos/neg

---

1. arc face gan
    * hparam search!
2. pristine focused loss
3. inference metrics form arcface paper and sarah.43 paper

4. change backbone ? LaMa, swin
