This repository is used to train, prune, generate explanations and evaluate explantions to determine the effect of model pruning on explanation correctness in multitask learning. This is done on the NYUv2 dataset with the MTAN (Multitask Attention Network) model, pruned with HRank pruningm using GradCAM and variations of SegGradCAM to generate attribution based explanations that were evaluated in terms of correctness using ROAD (Remove and Debias)

1. TRAIN BASE MODEl
  2. Download NYUv2 dataset from external source
  3. Train Chosen MTAN model on NYUv2 dataset using NYUv2 directory
  4. More details on: https://github.com/Cranial-XIX/CAGrad
5. PRUNE MODEL(S)
  6. Generate convolutional ranks
  7. Prune the model using HRank at desired pruning ratio
  8. More details on: https://github.com/lmbxmu/HRank
9. GENERATE & EVALUATION EXPLANATIONS
  10. Generate explanations and evalute using ROAD on xAI directory using following sh command "python -u model_segnet_mtan.py --model [model_name]"
  11. Results in csv file with 'ROAD percentile', 'Class', 'Class Metric', 'Extra Class Metric', 'Task' and 'Image Number'
12. PLOT RESULTS
  13. Plot using ROAD_Analysis_and_Plotting directory
  14. run 'results.py' with corresponding results csv file previously obtained (step 11)
