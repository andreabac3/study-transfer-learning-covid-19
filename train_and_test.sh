#!/bin/bash
for model_name in "resnext" "mobilenet_v2" "mobilenet_v3" "mnasnet" "wide_resnet50" "densenet" "shufflenet" "googlenet" "alexnet"  "vgg16" "resnet50" "squeezenet" 
do
    PYTHONPATH=. python3 ./src/train.py  -m train.model_type=$model_name  train.pl_trainer.max_epochs=40 data.subset_percentage=1.0
    PYTHONPATH=. python3 ./src/test.py train.model_type=$model_name  train.pl_trainer.max_epochs=40 data.subset_percentage=1.0
done