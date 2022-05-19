# HandRRNN

This is the implementation of the manuscript

1. Prepare dataset 

    please download the NYU dataset

    follow the instructions in the './preprocess_nyu/' for datasets preprocessing 

2. Install PointNet++ CUDA operations

    follow the instructions in the './pointnet2' for installation 

3. Evaluate

    execute ``` python3 eval_msra.py --model [saved model name] --iters [training iterations] --test_iters [testing iterations] --test_path [testing set path]```

    for example 
    ```python3 eval_msra.py --model best_model.pth --iters 3 --test_iters 5 --test_path ../preprocess_nyu/testing/```

    we provided the pre-trained models ('./pretrained_model/nyu_rrnn_3iters/best_model.pth') for NYU

4. If a new training process is needed, please execute the following instructions after step 1 and 2 are completed

   . for training NYU
    execute ``` python3 train_rrnn.py --iters [number of training iteration] --dataset_path [NYU training dataset path] --test_path [NYU testing dataset path]```
    example ``` python3 train_rrnn.py --iters 3 --dataset_path ../preprocess_nyu/training/ --test_path ../preprocess_nyu/testing/```
