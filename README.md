# Experiment code for PASSLEAF 
Code for the PASSLEAF paper.

The implementation is a fork of [UKGE](https://github.com/stasl0217/UKGE).
<!-- the implementation of deterministic knowledge graph embedding models is a fork of [OpenKE](https://github.com/thunlp/OpenKE). -->

## Environment
* python3 (>=3.6)
* tensorflow-gpu (==1.14)

For other dependencies, please see ```requirements.txt```. (Some of them may be UNNECESSARY!)


## Datasets
The base datasets are from [UKGE](https://github.com/stasl0217/UKGE).


Datasets for PASSLEAF models:
* PPI5K (ppi5k_no_psl)
* NL27K (nl27k_no_psl)
* CN15K (cn15k_no_psl)
<!-- 
Filtered datasets for deterministic KG embedding models:
* ppi5k_openke_threshold_[threshold]_gen
* nl27k_openke_threshold_[threshold]_gen
* cn15k_openke_threshold_[threshold]_gen

[threshold] is the threshold for keeping samples. The value may be one of the following: {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}. -->

The datasets locate in the ```data``` folder.

## Models
In the following script examples, we use a batch size of 512, embedding dimension of 512, and save the checkpoint every 40 epochs. 

### PASSLEAF Model **with** pool-based semi-supervised learning
* ComplEx_m5_4: Uncertain ComplEx
* RotatE_m5: Uncertain RotatE
* RotatE_m3_3: Simplified Uncertain Rotate
* UKGE_logi_m2: UKGE + pool-based semi-supervised learning

Scripts: 
* Train:

    ```
    python3 run/run.py --data [dataset] --batch_size 512 --epoch 2000 -d 512 --no_psl --models_dir  [base path to save model] -m [model name]  --semisupervised_v2 --save_freq 40
    ```

* Test:
    
    ```
    python3 run/test.py --data [dataset] --batch_size 512 -d 512 --no_psl --resume_model_path [saved model directory path] -m [model name] --start [starting epoch] --to [ending epoch] --step 40
    ```

* Example:

    ```
    python3 run/run.py --data cn15k_no_psl --batch_size 512 --epoch 3000 -d 512 --no_psl --models_dir  ./trained_model_batch512_dim_512_semisupervised_v2 -m ComplEx_m5_4  --semisupervised_v2 --save_freq 40

    python3 run/test.py --data cn15k_no_psl --batch_size 512 -d 512 --no_psl --resume_model_path ./trained_model_batch512_dim_512_semisupervised_v2/cn15k_no_psl/ComplEx_m5_4_0930 -m ComplEx_m5_4 --start 10 --to 3000 --step 20
    ```
    ＊ the model saving path name varies according to the current date.


### PASSLEAF Model **without** pool-based semi-supervised learning 
* ComplEx_m5_1: Uncertain ComplEx (no SS)
* RotatE_m5_1: Uncertain RotatE (no SS)
* RotatE_m3_1: Simplified Uncertain Rotate (no SS)
* logi: UKGE

Scripts: 
* Train:

    Same as above but without the ```--semisupervised_v2``` flag
    ```
    python3 run/run.py --data [dataset] --batch_size 512 --epoch 2000 -d 512 --no_psl --models_dir  [base path to save model] -m [model name] --save_freq 40
    ```

* Test:
    
    Same as above.


### Deterministic KG Embedding Models
TODO. Not available in the repo yet


## Reference
The PASSLEAF paper.
Chen, Z.-M., Yeh, M.-Y., & Kuo, T.-W. (2021). PASSLEAF: A Pool-bAsed Semi-Supervised LEArning Framework for Uncertain Knowledge Graph Embedding. Proceedings of the AAAI Conference on Artificial Intelligence, 35(5), 4019-4026. 


    @article{
        Chen_Yeh_Kuo_2021, 
        title={PASSLEAF: A Pool-bAsed Semi-Supervised LEArning Framework for Uncertain Knowledge Graph Embedding}, 
        volume={35}, 
        url={https://ojs.aaai.org/index.php/AAAI/article/view/16522}, 
        number={5}, 
        journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
        author={Chen, Zhu-Mu and Yeh, Mi-Yen and Kuo, Tei-Wei}, 
        year={2021}, 
        month={May}, 
        pages={4019-4026} 
    }


<!-- 

### Deterministic KG Embedding Models
* ComplEx
* DistMult
* RotatE

Please note that only __Filtered datasets__ are applicable here.

Scripts:
* Train:

    ```
    python3 run/runOpenKE.py --data [dataset] -m [model name]
    ```

* Test:

    ```
    python3 run/test_openke.py --data [dataset]  --resume_model_path [saved model directory path]  --start [starting epoch] --to [ending epoch] --step 40 -m [model]
    `` -->


## The output files and Analyses
### Outputs of ```run/test.py```
By default, ```run/test.py``` generates the following files based on saved models per training step (checkpoints):

Validation:
* ```val%s_mean_rank_accurate.csv```: mean rank (TEP) and hit@K (TEP) for each specified training steps.
* ```val_loss_accurate.csv```:  MSE (CSP) and nDCG (TEP) for each specified training steps.
* ```val_detail_[epoch].csv```: detailed predictions. One file per training steps.

Testing: for each best checkpoint according to the validation MSE, nDCG(linear), and Hit@20
* ```test_mean_rank_accurate.csv```: mean rank (TEP) and hit@K (TEP) on testing set.
* ```test_loss_accurate.csv```: MSE (CSP) and nDCG (TEP) on testing set.
* ```test_test_only_detail_[epoch].csv```: detailed predictions on testing set .
* ```test_mean_rank_training_included.csv```: mean rank (TEP) and hit@K (TEP) on testing set. (training set candidates INCLUDED)
* ```test_loss_training_included.csv```: MSE (CSP) and nDCG (TEP) on testing set. (training set candidates INCLUDED)
* ```test_detail_[epoch]_training_included.csv```: detailed predictions on testing set. (training set candidates INCLUDED)

### Automatized analysis tool
To automatize the analysis, use the ```autorecord_openke+ukge.py``` script.
```
python3 autorecord_openke+ukge.py > ./records.csv
```
Please see the script for details about the options.


## Commands for reproduction
### Saved trained model

Pretrained model files are missing :(

Trying to fix that.

### Evaluation commands
#### Simplified Uncertain RotatE + SS
```
    python3 run/test.py --data ppi5k_no_psl --batch_size 512 --epoch 400 -d 512 --no_psl --resume_model_path ./trained_model_batch512_dim_512_semisupervised_v2/ppi5k_no_psl/RotatE_m3_3_0306 -m RotatE_m3_3 --start 10 --to 2000 --step 40
    python3 run/test.py --data nl27k_no_psl --batch_size 512 --epoch 400 -d 512 --no_psl --resume_model_path ./trained_model_batch512_dim_512_semisupervised_v2/nl27k_no_psl/RotatE_m3_3_0304 -m RotatE_m3_3 --start 10 --to 2000 --step 40
    python3 run/test.py --data cn15k_no_psl --batch_size 512 --epoch 400 -d 512 --no_psl --resume_model_path ./trained_model_batch512_dim_512_semisupervised_v2/cn15k_no_psl/RotatE_m3_3_0306 -m RotatE_m3_3 --start 10 --to 3000 --step 40
```
#### UKGE logi + SS
```
    python3 run/test.py --data ppi5k_no_psl --batch_size 512 --epoch 400 -d 512 --no_psl --resume_model_path ./trained_model_batch512_dim_512_semisupervised_v2/ppi5k_no_psl/UKGE_logi_m2_0306 -m UKGE_logi_m2 --start 10 --to 2000 --step 40
    python3 run/test.py --data nl27k_no_psl --batch_size 512 --epoch 400 -d 512 --no_psl --resume_model_path ./trained_model_batch512_dim_512_semisupervised_v2/nl27k_no_psl/UKGE_logi_m2_0301 -m UKGE_logi_m2 --start 10 --to 2000 --step 40
    python3 run/test.py --data cn15k_no_psl --batch_size 512 --epoch 400 -d 512 --no_psl --resume_model_path ./trained_model_batch512_dim_512_semisupervised_v2/cn15k_no_psl/UKGE_logi_m2_0301 -m UKGE_logi_m2 --start 10 --to 3000 --step 20
```
#### Uncertain ComplEx + SS
```
    python3 run/test.py --data ppi5k_no_psl --batch_size 512 --epoch 400 -d 512 --no_psl --resume_model_path ./trained_model_batch512_dim_512_semisupervised_v2/ppi5k_no_psl/ComplEx_m5_4_0306 -m ComplEx_m5_4 --start 10 --to 2000 --step 40
    python3 run/test.py --data nl27k_no_psl --batch_size 512 --epoch 400 -d 512 --no_psl --resume_model_path ./trained_model_batch512_dim_512_semisupervised_v2/nl27k_no_psl/ComplEx_m5_4_0301 -m ComplEx_m5_4 --start 10 --to 2000 --step 40
    python3 run/test.py --data cn15k_no_psl --batch_size 512 --epoch 400 -d 512 --no_psl --resume_model_path ./trained_model_batch512_dim_512_semisupervised_v2/cn15k_no_psl/ComplEx_m5_4_0301 -m ComplEx_m5_4 --start 10 --to 3000 --step 20
```
#### Simplified Uncertain Rotate (no pool-based semi-supervised learning)
```
    python3 run/test.py --data ppi5k_no_psl --batch_size 512 --epoch 400 -d 512 --no_psl --resume_model_path ./trained_model_batch512_dim_512/ppi5k_no_psl/RotatE_m3_1_0116 -m RotatE_m3_1 --start 10 --to 2000 --step 40
    python3 run/test.py --data nl27k_no_psl --batch_size 512 --epoch 400 -d 512 --no_psl --resume_model_path ./trained_model_batch512_dim_512/nl27k_no_psl/RotatE_m3_1_0116 -m RotatE_m3_1 --start 10 --to 2000 --step 40
    python3 run/test.py --data cn15k_no_psl --batch_size 512 --epoch 400 -d 512 --no_psl --resume_model_path ./trained_model_batch512_dim_512/cn15k_no_psl/RotatE_m3_1_0116 -m RotatE_m3_1 --start 10 --to 2000 --step 40
```
#### Uncertain Complex (no pool-based semi-supervised learning)
```
    python3 run/test.py --data cn15k_no_psl --batch_size 512 --epoch 400 -d 512 --no_psl --resume_model_path ./trained_model_batch512_dim_512/cn15k_no_psl/ComplEx_m5_1_0117 -m ComplEx_m5_1 --start 10 --to 2000 --step 40
    python3 run/test.py --data nl27k_no_psl --batch_size 512 --epoch 400 -d 512 --no_psl --resume_model_path ./trained_model_batch512_dim_512/nl27k_no_psl/ComplEx_m5_1_0117 -m ComplEx_m5_1 --start 10 --to 2000 --step 40
    python3 run/test.py --data ppi5k_no_psl --batch_size 512 --epoch 400 -d 512 --no_psl --resume_model_path ./trained_model_batch512_dim_512/ppi5k_no_psl/ComplEx_m5_1_0117 -m ComplEx_m5_1 --start 10 --to 2000 --step 40
```
#### UKGE (no pool-based semi-supervised learning)
```
    python3 run/test.py --data cn15k_no_psl --batch_size 512 --epoch 400 -d 512 --no_psl --resume_model_path ./trained_model_batch512_dim_512/cn15k_no_psl/logi_0117 -m logi --start 10 --to 2000 --step 40
    python3 run/test.py --data nl27k_no_psl --batch_size 512 --epoch 400 -d 512 --no_psl --resume_model_path ./trained_model_batch512_dim_512/nl27k_no_psl/logi_0117 -m logi --start 10 --to 2000 --step 40
    python3 run/test.py --data ppi5k_no_psl --batch_size 512 --epoch 400 -d 512 --no_psl --resume_model_path ./trained_model_batch512_dim_512/ppi5k_no_psl/logi_0117 -m logi --start 10 --to 2000 --step 40
```

