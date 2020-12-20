## DAPR Instruction
A pytorch implementation for "Beyond Modality Alignment: Learning Part-level Representation for Visible-Infrared Person Re-identification" on python==3.6, pytorch==0.4.

## Data Preparation
Two datasets are used to train and evaluate the model, i.e.,

* [SYSU-MM01](https://github.com/wuancong/SYSU-MM01)[1]
* [RegDB](http://dm.dongguk.edu/link.html) (A private download link can be requested from [mangye16@gmail.com](mangye16@gmail.com)) [2]

The datasets are organized by ```data/data_manager.py```, please change the path to your own dataset path.
## Requirement
* python==3.6
* pytorch>=0.4
* argparse
* shutil
* tensorboardX

## Method
DAPR is trained in an end-to-end manner. It gradually reduces modality discrepancy of high-level features by back-propagating reversal gradients from a modality classifier, in order to learn a modality-consistent feature space. 

And simultaneously, multiple head of classifiers with the improved part-aware BNNeck are integrated into DAPR to guide the network producing identity-discriminant representations over both local and global perspective in the aforementioned modality-consistent space.

## Training
Train the model by running
```python train.py --dataset='sysu' --train_batch=64 --num_instance=4 --optim='sgd' --lr=0.1 --num_strips=6```
* ```--dataset``` : which dataset "sysu" or "regdb"
* ```--train_batch```: batch size 
* ```--num_instance```: number of each identity in a batch
* ```--optim```: optimizer
* ```--lr```: learning rate
* ```--num_strips```: number of strips of human body

Other parameters can be set in the ```train.py``` 
## Evaluation
Test the model by running
```python test.py --dataset='sysu' --mode='eval' --eval_mode='all' --num_shot='single' --save_path='./logs/log'
```
* ```--dataset```: which dataset "sysu" or "regdb"
* ```--mode```: which mode, "train" or "eval"
* ```--eval_mode```: "all" or "indoor" search for SYSU only
* ```--num_shot```: 'single' or 'all' for SYSU only
* ```--save_path```: the saved model path

## Reference 
[1] Wu, Ancong, Wei-Shi Zheng, Hong-Xing Yu, Shaogang Gong, and Jianhuang Lai. "Rgb-infrared cross-modality person re-identification." In CVPR, pp. 5380-5389, 2017.
 
[2] Ye, Mang, Zheng Wang, Xiangyuan Lan, and Pong C. Yuen. "Visible thermal person re-identification via dual-constrained top-ranking." In IJCAI, 2018.


