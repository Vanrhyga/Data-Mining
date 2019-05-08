# NeuACF
This is an implementation of paper  [(Aspect-Level Deep Collaborative Filtering via Heterogeneous Information Networks)](https://www.ijcai.org/proceedings/2018/0471.pdf). 

Please refer our paper if you use this code and the bibtex of this paper is:
```
@inproceedings{han2018aspect,
   title={Aspect-Level Deep Collaborative Filtering via Heterogeneous Information Networks.},
   author="\textbf{Han, Xiaotian} and Shi, Chuan and Wang, Senzhang and Philip, S Yu and Song, Li",
   booktitle={IJCAI},
   pages={3393--3399},
   year={2018}
 }
```


### Requirements
- Python 3.6
- Tensorflow 1.2.1
- docopt 0.6.2
- numpy 1.13.3
- sklearn 0.18.1
- pandas 0.20.1
- scipy 1.0.0

### How to Run
1. unzip dataset.7z
2. Compute the aspect-level similarity matrix with the matlab code
3. Run the model with the python code acf.py

example:
```
 python ./NeuACF++.py ../dataset/amazon/ amazon --mat "U.UIU,I.IUI,U.UIBIU,I.IBI,U.UICIU,I.ICI,U.UIVIU,I.IVI" --epochs 250 --last_layer_size 32 --batch_size 1024 --num_of_neg 10 --learn_rate 0.00005 --num_of_layers 2 --margin 1.0 --mat_select median --merge attention

 python ./NeuACF++.py ../dataset/ml-100k/ ml-100k --mat "U.UMU,M.MUM,U.UMDMU,M.MDM,U.UMAMU,M.MAM" --epochs 250 --last_layer_size 32 --batch_size 1024 --num_of_neg 10 --learn_rate 0.00005 --num_of_layers 2 --margin 1.0 --mat_select median --merge attention

 python ./NeuACF++.py ../dataset/ml-1m/ ml-1m --mat "U.UMU,M.MUM,U.UMDMU,M.MDM,U.UMAMU,M.MAM" --epochs 250 --last_layer_size 32 --batch_size 1024 --num_of_neg 10 --learn_rate 0.00005 --num_of_layers 2 --margin 1.0 --mat_select median --merge attention

```

### Parameters

Parameter | Note  
|:---|:---|
|--mat|sim_mat [default: ""]|
|--epochs|Embedding size [default: 40]|
|--last_layer_size| The number of iterations [default: 64]|
|--num_of_layers|                The number of layers [default: 2]|
|--num_of_neg|               The number of negs [default: 2]|
|--learn_rate|                The learn_rate [default: 0.00005]|
|--batch_size|                batch_size [default: 1024]|
|--mat_select|                mat select type [default: median]|
|--merge|                batch_size [default: attention]|


#### Link
For more information, visit the webpage http://www.shichuan.org