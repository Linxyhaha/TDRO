# Temporally and Distributionally Robust Optimization for Cold-start Recommendation
:bulb: This is the pytorch implementation of our paper 
> [Temporally and Distributionally Robust Optimization for Cold-start Recommendation](https://arxiv.org/pdf/2312.09901.pdf)
>
> Xinyu Lin, Wenjie Wang, Jujia Zhao, Yongqi Li, Fuli Feng, Tat-Seng Chua

## Environment
- Anaconda 3
- python 3.7.11
- pytorch 1.10.0
- numpy 1.21.4
- kmeans_pytorch

## Usage

### Data
The experimental data are in './data' folder, including Amazon, Micro-video, and Kwai.

### :red_circle: Training 
```
python main.py --model_name=$1 --data_path=$2 --batch_size=$3 --l_r=$4 --reg_weight=$5 --num_group=$6 --num_period=$7 --mu=$8 --eta=$9 --lam=$10 --split_mode=$11 --log_name=$12 --gpu=$13
```
or use run.sh
```
sh run.sh <model_name> <dataset> <batch_size> <lr> <reg_weight> <num_group> <num_period> <mu> <eta> <lam> <split_mode> <logname> <gpu_id>
```
- The log file will be in the './code/log/' folder. 
- The explanation of hyper-parameters can be found in './code/main.py'. 
- The default hyper-parameter settings are detailed in './code/hyper-parameters.txt'.

:star2: TDRO is a model-agnostic training framework and can be applied to any cold-start recommender model. You can simply create your cold-start recommender model script in './code' folder, in a similar way to "model_CLCRec.py". Alternatively, you may adopt the function ``train_TDRO`` in "Train.py" to your own code for training your cold-start recommender model via TDRO.

### :large_blue_circle: Inference
Get the results of TDRO by running inference.py:

```
python inference.py --inference --data_path=$1 --ckpt=$2 --gpu=$3
```
or use inference.sh
```
sh inference.sh dataset <ckpt_path> <gpu_id>
```

### :white_circle: Examples
1. Train on Amazon dataset
```
cd ./code
sh run.sh TDRO amazon 1000 0.001 0.001 5 5 0.2 0.2 0.3 global log 0
```
2. Inference 
```
cd ./code
sh inference.sh amazon <ckpt_path> 0
```
## Citation
If you find our work is useful for your research, please consider citing:
```
@inproceedings{lin2023temporally,
      title={Temporally and Distributionally Robust Optimization for Cold-start Recommendation}, 
      author={Xinyu Lin, Wenjie Wang, Jujia Zhao, Yongqi Li, Fuli Feng, and Tat-Seng Chua},
      booktitle={AAAI},
      year={2024}
}
```

## License

NUS © [NExT++](https://www.nextcenter.org/)