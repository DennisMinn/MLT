1. To run our project please run the following:

```bash
python -m venv ml_toolkit_env

source ml_toolkit_env/bin/activate

pip install -r requirements.txt
```

* To training and log a model, please use either the
```tensorflow``` or ```wandb``` branch and run the demo notebook in its entirety

2. To view the results of your or our experiments
* To view the logs of Tensorboard
```
tensorboard --logdir=logs/tensorboard_logs
```
* To view the logs of Weights & Bias please check your Weights & Bias dashboard

* Our WandB experiment logs
[MNIST](https://wandb.ai/dminn/MNIST?workspace=user-dminn) \\
[Fashion MNIST](https://wandb.ai/dminn/Fasion?workspace=user-dminn) \\
[CIFAR 10](https://wandb.ai/dminn/CIFAR?workspace=user-dminn)
