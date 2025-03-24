# Surrogate Learning in Meta-Black-Box Optimization: A Preliminary Study

Here we provide source codes of Surr-RLDE, which has been recently accepted by GECCO 2025.

## Citation

The PDF version of the paper is available [here](
). If you find our Surr-RLDE useful, please cite it in your publications or projects.

```latex

```

## Requirements
You can install all of dependencies of Surr-RLDE via the command below.
```bash
pip install -r requirements.txt
```

## Train
The agent training process can be activated via the command below, which is just an example.
```bash
python main.py --run_experiments --problem bbob-surrogate 
```
For more adjustable settings, please refer to `main.py` and `config.py` for details.

Recording results: Log files will be saved to `./outputs/logs/train/` . The saved checkpoints will be saved to `./outputs/model/train/`. The file structure is as follow:
```
outputs
|--logs
   |--train
      |--run_name
         |--...
|--models
   |--train
      |--run_name
         |--Epoch
            |--epoch1.pkl
            |--epoch2.pkl
            |--...
```

## Rollout
The rollout process can be easily activated via the command below.
```bash
python main.py --test --problem bbob 
```
To use the test_model.pkl file located in the home directory as the target model, you can modify the command as follows:
```bash
python main.py --test --problem bbob 
```