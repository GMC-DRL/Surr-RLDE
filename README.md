# Surrogate Learning in Meta-Black-Box Optimization: A Preliminary Study

Here we provide source codes of Surr-RLDE, which has been recently accepted by GECCO 2025.

## Citation

The PDF version of the paper is available [here](https://arxiv.org/abs/2503.18060
). If you find our Surr-RLDE useful, please cite it in your publications or projects.

```latex

```

## Requirements
You can install all of dependencies of Surr-RLDE via the command below.
```bash
pip install -r requirements.txt
```

## Train
The Surr-RLDE agent training process can be activated via the command below, which is just an example.
```bash
python main.py --run_experiments --problem bbob-surrogate 
```
For more adjustable settings, please refer to `main.py` and `config.py` for details.

Recording results: Log files will be saved to `./outputs/train/` . The saved checkpoints will be saved to `./agent_model/train/`. The file structure is as follow:
```
todo
```

## Rollout
The rollout process can be easily activated via the command below.
```bash
todo
```
To use the test_model.pkl file located in the home directory as the target model, you can modify the command as follows:
```bash
todo 
```