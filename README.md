# Surrogate Learning in Meta-Black-Box Optimization: A Preliminary Study

Here we provide source codes of Surr-RLDE, which has been recently accepted by GECCO 2025.

## Citation

The PDF version of the paper is available [here](https://arxiv.org/abs/2503.18060 ). If you find our Surr-RLDE useful, please cite it in your publications or projects.

```latex
@misc{ma2025surrogatelearningmetablackboxoptimization,
      title={Surrogate Learning in Meta-Black-Box Optimization: A Preliminary Study}, 
      author={Zeyuan Ma and Zhiyang Huang and Jiacheng Chen and Zhiguang Cao and Yue-Jiao Gong},
      year={2025},
      eprint={2503.18060},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.18060}, 
}
```

## Requirements

You can install all of dependencies of Surr-RLDE via the command below.

```bash
pip install -r requirements.txt
```

## Train

### Surrogate Learning Stage

The surrogate learning process can be activated via te command below

~~~bash
python main.py --train_surrogate 
~~~

The trained model will be saved to`output/surrogate_model/`

### Policy Learning Stage

The Surr-RLDE agent training process can be activated via the command below, which is just an example.

```bash
python main.py --run_experiments --problem bbob-surrogate 
```

For more adjustable settings, please refer to `main.py` and `config.py` for details.

Recording results: Log files will be saved to `./output/train/` . The saved checkpoints will be saved to `./agent_model/train/`. The file structure is as follow:

```
|--agent_model
   |--train
      |--Surr_RLDE_Agent
         |--run_Name
            |--checkpoint0.pkl
            |--checkpoint1.pkl
            |--...

|--output
   |--train
      |--Surr_RLDE_Agent
         |--runName
            |--log
            |--pic
```

## Test

The test process can be easily activated via the command below. The defalt agent load path is `agent_model/test/`

```bash
python main.py --test --agent_load_dir YourAgentDir --agent_for_cp Surr_RLDE_Agent --l_optimizer_for_cp Surr_RLDE_Optimizer 

```

You can compare Surr-RLDE with DEDQN, DEDDQN, GLEET by adding them into the agent_for_cp and l_optimizer_for_cp

```bash
python main.py --test --agent_load_dir YourAgentDir --agent_for_cp Surr_RLDE_Agent DEDQN_Agent --l_optimizer_for_cp Surr_RLDE_Optimizer DEDQN_optimizer
```
