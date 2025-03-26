import argparse
import time


def get_config(args=None):
	parser = argparse.ArgumentParser()
	# Common config
	parser.add_argument('--problem', default='bbob-surrogate', choices=['bbob-surrogate'],
						help='specify the problem suite')
	parser.add_argument('--dim', type=int, default=10, help='dimension of search space')
	parser.add_argument('--upperbound', type=float, default=5, help='upperbound of search space')
	parser.add_argument('--difficulty', default='easy', choices=['easy', 'difficult'], help='difficulty level')
	parser.add_argument('--device', default='cpu', help='device to use')
	parser.add_argument('--train', default=None, action='store_true', help='switch to train mode')
	parser.add_argument('--test', default=None, action='store_true', help='switch to inference mode')
	parser.add_argument('--rollout', default=None, action='store_true', help='switch to rollout mode')
	parser.add_argument('--run_experiment', default=None, action='store_true', help='switch to run_experiment mode')
	parser.add_argument('--train_surrogate', default=None, action='store_true',
						help='switch to surrogate model training mode')

	# Training parameters
	parser.add_argument('--max_learning_step', type=int, default=1500000, help='the maximum learning step for training')
	parser.add_argument('--model_path', default='./problem/Surrogate_KAN/Dim10/', type=str,
						help='the load path of surrogate models')
	parser.add_argument('--agent_save_dir', type=str, default='agent_model/train/',
						help='save your own trained agent model')
	parser.add_argument('--log_dir', type=str, default='output/',
						help='logging testing output')
	parser.add_argument('--draw_interval', type=int, default=3, help='interval epochs in drawing figures')
	parser.add_argument('--agent_for_plot_training', type=str, nargs='+', default=['RL_HPSDE_Agent'],
						help='learnable optimizer to compare')
	parser.add_argument('--n_checkpoint', type=int, default=20, help='number of training checkpoints')
	parser.add_argument('--resume_dir', type=str, help='directory to load previous checkpoint model')

	# Surrogate model training parameters
	parser.add_argument('--surrogate_save_dir', default='output/', type=str, help='the save path of your surrogate model')
	parser.add_argument('--surrogate_train_mode', default='ROA', choices=['ROA', 'MSE'], type=str,
						help='surrogate model training mode')

	# Testing parameters
	parser.add_argument('--agent', default=None, help='None: traditional optimizer, else Learnable optimizer')
	parser.add_argument('--agent_load_dir', type=str,
						help='load your own agent model')
	parser.add_argument('--optimizer', default=None, help='your own learnable or traditional optimizer')
	parser.add_argument('--agent_for_cp', type=str, nargs='+', default=[],
						help='learnable optimizer to compare')
	parser.add_argument('--l_optimizer_for_cp', type=str, nargs='+', default=[],
						help='learnable optimizer to compare')  # same length with "agent_for_cp"
	parser.add_argument('--t_optimizer_for_cp', type=str, nargs='+', default=[],
						help='traditional optimizer to compare')
	parser.add_argument('--test_batch_size', type=int, default=1, help='batch size of test set')

	# Rollout parameters
	parser.add_argument('--agent_for_rollout', type=str, nargs='+', help='learnable agent for rollout')
	parser.add_argument('--optimizer_for_rollout', type=str, nargs='+', help='learnabel optimizer for rollout')
	parser.add_argument('--plot_smooth', type=float, default=0.8,
						help='a float between 0 and 1 to control the smoothness of figure curves')

	config = parser.parse_args(args)
	config.maxFEs = 2000 * config.dim
	# for bo, maxFEs is relatively smaller due to time limit
	config.bo_maxFEs = 10 * config.dim
	config.n_logpoint = 50

	config.train_agent = 'Surr_RLDE_Agent'
	config.train_optimizer = 'Surr_RLDE_Optimizer'

	if config.run_experiment and len(config.agent_for_cp) >= 1:
		assert config.agent_load_dir is not None, "Option --agent_load_dir must be given since you specified option --agent_for_cp."

	config.run_time = f'{time.strftime("%Y%m%dT%H%M%S")}_{config.problem}_{config.difficulty}_{config.dim}D'
	config.test_log_dir = config.log_dir + '/test/' + config.run_time + '/'
	config.rollout_log_dir = config.log_dir + '/rollout/' + config.run_time + '/'

	if config.train or config.run_experiment:
		config.agent_save_dir = config.agent_save_dir + config.train_agent + '/' + config.run_time + '/'

	config.save_interval = config.max_learning_step // config.n_checkpoint
	config.log_interval = config.maxFEs // config.n_logpoint
	if 'Random_search' not in config.t_optimizer_for_cp:
		config.t_optimizer_for_cp.append('Random_search')

	return config
