import copy
from agent.basic_agent import Basic_Agent
from agent.networks import MLP
from agent.utils import *


class Surr_RLDE_Agent(Basic_Agent):
	def __init__(self, config):
		super().__init__(config)
		config.state_size = 9
		config.n_act = 15
		config.lr = 1e-4
		config.batch_size = 64
		config.epsilon = 0.5  # 0.5 - 0.05
		config.gamma = 0.99
		config.update_target_steps = 1000
		config.memory_size = 100000
		config.warm_up_size = config.batch_size
		config.net_config = [{'in': config.state_size, 'out': 32, 'drop_out': 0, 'activation': 'ReLU'},
							 {'in': 32, 'out': 64, 'drop_out': 0, 'activation': 'ReLU'},
							 {'in': 64, 'out': 32, 'drop_out': 0, 'activation': 'ReLU'},
							 {'in': 32, 'out': config.n_act, 'drop_out': 0, 'activation': 'None'}]
		self.device = config.device
		self.config = config
		self.pred_Qnet = MLP(config.net_config).to(self.device)
		self.target_Qnet = copy.deepcopy(self.pred_Qnet).to(self.device)
		self.optimizer = torch.optim.AdamW(self.pred_Qnet.parameters(), lr=config.lr)
		self.criterion = torch.nn.MSELoss()
		self.n_act = config.n_act
		self.epsilon = config.epsilon
		self.gamma = config.gamma
		self.update_target_steps = config.update_target_steps
		self.batch_size = config.batch_size
		self.replay_buffer = ReplayBuffer_torch(config.memory_size, device=self.device, state_dim=self.config.state_size)
		self.warm_up_size = config.warm_up_size
		self.max_learning_step = config.max_learning_step


		self.cur_checkpoint = 0

		"""
		Parameter
		----------
		config: An argparse. Namespace object for passing some core configurations such as max_learning_step.

		Must To Do
		----------
		1. Save the model of initialized agent, which will be used in "rollout" to study the training process.
		2. Initialize a counter to record the number of accumulated learned steps
		3. Initialize a counter to record the current checkpoint of saving agent
		"""
		save_class(self.config.agent_save_dir, 'checkpoint0', self)  # save the model of initialized agent.
		self.learned_steps = 0  # record the number of accumulated learned steps
		self.learned_steps_history = 0
		self.cur_checkpoint += 1  # record the current checkpoint of saving agent

	def get_action(self, state, options=None):
		"""
		Parameter
		----------
		state: state features defined by developer.

		Return
		----------
		action: the action inferenced by using state.
		"""
		state = torch.tensor(state).to(self.device)
		action = None

		with torch.no_grad():
			Q_list = self.pred_Qnet(state)
		if options['epsilon_greedy'] and torch.rand(1) < self.epsilon:
			action = torch.randint(0, self.n_act, (1,)).item()
		if action is None:
			action = torch.argmax(Q_list).item()
		# print(action)
		return action


	def train_episode(self, env):
		""" Called by Trainer.
			Optimize a problem instance in training set until reaching max_learning_step or satisfy the convergence condition.
			During every train_episode,you need to train your own network.

		Parameter
		----------
		env: an environment consisting of a backbone optimizer and a problem sampled from train set.

		Must To Do
		----------
		1. record total reward
		2. record current learning steps and check if reach max_learning_step
		3. save agent model if checkpoint arrives

		Return
		----------
		A boolean that is true when fes reaches max_learning_step otherwise false
		A dict: {'normalizer': float,
				 'gbest': float,
				 'return': float,
				 'learn_steps': int
				 }
		"""
		# 更新一下epsilon 一个epoch更新一次
		if self.learned_steps == 3136 or self.learned_steps - 3200 == self.learned_steps_history:
			self.learned_steps_history = self.learned_steps
			self.epsilon = self.epsilon - (0.5 - 0.05) / 468
			# print(self.epsilon)

		state = env.reset()
		R = 0  # total reward
		is_done = False

		while not is_done:
			action = self.get_action(state, {'epsilon_greedy': True})
			next_state, reward, is_done = env.step(action)
			R += reward
			self.replay_buffer.append(state, action, reward, next_state, is_done)
			if len(self.replay_buffer) > self.warm_up_size:

				# print('q')
				batch_state, batch_action, batch_reward, batch_next_state, batch_done = self.replay_buffer.sample(self.batch_size)
				# print(batch_state, batch_action, batch_reward, batch_next_state, batch_done)

				# batch_state = batch_state.to(self.device)
				# batch_action = batch_action.to(self.device)
				# batch_reward = batch_reward.to(self.device)
				# batch_next_state = batch_next_state.to(self.device)
				# batch_done = batch_done.to(self.device)
				# print(batch_done)
				pred_Q = (self.pred_Qnet(batch_state)).gather(1, batch_action.unsqueeze(-1)).squeeze(-1)
				# print(pred_Q.shape)
				# target_Q = batch_reward + self.gamma * (1 - batch_done) * pred_Q

				# Target Q values
				with torch.no_grad():
					max_actions = self.pred_Qnet(batch_next_state).argmax(dim=1)
					target_q_values = self.target_Qnet(batch_next_state).gather(1, max_actions.unsqueeze(-1)).squeeze(-1)
					targets = batch_reward + (self.gamma * target_q_values * (1 - batch_done))

				# Loss and optimization
				loss = self.criterion(pred_Q, targets)
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
				self.learned_steps += 1
				# print(self.max_learning_step)
				# print(self.cur_checkpoint,self.config.save_interval)

				# save agent model if checkpoint arrives
				if self.learned_steps >= (self.config.save_interval * self.cur_checkpoint):
					save_class(self.config.agent_save_dir, 'checkpoint' + str(self.cur_checkpoint), self)
					self.cur_checkpoint += 1

				if self.learned_steps >= self.max_learning_step:
					break

			state = next_state
			if self.learned_steps % self.update_target_steps == 0:
				for target_parma, parma in zip(self.target_Qnet.parameters(), self.pred_Qnet.parameters()):
					target_parma.data.copy_(parma.data)

		return self.learned_steps >= self.config.max_learning_step, {'normalizer': env.optimizer.cost[0],
																	 'gbest': env.optimizer.cost[-1],
																	 'return': R,
																	 'learn_steps': self.learned_steps}

	def rollout_episode(self, env):
		""" Called by method rollout and Tester.test

		Parameter
		----------
		env: an environment consisting of a backbone optimizer and a problem sampled from test set

		Return
		----------
		A dict: {'cost': list,
		'fes': int,
		'return': float
		}
		"""
		state = env.reset()
		is_done = False
		R = 0  # total reward
		while not is_done:
			action = self.get_action(state, {'epsilon_greedy': False})
			next_state, reward, is_done = env.step(action)  # feed the action to environment
			R += reward  # accumulate reward
			state = next_state

		return {'cost': env.optimizer.cost, 'fes': env.optimizer.fes, 'return': R}
