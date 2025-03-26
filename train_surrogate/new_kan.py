import torch
import numpy as np
import torch.nn.functional as F
from problem.kan import *


class new_KAN(KAN):

	def __init__(self, width=None, grid=3, k=3, mult_arity=2, noise_scale=0.3, scale_base_mu=0.0, scale_base_sigma=1.0,
				 base_fun='silu', symbolic_enabled=True, affine_trainable=False, grid_eps=0.02, grid_range=[-1, 1],
				 sp_trainable=True, sb_trainable=True, seed=1, save_act=True, sparse_init=False, auto_save=True,
				 first_init=True, ckpt_path='./model', state_id=0, round=0, device='cpu'):

		super(new_KAN, self).__init__(width=width, grid=grid, k=k, mult_arity=mult_arity, noise_scale=noise_scale,
									  scale_base_mu=scale_base_mu, scale_base_sigma=scale_base_sigma,
									  base_fun=base_fun, symbolic_enabled=symbolic_enabled,
									  affine_trainable=affine_trainable, grid_eps=grid_eps,
									  grid_range=grid_range,
									  sp_trainable=sp_trainable, sb_trainable=sb_trainable, seed=seed,
									  save_act=save_act, sparse_init=sparse_init,
									  auto_save=auto_save,
									  first_init=first_init, ckpt_path=ckpt_path, state_id=state_id, round=round,
									  device=device)

	def adjusted_loss_fn(self, step, total_steps, mode):

		if mode == 'MSE':
			alpha = 1
			beta = 0

		elif mode == 'ROA':
			if step <= 300:
				alpha = 1
				beta = 0
			else:
				beta = 1
				alpha = 1 - (step - 300) / (total_steps - 300)

		def order_loss_fn(pred, target):
			# print('1')
			target = target.unsqueeze(1)
			mse_loss = F.mse_loss(pred, target)
			if beta == 0:
				return alpha * mse_loss

			sorted_y, indices = torch.sort(target.flatten(), descending=True)
			# print(sorted_y, indices)
			sorted_y = sorted_y.to(self.device)
			sorted_yhat = pred.flatten()[indices].to(self.device)
			total_order_loss = torch.tensor([0.0], device=self.device)
			zero_tensor = torch.tensor([0.0], device=self.device)
			total_order_loss += torch.max(sorted_y[1] - sorted_yhat[0], zero_tensor)  # 第一个
			total_order_loss += torch.max(sorted_yhat[-1] - sorted_yhat[-2], zero_tensor)  # 最后一个

			distance = torch.abs(sorted_y[2:] - sorted_yhat[:-2]).to(self.device)
			# 计算排序损失
			sorted_diff1 = sorted_y[2:] - sorted_yhat[1:-1]  # (sorted_y[i] - sorted_yhat[i+1])
			sorted_diff2 = sorted_yhat[1:-1] - sorted_yhat[:-2]  # (sorted_yhat[i] - sorted_y[i+1])
			loss = torch.abs(sorted_diff1) + torch.abs(sorted_diff2) - distance
			# 计算最大损失
			order_loss = torch.maximum(loss, torch.zeros_like(loss))
			# order_loss_next = torch.maximum(sorted_diff_next, torch.zeros_like(sorted_diff_next))

			total_order_loss += torch.sum(order_loss)
			return alpha * mse_loss + beta * total_order_loss

		return order_loss_fn

	def fit(self, dataset, opt="LBFGS", steps=100, log=1, lamb=0., lamb_l1=1., lamb_entropy=2., lamb_coef=0.,
			lamb_coefdiff=0., update_grid=True, grid_update_num=10, loss_fn=None, lr=1., start_grid_update_step=-1,
			stop_grid_update_step=50, batch=-1,
			metrics=None, save_fig=False, in_vars=None, out_vars=None, beta=3, save_fig_freq=1, img_folder='./video',
			singularity_avoiding=False, y_th=1000., reg_metric='edge_forward_spline_n', display_metrics=None,
			mode=None):
		'''
		training

		Args:
		-----
			dataset : dic
				contains dataset['train_input'], dataset['train_label'], dataset['test_input'], dataset['test_label']
			opt : str
				"LBFGS" or "Adam"
			steps : int
				training steps
			log : int
				logging frequency
			lamb : float
				overall penalty strength
			lamb_l1 : float
				l1 penalty strength
			lamb_entropy : float
				entropy penalty strength
			lamb_coef : float
				coefficient magnitude penalty strength
			lamb_coefdiff : float
				difference of nearby coefficits (smoothness) penalty strength
			update_grid : bool
				If True, update grid regularly before stop_grid_update_step
			grid_update_num : int
				the number of grid updates before stop_grid_update_step
			start_grid_update_step : int
				no grid updates before this training step
			stop_grid_update_step : int
				no grid updates after this training step
			loss_fn : function
				loss function
			lr : float
				learning rate
			batch : int
				batch size, if -1 then full.
			save_fig_freq : int
				save figure every (save_fig_freq) steps
			singularity_avoiding : bool
				indicate whether to avoid singularity for the symbolic part
			y_th : float
				singularity threshold (anything above the threshold is considered singular and is softened in some ways)
			reg_metric : str
				regularization metric. Choose from {'edge_forward_spline_n', 'edge_forward_spline_u', 'edge_forward_sum', 'edge_backward', 'node_backward'}
			metrics : a list of metrics (as functions)
				the metrics to be computed in training
			display_metrics : a list of functions
				the metric to be displayed in tqdm progress bar

		Returns:
		--------
			results : dic
				results['train_loss'], 1D array of training losses (RMSE)
				results['test_loss'], 1D array of test losses (RMSE)
				results['reg'], 1D array of regularization
				other metrics specified in metrics

		Example
		-------
		>>> from kan import *
		>>> model = KAN(width=[2,5,1], grid=5, k=3, noise_scale=0.3, seed=2)
		>>> f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
		>>> dataset = create_dataset(f, n_var=2)
		>>> model.fit(dataset, opt='LBFGS', steps=20, lamb=0.001);
		>>> model.plot()

		# Most examples in toturals involve the fit() method. Please check them for useness.
		'''

		if lamb > 0. and not self.save_act:
			print('setting lamb=0. If you want to set lamb > 0, set self.save_act=True')

		old_save_act, old_symbolic_enabled = self.disable_symbolic_in_fit(lamb)

		pbar = tqdm(range(steps), desc='description', ncols=100)

		grid_update_freq = int(stop_grid_update_step / grid_update_num)

		if opt == "Adam":
			optimizer = torch.optim.Adam(self.get_params(), lr=lr)
		elif opt == "LBFGS":
			optimizer = LBFGS(self.get_params(), lr=lr, history_size=10, line_search_fn="strong_wolfe",
							  tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)

		results = {}
		results['train_loss'] = []
		results['test_loss'] = []
		results['reg'] = []
		if metrics != None:
			for i in range(len(metrics)):
				results[metrics[i].__name__] = []

		if batch == -1 or batch > dataset['train_input'].shape[0]:
			batch_size = dataset['train_input'].shape[0]
			batch_size_test = dataset['test_input'].shape[0]
		else:
			batch_size = batch
			batch_size_test = batch

		global train_loss, reg_

		def closure():
			global train_loss, reg_
			optimizer.zero_grad()
			pred = self.forward(dataset['train_input'][train_id], singularity_avoiding=singularity_avoiding, y_th=y_th)
			train_loss = loss_fn(pred, dataset['train_label'][train_id])
			if self.save_act:
				if reg_metric == 'edge_backward':
					self.attribute()
				if reg_metric == 'node_backward':
					self.node_attribute()
				reg_ = self.get_reg(reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff)
			else:
				reg_ = torch.tensor(0.)
			objective = train_loss + lamb * reg_
			objective.backward()
			return objective

		if save_fig:
			if not os.path.exists(img_folder):
				os.makedirs(img_folder)

		for _ in pbar:
			# print(_)
			# if loss_fn == None:
			# 	loss_fn = loss_fn_eval = lambda x, y: torch.mean((x - y) ** 2)
			# elif loss_fn == 'order_loss':
			# 	loss_fn = loss_fn_eval = self.adjusted_loss_fn(_, steps, mode=mode)
			# else:
			# 	loss_fn = loss_fn_eval = loss_fn

			loss_fn = loss_fn_eval = self.adjusted_loss_fn(_, steps, mode=mode)

			# loss_fn  = lambda x, y: torch.mean((x - y) ** 2)
			if _ == steps - 1 and old_save_act:
				self.save_act = True

			train_id = np.random.choice(dataset['train_input'].shape[0], batch_size, replace=False)
			test_id = np.random.choice(dataset['test_input'].shape[0], batch_size_test, replace=False)

			if _ % grid_update_freq == 0 and _ < stop_grid_update_step and update_grid and _ >= start_grid_update_step:
				self.update_grid(dataset['train_input'][train_id])

			if opt == "LBFGS":
				optimizer.step(closure)

			if opt == "Adam":
				pred = self.forward(dataset['train_input'][train_id], singularity_avoiding=singularity_avoiding,
									y_th=y_th)
				train_loss = loss_fn(pred, dataset['train_label'][train_id])
				if self.save_act:
					if reg_metric == 'edge_backward':
						self.attribute()
					if reg_metric == 'node_backward':
						self.node_attribute()
					reg_ = self.get_reg(reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff)
				else:
					reg_ = torch.tensor(0.)
				loss = train_loss + lamb * reg_
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

			test_loss = loss_fn_eval(self.forward(dataset['test_input'][test_id]), dataset['test_label'][test_id])

			if metrics != None:
				for i in range(len(metrics)):
					results[metrics[i].__name__].append(metrics[i]().item())

			results['train_loss'].append(torch.sqrt(train_loss).cpu().detach().numpy())
			results['test_loss'].append(torch.sqrt(test_loss).cpu().detach().numpy())
			results['reg'].append(reg_.cpu().detach().numpy())

			if _ % log == 0:
				if display_metrics == None:
					pbar.set_description("| train_loss: %.2e | test_loss: %.2e | reg: %.2e | " % (
						torch.sqrt(train_loss).cpu().detach().numpy(), torch.sqrt(test_loss).cpu().detach().numpy(),
						reg_.cpu().detach().numpy()))
				else:
					string = ''
					data = ()
					for metric in display_metrics:
						string += f' {metric}: %.2e |'
						try:
							results[metric]
						except:
							raise Exception(f'{metric} not recognized')
						data += (results[metric][-1],)
					pbar.set_description(string % data)

			if save_fig and _ % save_fig_freq == 0:
				self.plot(folder=img_folder, in_vars=in_vars, out_vars=out_vars, title="Step {}".format(_), beta=beta)
				plt.savefig(img_folder + '/' + str(_) + '.jpg', bbox_inches='tight', dpi=200)
				plt.close()

		self.log_history('fit')
		# revert back to original state
		self.symbolic_enabled = old_symbolic_enabled
		return results
