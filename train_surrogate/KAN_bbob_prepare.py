from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
import os
from torch.utils.data import TensorDataset, DataLoader
import problem.bbob as bbob
from .pflacco.sampling import create_initial_sample
from problem.kan import *

class BBOB_Dataset(Dataset):
	def __init__(self,
				 data,
				 batch_size=1):
		super().__init__()
		self.data = data
		self.batch_size = batch_size  # 每个batch的样本数
		self.N = len(self.data)  # 样本数
		self.ptr = [i for i in range(0, self.N, batch_size)]
		self.index = np.arange(self.N)

	@staticmethod  # 静态方法，用于根据指定的参数创建训练集和测试集
	def get_datasets(suit,
					 Dim,
					 upperbound,
					 Shift,
					 Bias,
					 shifted=False,
					 rotated=False,
					 biased=False,
					 batch_size=1):

		if (suit is None):
			func_id = [i for i in range(1, 25)]
		else:
			func_id = [i for i in suit]  # [1, 24]  BBOB 基准测试集中所有问题的编号

		# get problem instances
		data_set = []
		assert upperbound >= 5., f'Argument upperbound must be at least 5, but got {upperbound}.'
		ub = upperbound
		lb = -upperbound
		for id in func_id:
			for dim in Dim:
				for bias_value in Bias:
					for shift_value in Shift:
						if shifted:
							shift = 0.8 * (np.random.random(dim) * (ub - lb) + lb)
							# shift = 0.8 * (np.full(dim, shift_value) * (ub - lb) + lb)
						else:
							shift = np.zeros(dim)
						if rotated:
							H = bbob.rotate_gen(dim)
						else:
							H = np.eye(dim)
						if biased:
							bias = np.random.randint(1, 26) * 100
						else:
							bias = 0
						instance = eval(f'bbob.F{id}')(dim=dim, shift=shift, rotate=H, bias=bias, lb=lb, ub=ub)
						data_set.append(instance)
		return BBOB_Dataset(data_set, batch_size)

	def __getitem__(self, item):
		if self.batch_size < 2:
			return self.data[self.index[item]]
		ptr = self.ptr[item]
		index = self.index[ptr: min(ptr + self.batch_size, self.N)]
		res = []
		for i in range(len(index)):
			res.append(self.data[index[i]])
		return res

	def __len__(self):
		return self.N

	def __add__(self, other: 'BBOB_Dataset'):
		return BBOB_Dataset(self.data + other.data, self.batch_size)

	def shuffle(self):
		self.index = np.random.permutation(self.N)


def construct_bbob_problem_set(func, dim, shift, bias, shifted, biased, rotated, upperbound=5):
	return BBOB_Dataset.get_datasets(suit=func, Dim=dim, Shift=shift, Bias=bias, upperbound=upperbound, shifted=shifted,
									 biased=biased, rotated=rotated)


def BBOB_Xy_Dataset(instance, n):


	x = create_initial_sample(dim=instance.dim, sample_type='lhs', n=n,
							  lower_bound=instance.lb, upper_bound=instance.ub)
	# Calculate the objective values of the initial sample using an arbitrary objective function

	if isinstance(x, list):
		x = np.array(x, dtype=float)
	if isinstance(x, pd.DataFrame):
		x = x.values
	if x.ndim == 1:
		x = x.reshape(-1, instance.dim)
	elif x.ndim == 2:
		if x.shape[1] != instance.dim:
			raise ValueError(f"Input x must have {instance.dim} columns, got {x.shape[1]} columns.")

	y = instance.func(x)
	return x, y


def gen_KAN_dataset(train_loader, test_loader, device='cpu'):
	train_input = []
	test_input = []
	train_output = []
	test_output = []

	for batch in train_loader:
		train_input.append(batch[0])
		train_output.append(batch[1])

	for batch in test_loader:
		test_input.append(batch[0])
		test_output.append(batch[1])

	# torch.float64
	kan_train_input = torch.cat(train_input, dim=0).double().to(device)
	kan_train_output = torch.cat(train_output, dim=0).double().to(device)
	kan_test_input = torch.cat(test_input, dim=0).double().to(device)
	kan_test_output = torch.cat(test_output, dim=0).double().to(device)

	kan_dataset = {
		'train_input': kan_train_input,
		'train_label': kan_train_output,
		'test_input': kan_test_input,
		'test_label': kan_test_output
	}


	return kan_dataset

def normalize(train_X, train_y, test_X, test_y, instance):
	data = torch.load(
		f'train_surrogate/max_value/Dim{instance.dim}/{instance}/max_value.pth')

	max_value = data['max_value']

	x_train_norm = (train_X - instance.lb) / (instance.ub - instance.lb)
	x_test_norm = (test_X - instance.lb) / (instance.ub - instance.lb)
	y_train_norm = train_y / max_value
	y_test_norm = test_y / max_value

	return x_train_norm, y_train_norm, x_test_norm, y_test_norm


def gen_instance_max(seed):
	np.random.seed(seed=seed)
	instances = construct_bbob_problem_set(func=None, dim=[10],
										   shift=[0], bias=[0], shifted=False, biased=False,
										   rotated=False, upperbound=5)
	for instance in instances:
		X = create_initial_sample(dim=instance.dim, sample_type='lhs', n=200000,
								  lower_bound=instance.lb, upper_bound=instance.ub)

		X_np = X.values if isinstance(X, pd.DataFrame) else X
		y = instance.func(X_np)

		y_np = y.values if isinstance(y, pd.DataFrame) else y
		y_tensor = torch.tensor(y_np, dtype=torch.float32)
		max_value = torch.max(y_tensor)
		save_dir = f'./max_value/Dim{instance.dim}/{instance}'

		os.makedirs(save_dir, exist_ok=True)

		torch.save({
			'max_value': max_value,
		}, f'./max_value/Dim{instance.dim}/{instance}/max_value.pth')


if __name__ == '__main__':
	gen_instance_max(seed=46)