from .kan import *
from problem.bbob import *


class Kan_bbob_model(Basic_Problem):
	def __init__(self, dim, func_id, lb, ub, config):
		self.dim = dim
		self.func_id = func_id
		shift = np.zeros(dim)
		bias = 0
		rotate = np.eye(dim)
		self.instance = eval(f'F{func_id}')(dim=dim, shift=shift, rotate=rotate, bias=bias, lb=lb, ub=ub)
		self.model = KAN.loadckpt(f'./problem/Surrogate_KAN/Dim10/fun_{self.instance}/model')
		self.device = config.device
		self.model.to(self.device)

		self.ub = ub
		self.lb = lb

		self.optimum = None

	def eval(self, x):
		input_x = (x - self.lb) / (self.ub - self.lb)
		input_x.to(self.device)
		with torch.no_grad():
			y = self.model(input_x)
		return y

	def __str__(self):
		return f'KAN_srg_{self.instance}'


class bbob_surrogate_Dataset(Dataset):
	def __init__(self,
				 data,
				 batch_size=1):
		super().__init__()
		self.data = data
		self.batch_size = batch_size
		self.N = len(self.data)
		self.ptr = [i for i in range(0, self.N, batch_size)]
		self.index = np.arange(self.N)

	@staticmethod
	def get_datasets(config, dim, train_id, test_id, upperbound=5, train_batch_size=1,
					 test_batch_size=1,shifted=True, rotated=True, biased=True):

		# np.random.seed(4)
		train_set = []
		test_set = []
		ub = upperbound
		lb = -upperbound

		is_train = config.is_train


		func_id = [i for i in range(1, 25)]
		for id in func_id:
			np.random.seed(id+100)
			if shifted:
				shift = 0.8 * (np.random.random(dim) * (ub - lb) + lb)
			else:
				shift = np.zeros(dim)
			if rotated:
				H = rotate_gen(dim)
			else:
				H = np.eye(dim)
			if biased:
				bias = np.random.randint(1, 26) * 100
			else:
				bias = 0
			# instance = eval(f'F{id}')(dim=dim, shift=shift, rotate=H, bias=bias, lb=lb, ub=ub)
			# print(bias)

			if id in train_id:
				if is_train:
					train_instance = Kan_bbob_model(dim, id, lb, ub, config)
				else:
					train_instance = eval(f'F{id}')(dim=dim, shift=shift, rotate=H, bias=bias, lb=lb, ub=ub)
				train_set.append(train_instance)
			if id in test_id:
				test_instance = eval(f'F{id}')(dim=dim, shift=shift, rotate=H, bias=bias, lb=lb, ub=ub)
				test_set.append(test_instance)

		return bbob_surrogate_Dataset(train_set, train_batch_size), bbob_surrogate_Dataset(test_set, test_batch_size)

	def __len__(self):
		return self.N

	def __getitem__(self, item):

		if self.batch_size < 2:
			return self.data[self.index[item]]
		ptr = self.ptr[item]
		index = self.index[ptr: min(ptr + self.batch_size, self.N)]
		res = []
		for i in range(len(index)):
			res.append(self.data[index[i]])
		return res

	def __add__(self, other: 'bbob_surrogate_Dataset'):
		return bbob_surrogate_Dataset(self.data + other.data, self.batch_size)

	def shuffle(self):
		self.index = torch.randperm(self.N)
