from .KAN_bbob_prepare import *
from train_surrogate.new_kan import *


def train_KAN_surrogate(config):
	dim = config.dim
	train_size = 50000
	np.random.seed(46)
	instances = construct_bbob_problem_set(func=None, dim=[dim],
										   shift=[0], bias=[0], shifted=False, biased=False,
										   rotated=False, upperbound=5)
	train_mode = config.surrogate_train_mode
	save_path = config.surrogate_save_dir
	print('Surrogate Training:')
	for instance in instances:

		print(instance)

		X, y = BBOB_Xy_Dataset(instance, n=train_size * 2)
		# print(X,y)
		X_np = X.values if isinstance(X, pd.DataFrame) else X
		y_np = y.values if isinstance(y, pd.DataFrame) else y

		X_tensor = torch.tensor(X_np, dtype=torch.float32)
		y_tensor = torch.tensor(y_np, dtype=torch.float32)

		split_index = int(len(X_tensor) * 0.5)
		train_X = X_tensor[:split_index]
		train_y = y_tensor[:split_index]
		test_X = X_tensor[split_index:]
		test_y = y_tensor[split_index:]

		x_train_norm, y_train_norm, x_test_norm, y_test_norm = normalize(train_X, train_y, test_X, test_y, instance)

		train_dataset = TensorDataset(x_train_norm, y_train_norm)
		test_dataset = TensorDataset(x_test_norm, y_test_norm)

		batch_size = 1
		train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
		test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

		kan_dataset = gen_KAN_dataset(train_loader, test_loader, device='cuda')

		torch.set_default_dtype(torch.float64)
		device = 'cuda'

		# KAN model
		model = new_KAN(width=[instance.dim, 10, 1], grid=5, k=5, seed=46, device=device)
		_ = model.fit(kan_dataset, opt="LBFGS", steps=1300, lr=0.01, batch=100, mode=train_mode,
						   loss_fn='order_loss')

		path = save_path + f'/surrogate_model/Dim{instance.dim}/{instance}/model'

		if not os.path.exists(path):
			# Create the directory
			os.makedirs(path)

		model.saveckpt(path)
