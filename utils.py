from problem import bbob, bbob_surrogate

def construct_problem_set(config):
	problem = config.problem
	# if problem in ['bbob', 'bbob-noisy']:
	# 	return bbob.BBOB_Dataset.get_datasets(suit=config.problem,
	# 										  dim=config.dim,
	# 										  upperbound=config.upperbound,
	# 										  train_batch_size=config.train_batch_size,
	# 										  test_batch_size=config.test_batch_size,
	# 										  difficulty=config.difficulty)

	if problem in ['bbob-surrogate']:
		return bbob_surrogate.bbob_surrogate_Dataset.get_datasets(config=config, dim=config.dim, upperbound=config.upperbound, train_batch_size=config.train_batch_size,
															test_batch_size=config.test_batch_size,
															train_id=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
																	20],
															test_id=[16, 17, 18, 19, 21, 22, 23, 24],
															shifted=False, rotated=False, biased=False)
	else:
		raise ValueError(problem + ' is not defined!')
