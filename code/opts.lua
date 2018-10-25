
function table.ismember(t, item)
	for key, value in next, t do
		if value == item then
			return true
		end
	end
	return false
end

local M = {}

function M.parse(arg)
	print 'Non-uniform blind deblurring with CNN'
	print '==> processing options'
	
	local cmd = torch.CmdLine()
	cmd:text()
	cmd:text('Options:')
	-- global:
	cmd:option('-seed', 0, 'If nonzero, fixed input seed for repeatable experiments. If 0, then random seed')
	cmd:option('-threads', 2, 'number of main threads')
	cmd:option('-nDonkeys', 4, 'number of donkeys to initialize (data loading threads)')
	cmd:option('-gpuid', 1, 'GPU id to use, 1-based')
	-- data:
	cmd:option('-dataset', 'GOPRO_Large', 'dataset to use for training : GOPRO | GOPRO_Large')
	cmd:option('-blur_type', 'gamma2.2', 'camera response function: linear | gamma2.2')
	cmd:option('-scale_levels', 3, '1 for base scale only, multiply 0.5 scale_levels times')
	-- model:
	cmd:option('-model', 'ResNet', 'type of model to construct: ConvNet | ResNet')
	cmd:option('-supWidth', 256, 'supWidth')
	cmd:option('-nStates', 64, '# of hidden units')
	cmd:option('-filtsize', 5, 'filter size')
	cmd:option('-nlayers', 40, 'number of conv layers at each scale: at least 1')
	cmd:option('-reduce_model', false, 'reduce pre-trained model features')
	cmd:option('-reduce_method', 'simple', 'model reduction method: simple | cluster')
	cmd:option('-prune_ratio', 0, 'ratio of feature maps to be zeroed out. 0 <= r < 1')
	-- loss:
	cmd:option('-abs_weight', 0, 'weight of L1 loss. At least one loss should be positive')
	cmd:option('-mse_weight', 1, 'weight of L2 loss. At least one loss should be positive')
	cmd:option('-adv_weight', 1e-4, 'weight of adversarial loss. At least one loss should be positive')
	-- training:
	now = os.date("%Y-%m-%d %H-%M-%S")
	cmd:option('-save', now, 'subdirectory to save/log experiments in')
	cmd:option('-nEpochs', -1, 'Number of total epochs to run')
	cmd:option('-epochNumber', 1, 'Manual epoch number (useful on restarts)')
	cmd:option('-epochbatchSize', 4000, 'epoch batch size')
	cmd:option('-minibatchSize', 4, 'mini-batch size')
	cmd:option('-loadmodel', '.', 'load pretrained network')
	cmd:option('-train_only', false, 'if true, do not test while training: true | false')
	-- optimization
	cmd:option('-optimization', 'ADAM', 'optimization method: SGD | ADADELTA | ADAM | RMSPROP')
	cmd:option('-rateLearning', 5e-5, 'initial learning rate')
	cmd:option('-weightDecay', 0, 'weight decay (SGD only)')--1e-6
	cmd:option('-momentum', 0.9, 'momentum (SGD only)')
	cmd:option('-beta1', 0.9, 'first momentum coefficient (ADAM)')
	cmd:option('-beta2', 0.999, 'first momentum coefficient (ADAM)')
	cmd:option('-epsilon', 1e-8, 'first momentum coefficient (ADAM)')
	cmd:option('-type', 'cuda', 'type: float | cuda | cudaHalf')
	
	-- continue experiment
	cmd:option('-load', false, 'load trained data. You may continue training')
	cmd:option('-continue', false, 'continue experiment.')
	
	cmd:text()
	local opt = cmd:parse(arg or {})
	opt.save = paths.concat('../experiment', opt.save)
	if opt.load or opt.continue then
		local opt_old = torch.load(paths.concat(opt.save, 'opt'));
		local update_list = {}
		table.insert(update_list, 'save')
		table.insert(update_list, 'gpuid')
		table.insert(update_list, 'optimization')
		table.insert(update_list, 'rateLearning')
		table.insert(update_list, 'load')
		table.insert(update_list, 'continue')
		table.insert(update_list, 'nEpochs')
		table.insert(update_list, 'epochNumber')
		table.insert(update_list, 'dataset')
		table.insert(update_list, 'seed')
		table.insert(update_list, 'threads')
		table.insert(update_list, 'epochbatchSize')
		table.insert(update_list, 'minibatchSize')
		table.insert(update_list, 'train_only')
		table.insert(update_list, 'blur_type')
		table.insert(update_list, 'type')
		if opt.reduce_model then
			table.insert(update_list, 'nStates')
		end
		
		for key, value in next, opt_old do -- do not use ipairs
			if not table.ismember(update_list, key) then
				opt[key] = value
			end
		end

		if opt.epochNumber == 1 then -- if not set, then continue from the end
			local model_dir = paths.concat(opt.save, 'models')
			local max_epoch = 1
			for modelname in paths.iterfiles(model_dir) do
				local iter = tonumber(modelname:sub(7, -4))
				if iter then	-- nil if not number
					max_epoch = math.max(max_epoch, iter)
				end
			end
			opt.epochNumber = max_epoch + 1
		end
	end
	
	-- nb of threads and fixed seed (for repeatable experiments)
	if opt.threads <= 0 then
		opt.threads = 1
	end
	torch.setnumthreads(opt.threads)
	
	if opt.seed == 0 then -- not fixed
		opt.seed = torch.seed()
	else
		torch.manualSeed(opt.seed)
	end
	print(string.format('Starting main thread with seed: %d', opt.seed))
	
	if opt.nEpochs <= 0 then
		opt.nEpochs = math.huge	-- train forever
	end
	
	if opt.type == 'float' then
		print('==> switching to floats')
		operate_type = default_type
	elseif opt.type:find('cuda') then
		print('==> switching to CUDA')
		if opt.type == 'cuda' then
			operate_type = 'torch.CudaTensor'
		elseif opt.type == 'cudaHalf' then
			operate_type = 'torch.CudaHalfTensor'
			if not (opt.load or opt.continue) then
				opt.epsilon = math.sqrt(opt.epsilon)
			end
		end
		cutorch.setDevice(opt.gpuid)
		-- if cutorch.getDeviceCount() >= (opt.gpuid + opt.ngpu - 1) then
		-- 	cutorch.setDevice(opt.gpuid)
		-- end
	end

	return opt
end

return M
