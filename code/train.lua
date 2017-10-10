require 'torch'   -- torch
require 'torchx'
require 'cutorch'
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'image'   -- for rotating and flipping patches
require 'math'    -- to calculate base kernels
require 'gnuplot' -- to visualize error plot

function draw_error_plot(error_table, mode, error_type)
	-- visualize error plot
	
	local mode = mode or 'Train'	-- 'Train' or 'Test'
	local legend = mode .. ' ' .. error_type
	local title = mode .. ' Plot (' .. error_type .. ')'
	local filename =  paths.concat(opt.save, title .. '.pdf')
	local n = gnuplot.pdffigure(filename)
	
	if error_type:lower() == 'entropy' then
		local legend = {'gen', 'fake', 'real'}
		local error_tensors = {}
		error_tensors.gen = torch.Tensor(error_table.gen)
		error_tensors.fake = torch.Tensor(error_table.fake)
		error_tensors.real = torch.Tensor(error_table.real)

		gnuplot.plot(
			{legend[1], error_tensors.gen, '-'},
			{legend[2], error_tensors.fake, '-'},
			{legend[3], error_tensors.real, '-'}
		)
	else
		local error_tensor = torch.Tensor(error_table)
		gnuplot.plot(legend, error_tensor, '-')
	end

	gnuplot.grid(true)
	gnuplot.title(title)
	if error_type == 'PSNR' then
		gnuplot.movelegend('right', 'bottom')
	elseif error_type == 'MSE' then
		gnuplot.movelegend('right', 'top')
	-- else
	-- 	if error_table[#error_table] > error_table[1] then
	-- 		gnuplot.movelegend('right', 'bottom')
	-- 	else
	-- 		gnuplot.movelegend('right', 'top')
	-- 	end
	end
	gnuplot.xlabel('iteration')
	gnuplot.plotflush(n)
	
	gnuplot.closeall()
end

----------------------------------------------------------------------
print '==> defining some tools'
----------------------------------------------------------------------
print '==> configuring optimizer'
print('optimization algorithm : '..opt.optimization)

function set_state(optimization)
	local optimization = optimization or opt.optimization
	local optimState, optimMethod
	if optimization == 'SGD' then
		optimState = {
			learningRate = opt.rateLearning,
			weightDecay = opt.weightDecay,
			momentum = opt.momentum,
			dampening = 0,
			learningRateDecay = 1e-5,
			nesterov = true
		}
		optimMethod = optim.sgd
	elseif optimization == 'ADADELTA' then
		optimState = {
			weightDecay = opt.weightDecay
		}
		optimMethod = optim.adadelta;
	elseif optimization == 'ADAM' then
		optimState = {
			learningRate = opt.rateLearning,
			beta1 = opt.beta1,
			beta2 = opt.beta2,
			epsilon = opt.epsilon,
			weightDecay = opt.weightDecay
		}
		optimMethod = optim.adam
	elseif optimization == 'RMSPROP' then
		optimState = {
			learningRate = opt.rateLearning
		}
		optimMethod = optim.rmsprop
	else
		error('unknown optimization method')
	end

	return optimState, optimMethod
end

function load_state()
	-- assume set_state is called before
	local new_state = torch.load(paths.concat(opt.save, 'train_state.t7'))
	new_state.G.learningRate = optimState.G.learningRate
	if adv_train then
		new_state.D.learningRate = optimState.D.learningRate
	end

	return new_state
end

optimState, optimMethod = {}, {}
optimState.G, optimMethod.G = set_state(opt.optimization)
if adv_train then
	optimState.D, optimMethod.D = set_state(opt.optimization)
end
----------------------------------------------------------------------
function load_record(mode, type)
	local mode = mode or 'train'	-- train or test
	local filename = paths.concat(opt.save, mode .. '_' .. type ..'.t7')	--	ex) train_error.t7
	assert(paths.filep(filename), 'No ' .. mode .. ' ' .. type .. ' record found!')
	
	return torch.load(filename)
end

function save_record(mode, type, loss)
	local mode = mode or 'train'	-- train or test
	local filename = paths.concat(opt.save, mode .. '_' .. type ..'.t7')
	torch.save(filename, loss)

	return
end

if opt.load or opt.continue then
	
	if paths.filep(paths.concat(opt.save, 'state.t7')) then
		optimState = load_state()
	end

	train_error = load_record('train', 'error')
	train_psnr = load_record('train', 'psnr')
	if adv_train then
		train_entropy = load_record('train','entropy')
	end
else
	train_error = {}
	train_psnr = {}
	if adv_train then
		train_entropy = {}
		train_entropy.gen, train_entropy.fake, train_entropy.real = {}, {}, {}
	end

end

----------------------------------------------------------------------
print '==> defining training procedure'

local abs, mse = 0, 0
local entropy = {gen = 0, fake = 0, real = 0}
local loss = 0

local blur, sharp, deblurred
local gt = {}
local true_label, false_label
if adv_train then
	true_label = torch.ones(opt.minibatchSize):cuda()
	false_label = torch.zeros(opt.minibatchSize):cuda()
	gt[2] = true_label
end

local feval = {}
feval.G = function(x)
	model.G:zeroGradParameters()

	local output = model_container:forward(blur)
	if adv_train == false then
		deblurred, gt = output, sharp
	elseif adv_train == true then
		deblurred, gt[1] = output[1], sharp
		output_label = output[2]
	end

	loss = criterion_container(output, gt)
	abs = criterion_container.criterions[1].criterions[1].criterions[1].output
	mse = criterion_container.criterions[1].criterions[1].criterions[2].output
	if adv_train then
		entropy.gen = criterion_container.criterions[2].output
	end

	model_container:backward(blur, criterion_container.gradInput)

	return loss, gradParameters.G
end

feval.D = function(x)
	model.D:zeroGradParameters()

	-- train with Generator output as negative example
	entropy.fake = criterion.D(output_label, false_label)
	model.D:backward(deblurred, criterion.D.gradInput)
	-- train with GT as a positive example
	output_label = model.D:forward(sharp)
	entropy.real = criterion.D(output_label, true_label)
	model.D:backward(sharp, criterion.D.gradInput)

	return entropy.fake + entropy.real, gradParameters.D

end

function trainBatch(inputs, targets, shuffle)
	cutorch.synchronize()
	
	blur = inputs
	sharp = {}
	for lv, lv_patch in ipairs(targets) do
		sharp[lv] = lv_patch:cuda()
	end
	
	optimMethod.G(feval.G, parameters.G, optimState.G)
	if adv_train then
		optimMethod.D(feval.D, parameters.D, optimState.D)
	end
	cutorch.synchronize()
	
	return
end

function train()
	print('==> doing epoch on training data:')
	print("==> online epoch # " .. epoch .. ' [mini-batchSize = ' .. opt.minibatchSize .. ']')

	-- local vars
	local timer = torch.Timer()
	
	-- set model to training mode (for modules that differ in training and testing, like Dropout)
	cutorch.synchronize()
	model.G:training()
	if adv_train then	model.D:training()	end

	local cABS, cMSE = 0, 0
	local cLoss = 0
	local cPSNR = 0
	local cEntropy = {gen = 0, fake = 0, real = 0}
	local minibatch_count = 0

	local function cumulate_error()
		cABS = cABS + abs
		cMSE = cMSE + mse
		cPSNR = cPSNR - 10*math.log10(mse)

		cLoss = cLoss + loss

		if adv_train then
			cEntropy.gen = cEntropy.gen + entropy.gen
			cEntropy.fake = cEntropy.fake + entropy.fake
			cEntropy.real = cEntropy.real + entropy.real
		end
		
	end
	do
		local opt = opt
		local train_list = train_list
		for i = 1, opt.epochbatchSize, opt.minibatchSize do
			-- queue jobs to data-workers
			donkeys:addjob(
				-- the job callback (runs in data-worker thread)
				function()
					return generate_batch()
				end,
				-- the end callback (runs in the main thread)
				function(input_batch, target_batch)
					trainBatch(input_batch, target_batch)
					cumulate_error()
					minibatch_count = minibatch_count + 1
					xlua.progress(minibatch_count*opt.minibatchSize, opt.epochbatchSize)
				end
			)
		end
	end
	
	donkeys:synchronize()
	cutorch.synchronize()
	
	train_error[epoch] = cLoss / minibatch_count
	draw_error_plot(train_error, 'Train', 'Loss')
	
	train_psnr[epoch] = cPSNR / minibatch_count	-- this is meaningless when data term is not mse
	draw_error_plot(train_psnr, 'Train', 'PSNR')
	print('average PSNR(train) : ' .. train_psnr[epoch])
	
	if adv_train then
		train_entropy.gen[epoch] = cEntropy.gen / opt.adv_weight / minibatch_count
		train_entropy.fake[epoch] = cEntropy.fake / opt.adv_weight / minibatch_count
		train_entropy.real[epoch] = cEntropy.real / opt.adv_weight / minibatch_count
		draw_error_plot(train_entropy, 'Train', 'Entropy')
		print('average Entropy(gen) : ' .. train_entropy.gen[epoch])
		print('average Entropy(fake) : ' .. train_entropy.fake[epoch])
		print('average Entropy(real) : ' .. train_entropy.real[epoch])
	end
	-- time taken
	local time = timer:time().real / 60
	print("==> time to learn 1 epoch = " .. time .. ' min')
	
	collectgarbage()
	collectgarbage()
	
	-- save/log current net
	do
		opt.epochNumber = epoch
		filename = paths.concat(opt.save, 'opt')
		torch.save(filename, opt)

		save_main_model(epoch)
		save_record('train', 'state', optimState)
		
		save_record('train', 'error', train_error)
		save_record('train', 'psnr', train_psnr)
		if adv_train then
			save_record('train', 'entropy', train_entropy)
		end
	end
	
end

