require 'torch'
require 'cutorch'
require 'paths'
require 'xlua'
require 'optim'

function record_params(opt)

	if not paths.dirp('../experiment') then
		paths.mkdir('../experiment')
	end
	if not paths.dirp(opt.save) then
		paths.mkdir(opt.save)
		torch.save(paths.concat(opt.save, 'opt'), opt)
	end

	local today = now:sub(1, 11)
	local nowtime = now:sub(12, 13) .. ":" .. now:sub(15, 16) .. ":" .. now:sub(18, 19)
	
	local modelparam = io.open(opt.save .. "/modelparam.txt", "a+")
	modelparam:write("Experiment at " .. today .. nowtime .. "\n\n")
	if not (opt.load or opt.continue) then 
		modelparam:write("Dataset : ".. opt.dataset.."\n")
		modelparam:write("Camera response function : ".. opt.blur_type.."\n")
		modelparam:write("model : " .. opt.model .. "\n")
		modelparam:write("scale_levels : " .. opt.scale_levels .. "\n\n")
		
		modelparam:write("supWidth : " .. opt.supWidth .. "\n")
		modelparam:write("nStates : " .. opt.nStates .. "\n")
		modelparam:write("filtsize : " .. opt.filtsize .. "\n")
		modelparam:write("nlayers : " .. opt.nlayers .. "\n")
	end
	modelparam:write("\n")
	modelparam:write("L1 loss weight : " .. opt.abs_weight .. "\n")
	modelparam:write("L2 loss weight : " .. opt.mse_weight .. "\n")
	modelparam:write("adversarial loss weight : " .. opt.adv_weight .. "\n")
	
	modelparam:write("optimization method : " .. opt.optimization .. "\n")
	if opt.optimization == 'SGD' or opt.optimization == 'ADAM' then
		modelparam:write("learning rate : " .. opt.rateLearning .. "\n")
		if opt.optimization == 'SGD' then
			modelparam:write("momentum : " .. opt.momentum .. "\n")
			modelparam:write("weight decay : " .. opt.weightDecay .. "\n")
		end
	end
	modelparam:write("batch size : " .. opt.epochbatchSize .. "\n")
	modelparam:write("mini-batch size : " .. opt.minibatchSize .. "\n")
	modelparam:write("\n\n")
	modelparam:close()

	return
end

default_type = 'torch.FloatTensor'
torch.setdefaulttensortype(default_type)

local opts = paths.dofile('opts.lua')
opt = opts.parse(arg)
record_params(opt)	-- save experiment parameters

----------------------------------------------------------------------
----------------------------------------------------------------------
print '==> executing all'
dofile 'data.lua'
dofile 'loss.lua'
dofile 'model.lua'
dofile 'train.lua'
dofile 'test.lua'
----------------------------------------------------------------------
----------------------------------------------------------------------

epoch = opt.epochNumber
print('epoch begins : '..epoch)
local epoch_threshold = opt.nEpochs
local slow_down_period = math.ceil(150 * 4000/opt.epochbatchSize)

if opt.load then
	require 'trepl'()
else
	if opt.continue and (not opt.train_only) then
		if follow_up_test(1, epoch-1) == true then
			model = load_main_model(epoch-1)
			parameters, gradParameters = get_model_parameters()
		end
	end
	
	print '==> training!'
	while epoch <= epoch_threshold do
		train()
		if not opt.train_only then
			test(nil, true)
		end
		if epoch == slow_down_step then
			opt.rateLearning = opt.rateLearning / 10
			optimState.G.learningRate = opt.rateLearning
			if adv_train then
				optimState.D.learningRate = opt.rateLearning
			end
		end
		-- next epoch
	    epoch = epoch + 1
	end
	epoch = epoch - 1
	if opt.train_only then
		test(nil, true)
	end
	if opt.train_only then
		follow_up_test(1, epoch-1)
		model = load_main_model(epoch)
		test(epoch, true)
	end
end
