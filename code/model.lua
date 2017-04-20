require 'torch'   -- torch
require 'cutorch' -- cudaTensor
require 'image'   -- for image transforms
require 'nn'      -- provides all sorts of trainable modules/layers
require 'cunn'
require 'cudnn'   -- cudnn

if not opt then
	opt = torch.load(paths.concat(opt.save, 'opt'));
end

cudnn.fastest = true
cudnn.benchmark = true

----------------------------------------------------------------------
print '==> define parameters'
nChannel = 3;

supWidth = opt.supWidth;
filtsize = opt.filtsize
nlayers = opt.nlayers
nStates = opt.nStates
inChannel, outChannel = nChannel, nChannel

model_dir = paths.concat(opt.save, 'models')
if not paths.dirp(model_dir) then
	paths.mkdir(model_dir)
end

function ResBlock(filtsize, nStates, inStates)
	
	local function shortcut(str)
		local str = str or 1
		if str == 1 then
			return nn.Identity()
		else
			local str = 2
			return nn.Sequential()
				:add(nn.SpatialAveragePooling(1,1, str,str))
				:add(nn.Concat(2))
					:add(nn.Identity())
					:add(nn.MulConstant(0))
		end
	end

	local filtsize = filtsize or opt.filtsize
	local padW, padH = (filtsize-1)/2, (filtsize-1)/2
	local nStates = nStates or opt.nStates
	local inStates = inStates or nStates
	local str = ((nStates == inStates) and 1) or 2 
	
	local block = nn.Sequential()
	local concat = nn.ConcatTable()

	local conv1 = nn.SpatialConvolution(inStates,nStates, filtsize,filtsize, str,str, padW,padH)
	local relu = nn.ReLU(true)
	local conv2 = nn.SpatialConvolution(nStates,nStates, filtsize,filtsize, 1,1, padW,padH)

	local path = nn.Sequential()
		:add(conv1):add(relu)
		:add(conv2)
	local concat = nn.ConcatTable()
		:add(path)
		:add(shortcut(str))
	
	local block = nn.Sequential()
		:add(concat)
		:add(nn.CAddTable(true))
	
	return block
end

function ResNet(nlayers, filtsize, inChannel, outChannel, nStates)
	
	local nlayers = nlayers or opt.nlayers
	local filtsize = filtsize or opt.filtsize
	local padW, padH = (filtsize-1)/2, (filtsize-1)/2
	local dW, dH = 1, 1
	local inChannel = inChannel or nChannel
	local outChannel = outChannel or nChannel
	local nStates = nStates or opt.nStates
	
	local model = nn.Sequential()
	local conv = nn.SpatialConvolution(inChannel,nStates, filtsize,filtsize, dW,dH, padW,padH)
	
	model:add(conv:clone())
	for layer = 1, (nlayers-2)/2 do
		model:add(ResBlock(filtsize, nStates))
	end
	conv = nn.SpatialConvolution(nStates,outChannel, filtsize,filtsize, dW,dH, padW,padH)
	model:add(conv:clone())
	
	return model
end

function generate_conv_end(inChannel, outChannel, ratio)
	
	local inChannel = inChannel or nChannel
	local outChannel = outChannel or nChannel
	local ratio = ratio or 2
	-- local filt, pad, adj = ratio, 0, 0
	
	local filt = 5 --3
	local dW,dH = 1,1
	local padW, padH = (filt-1)/2, (filt-1)/2
	local uppath = nn.Sequential()
		:add(nn.SpatialConvolution(outChannel, inChannel*ratio^2, filt,filt, dW,dH, padW,padH))
		:add(nn.PixelShuffle(ratio))
	local conv_end = nn.ConcatTable()
		:add(uppath)
		:add(nn.Identity())
		-- :add(nn.SpatialFullConvolution(outChannel,inChannel,filt,filt,ratio,ratio,pad,pad,adj,adj))
	
	return conv_end
end

function generate_main_model(modeltype)
	local modeltype = modeltype or opt.model
	local scale_levels = opt.scale_levels
	local generate_net

	if modeltype == 'ConvNet' then
		generate_net = ConvNet
	elseif modeltype == 'ResNet' then
		generate_net = ResNet
	else
		error('unknown model type')
	end
	
	local model

	if scale_levels == 1 then
		
		model = generate_net(nlayers, filtsize, inChannel, outChannel, opt.nStates)
		
		model:insert(nn.Copy(default_type, operate_type), 1)
		model:insert(nn.AddConstant(-average, true), 2)
		model:add(nn.AddConstant(average, true))
		model = nn.ParallelTable():add(model)
		
	else
		local conv_coarse = generate_net(nlayers, filtsize, inChannel, outChannel, opt.nStates)
		conv_coarse:add(generate_conv_end(inChannel, outChannel, ratio))
		
		local conv_fine = generate_net(nlayers, filtsize, inChannel+outChannel, outChannel, opt.nStates)
		conv_fine:insert(nn.JoinTable(2), 1)
		conv_fine:add(generate_conv_end(inChannel, outChannel))
		
		local conv_finest = generate_net(nlayers, filtsize, inChannel+outChannel, outChannel, opt.nStates)
		conv_finest:insert(nn.JoinTable(2), 1)
		
		-- local conv_finest = conv_fine:clone()
		-- conv_finest:remove()
		
		model = nn.Sequential()
		do -- coarse
			local submodel = nn.Sequential()
			local submodel_par = nn.ParallelTable()
			for lv = 1, scale_levels do
				local subpath = nn.Sequential()
				subpath:add(nn.Copy(default_type, operate_type))
				subpath:add(nn.AddConstant(-average, true))
				if lv == scale_levels then
					subpath:add(conv_coarse)
				end
				submodel_par:add(subpath)
			end
			submodel:add(submodel_par)
			submodel:add(nn.FlattenTable())
			model:add(submodel)
		end
		for i = scale_levels-1, 2, -1 do -- fine
			local submodel = nn.Sequential()
			local subconcat = nn.ConcatTable()
			
			local subconcat_path1 = nn.NarrowTable(1,i-1)
			local subconcat_path2 = nn.Sequential():add(nn.NarrowTable(i, 2))
			subconcat_path2:add(conv_fine:clone())
			local subconcat_path3 = nn.NarrowTable(i+2, scale_levels-i)
			
			subconcat:add(subconcat_path1)
			subconcat:add(subconcat_path2)
			subconcat:add(subconcat_path3)
			submodel:add(subconcat)
			submodel:add(nn.FlattenTable())
			
			model:add(submodel:clone())
		end
		do -- finest
			local submodel = nn.Sequential()
			local subconcat = nn.ConcatTable()
			local subconcat_path1 = nn.Sequential():add(nn.NarrowTable(1,2))
			subconcat_path1:add(conv_finest)
			
			subconcat:add(subconcat_path1)
			for j = 2, scale_levels do
				subconcat:add(nn.SelectTable(j+1))
			end
			submodel:add(subconcat)
			model:add(submodel)
		end
		do -- post processing
			local endmodel_par = nn.ParallelTable()
			for lv = 1, scale_levels do
				endmodel_par:add(nn.AddConstant(average, true))
			end
			model:add(endmodel_par)
		end
		
	end

	model = cudnn.convert(model, cudnn):cuda()
	model:reset()

	return model
end

function generate_discriminator()
	
	local function conv_block(filtsize, inStates, nStates, str, negval, pad)
		local filtsize = filtsize or 3
		local pad = pad or (filtsize-1)/2
		local str = str or 1
		-- local negval = nil
		local block = nn.Sequential()
			:add(nn.SpatialConvolution(inStates,nStates, filtsize,filtsize, str,str, pad,pad):noBias())
			:add(nn.LeakyReLU(negval, true))
		
		return block
	end

	local filtsize = filtsize or 5
	local pad = (filtsize-1)/2

	local conv_front = nn.SpatialConvolution(nChannel,32, filtsize,filtsize, 1,1, pad,pad):noBias()
	local negval = 0.2 -- nil
	--[[
		local dense = nn.SpatialConvolution(1024,1, 1,1)
		local model = nn.Sequential()
			:add(conv_front):add(nn.LeakyReLU(negval, true))
			:add(conv_block(filtsize, 32,32, 2, negval))	-- 128
			:add(conv_block(filtsize, 32,64, 1, negval))
			:add(conv_block(filtsize, 64,64, 2, negval))	-- 64
			:add(conv_block(filtsize, 64,128, 1, negval))
			:add(conv_block(filtsize, 128,128, 2, negval))	-- 32
			:add(conv_block(filtsize, 128,256, 1, negval))
			:add(conv_block(filtsize, 256,256, 4, negval))	-- 8
			:add(conv_block(filtsize, 256,512, 1, negval))
			:add(conv_block(filtsize, 512,512, 4, negval))	-- 2
			:add(conv_block(filtsize, 512,1024, 2, negval))	-- 1
			:add(dense):add(nn.Sigmoid())
	]]--
	local dense = nn.SpatialConvolution(512,1, 1,1)
	local model = nn.Sequential()
		:add(nn.SelectTable(1))
		:add(conv_front):add(nn.LeakyReLU(negval, true))
		:add(conv_block(filtsize, 32,32, 2, negval))	-- 128
		:add(conv_block(filtsize, 32,64, 1, negval))
		:add(conv_block(filtsize, 64,64, 2, negval))	-- 64
		:add(conv_block(filtsize, 64,128, 1, negval))
		:add(conv_block(filtsize, 128,128, 4, negval))	-- 16
		:add(conv_block(filtsize, 128,256, 1, negval))
		:add(conv_block(filtsize, 256,256, 4, negval))	-- 4
		:add(conv_block(filtsize, 256,512, 1, negval))
		:add(conv_block(4, 512,512, 4, negval, 0))	-- 1	filtsize 5 is equivalent to 4 here
		:add(dense):add(nn.Sigmoid())
		
	model = cudnn.convert(model, cudnn):cuda()
	model:reset()

	return model
end

function load_main_model(epochNumber)
	-- load trained net
	model = nil
	model_container = nil
	parameters, gradParameters = nil, nil
	collectgarbage()
	collectgarbage()

	local epochNumber = epochNumber or opt.epochNumber - 1
	local modelname = paths.concat(opt.save, 'models', 'model-'.. epochNumber .. '.t7')
	print('==> loading model from ' .. modelname)
    
	assert(paths.filep(modelname), 'no trained model found!')
	local model = torch.load(modelname)
	if torch.type(model) ~= 'table' then	-- backward compatibility
		model = {G = model}
		if adv_train then
			model.D = generate_discriminator()
		end
	end
	
	return model
    
end

function save_main_model(epochNumber)

	model.G:clearState()	-- do not use model_container:clearState() 
	if adv_train then
		model.D:clearState()
	end
	collectgarbage()
	collectgarbage()
	local filename = paths.concat(model_dir, 'model-'..epochNumber..'.t7')
	print('==> saving model to ' .. filename..'\n')
	torch.save(filename, model)

	return
end

function get_model_parameters()
	-- assume there is a global variable: model, model.G, (model.D)
	parameters, gradParameters = {}, {}
	for k, v in next, model do
		parameters[k], gradParameters[k] = model[k]:getParameters()
	end

	return parameters, gradParameters
end

----------------------------------------------------------------------
print '==> construct model'

if opt.load or opt.continue then
	model = load_main_model(opt.epochNumber-1)
else
	if paths.filep(opt.loadmodel) then
		model = torch.load(opt.loadmodel)
	else
		model = {}
		model.G = generate_main_model(opt.model)
		if adv_train then
			model.D = generate_discriminator()
		end
	end
end

parameters, gradParameters = get_model_parameters()
----------------------------------------------------------------------
do
	model_container = nn.Sequential()
		:add(model.G)
	if adv_train then
		model_container:add(
			nn.ConcatTable()
				:add(nn.Identity())
				:add(model.D)
		)
	end
end

if not (opt.load or opt.continue) then
	print '==> here is the model: Generator'
	print(model.G)
	if adv_train then
		print '==> here is the model: Discriminator'
		print(model.D)
	end
end

collectgarbage()
collectgarbage()