require 'torch'		-- torch
require 'torchx'
require 'cutorch'
require 'xlua'		-- xlua provides useful tools, like progress bars
require 'optim'		-- an optimization package, for online and batch methods
require 'image'		-- for rotating and flipping patches
require 'math'		-- to calculate base kernels

function generate_pyramid(img, scale_levels)
	
	local scale_levels = scale_levels or 1
	local scales = {}
	for i = 1, scale_levels do
		scales[i] = 0.5^(i-1)
	end

	local average = img:mean(2):mean(3):squeeze()
	for channel = 1, img:size(1) do
		img[channel] = img[channel] - average[channel] 
	end

	local pyramid = image.gaussianpyramid(img, scales)
	for lv = 1, scale_levels do
		for channel = 1, img:size(1) do
			pyramid[lv][channel] = pyramid[lv][channel] + average[channel]
		end
	end
	
	return pyramid
end

-- function extract_patch(train_list, seq_id, frame_id, merge_blur, supWidth, supHeight, scale_levels)
function extract_patch(seq_id, frame_id, supWidth, supHeight, scale_levels)
	
	local supWidth = supWidth or opt.supWidth
	local supHeight = supHeight or supWidth
	local scale_levels = scale_levels or opt.scale_levels

	local imsize = image.getSize(train_list[seq_id][sharp_key][frame_id])
	local scale = 0.5^(scale_levels-1)

	local lux_s = torch.random(0, math.floor((imsize[3]-supWidth)*scale))
	local luy_s = torch.random(0, math.floor((imsize[2]-supHeight)*scale))
	local lux, luy = lux_s/scale, luy_s/scale	-- prevent translation while downsampling

	local input_patch = image.crop(image.load(train_list[seq_id][blur_key][frame_id]), lux, luy, lux+supWidth, luy+supHeight)
	local target_patch = image.crop(image.load(train_list[seq_id][sharp_key][frame_id]), lux, luy, lux+supWidth, luy+supHeight)

	collectgarbage()
	collectgarbage()

	return input_patch, target_patch

end

function augment_patch(input_patch, target_patch)
	
	local target_input = torch.random(1, 10) == 1 -- sharp input to sharp output
	local change_saturation = torch.random(1, 10) == 1
	local flip_h = torch.random(0, 1) == 1
	local rotate = torch.random(0, 3)
	
	local shuffle_color = true
	local add_noise = true

	if target_input then
		input_patch = target_patch:clone()
	end

	if flip_h then
		input_patch = image.hflip(input_patch)
		target_patch = image.hflip(target_patch)
	end
	
	if rotate > 0 then
		local theta = math.pi/2 * rotate
		input_patch = image.rotate(input_patch, theta)
		target_patch = image.rotate(target_patch, theta)
	end

	if shuffle_color then
		local nChannel = input_patch:size(1)
		local perm = torch.randperm(nChannel):long()

		input_patch = input_patch:index(1, perm)
		target_patch = target_patch:index(1, perm)
	end

	if change_saturation then
		local amp_factor =  1 + torch.uniform(-0.5, 0.5)
		local input_hsv = image.rgb2hsv(input_patch)
		local target_hsv = image.rgb2hsv(target_patch)

		input_hsv[2]:mul(amp_factor)
		target_hsv[2]:mul(amp_factor)

		input_patch = image.hsv2rgb(input_hsv)
		target_patch = image.hsv2rgb(target_hsv)
	end

	if add_noise then
		local sigma_sigma = 2/255
		local sigma = torch.randn(1)[1] * sigma_sigma
		local noise = torch.randn(input_patch:size()) * sigma
		input_patch:add(noise)
	end

	input_patch:clamp(0, 1)
	target_patch:clamp(0, 1)

	collectgarbage()
	collectgarbage()
	
	return input_patch, target_patch

end

function generate_batch(batch_size, scale_levels, supWidth, supHeight)
	
	local batch_size = batch_size or opt.minibatchSize
	local scale_levels = scale_levels or opt.scale_levels
	local supWidth = supWidth or opt.supWidth
	local supHeight = supHeight or supWidth
	-- local merge_blur = merge_blur or 0	-- merge subsequent blurry images to generate even larger blurs

	local input_batch, target_batch = {}, {}
	for lv = 1, scale_levels do
		local scale = 0.5^(lv-1)
		local supHeight_lv = supHeight * scale
		local supWidth_lv = supWidth * scale
		input_batch[lv] = torch.zeros(batch_size, 3, supHeight_lv, supWidth_lv)
		target_batch[lv] = torch.zeros(batch_size, 3, supHeight_lv, supWidth_lv)
	end
	
	local input_patch, target_patch
	local input_patch_pyramid, target_patch_pyramid

	local seq_prob = torch.ones(#train_list)
	local patch_seq_id = torch.multinomial(seq_prob, batch_size, false)
	for patch_id = 1, batch_size do
		local seq_id = patch_seq_id[patch_id]
		local frame_id = torch.random(#train_list[seq_id][sharp_key])
		
		-- extract
		input_patch, target_patch = extract_patch(seq_id, frame_id, supWidth, supHeight, scale_levels)
		-- augment
		input_patch, target_patch = augment_patch(input_patch, target_patch)
		-- tug in
		input_patch_pyramid = generate_pyramid(input_patch, scale_levels)
		target_patch_pyramid = generate_pyramid(target_patch, scale_levels)
		for lv = 1, scale_levels do
			input_batch[lv][patch_id] = input_patch_pyramid[lv]
			target_batch[lv][patch_id] = target_patch_pyramid[lv]
		end
	end
	
	collectgarbage()
	collectgarbage()

	return input_batch, target_batch
	
end

function generate_testpair(seq_id, frame_id, scale_levels)

	local scale_levels = scale_levels or opt.scale_levels

	local input_img = image.load(test_list[seq_id][blur_key][frame_id])
	local target_img = image.load(test_list[seq_id][sharp_key][frame_id])
	
	local input_img_pyramid = generate_pyramid(input_img, scale_levels)
	local target_img_pyramid = generate_pyramid(target_img, scale_levels)
	for lv = 1, scale_levels do
		input_img_pyramid[lv] = input_img_pyramid[lv]:repeatTensor(1,1,1,1) 
		target_img_pyramid[lv] = target_img_pyramid[lv]:repeatTensor(1,1,1,1)
	end

	collectgarbage()
	collectgarbage()
	
	return input_img_pyramid, target_img_pyramid
end