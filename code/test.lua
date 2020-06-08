require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'image'

----------------------------------------------------------------------
-- parse command line arguments

if not opt then
	local opts = paths.dofile('opts.lua')
	opt = opts.parse(arg)
end

----------------------------------------------------------------------
print '==> defining test procedure'

if opt.load or opt.continue then
	test_error = load_record('test', 'error')
	test_psnr = load_record('test', 'psnr')
else
	test_error = {}
	test_psnr = {}
	save_record('test', 'error', test_error)
	save_record('test', 'psnr', test_psnr)
end

local test_dir = paths.concat(opt.save, 'test')
if not paths.dirp(test_dir) then
	paths.mkdir(test_dir)
end

function get_output(input_img_pyramid)

	local temp = input_img_pyramid
	local temp_img_pyramid = {}
	local output_img_pyramid = {}
	if opt.scale_levels == 1 then
		temp = input_img_pyramid[1]
		for i = 1, #model.G:get(1) do
			temp = model.G:get(1):get(i):clone():forward(temp)
			collectgarbage()
			collectgarbage()
		end
		output_img_pyramid[1] = temp:float()
	else	-- opt.scale_levels > 1
		for i = 1, #model.G do
			if i < opt.scale_levels then
				temp = model.G:get(i):clone():forward(temp)
				temp_img_pyramid[opt.scale_levels-i+1] = temp[opt.scale_levels-i+2]:clone() 
			elseif i == opt.scale_levels then
				local finemodel = model.G:get(i):get(1):get(1):get(2):clone()
				for j = 1, #finemodel do
					if j == 1 then
						temp = finemodel:get(j):clone():forward({table.unpack(temp, 1, 2)})
					else
						temp = finemodel:get(j):clone():forward(temp)
					end
					collectgarbage()
					collectgarbage()
				end
				temp_img_pyramid[1] = temp:clone()
			elseif i == opt.scale_levels + 1 then
				output_img_pyramid = model.G:get(i):clone():forward(temp_img_pyramid)
				for lv, img in ipairs(output_img_pyramid) do
					output_img_pyramid[lv] = img:float()
				end
			end
		end
	end
	collectgarbage()
	collectgarbage()
	
	return output_img_pyramid
end

function gen_pyramid(img, scale_levels)
	local scale_levels = scale_levels or opt.scale_levels
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
		pyramid[lv] = pyramid[lv]:repeatTensor(1,1,1,1) 
	end
	
	collectgarbage()
	collectgarbage()

	return pyramid
end

function get_output_img(input_img, input_img_name)
	local input_img = input_img
	if not input_img then
		input_img = image.load(input_img_name, 3)
	end

	local orig_input_img = input_img:clone()
	local orig_imsize = orig_input_img:size()

	local height, width = orig_imsize[2], orig_imsize[3]
	local inv_scale = 2^(opt.scale_levels-1)
	local pad_height, pad_width = math.fmod(-height, inv_scale), math.fmod(-width, inv_scale)
	if pad_height < 0 then
		pad_height = pad_height + inv_scale
		local row_to_pad = input_img:sub(1,-1, -1,-1, 1,-1):repeatTensor(1,pad_height,1)
		input_img = torch.cat({input_img, row_to_pad}, 2)
	end
	if pad_width < 0 then
		pad_width = pad_width + inv_scale
		local col_to_pad = input_img:sub(1,-1, 1,-1, -1,-1):repeatTensor(1,1,pad_width)
		input_img = torch.cat({input_img, col_to_pad}, 3)
	end

	local imsize = input_img:size()
	local input_img_pyramid = gen_pyramid(input_img)

	local output_img_pyramid = get_output(input_img_pyramid)
	local output_img = output_img_pyramid[1][1]:sub(1,-1, 1,height, 1,width):contiguous()

	collectgarbage()
	collectgarbage()

	return output_img
end

function get_output_img_part(input_img, input_img_name, xmin, xmax, ymin, ymax, margin)
	local input_img = input_img
	if not input_img then
		input_img = image.load(input_img_name, 3)
	end
	local margin = margin or 16
	
	local output_img = input_img:clone()
	local orig_imsize = output_img:size()

	local height, width = orig_imsize[2], orig_imsize[3]

	local xmin, ymin = xmin or 1, ymin or 1
	local xmax, ymax = xmax or width, ymax or height
	xmin, ymin = math.max(1, xmin), math.max(1, ymin)
	xmax, ymax = math.min(xmax, width), math.min(ymax, height)
	local xmin_, ymin_ = math.max(1, xmin-margin), math.max(1, ymin-margin)
	local xmax_, ymax_ = math.min(xmax+margin, width), math.min(ymax+margin, height)
	local margin_xmin, margin_ymin = xmin - xmin_, ymin - ymin_
	local margin_xmax, margin_ymax = xmax_ - xmax, ymax_ - ymax

	local blur_part = input_img:sub(1,-1, ymin_,ymax_, xmin_,xmax_):contiguous()

	local deblurred_part = get_output_img(blur_part)
	output_img[{{1,-1}, {ymin,ymax}, {xmin,xmax}}]:copy(
		deblurred_part:sub(1,-1, margin_ymin+1,-margin_ymax-1, margin_xmin+1,-margin_xmax-1)
		)

	collectgarbage()
	collectgarbage()

	return output_img
end

function deblur_dir(image_dir, output_dir)
	model.G:evaluate()
	local image_dir = image_dir
	local output_dir = output_dir or paths.concat(opt.save, 'deblur_result')
	if not paths.dirp(output_dir) then
		paths.mkdir(output_dir)
	end
	
	for img_name in paths.iterfiles(image_dir) do
		local fullname = paths.concat(image_dir, img_name)
		local output_img = get_output_img(nil, fullname)
		image.save(paths.concat(output_dir, img_name), output_img)
	end
	collectgarbage()
	collectgarbage()

	return
end

-- test function
function test(epochNumber, save_result, full_data)

	local epochNumber = epochNumber or epoch 
    model.G:evaluate()
	-- local vars
    local timer = torch.Timer()
	
    -- test over test data
    print('==> testing on test set:')
    local cError = 0
	local cPSNR = 0

	local test_list = test_list
	local scale_levels = opt.scale_levels

    local test_count = 0
    for seq_id, seq_name in ipairs(test_list) do
		local test_size = #test_list[seq_id][blur_key]
		if not full_data then
			test_size = math.min(test_size, 10)
		end
		
		for frame_id = 1, test_size do
			-- queue jobs to data-workers
			donkeys:addjob(
				function()
					local seq_id, frame_id = seq_id, frame_id
					local scale_levels = scale_levels
					return generate_testpair(seq_id, frame_id, scale_levels)
				end,
				-- the end callback (runs in the main thread)
				function(input_img_pyramid, target_img_pyramid)
					
					local output_img_pyramid = get_output(input_img_pyramid)
					local mse = (output_img_pyramid[1]:cuda() - target_img_pyramid[1]:cuda()):pow(2):mean()
					local psnr = -10*math.log10(mse)
					
					cError = cError + mse
					cPSNR = cPSNR + psnr
					
					test_count = test_count + 1

					if save_result then
						local input_img_name = test_list[seq_id][blur_key][frame_id]
						local inputname = 'seq_id-'..seq_id..' frame-'.. frame_id .. ' Input.png';
						local outputname = 'seq_id-'..seq_id..' frame-'.. frame_id .. ' Output.png';
						
						file.copy(input_img_name, paths.concat(test_dir, inputname))
						image.save(paths.concat(test_dir, outputname), output_img_pyramid[1]:squeeze(1))
					end
				end
			)
		end
	end
	donkeys:synchronize()
	cutorch.synchronize()
	collectgarbage()
	collectgarbage()

	test_error[epochNumber] = cError / test_count 
	test_psnr[epochNumber] = cPSNR / test_count
	print('average PSNR(test) : ' .. test_psnr[epochNumber])
	
	save_record('test', 'error', test_error)
	save_record('test', 'psnr', test_psnr)
	
	draw_error_plot(test_error, 'Test', 'MSE')
	draw_error_plot(test_psnr, 'Test', 'PSNR')
    

	local time = timer:time().real
	print("==> time to test = " .. time/60 .. ' min')
	print("==> time per image = " .. time/test_count .. ' sec\n')
	
	return
end

-- fill in the missing test error
function follow_up_test(test_begin, test_end, force_all, full_data)
	
	local test_begin = test_begin or 1
	local test_end = test_end or opt.epochNumber - 1
	
	local tested = false 
	for epochNumber = test_begin, test_end do
		if force_all or test_error[epochNumber] == nil then
			model = load_main_model(epochNumber)
			test(epochNumber, false, full_data)
			tested = true
		end
	end
	
	return tested
end

function demo(image_dir, output_dir)
	model.G:evaluate()
	local image_dir = image_dir
	local output_dir = output_dir or paths.concat(opt.save, 'deblur_result')
	if not paths.dirp(output_dir) then
		paths.mkdir(output_dir)
	end
	
	local window
	
	for img_name in paths.iterfiles(image_dir) do
		
		local fullname = paths.concat(image_dir, img_name)
		img = image.load(fullname)
		window = image.display{image = img, min=0, max=1, offscreen = false, win = window}--, gui = false}
		
		local timer = torch.Timer()
		local output_img = get_output_img(nil, fullname)
		local deblur_time = timer:time().real
		print(' ' .. deblur_time .. ' s taken')
		-- window = image.display{image = output_img, offscreen = false, win = window}
		window = image.display{image = output_img, min=0, max=1, offscreen = false, win = window}--, gui = false}
		image.save(paths.concat(output_dir, img_name), output_img)
		sys.sleep(3)
	end
	
end






