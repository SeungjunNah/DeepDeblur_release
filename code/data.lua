require 'torch'   -- torch
require 'torchx'

local threads = require 'threads'
threads.serialization('threads.sharedserialize')

print('==> Reading Dataset '..opt.dataset)

function get_imglist(directory, extlist)
	
	local temp_list = paths.indexdir(directory, extlist)
	local list = {}
	for i = 1, temp_list:size() do
		table.insert(list, temp_list:filename(i))
	end
	table.sort(list)

	return list
end

----------------------------------------------------
--train_list[seq_id]['blur'][frame_id]
--train_list[seq_id]['blur_gamma'][frame_id]
--train_list[seq_id]['sharp'][frame_id]
--test_list[seq_id]['blur'][frame_id]
--test_list[seq_id]['blur_gamma'][frame_id]
--test_list[seq_id]['sharp'][frame_id]
----------------------------------------------------
local blur_key_linear, blur_key_gamma = 'blur', 'blur_gamma'
sharp_key = 'sharp'
if opt.blur_type == 'linear' then
	blur_key = blur_key_linear
elseif opt.blur_type == 'gamma2.2' then
	blur_key = blur_key_gamma
else
	error('unknown camera response function')
end


local datadir = paths.concat('../dataset', opt.dataset)
local train_dir = paths.concat(datadir, 'train')
local test_dir = paths.concat(datadir, 'test')

local data_list
train_list, test_list = {}, {}

for subset in paths.iterdirs(datadir) do
	local subdir
	local sublist
	if subset == 'train' then
		subdir = train_dir
		sublist = train_list
	elseif subset == 'test' then
		subdir = test_dir
		sublist = test_list
	else	-- no train / test division
		subdir = datadir
		sublist = data_list
	end

	local sequences = {}
	for seq_name in paths.iterdirs(subdir) do
		table.insert(sequences, seq_name)
	end
	table.sort(sequences)

	for seq_id, seq_name in ipairs(sequences) do
		local sequence_name = paths.concat(subdir, seq_name)
		
		local blur_dir = paths.concat(sequence_name, blur_key)
		local sharp_dir = paths.concat(sequence_name, sharp_key) 
		
		sublist[seq_id] = {}
		sublist[seq_id][blur_key] = get_imglist(blur_dir)
		sublist[seq_id][sharp_key] = get_imglist(sharp_dir)
	end
end

if #train_list == 0 or #test_list == 0 then
	train_list = data_list
	test_list = data_list
	-- full_data = true
end

average = 0.5

do	-- initialize data loading threads
	if opt.nDonkeys > 0 then
		local def_type = default_type
		local options = opt -- make an upvalue to serialize over to donkey threads
		local list_train, list_test = train_list, test_list
		local key_blur, key_sharp = blur_key, sharp_key
		donkeys = threads.Threads(
			opt.nDonkeys,
			function(threadid)
				require 'torch'
				require 'image'
				
				return threadid
			end,
			function(threadid)
				default_type = def_type
				torch.setdefaulttensortype(default_type)
				opt = options
				blur_key, sharp_key = key_blur, key_sharp
				tid = threadid
				local seed = torch.seed()
				print(string.format('Starting donkey with id: %d seed: %d', tid, seed))
				train_list, test_list = list_train, list_test
				blur_key, sharp_key = key_blur, key_sharp
				paths.dofile('donkey.lua')
			end
		)
	else	-- single threaded data loading. Useful for debugging.
		paths.dofile('donkey.lua')
		donkeys = {}
		function donkeys:addjob(f1, f2) f2(f1()) end
		function donkeys:synchronize() end
	end
end

collectgarbage()
collectgarbage()