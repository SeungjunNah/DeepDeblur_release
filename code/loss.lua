require 'torch'
require 'nn'
require 'cunn'

if not opt then
	local opts = paths.dofile('opts.lua')
	opt = opts.parse(arg)
    dofile 'data.lua'
end

adv_train = opt.adv_weight > 0

local data_term = nn.MultiCriterion()
do
	local abs = nn.AbsCriterion();	abs.sizeAverage = true;
	data_term:add(abs, opt.abs_weight)
	local mse = nn.MSECriterion();	mse.sizeAverage = true;
	data_term:add(mse, opt.mse_weight)
end

local adv_loss
if adv_train then
	local weights = torch.Tensor(opt.minibatchSize):fill(1/torch.log(2))
	adv_loss = nn.BCECriterion(weights * opt.adv_weight)
end

criterion = {}
criterion.G = nn.ParallelCriterion()
for lv = 1, opt.scale_levels do
	criterion.G:add(data_term:clone())
end
criterion.D = adv_loss

criterion_container = nn.ParallelCriterion()
	:add(criterion.G)
if adv_train then
	criterion_container:add(criterion.D)
end

if opt.type == 'cuda' then
	criterion.G = criterion.G:cuda()
	if adv_train then
		criterion.D = criterion.D:cuda()
	end
	criterion_container = criterion_container:cuda()
end

----------------------------------------------------------------------
print '==> here is the loss function:'
print(criterion_container)
