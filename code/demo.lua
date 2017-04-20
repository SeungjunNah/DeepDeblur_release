-- To run demo, type
-- qlua -i demo.lua -load -save 'scale3-depth40_adv'

require 'torch'
require 'cutorch'
require 'paths'
require 'xlua'
require 'optim'

default_type = 'torch.FloatTensor'
torch.setdefaulttensortype(default_type)

local opts = paths.dofile('opts.lua')
opt = opts.parse(arg)

-- nb of threads and fixed seed (for repeatable experiments)
torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)
----------------------------------------------------------------------
----------------------------------------------------------------------
dofile 'model.lua'
dofile 'train.lua'
dofile 'test.lua'
----------------------------------------------------------------------
----------------------------------------------------------------------

local example_dir = paths.concat('..', 'dataset', 'examples')
local image_dir, output_dir

if opt.blur_type == 'linear' then
    image_dir = paths.concat(example_dir, 'blur_lin')
    output_dir = paths.concat(example_dir, 'deblurred_lin')
elseif opt.blur_type == 'gamma2.2' then
    image_dir = paths.concat(example_dir, 'blur_gamma')
    output_dir = paths.concat(example_dir, 'deblurred_gamma')
end

demo(image_dir, output_dir)

require 'trepl'()