require 'torch'
require 'xlua'
require 'optim'
require 'nn'
require 'nnx'
require 'nngraph'
require 'hdf5'
require 'string'
require 'image'
require 'paths'

ffi = require 'ffi'
eval = require('evaluate');

local term = require 'term'
local colors = term.colors
function writeformatted(set, loss, f1, color)
	local f1string
	if type(f1) == "table" then
		f1string = string.format("%.7f" % {f1[1].f1})
		for i = 2,#f1 do
			f1string = string.format(f1string .. ",%.7f" % f1[i].f1)
		end
	else
		f1string = string.format("%.7f" % f1)
	end
	print(string.format("      %s%s%s : Loss: %.7f, F1: %s%s%s" % { color, set, colors.reset, loss, color, f1string, colors.reset }))
end

torch.setdefaulttensortype('torch.FloatTensor')

-- Options
paths.dofile('opts.lua')
opt.nOutChannels = 1

-- Output directory
local workingDir = './snapshots/'
if opt.directory ~= '' then
	workingDir = paths.concat(workingDir, opt.directory)
end

-- Data loading
local DataLoader = require 'dataloader'
trainLoader, validLoader = DataLoader.create(opt)

-- Initialise Logging 
if not Logger then paths.dofile('Logger.lua') end

if opt.model == 'none' then
	--- Load up network model or initialize from scratch
	paths.dofile('models/' .. opt.netType .. '.lua')

	print('==> Creating model from file: models/' .. opt.netType .. '.lua')
	model = createModel(modelArgs)

else
	print ("==> Loading model from existing file:", opt.model)
	model = torch.load(opt.model)
end

if opt.GPU ~= -1 then
	-- Convert model to CUDA
	print('==> Converting model to CUDA')
	model:cuda()
	
	cudnn.fastest = true
	cudnn.benchmark = true
end

-- Criterion
criterion = nn.ParallelCriterion()
for i = 1,opt.nStack do
	criterion:add(nn[opt.crit .. 'Criterion'](), 0.5)
end
criterion:add(nn[opt.crit .. 'Criterion'](), 1.0)
criterion:add(nn[opt.crit .. 'Criterion'](), 1.0)


criterion:cuda()

if opt.optimState == 'none' then
	-- Optimisation
	print('==> Creating optimState from scratch')
	optfn = optim[opt.optMethod]
	if not optimState then
		optimState = {
			learningRate = opt.LR,
			learningRateDecay = opt.LRdecay,
			momentum = opt.momentum,
			weightDecay = opt.weightDecay,
			alpha = opt.alpha,
			epsilon = opt.epsilon
		}
	end
else
	print ("==> Loading optimState from existing file: ", opt.optimState)
	optimState = torch.load(opt.optimState)
	optfn = optim[opt.optMethod]
end

paths.dofile('train.lua')

-- Train and validate for a while
local loss, f1 = 0.0, 0.0
local tp, fp, tn , fn = 0, 0, 0
local scores, f1scores, classScores, classAcc

local epoch = 1
if optimState.epoch then
	epoch = optimState.epoch + 1
	print ("Resuming from optimState at epoch " .. epoch)
	
end

local continue = epoch > 1
log = {}
log.train = Logger(paths.concat(workingDir, 'train.log'), continue)
log.train:setNames{'epoch', 'loss', 'tp', 'fp' ,'fn', 'tn', 'recall', 'precision', 'f1', 'accuracy', 'lr'}
log.valid = Logger(paths.concat(workingDir, 'valid.log'), continue)
log.valid:setNames{'epoch', 'loss', 'tp', 'fp' ,'fn', 'tn', 'recall', 'precision', 'f1', 'accuracy', 'lr'}

print ("Working directory: " .. workingDir)

while epoch <= opt.nEpochs do
	print ("Epoch " .. epoch)

  loss, scores = train()

	f1scores = eval.calculateMultiF1(scores)

	writeformatted('Train', loss, f1scores, colors.green)
  print(scores)
  print(f1scores)
  
	-- Logging
	if log['train'] then
		log['train']:add{
			string.format("%d" % epoch),
			string.format("%.6f" % loss),
			
			string.format("%d" % scores[1].tp),
			string.format("%d" % scores[1].fp),
			string.format("%d" % scores[1].fn),
			string.format("%d" % scores[1].tn),
      
			string.format("%.6f" % f1scores[1].recall),
			string.format("%.6f" % f1scores[1].precision),
			string.format("%.6f" % f1scores[1].f1),
			string.format("%.6f" % f1scores[1].accuracy),

			string.format("%g" % optimState.learningRate)
		}
	end

	-- If we are validating this epoch
	if (opt.validate ~= 0 and epoch % opt.validate == 0) then
		local multiscore
		local multiloss = 0
		

		for i = 1,opt.validateIterations do

			loss, scores = valid();
      
			multiloss = multiloss + loss
			if not multiscore then
				multiscore = scores
			else
				for s = 1,#multiscore do
					multiscore[s].tp = multiscore[s].tp + scores[s].tp
					multiscore[s].fp = multiscore[s].fp + scores[s].fp
          multiscore[s].tn = multiscore[s].tn + scores[s].tn
					multiscore[s].fn = multiscore[s].fn + scores[s].fn
				end
        
			end
      
		end

		f1scores = eval.calculateMultiF1(multiscore)
		writeformatted('Valid', multiloss / opt.validateIterations, f1scores, colors.cyan)
    print(multiscore)
    print(f1scores)
    
    collectgarbage()
    
	-- Logging
		if log['valid'] then
			log['valid']:add{
				string.format("%d" % epoch),
				string.format("%.6f" % loss),
				
				string.format("%d" % multiscore[1].tp),
				string.format("%d" % multiscore[1].fp),
				string.format("%d" % multiscore[1].fn),
				string.format("%d" % multiscore[1].tn),
        
				string.format("%.6f" % f1scores[1].recall),
				string.format("%.6f" % f1scores[1].precision),
				string.format("%.6f" % f1scores[1].f1),
				string.format("%.6f" % f1scores[1].accuracy),
        
				string.format("%g" % optimState.learningRate)
			}
		end--]]
	end

	optimState.epoch = epoch

	-- If snapshotting on this epoch
	if epoch % opt.snapshot == 0 then
		print ("Saving model")
		model:clearState()
		torch.save(paths.concat(workingDir, 'model_' .. epoch .. '.t7') , model)
		torch.save(paths.concat(workingDir, 'optimState_' .. epoch .. '.t7'), optimState)
	end

	if (epoch % opt.LRStep == 0) then
		optimState.learningRate = optimState.learningRate * opt.LRStepGamma
		print ("Reducing learning rate to " .. optimState.learningRate)
	end
	epoch = epoch + 1
end
