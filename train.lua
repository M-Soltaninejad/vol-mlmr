-- Main training function
local pool = nn.VolumetricAveragePooling(2,2,2,2,2,2)

function step(tag)
  local avgLoss, avgAcc = 0.0, 0.0
  local avgLoss = 0.0
  local score = {}
  local classScore = { tp = 0, fp = 0, tn = 0, fn = 0 }
  local output, err, idx
  local param, gradparam = model:getParameters()
  local function evalFn(x) return criterion.output, gradparam end
  local images, labels, batch

  if tag == 'train' then
    model:training()
    set = 'train'
    loader = trainLoader
  else
    model:evaluate()
    set = 'valid'
    loader = validLoader
  end

  local nIters,datasetsize,batchsize = loader:size(), loader:datasetsize(), loader:batchsize()

  for i,sample in loader:run() do
    local input = sample.input;
    local masks = sample.mask;

    xlua.progress(i, nIters)

    if opt.GPU ~= -1 then
      -- Convert to CUDA
      input = input:cuda()
      masks = masks:cuda()
    end

    -- Multiple supervision table
    local label;
    if opt.nStack > 1 then
      label = {}
      for t = 1,opt.nStack do
        label[t] = masks
      end
    else
      label = masks 
    end
    -- Do a forward pass and calculate loss
    local output = model:forward(input)


    --local o = {}
    --for j = 1,4 do o[j] = output[j] end
   
   
    -- ************ Depends on initial sub-sampling  *******************
    -- No subsampling
--    local err = criterion:forward(output, label)
    
    -- With 3 subsampling
    local label1 = label:float()

    local new_label = label1:narrow(3, 57, 16):narrow(4, 57, 16):narrow(5, 57, 16)
    
    new_label = new_label:cuda()
    
    local label_total = {}
    -- ********************* comment for original big patch size otherwsie it will resize it
    
    -- *********************
    table.insert(label_total,label)
    table.insert(label_total,new_label)
    table.insert(label_total,new_label)
    
    
--      I = input
--      O = output
--     A = output
--     B = label_total

--    do return end

    local err = criterion:forward(output, label_total)
    
    -- *************************************************************

    avgLoss = avgLoss + err / nIters

    if tag == 'train' then
      -- Training: Do backpropagation and optimization
      model:zeroGradParameters()
      model:backward(input, criterion:backward(output, label_total))
      optfn(evalFn, param, optimState)
    end
--print(#output)

    -- Calculate accuracy
    local predMasks;
    if opt.nStack > 1 then
      predMasks = eval.nms3D(output[opt.nStack]:float(), 0.5)
    else
      predMasks = eval.nms3D(output[3]:float(), 0.5)
    end

    -- ************ Depends on initial sub-sampling  *******************
    -- No subsampling
--    local evaluation = eval.evaluate3D(predMasks, masks:float())
    
    -- With 3 subsampling

    local evaluation = eval.evaluate3D(predMasks, new_label:float())
    -- *************************************************************
    

    if #score == 0 then
      for s = 1,#evaluation do
        table.insert(score, { tp = 0, fp = 0, tn = 0, fn = 0 })
      end
    end

    for s = 1,#evaluation do
      score[s].tp = score[s].tp + evaluation[s].tp
      score[s].tn = score[s].tn + evaluation[s].tn
      score[s].fp = score[s].fp + evaluation[s].fp
      score[s].fn = score[s].fn + evaluation[s].fn
    end

  end

  return avgLoss , score

end

function train() return step('train') end
function valid() return step('valid') end
