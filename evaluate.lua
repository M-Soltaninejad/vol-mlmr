require 'torch'

local M = {}

local function nms3D(pred, threshold)
    local batchcount = pred:size(1)
    local channelcount = pred:size(2)
    
    local pred_mask = {}

    pred_mask = torch.gt(pred,threshold):float()

    return pred_mask
end

local function calculateF1(tp, fp, tn, fn)
    local precision, recall, accuracy = 0.0, 0.0, 0.0
    
    if tp + fp > 0 then precision = tp / (tp + fp) end
    if tp + fn > 0 then recall = tp / (tp + fn) end
    if tp + tn + fp + fn > 0 then accuracy = (tp + tn) / (tp + tn + fp + fn) end

    local f1 = 0.0
    if precision + recall > 0 then f1 = 2 * (precision * recall) / (precision + recall) end
    return precision, recall, accuracy, f1
end

local function calculateMultiF1(score)
    assert(type(score) == 'table', "Score is not a table")

    local prf1 = {}

    for i = 1,#score do
        local p,r, acc,f1
        s = score[i]
        p,    r,acc, f1 = calculateF1(s.tp, s.fp, s.tn, s.fn)
        table.insert(prf1, { precision = p, recall = r, accuracy = acc, f1 = f1 })
    end

    return prf1
end



local function evaluateclassification(pred, gt)
    assert(pred:numel() == gt:numel(), "Class prediction and scores are different sizes")
    
    local pred_data = pred:contiguous():storage() -- Much faster than normal tensor indexing
    local gt_data = gt:contiguous():storage() 
    
    local length = pred_data:size()
    
    local tp, fp, tn, fn = 0,0,0,0
    
    for i = 1,length do
      if gt_data[i] == 0 then
        if pred_data[i] == 0 then
          tn = tn + 1
        else
          fp = fp + 1
        end
      else
        if pred_data[i] == 0 then
          fn = fn + 1
        else
          tp = tp + 1
        end
      
      end

    end
 
      return {
        tp = tp,
        fp = fp,
        fn = fn,
        tn = tn
        }
end

local function evaluate3D(pred, gt)
  local tfpn = {}
  tfpn = evaluateclassification(pred, gt)
  local tfpnTable = {}
  table.insert(tfpnTable, tfpn)
  return tfpnTable
end

M.calculateF1 = calculateF1
M.calculateMultiF1 = calculateMultiF1
M.evaluateclassification = evaluateclassification
M.nms3D = nms3D
M.evaluate3D = evaluate3D

return M