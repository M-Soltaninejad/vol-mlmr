--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Multi-threaded data loader
--

local datasets = require 'datasets/init'
local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local M = {}
local DataLoader = torch.class('resnet.DataLoader', M)

function DataLoader.create(opt)
    -- The train and val loader
    local loaders = {}

    for i, split in ipairs{'train', 'valid'} do
        local dataset = datasets.create(opt, split)
        loaders[i] = M.DataLoader(dataset, opt, split)
    end
    return table.unpack(loaders)
end

function DataLoader:__init(dataset, opt, split)
    local manualSeed = opt.manualSeed
    local function init()
        torch.setdefaulttensortype('torch.FloatTensor')
        require('datasets/' .. opt.dataset)
    end
    local function main(idx)
        local utils = require('dataloaderutil')
        
        if manualSeed ~= 0 then
         torch.manualSeed(manualSeed + idx)
        end
        torch.setnumthreads(1)
        _G.dataset = dataset
        return dataset:size()
    end
    
    local threads, sizes = Threads(opt.nThreads, init, main)
    self.threads = threads
    self.__size = sizes[1][1]
    self.split = split
    self.batchSize = opt[split .. 'Batch']
end

function DataLoader:size()
    return math.floor(self.__size / self.batchSize)
end

function DataLoader:datasetsize()
    return math.floor(self.__size)
end

function DataLoader:batchsize()
    return math.floor(self.batchSize)
end

function DataLoader:run()
    local threads = self.threads
    local size, batchSize = self.__size, self.batchSize
    local perm = torch.randperm(size)

    local idx, sample = 1, nil
    local function enqueue()
        while idx <= size and threads:acceptsjob() do
         local indices = perm:narrow(1, idx, math.min(batchSize, size - idx + 1))
         threads:addjob(
            function(indices, useAugmentation, opt)
                local sz = indices:size(1)
                local inputbatch, imageSize
                local maskbatch

                -- Fixed capture and augmentation parameters
                local captureDimXY = opt.captureResXY
                local captureDimZ = opt.captureResZ

                local inputDimXY = opt.inputResXY
                local inputDimZ = opt.inputResZ


                local randomOffset = 8 --captureDimZ/2
                local randomOffsets = torch.Tensor({randomOffset, randomOffset, randomOffset})

                for i, idx in ipairs(indices:totable()) do
                    
                    -- Get next image from dataset
                    local sample = _G.dataset:get(idx)

                    local input = sample.input
                    local centre = sample.centre
                    local mask = sample.mask
                    
                    local imgwidth, imgheight, imgdepth = input:size(3), input:size(2), input:size(1)
                    
                    local chx = math.random()
                    -- **% of the time use a random position along the image edge
                    if chx < 0.30 then --0.60
                        centre = torch.rand(3):cmul(torch.Tensor({imgwidth,imgheight,imgdepth}))
                        

                    -- **% of the time, use the chosen image position and add jitter
                    else
                        centre:add(torch.randn(3):cmul(randomOffsets))
                    end

                      -- Create CB based on centre point
                    local cb = torch.Tensor({
                        centre[1] - (captureDimXY / 2),
                        centre[2] - (captureDimXY / 2),
                        centre[3] - (captureDimZ / 2),
                        captureDimXY, captureDimXY, captureDimZ});
                    
                   
                    -- Round cb into pixel coordinates
                    cb:round()

                    -- Standard XYZ bounds checks
                    if cb[1] < 1 then cb[1] = 1 end
                    if cb[2] < 1 then cb[2] = 1 end
                    if cb[3] < 1 then cb[3] = 1 end

                    if cb[1] + cb[4] > imgwidth then cb[1] = imgwidth - cb[4] end
                    if cb[2] + cb[5] > imgheight then cb[2] = imgheight - cb[5] end
                    if cb[3] + cb[6] > imgdepth then cb[3] = imgdepth - cb[6] end
                    
                    -- Crop and Clone
                    input = input[{ {cb[3], cb[3] + cb[6] - 1}, {cb[2], cb[2] + cb[5] - 1}, {cb[1], cb[1] + cb[4] - 1} }]:float():div(255):clone()
                    mask = mask[{ {cb[3], cb[3] + cb[6] - 1}, {cb[2], cb[2] + cb[5] - 1}, {cb[1], cb[1] + cb[4] - 1} }]:float():div(255):clone()
                    mask[mask:gt(0)] = 1.0
                    
                    if not inputbatch then
                        imageSize = input:size():totable()
                        inputbatch = torch.Tensor(sz, 1, table.unpack(imageSize))
                    end
                    
                    if not maskbatch then
                        imageSize = input:size():totable()
                        maskbatch = torch.Tensor(sz, 1, table.unpack(imageSize))
                    end

                    inputbatch[i][1]:copy(input)
                    maskbatch[i][1]:copy(mask)      
                end

                collectgarbage()
                return {
                    input = inputbatch,
                    mask = maskbatch,
                }
            end,
            function(_sample_)
                 sample = _sample_
            end,
            indices, self.split == 'train', opt
         )
         idx = idx + batchSize
        end
    end

    local n = 0
    local function loop()
        enqueue()
        if not threads:hasjob() then
         return nil
        end
        threads:dojob()
        if threads:haserror() then
         threads:synchronize()
        end
        enqueue()
        n = n + 1
        return n, sample
    end

    return loop
end

return M.DataLoader

