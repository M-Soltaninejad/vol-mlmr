--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--

local M = {}
local RootDataset = torch.class('resnet.RootDataset', M)

require 'image'
require 'paths'
require 'math'

function RootDataset:__init(imageInfo, opt, split)
   assert(imageInfo[split], split)

   self.imageInfo = imageInfo[split]
   self.split = split
   self.sourcedir = paths.concat(os.getenv('HOME'),'Data/Images/TH7')

   -- Load volumes
   self.volumes = {}
   self.masks = {}
   for i = 1, #self.imageInfo do
      
      local headerpath =  paths.concat(self.sourcedir, self.imageInfo[i].filename)
      local datapath = headerpath:gsub(".hdr",".data")
      local header = torch.load(headerpath)
      
      local maskheaderpath =  paths.concat(self.sourcedir, self.imageInfo[i].maskname)
      local maskpath = maskheaderpath:gsub(".hdr",".data")
      local maskheader = torch.load(maskheaderpath)
      
      local size = torch.LongTensor(header):prod()  
      
--      datastorage = torch.FloatStorage(datapath, false, size)
--      datatensor = torch.FloatTensor(datastorage, 1, header)
      datastorage = torch.ByteStorage(datapath, false, size)
      datatensor = torch.ByteTensor(datastorage, 1, header)      
      table.insert(self.volumes, datatensor)
      
      maskstorage = torch.ByteStorage(maskpath, false, size)
      masktensor = torch.ByteTensor(maskstorage, 1, maskheader)
      table.insert(self.masks, masktensor)      
      
   end


   -- Load annotations
   self.global_index = nil
   self.annotations = {}
   for i = 1, #self.imageInfo do
      -- Add to global_index
      local current_annotation = self.imageInfo[i].points
      local count = current_annotation:size(1)
      local indexed_annotation = torch.cat(torch.Tensor(count,1):fill(i), current_annotation, 2)

      if not self.global_index then
         self.global_index = indexed_annotation   
      else
         self.global_index = torch.cat(self.global_index, indexed_annotation, 1)
      end

   end

end


function RootDataset:get(i)
    local ann = self.global_index[i]

    local volume_index = ann[1]

    local position = ann[{{2,-1}}]:clone()
    
    return {
      input = self.volumes[volume_index],
      mask = self.masks[volume_index],
      centre = position
   }
end

function RootDataset:size()
   return self.global_index:size(1)
end

return M.RootDataset
