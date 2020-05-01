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
local batch_size = 10
local State = '**' -- The desired saved model state
local data_List = {103} -- The list of data IDs to be segmented

-- Output directory
local workingDir = './snapshots'

opt.model = paths.concat(workingDir, 'model_' .. State  .. '.t7')
opt.optimState = paths.concat(workingDir, 'optimState_' .. State  .. '.t7')

local outDir = paths.concat(os.getenv('HOME'),'Data/Outputs' 

if opt.directory ~= '' then
  workingDir = paths.concat(workingDir, opt.directory)
end


-- Data loading

testinput = torch.Tensor(16,1,256,256)

if opt.model == 'none' then
  print('==> Please select a model first')
  do return end 
else
  print ("==> Loading model from existing file:", opt.model)
  model = torch.load(opt.model)
end

model:cuda()
model:evaluate()


function pngsearch(file) 
  return file:find('.png') 
end

function file_exists(name)
  local f=io.open(name,"r")
  if f~=nil then io.close(f) return true else return false end
end

for data_id = 1,#data_List do 
local dataNumber = data_List[data_id]
local ouputDir = paths.concat(outDir , State, 'Case' .. dataNumber)
os.execute("mkdir " .. ouputDir )

local imgDir = paths.concat(os.getenv('HOME'),'Data/Images/Input/data' .. dataNumber .. '/data/'

allfiles = {}
fileidx = 1
-- All PNG files  
for f in paths.files(imgDir, pngsearch) do
  allfiles[fileidx] = f;
  fileidx = fileidx + 1
end
imagecount = #allfiles

-- Sort the allfiles
allfiles_sorted = {}
for zz = 1, #allfiles do
  if zz<=10 then
    fileNumber = '000' .. tostring(zz-1)
  elseif zz <=100 then 
    fileNumber = '00' .. tostring(zz-1)
  elseif zz <=1000 then
    fileNumber = '0' .. tostring(zz-1)
  else
    fileNumber = tostring(zz-1)
  end
  allfiles_sorted[zz] = 'image_' .. fileNumber .. '.png'  -- reco image_
end

imageFile = paths.concat(imgDir,'image_0000.png')  -- reco image_
img = image.load(imageFile,1,'float')
img = img:squeeze()  -- for Nano dataset

X = img:size(2)
Y = img:size(1)
Z = #allfiles

xs = 16
ys = 16
zs = 16

xs2 = 128
ys2 = 128
zs2 = 128
--for zz = 1,16 do
--  load image
--  insert in volume
--end

local zz_end = math.floor((#allfiles-56)/zs)*zs 

for zz = 1+56, zz_end, zs do -- #allfiles
print(zz)
  init = zz
  idx_temp = 0;
  Data = torch.FloatTensor(zs2,Y,X):zero()
  for idx = init-56,(init+zs-1) +56 do
    -- Load PNG
    imageFileName = allfiles_sorted[idx]
    imageFile = imgDir .. imageFileName
    img = image.load(imageFile,1,'float')
    idx_temp = idx_temp+1
    Data[idx_temp]:copy(img)
    print(idx)
  end

  local outImage = torch.FloatTensor(zs,Y,X):zero()

  local input = torch.FloatTensor(batch_size,1,zs2,ys2,xs2)
  local cbout_table = torch.Tensor(batch_size,6):zero()
  -- coord array
  local batch_index = 1
  for ii= 1+56,X-xs-56,xs do
    for jj=1+56,Y-ys-56,ys do
	  local cbin = torch.Tensor({ii-56, jj-56, 1, xs2, ys2, zs2});
	  cbout_table[{batch_index,{}}]  = torch.Tensor({ii, jj, 1, xs, ys, zs});
      input[{batch_index,1,{},{},{}}] = Data[{ {cbin[3], cbin[3] + cbin[6] - 1}, {cbin[2], cbin[2] + cbin[5] - 1}, {cbin[1], cbin[1] + cbin[4] - 1} }];
	  -- COPY COORDINATES
	  batch_index = batch_index + 1
       
	  if (batch_index == batch_size + 1) then
		-- USE NETWORK
		local cudaInput = input:cuda() -- CHECK IF FLOAT CAST IS NEEDED
		-- GO THROUGH NETWORK
      -- ********* Depends on the number of HG stacks **************
      
      -- NStack =1
  		 local output = model:forward(cudaInput);
         output = output[3]:float();   
		 
      --NStack > 1
--      local output = model:forward(cudaInput);
--      output = output[2]:float();
	
      -- ********************************************************
		-- COPY BACK
		for b_i = 1,batch_size do
		 -- copy output mask into output using coordinates
		  cbout = cbout_table[{b_i,{}}]
		  outImage[{ {cbout[3], cbout[3] + cbout[6] - 1}, {cbout[2], cbout[2] + cbout[5] - 1}, {cbout[1], cbout[1] + cbout[4] - 1} }] = output[{b_i,1,{},{},{}}]

		end
		batch_index = 1
		-- clean up
		cudaInput = nil
		output = nil
		collectgarbage()
	  end

      --input1 = nil
      --input = nil
      --cudaInput = nil
      --output = nil
    end  -- jj
--    print(ii )
--    if ii>50 then
--      break 
--    end
  end -- ii

  if (batch_index <= batch_size and batch_index > 1) then
		-- USE NETWORK
		local cudaInput = input:cuda() -- CHECK IF FLOAT CAST IS NEEDED
		-- GO THROUGH NETWORK
		 local output = model:forward(cudaInput);
         output = output[3]:float();
		-- COPY BACK
		for b_i = 1,batch_index - 1 do
		 -- copy output mask into output using cordinates
		  cbout = cbout_table[{b_i,{}}]
		  outImage[{ {cbout[3], cbout[3] + cbout[6] - 1}, {cbout[2], cbout[2] + cbout[5] - 1}, {cbout[1], cbout[1] + cbout[4] - 1} }] = output[{b_i,1,{},{},{}}]

		end
		-- clean up
		cudaInput = nil
		output = nil
		collectgarbage()
  end
  
  for slice =1,zs do
    idx_s = init + slice - 1
    outImageSlice = outImage[slice]
    imageFileName = allfiles_sorted[idx_s]
    imageFile = paths.concat(ouputDir, imageFileName)

    image.save(imageFile, outImageSlice)

  end
  

  img = nil

  collectgarbage()
end

end -- dataNumber

do return end



