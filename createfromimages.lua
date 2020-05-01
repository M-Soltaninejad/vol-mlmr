require 'paths'
require 'torch'
require 'hdf5'
require 'string'
require 'image'

torch.setdefaulttensortype('torch.FloatTensor')

-- Project directory
for data_id = 101,103 do -- Data IDs
  dataNumber = tostring(data_id)
  CurrentDir = os.getenv('HOME')
  imgDir = paths.concat(CurrentDir,'Data/Images/Input'
  maskDir = paths.concat(CurrentDir,'Data/Images/Manual'

  outImgDir= '/Data/TH7'
  outMaskDir = outImgDir
  headerOutImgDir = outImgDir
  headerOutMaskDir = outMaskDir

  function pngsearch(file) 
    return file:find('.png') 
  end

  function file_exists(name)
    local f=io.open(name,"r")
    if f~=nil then io.close(f) return true else return false end
  end

-- function scanfile(f)
--    return nil
-- end

  allfiles = {}
  fileidx = 1
  idx_new = 1
-- All PNG files  
  for f in paths.files(imgDir, pngsearch) do
    allfiles[fileidx] = f;
    fileidx = fileidx + 1
  end

  imagecount = #allfiles


-- Load one PNG and check size
  imageFile = paths.concat(imgDir,'image_0000.png')

  img = image.load(imageFile,1,'byte')
  img = img:squeeze()


-- ZYX
  X = img:size(2)
  Y = img:size(1)
  Z = #allfiles

  Data = torch.ByteTensor(Z, Y, X)
  MaskData = torch.ByteTensor(Z, Y, X)

  for idx = 1, #allfiles do -- #allfiles
    -- Load PNG
    imageFileName = allfiles[idx]
    imageFile = imgDir .. imageFileName
    maskFile = maskDir .. imageFileName
    img = image.load(imageFile,1,'byte')
    img = img:squeeze()
    mask = image.load(maskFile,1,'byte')
    mask = mask:squeeze()
    current_data = img
    current_mask = mask
    idx_new = tonumber(imageFileName:sub(7,10))+1
    Data[idx_new]:copy(current_data)
    MaskData[idx_new]:copy(current_mask)
    print(idx)

    img = nil
    mask = nil
    collectgarbage()
  end

  outPathImg = paths.concat(outImgDir,'data' .. dataNumber .. '.data')
  outPathMask = paths.concat(outMaskDir,'data' .. dataNumber .. '_Mask.data')

  a = torch.DiskFile(outPathImg, 'w'):binary()
  a:writeByte(Data:storage())
  a:close()

  a_mask = torch.DiskFile(outPathMask, 'w'):binary()
  a_mask:writeByte(MaskData:storage())
  a_mask:close()

-- Write Header
  headerOutPath = paths.concat(headerOutImgDir,'data' .. dataNumber .. '.hdr')
  torch.save(headerOutPath, Data:size())
  headerOutMaskPath = paths.concat(headerOutMaskDir,'data' .. dataNumber .. '_Mask.hdr')
  torch.save(headerOutMaskPath, Data:size())

end
