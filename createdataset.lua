require 'paths'
require 'torch'
require 'string'

torch.setdefaulttensortype('torch.FloatTensor')

-- Project directory
rootDir = paths.concat(os.getenv('HOME'),'ctRoots')
imgDir = paths.concat(rootDir,'datasets/ctRoots')
genDir = paths.concat(rootDir,'gen')
genDir = paths.concat('./gen')

train_cases = {101,102}
valid_cases = {103}

function file_exists(name)
  local f=io.open(name,"r")
  if f~=nil then io.close(f) return true else return false end
end

function lines_from(file)
  if not file_exists(file) then return {} end
  lines = {}
  for line in io.lines(file) do 
    lines[#lines + 1] = line
  end
  return lines
end

data = {}
data.train = {}
data.valid = {}
function scanfile(f)

  local lines
  if file_exists(f) then
    lines = lines_from(f)
  end

  local pts, ptensor = {}, nil
  if lines and #lines > 0 then
    -- Read lines x y z 
    for i = 1, #lines do -- Ignore header line
      s = lines[i]
      s = s:split(',')

      table.insert(pts, s)
    end
    ptensor = torch.Tensor(pts)
  end


  return ptensor
end

j_counter = 0
idx_case = train_cases

for k = 1, #idx_case do
  j_counter = j_counter +1
  j = idx_case[k]
  print(j)
  local points = nil
  local points_decreased = nil
  local idx = nill
  f = 'data' .. tostring(j) .. '.hdr'
  csvname = f:gsub(".hdr", ".csv")
  csvpath = paths.concat('datasets/ctRoots', csvname)
  points = scanfile(csvpath)
  data.train[j_counter] = { image_path = './roots/TH7/data' .. tostring(j) .. '.hdr',
    mask_path = './roots/TH7/data' .. tostring(j) .. '_Mask.hdr' ,
    filename = 'data' .. tostring(j) .. '.hdr', maskname = 'data' .. tostring(j) .. '_Mask.hdr' }
  data.train[j_counter].points = points
  collectgarbage()
end

j_counter = 0
idx_case = valid_cases 

for k = 1, #idx_case do 
  j_counter = j_counter +1
  j = idx_case[k]
  print(j)
  local points = nil
  local points_decreased = nil
  local idx = nil
  f = 'data' .. tostring(j) .. '.hdr'
  csvname = f:gsub(".hdr", ".csv")
  csvpath = paths.concat('datasets/ctRoots', csvname)
  points = scanfile(csvpath)

  data.valid[j_counter] = { image_path = './roots/TH7/data' .. tostring(j) .. '.hdr', 
    mask_path = './roots/TH7/data' .. tostring(j) .. '_Mask.hdr' , 
    filename = 'data' .. tostring(j) .. '.hdr', maskname = 'data' .. tostring(j) .. '_Mask.hdr' }
  data.valid[j_counter].points = points
  collectgarbage()
end

torch.save(paths.concat(genDir,'ctRoots.t7'), data)


