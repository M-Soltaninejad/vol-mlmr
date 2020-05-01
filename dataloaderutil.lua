local M = {}

local cache = {}

local function gaussianKernel3D(size, sigma)
	local gaussian = torch.FloatTensor(size, size, size)
	local centre = (size / 2.0) + 0.5

	local twos2 = 2 * math.pow(sigma, 2)

	for x = 1,size do
		for y = 1,size do
			for z = 1,size do
				gaussian[z][y][x] = -((math.pow(x - centre, 2)) + (math.pow(y - centre, 2)) + (math.pow(z - centre, 2))) / twos2
			end
		end
	end

	return gaussian:exp()
end

local function drawGaussian3D(img, x, y, z, sigma)
	local depth,height,width = img:size(1),img:size(2),img:size(3)
	
	-- Draw a 2D gaussian
	-- Check that any part of the gaussian is in-bounds
	local tmpSize = math.ceil(3*sigma)

	local ful = {math.floor(z - tmpSize), math.floor(y - tmpSize), math.floor(x - tmpSize)}
	local blr = {math.floor(z + tmpSize), math.floor(y + tmpSize), math.floor(x + tmpSize)}

	-- If not, return the image as is
	if (ful[3] > width or ful[2] > height or ful[1] > depth or blr[3] < 1 or blr[2] < 1 or blr[1] < 1) then return img end

	-- Generate gaussian
	local size = 2*tmpSize + 1

	if not cache[size] then
		cache[size] = gaussianKernel3D(size, sigma)
	end

	local g = cache[size]
	-- Usable gaussian range
	local g_x = {math.max(1, 2-ful[3]), math.min(size, size + (width - blr[3]))}
	local g_y = {math.max(1, 2-ful[2]), math.min(size, size + (height - blr[2]))}
	local g_z = {math.max(1, 2-ful[1]), math.min(size, size + (depth - blr[1]))}

	-- Image range
	local img_x = {math.max(1, ful[3]), math.min(blr[3], width)}
	local img_y = {math.max(1, ful[2]), math.min(blr[2], height)}
	local img_z = {math.max(1, ful[1]), math.min(blr[1], depth)}
	
	img:sub(img_z[1], img_z[2], img_y[1], img_y[2], img_x[1], img_x[2]):cmax(g:sub(g_z[1], g_z[2], g_y[1], g_y[2], g_x[1], g_x[2]))
end

local function renderheatmap3D(img, pts, sd)
	for g = 1, pts:size(1) do
		drawGaussian3D(img, pts[g][1], pts[g][2], pts[g][3], sd)
	end
end

M.renderheatmap = renderheatmap
M.drawGaussian = drawGaussian
M.gaussianKernel3D = gaussianKernel3D
M.drawGaussian3D = drawGaussian3D
M.renderheatmap3D = renderheatmap3D
return M