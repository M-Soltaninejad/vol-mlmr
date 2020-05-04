paths.dofile('layers/Residual.lua')

local function hourglass(n, f, inp)
    -- Upper branch
    local up1 = inp
    for i = 1,opt.nModules do up1 = Residual(f,f)(up1) end

    -- Lower branch
    local low1 = nnlib.VolumetricMaxPooling(2,2,2,2,2,2)(inp)

    for i = 1,opt.nModules do low1 = Residual(f,f)(low1) end
    local low2

    --local features
    if n > 1 then
        local sub = hourglass(n-1,f,low1)
        low2 = sub.hg
        --features = sub.feat
    else
        low2 = low1
        for i = 1,opt.nModules do low2 = Residual(f,f)(low2) end
        --features = low2
    end

    local low3 = low2
    for i = 1,opt.nModules do low3 = Residual(f,f)(low3) end
    local up2 = nn.UpSampling(2)(low3)

    -- Bring two branches together
    return { hg = nn.CAddTable()({up1,up2})} --, feat = features }
end

local function lin(numIn,numOut,inp)
    -- Apply 1x1 convolution, stride 1, no padding
    local l = nnlib.VolumetricConvolution(numIn,numOut,1,1,1,1,1,1,0,0,0)(inp)
    return nnlib.ReLU(true)(nn.VolumetricBatchNormalization(numOut)(l))
end

function createModel()

    local inp = nn.Identity()()

    -- More than one pooling
    local r1 = Residual(1,16)(inp) --r1 = Residual(1,128)(inp)
    local pool1 = nnlib.VolumetricMaxPooling(2,2,2)(r1)
    local r3 = Residual(16,32)(pool1)  -- r3 = Residual(128,256)(pool1)
    local pool2 = nnlib.VolumetricMaxPooling(2,2,2)(r3)
    local r4 = Residual(32,64)(pool2)
    local pool3 = nnlib.VolumetricMaxPooling(2,2,2)(r4)
    local r5 = Residual(64,opt.nFeats)(pool3)  --r5 = Residual(256,opt.nFeats)(pool2)
    local inter1 = r5
    
    local out = {}
    local out1_heat = {}
    local out1 = {}
    local out2_heat = {}    
    local out2 = {}
    
    -- Hourgalss one: Low resolution

    for i = 1,opt.nStack do
        local sub = hourglass(4,opt.nFeats,inter1)
        local hg = sub.hg

        -- Residual layers at output resolution
        local ll = hg
        for j = 1,opt.nModules do ll = Residual(opt.nFeats,opt.nFeats)(ll) end

        local ll1 = ll
        ll = lin(opt.nFeats,opt.nFeats,ll)
		
        -- Predicted heatmaps
        local tmpOut = nnlib.VolumetricConvolution(opt.nFeats,opt.nOutChannels,1,1,1,1,1,1,0,0,0)(ll)
        local out1_heat_small = nnlib.Sigmoid()(tmpOut)
        out1_heat = nn.UpSampling(8, 'linear')(out1_heat_small)
        
        local out_up1 = nn.Narrow(3, 7, 2)(ll1)
        out_up1 = nn.Narrow(4, 7, 2)(out_up1)
        out_up1 = nn.Narrow(5, 7, 2)(out_up1)
        local out_up2 = nn.UpSampling(8, 'linear')(out_up1)

        out1 = out_up2
 
    end
    
    -- Hourgalss two: High resolution
    local rr0 = nn.Narrow(3, 57, 16)(inp) --:narrow(3, 57, 16):narrow(4, 57, 16):narrow(5, 57, 16)
    rr0 = nn.Narrow(4, 57, 16)(rr0)
    rr0 = nn.Narrow(5, 57, 16)(rr0)
    local rr1 = Residual(1,16)(rr0) --r1 = Residual(1,128)(inp)
    local rr2 = Residual(16,opt.nFeats)(rr1)    
    local inter2= rr2

    for i = 1,opt.nStack do
        local sub = hourglass(4,opt.nFeats,inter2)
        local hg = sub.hg

        -- Residual layers at output resolution
        local ll = hg
        for j = 1,opt.nModules do ll = Residual(opt.nFeats,opt.nFeats)(ll) end

        out2 = ll
        
        ll = lin(opt.nFeats,opt.nFeats,ll)
        -- Predicted heatmaps
        local tmpOut = nnlib.VolumetricConvolution(opt.nFeats,opt.nOutChannels,1,1,1,1,1,1,0,0,0)(ll)
        out2_heat = nnlib.Sigmoid()(tmpOut)
    end
    
    local combined_features = nn.JoinTable(2)({out1,out2})
           
    -- Linear layer to produce first set of predictions
    local ll_out = lin(opt.nFeats * 2,opt.nFeats * 2, combined_features)

    -- Predicted heatmaps
    local tmpOut = nnlib.VolumetricConvolution(opt.nFeats*2,opt.nOutChannels,1,1,1,1,1,1,0,0,0)(ll_out)

    local stmp = nnlib.Sigmoid()(tmpOut)
    
    -- table.insert(out,stmp)
    table.insert(out,out1_heat)
    table.insert(out,out2_heat)
    table.insert(out,stmp)

    -- Final model
    local model = nn.gModule({inp}, out)

    return model
end



