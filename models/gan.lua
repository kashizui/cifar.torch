require 'nn'
require 'nngraph'

local MaxPooling = nn.SpatialMaxPooling


--- DISCRIMINATOR --
local modelD = nn.Sequential()

-- building block
local function ConvReLU(net, nInputPlane, nOutputPlane)
  net:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
  net:add(nn.ReLU(true))
  return net
end

-- in: 3 x 32 x 32
ConvReLU(modelD, 3,32)
modelD:add(MaxPooling(2,2,2,2))

-- 32 x 16 x 16
ConvReLU(modelD, 32,32)
modelD:add(MaxPooling(2,2,2,2))
modelD:add(nn.SpatialDropout(0.2))

-- 32 x 8 x 8
ConvReLU(modelD, 32,64)
modelD:add(MaxPooling(2,2,2,2))
modelD:add(nn.SpatialDropout(0.2))

-- 4 x 4 x 64
modelD:add(nn.View(4 * 4 * 64))
modelD:add(nn.Linear(4 * 4 * 64, 64))
modelD:add(nn.ReLU(true))
modelD:add(nn.Dropout(0.5))
modelD:add(nn.Linear(64,1))
modelD:add(nn.Sigmoid())
-- out: 1


--- COLORIZATION GENERATOR ---
-- building block
local function ConvBNReLU(node, nInputPlane, nOutputPlane)
  node = nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1)(node)
  node = nn.SpatialBatchNormalization(nOutputPlane, 1e-3)(node)
  node = nn.ReLU(true)(node)
  return node
end

-- EXTRACT FEATURES FROM GRAYSCALE
-- noiseInput: opt.noiseDim
local noiseInput = nn.Identity()()
local noiseNode = nn.Linear(opt.noiseDim, 32 * 32)(noiseInput)
noiseNode = nn.Reshape(1, 32, 32)(noiseNode)

-- grayInput: 1 x 32 x 32
-- noiseNode: 1 x 32 x 32
local grayInput = nn.Identity()()
local lg = nn.CAddTable()({grayInput, noiseNode})
lg = ConvBNReLU(lg, 1, 32)
lg = ConvBNReLU(lg, 32, 32)
lg = ConvBNReLU(lg, 32, 64)

-- CREATE COLOR CHANNELS
lg = nn.SpatialConvolution(64, 32, 1,1, 1,1, 0,0)(lg)
lg = nn.SpatialBatchNormalization(32, 1e-3)(lg)
lg = nn.ReLU(true)(lg)
lg = nn.SpatialConvolution(32, 2, 1,1, 1,1, 0,0)(lg)

-- out: 2 x 32 x 32 (UV)
local modelG = nn.gModule({grayInput, noiseInput}, {lg})


-- initialization from MSR
local function MSRinit(net)
  local function init(name)
    for k,v in pairs(net:findModules(name)) do
      local n = v.kW*v.kH*v.nOutputPlane
      v.weight:normal(0,math.sqrt(2/n))
      v.bias:zero()
    end
  end
  -- have to do for both backends
  init'nn.SpatialConvolution'
end

MSRinit(modelD)
MSRinit(modelG)

-- check that we can propagate forward without errors
print(#modelD:cuda():forward(torch.CudaTensor(16,3,32,32)))
print(#modelG:cuda():forward({torch.CudaTensor(16,1,32,32), torch.CudaTensor(16, opt.noiseDim)}))

return {
    D = modelD:cuda(),
    G = modelG:cuda(),
}
