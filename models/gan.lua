require 'nn'
require 'nngraph'

local MaxPooling = nn.SpatialMaxPooling

--- COLORIZATION GENERATOR ---
G = nn.Sequential()

-- building block
local function ConvBNReLU(net, nInputPlane, nOutputPlane)
  net:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
  net:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
  net:add(nn.ReLU(true))
  return net
end

-- EXTRACT FEATURES FROM GRAYSCALE
-- in: 1 x 32 x 32
ConvBNReLU(1,32)
ConvBNReLU(32,32)
ConvBNReLU(32,64)

-- CREATE COLOR CHANNELS
vgg:add(nn.SpatialConvolution(64, 32, 1,1, 1,1, 0,0))
vgg:add(nn.SpatialBatchNormalization(32,1e-3))
vgg:add(nn.ReLU(true))
vgg:add(nn.SpatialConvolution(32, 2, 1,1, 1,1, 0,0))
-- out: 2 x 32 x 32 (UV)


--- DISCRIMINATOR ---
modelD = nn.Sequential()

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

-- 4 x 4 x 64
modelD:add(nn.View(4 * 4 * 64))
modelD:add(nn.Dropout(0.5))
modelD:add(nn.Linear(4 * 4 * 64, 64))
modelD:add(nn.BatchNormalization(64))
modelD:add(nn.ReLU(true))
modelD:add(nn.Linear(64,1))
-- out: 1


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

MSRinit(vgg)

-- check that we can propagate forward without errors
-- should get 16x10 tensor
print(#vgg:cuda():forward(torch.CudaTensor(16,1,32,32)))

return vgg
