require 'nn'

local vgg = nn.Sequential()

-- building block
local function ConvBNReLU(nInputPlane, nOutputPlane)
  vgg:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
  vgg:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
  vgg:add(nn.ReLU(true))
  return vgg
end

local function ConvBNReLU1x1(nInputPlane, nOutputPlane)
  vgg:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 1,1, 1,1, 1,1))
  vgg:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
  vgg:add(nn.ReLU(true))
  return vgg
end

local MaxPooling = nn.SpatialMaxPooling

-- EXTRACT FEATURES FROM GRAYSCALE

-- 32 x 32 x 1
ConvBNReLU(1,32)
ConvBNReLU(32,32)
ConvBNReLU(32,64)

-- CREATE COLOR CHANNELS

vgg:add(nn.SpatialConvolution(64, 32, 1,1, 1,1, 0,0))
vgg:add(nn.SpatialBatchNormalization(32,1e-3))
vgg:add(nn.ReLU(true))
vgg:add(nn.SpatialConvolution(32, 2, 1,1, 1,1, 0,0))

-- 32 x 32 x 2 (UV)


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
