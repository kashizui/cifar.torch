require 'nn'

local vgg = nn.Sequential()

-- building block
local function ConvBNReLU(nInputPlane, nOutputPlane)
  vgg:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
  vgg:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
  vgg:add(nn.ReLU(true))
  return vgg
end

local MaxPooling = nn.SpatialMaxPooling

-- 32 x 32 x 3
ConvBNReLU(3,32)
vgg:add(MaxPooling(2,2,2,2))

-- 16 x 16 x 32
ConvBNReLU(32,32)
vgg:add(MaxPooling(2,2,2,2))

-- 8 x 8 x 32
ConvBNReLU(32,64)
vgg:add(MaxPooling(2,2,2,2))

-- 4 x 4 x 64
vgg:add(nn.View(1024))

classifier = nn.Sequential()
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(1024, 64))
classifier:add(nn.BatchNormalization(64))
classifier:add(nn.ReLU(true))
classifier:add(nn.Linear(64,10))
vgg:add(classifier)

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
print(#vgg:cuda():forward(torch.CudaTensor(16,3,32,32)))

return vgg
