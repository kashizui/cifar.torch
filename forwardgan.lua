require 'image'
require 'cudnn'
require 'cunn'
require 'nngraph'
dofile './provider.lua'

opt = lapp[[
   -l,--logs                  (default "/mnt/logs/gan")      subdirectory to read logs
   -p,--provider              (default "/mnt/provider.t7")  provider
   -i,--image                 (default 100)     index into image
   -t,--test                                        use test set
    -n,--noiseDim              (default 256)            dimensions of noise vector
    --noiseStd              (default 0.1)            std of the noise
    --naiveModel                                    use naive model
]]

-- init
provider = torch.load(opt.provider)
if opt.naiveModel then
    model = torch.load(paths.concat(opt.logs, 'naiveModel.net'))
else
    model = torch.load(paths.concat(opt.logs, 'model.net')).G
end

trainData = provider.trainData
if opt.test then
    data = provider.testData
else
    data = provider.trainData
end

-- rescuscitate original color image
yuvTrue = data.data[opt.image]
yuvTrue:select(1,1):div(256)
yuvTrue:select(1,2):mul(trainData.std_u)
yuvTrue:select(1,2):add(trainData.mean_u)
yuvTrue:select(1,2):div(256)
yuvTrue:select(1,3):mul(trainData.std_v)
yuvTrue:select(1,3):add(trainData.mean_v)
yuvTrue:select(1,3):div(256)
rgbTrue = image.yuv2rgb(yuvTrue)

-- get original unnormalized gray channel
grayOrig = torch.CudaTensor(1, 32, 32)
grayOrig:copy(yuvTrue:index(1, torch.LongTensor{1}):float())

-- get normalized gray image
gray = torch.CudaTensor(1, 1, 32, 32)
gray:copy(data.gray:index(1, torch.LongTensor{opt.image}):float())

-- generate noise
noiseInputs = torch.CudaTensor(1, opt.noiseDim)
noiseInputs:normal(0, opt.noiseStd)

-- forward through network
uvPred = model:forward({gray, noiseInputs})
yuvPred = torch.cat(grayOrig, uvPred[1], 1) -- cat along channel dimension and pull out single test point
yuvPred:select(1,2):mul(trainData.std_u)
yuvPred:select(1,2):add(trainData.mean_u)
yuvPred:select(1,2):div(256)
yuvPred:select(1,3):mul(trainData.std_v)
yuvPred:select(1,3):add(trainData.mean_v)
yuvPred:select(1,3):div(256)
rgbPred = image.yuv2rgb(yuvPred)

print(gray:size())
print(rgbPred:size())
print(rgbTrue:size())

-- Save image
if opt.test then
    basepath = paths.concat(opt.logs, 'test' .. opt.image)
else
    basepath = paths.concat(opt.logs, opt.image)
end
image.save(basepath .. 'in.png', grayOrig)
image.save(basepath .. 'out.png', rgbPred)
image.save(basepath .. 'true.png', rgbTrue)
