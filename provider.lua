require 'nn'
require 'image'
require 'xlua'

local Provider = torch.class 'Provider'

function Provider:__init(full)
    local trsize = 50000
    local tesize = 10000

    -- download dataset
    if not paths.dirp('/mnt/cifar-10-batches-t7') then
        local www = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar-10-torch.tar.gz'
        local tar = paths.basename(www)
        os.execute('wget ' .. www .. '; '.. 'tar xvf ' .. tar)
    end

    -- load dataset
    self.trainData = {
        data = torch.Tensor(50000, 3072),
        labels = torch.Tensor(50000),
        size = function() return trsize end
    }
    local trainData = self.trainData
    for i = 0,4 do
        local subset = torch.load('/mnt/cifar-10-batches-t7/data_batch_' .. (i+1) .. '.t7', 'ascii')
        trainData.data[{ {i*10000+1, (i+1)*10000} }] = subset.data:t()
        trainData.labels[{ {i*10000+1, (i+1)*10000} }] = subset.labels
    end
    trainData.labels = trainData.labels + 1

    local subset = torch.load('/mnt/cifar-10-batches-t7/test_batch.t7', 'ascii')
    self.testData = {
        data = subset.data:t():double(),
        labels = subset.labels[1]:double(),
        size = function() return tesize end
    }
    local testData = self.testData
    testData.labels = testData.labels + 1

    -- resize dataset (if using small version)
    trainData.data = trainData.data[{ {1,trsize} }]
    trainData.labels = trainData.labels[{ {1,trsize} }]

    testData.data = testData.data[{ {1,tesize} }]
    testData.labels = testData.labels[{ {1,tesize} }]

    -- reshape data
    trainData.data = trainData.data:reshape(trsize,3,32,32)
    testData.data = testData.data:reshape(tesize,3,32,32)
end

function Provider:normalize()
    ----------------------------------------------------------------------
    -- preprocess/normalize train/test sets
    --
    local trainData = self.trainData
    local testData = self.testData

    print '<trainer> preprocessing data (color space + normalization)'
    collectgarbage()

    local normalize = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))

    -- preprocess trainSet
    trainData.gray = torch.Tensor(50000, 1, 32, 32)
    trainData.grayOrig = torch.Tensor(50000, 1, 32, 32)
    for i = 1,trainData:size() do
        xlua.progress(i, trainData:size())
        -- rgb -> yuv
        local rgb = trainData.data[i]
        local yuv = image.rgb2yuv(rgb)
        local y = yuv:index(1, torch.LongTensor{1})
        trainData.data[i] = yuv
        -- normalize y locally:
        y[1] = normalize(y[{{1}}])
        trainData.gray[i] = y
    end

    ---- normalize u globally:
    local mean_u = trainData.data:select(2,2):mean()
    local std_u = trainData.data:select(2,2):std()
    trainData.data:select(2,2):add(-mean_u)
    trainData.data:select(2,2):div(std_u)
    ---- normalize v globally:
    local mean_v = trainData.data:select(2,3):mean()
    local std_v = trainData.data:select(2,3):std()
    trainData.data:select(2,3):add(-mean_v)
    trainData.data:select(2,3):div(std_v)

    -- save normalization factors
    trainData.mean_u = mean_u
    trainData.std_u = std_u
    trainData.mean_v = mean_v
    trainData.std_v = std_v

    -- preprocess testSet
    testData.gray = torch.Tensor(10000, 1, 32, 32)
    testData.grayOrig = torch.Tensor(10000, 1, 32, 32)
    for i = 1,testData:size() do
        xlua.progress(i, testData:size())
        -- rgb -> yuv
        local rgb = testData.data[i]
        local yuv = image.rgb2yuv(rgb)
        local y = yuv:index(1, torch.LongTensor{1})
        testData.data[i] = yuv
        -- normalize y locally:
        y[1] = normalize(y[{{1}}])
        testData.gray[i] = y
    end
    ---- normalize u globally:
    testData.data:select(2,2):add(-mean_u)
    testData.data:select(2,2):div(std_u)
    ---- normalize v globally:
    testData.data:select(2,3):add(-mean_v)
    testData.data:select(2,3):div(std_v)
end
