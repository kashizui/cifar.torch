require 'xlua'
require 'optim'
require 'cunn'
dofile './provider.lua'
local c = require 'trepl.colorize'

opt = lapp[[
    -s,--save                  (default "logs")      subdirectory to save logs
    -b,--batchSize             (default 128)          batch size
    -r,--learningRate          (default 1)        learning rate
    --learningRateDecay        (default 1e-7)      learning rate decay
    --weightDecay              (default 0.0005)      weightDecay
    -m,--momentum              (default 0.9)         momentum
    --epoch_step               (default 25)          epoch step
    --model                    (default vgg_bn_drop)     model name
    --max_epoch                (default 300)           maximum number of iterations
    --backend                  (default cudnn)            backend
    --params                   (default nil)         saved model if any
]]

print(opt)

print(c.blue '==>' ..' configuring model')
local model = nn.Sequential()
model:add(dofile('models/'..opt.model..'.lua'):cuda())

if opt.backend == 'cudnn' then
    require 'cudnn'
    cudnn.convert(model:get(1), cudnn)
end

print(model)

print(c.blue '==>' ..' loading data')
provider = torch.load '/mnt/provider.t7'

provider.trainData.data = provider.trainData.data:float()
provider.testData.data = provider.testData.data:float()
provider.trainData.gray = provider.trainData.gray:float()
provider.testData.gray = provider.testData.gray:float()

print('Will save at '..opt.save)
paths.mkdir(opt.save)
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
testLogger:setNames{'per-pixel mean-square error (train set)', 'per-pixel mean-square error (test set)'}
testLogger.showPlot = false

parameters,gradParameters = model:getParameters()


print(c.blue'==>' ..' setting criterion')
criterion = nn.MSECriterion():cuda()


print(c.blue'==>' ..' configuring optimizer')
optimState = {
    learningRate = opt.learningRate,
    weightDecay = opt.weightDecay,
    momentum = opt.momentum,
    learningRateDecay = opt.learningRateDecay,
}


function train()
    model:training()
    epoch = epoch or 1

    -- drop learning rate every "epoch_step" epochs
    if epoch % opt.epoch_step == 0 then
        optimState.learningRate = optimState.learningRate/2
    end

    print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

    local inputs = torch.CudaTensor(opt.batchSize, 1, 32, 32)
    local targets = torch.CudaTensor(opt.batchSize, 2, 32, 32)
    local indices = torch.randperm(provider.trainData.gray:size(1)):long():split(opt.batchSize)
    -- remove last element so that all the batches have equal size
    indices[#indices] = nil

    local tic = torch.tic()
    trainError = 0
    local numTrained = 0
    for t,v in ipairs(indices) do
        xlua.progress(t, #indices)

        -- get y and uv images
        inputs:copy(provider.trainData.gray:index(1,v))
        targets:copy(provider.trainData.data:index(1,v):index(2,torch.LongTensor{2,3}))

        local feval = function(x)
            if x ~= parameters then parameters:copy(x) end
            gradParameters:zero()

            -- Get UV output
            local outputs = model:forward(inputs)

            -- Compute loss
            local f = criterion:forward(outputs, targets)
            if f ~= f then
                print("nan detected in error! skipping minibatch..")
            else
                -- compute gradients
                local df_do = criterion:backward(outputs, targets)
                model:backward(inputs, df_do)

                -- Update error
                trainError = trainError + f
                numTrained = numTrained + opt.batchSize
            end

            return f, gradParameters
        end

        optim.adam(feval, parameters, optimState)
    end
    trainError = trainError / numTrained

    torch.toc(tic)
    print('Train average MSE error:', trainError)

    epoch = epoch + 1
end


function test()
    -- disable flips, dropouts and batch normalization
    model:evaluate()

    print(c.blue '==>'.." testing")
    local bs = 125
    local testError = 0
    for i=1,provider.testData.data:size(1),bs do
        -- Copy into Cuda
        local inputs = torch.CudaTensor(bs, 1, 32, 32)
        local targets = torch.CudaTensor(bs, 2, 32, 32)
        inputs:copy(provider.testData.gray:narrow(1, i, bs))
        targets:copy(provider.testData.data:narrow(1, i, bs):index(2, torch.LongTensor{2, 3}))

        -- forward through network
        local outputs = model:forward(inputs)

        -- add to running average
        local f = criterion:forward(outputs, targets)
        testError = testError + (f / provider.testData.data:size(1))
    end

    print('Test average MSE error:', testError)

    if testLogger then
        paths.mkdir(opt.save)
        testLogger:add{trainError, testError}
        testLogger:style{'-','-'}
        testLogger:plot()

        local base64im
        do
            os.execute(('convert -density 200 %s/test.log.eps %s/test.png'):format(opt.save,opt.save))
            os.execute(('openssl base64 -in %s/test.png -out %s/test.base64'):format(opt.save,opt.save))
            local f = io.open(opt.save..'/test.base64')
            if f then base64im = f:read'*all' end
        end

        local file = io.open(opt.save..'/report.html','w')
        file:write(([[
        <!DOCTYPE html>
        <html>
        <body>
        <title>%s - %s</title>
        <img src="data:image/png;base64,%s">
        <h4>optimState:</h4>
        <table>
        ]]):format(opt.save,epoch,base64im))
        for k,v in pairs(optimState) do
            if torch.type(v) == 'number' then
                file:write('<tr><td>'..k..'</td><td>'..v..'</td></tr>\n')
            end
        end
        file:write'</table><pre>\n'
        file:write(tostring(confusion)..'\n')
        file:write(tostring(model)..'\n')
        file:write'</pre></body></html>'
        file:close()
    end

    -- save model every epoch step
    if epoch % opt.epoch_step == 0 then
        local filename = paths.concat(opt.save, 'model.net')
        print('==> saving model to '..filename)
        torch.save(filename, model:get(1))
    end

end


for i=1,opt.max_epoch do
    train()
    test()
end


