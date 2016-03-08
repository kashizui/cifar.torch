require 'xlua'
require 'optim'
require 'cunn'
dofile './provider.lua'
local c = require 'trepl.colorize'

opt = lapp[[
    -s,--save                  (default "logs")      subdirectory to save logs
    -b,--batchSize             (default 256)          batch size
    -r,--learningRate          (default 1)        learning rate
    --learningRateDecay        (default 1e-7)      learning rate decay
    --weightDecay              (default 0.0005)      weightDecay
    -m,--momentum              (default 0.9)         momentum
    --epoch_step               (default 25)          epoch step
    --model                    (default gan)     model name
    --max_epoch                (default 300)           maximum number of iterations
    --backend                  (default cudnn)            backend
    -n,--noiseDim              (default 256)            dimensions of noise vector
]]

print(opt)
halfBatchSize = opt.batchSize / 2


-- Load process and display model
print(c.blue '==>' ..' configuring model')
local model = dofile('models/'..opt.model..'.lua'):cuda()

if opt.backend == 'cudnn' then
    require 'cudnn'
    cudnn.convert(model.G, cudnn)
    cudnn.convert(model.D, cudnn)
end

print(model.D)
print(model.G)


-- Load data from provider object
print(c.blue '==>' ..' loading data')
provider = torch.load '/mnt/provider.t7'

provider.trainData.data = provider.trainData.data:float()
provider.testData.data = provider.testData.data:float()
provider.trainData.gray = provider.trainData.gray:float()
provider.testData.gray = provider.testData.gray:float()


-- Configure logger
print('Will save at '..opt.save)
paths.mkdir(opt.save)
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
testLogger:setNames{'per-pixel mean-square error (train set)', 'per-pixel mean-square error (test set)'}
testLogger.showPlot = false


-- Initialize confusion matrix
classes = {'0','1'}
confusion = optim.ConfusionMatrix(classes)


-- Get pointers to parameters
parametersD, gradParametersD = model.D:getParameters()
parametersG, gradParametersG = model.G:getParameters()


-- Use negative log-likelihood loss function
print(c.blue'==>' ..' setting criterion')
criterion = nn.BCECriterion():cuda()


-- Configure optimizer/update algorithm
print(c.blue'==>' ..' configuring optimizer')
optimStateD = {
    learningRate = opt.learningRate,
    weightDecay = opt.weightDecay,
    momentum = opt.momentum,
    learningRateDecay = opt.learningRateDecay,
    optimize = true,
}
optimStateG = {
    learningRate = opt.learningRate,
    weightDecay = opt.weightDecay,
    momentum = opt.momentum,
    learningRateDecay = opt.learningRateDecay,
    optimize = true,
}


function train()
    model:training()
    epoch = epoch or 1

    print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

    -- Drop learning rate every "epoch_step" epochs
    if epoch % opt.epoch_step == 0 then
        optimState.learningRate = optimState.learningRate / 2
    end


    -- Generate random indices for minibatches
    local indicesG = torch.randperm(provider.trainData.gray:size(1)):long():split(opt.batchSize)
    local indicesD = torch.randperm(provider.trainData.gray:size(1)):long():split(opt.batchSize)
    -- remove last minibatch to ensure equal batch sizes
    indicesG[#indicesG] = nil
    indicesD[#indicesD] = nil


    -- Allocate tensors for minibatch
    local noiseInputs = torch.CudaTensor(opt.batchSize, opt.noiseDim)
    local grayInputs = torch.CudaTensor(opt.batchSize, 1, 32, 32)
    local uvInputs = torch.CudaTensor(opt.batchSize, 2, 32, 32)
    local dTargets = torch.CudaTensor(opt.batchSize)


    -- For each minibatch:
    for t, chunkG in ipairs(indices) do
        local chunkD = indicesD[t]
        xlua.progress(t, #indices)
        
        -----------------------------------------------------------------------
        -- Closure to evaluate and backprop through discriminator
        -----------------------------------------------------------------------
        local fevalD = function(x)
            collectgarbage()
            if x ~= parametersD then parametersD:copy(x) end
            gradParametersD:zero()

            -- Forward pass
            local yuvInputs = torch.cat(grayInputs, uvInputs, 2)
            local dOutputs = model.D:forward(yuvInputs)
            local errReal = criterion:forward(dOutputs:narrow(1, 1, halfBatchSize),
                                              dTargets:narrow(1, 1, halfBatchSize))
            local errFake = criterion:forward(dOutputs:narrow(1, halfBatchSize + 1, opt.batchSize),
                                              dTargets:narrow(1, halfBatchSize + 1, opt.batchSize))

            -- Compute heuristics on whether or not to optimize G/D
            local margin = 0.3
            optimStateD.optimize = true
            optimStateG.optimize = true      
            if errFake < margin or errReal < margin then
                optimStateD.optimize = false
            end
            if errFake > (1.0 - margin) or errReal > (1.0 - margin) then
                optimStateG.optimize = false
            end
            if optimStateG.optimize == false and optimStateD.optimize == false then
                optimStateG.optimize = true 
                optimStateD.optimize = true
            end

            -- Compute loss
            local f = criterion:forward(dOutputs, dTargets)

            -- Backward pass
            local df_dOutputs = criterion:backward(dOutputs, dTargets)
            model.D:backward(yuvInputs, df_dOutputs)

            -- Update confusion matrix
            confusion.batchAdd(dOutputs, dTargets)
            
            return f, gradParametersD
        end


        -----------------------------------------------------------------------
        -- Closure to evaluate and backprop through generator
        -----------------------------------------------------------------------
        local fevalG = function(x)
            collectgarbage()
            if x ~= parametersG then parametersG:copy(x) end
            gradParametersG:zero()

            -- Forward pass
            local uvSamples = model.G:forward({grayInputs, noiseInputs})
            local yuvSamples = torch.cat(grayInputs, uvSamples, 2)
            local dOutputs = model.D:forward(yuvSamples)
            local f = criterion:forward(dOutputs, dTargets)

            -- Backward pass
            local df_dOutputs = criterion:backward(dOutputs, dTargets)
            model.D:backward(yuvSamples, df_dOutputs)
            local df_dyuvSamples = model.D.modules[1].gradInput
            model.G:backward({grayInputs, noiseInputs}, df_dyuvSamples)

            return f, gradParametersG
        end

        -----------------------------------------------------------------------
        -- Update D network once (K=1): maximize log(D(x)) + log(1 - D(G(z)))
        -- Get half a minibatch of real, half fake
        -----------------------------------------------------------------------
        noiseInputs:normal(0, 1)
        grayInputs:index(provider.trainData.gray, 1, chunkD)

        -- Real color samples
        uvInputs[{{1, halfBatchSize}}] = provider.trainData.data:index(1, v):index(2, torch.LongTensor{2, 3})
        dTargets[{{1, halfBatchSize}}].fill(1)

        -- Generated color samples: forward half of the gray inputs through the generator
        uvInputs[{{halfBatchSize + 1, opt.batchSize}}] = model.G:forward({
            grayInputs[{{halfBatchSize + 1, opt.batchSize}}],
            noiseInputs[{{halfBatchSize + 1, opt.batchSize}}]
        })
        dTargets[{{halfBatchSize + 1, opt.batchSize}}].fill(0)

        optim.sgd(fevalD, parametersD, optimStateD)

        -----------------------------------------------------------------------
        -- Update G network: maximize log(D(G(z)))
        -----------------------------------------------------------------------
        grayInputs:index(provider.trainData.gray, 1, chunkG)
        noiseInputs:normal(0, 1)
        dTargets:fill(1)  -- goal is to fool the discriminator
        optim.sgd(fevalG, parametersG, optimStateG)

        -----------------------------------------------------------------------
        -- Some logging
        -----------------------------------------------------------------------
        print(confusion)
        confusion:zero()

    end

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
    --test()
end


