require 'xlua'
require 'optim'
require 'cunn'
dofile './provider.lua'
local c = require 'trepl.colorize'

opt = lapp[[
    -s,--save                  (default "logs")      subdirectory to save logs
    -b,--batchSize             (default 128)          batch size
    -r,--learningRate          (default 1e-3)        learning rate
    --learningRateDecay        (default 1e-7)      learning rate decay
    --weightDecay              (default 0.0005)      weightDecay
    -m,--momentum              (default 0.9)         momentum
    --epoch_step               (default 5)          epoch step
    --model                    (default gan)     model name
    --max_epoch                (default 300)           maximum number of iterations
    --backend                  (default cudnn)            backend
    -n,--noiseDim              (default 256)            dimensions of noise vector
    -g,--generator             (default "")          path to pretrained generator model
    --noiseStd                 (default 1)            std of noise
]]

print(opt)
halfBatchSize = opt.batchSize / 2


-- Load process and display model
print(c.blue '==>' ..' configuring model')
local model = dofile('models/'..opt.model..'.lua')

if opt.backend == 'cudnn' then
    require 'cudnn'
    cudnn.convert(model.G, cudnn)
    cudnn.convert(model.D, cudnn)
end

print(model.D)
--graph.dot(model.G.fg, 'G', 'G.png')


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
classes = {'0 (fake)','1 (real)'}
confusion = optim.ConfusionMatrix(classes)


-- Get pointers to parameters
parametersD, gradParametersD = model.D:getParameters()
parametersG, gradParametersG = model.G:getParameters()


-- Use negative log-likelihood loss function
print(c.blue'==>' ..' setting criterion')
criterion = nn.BCECriterion():cuda()
criterionNaive = nn.MSECriterion():cuda()


-- Configure optimizer/update algorithm
print(c.blue'==>' ..' configuring optimizer')
optimStateD = {
    learningRate = opt.learningRate,
    weightDecay = opt.weightDecay,
    momentum = opt.momentum,
    learningRateDecay = opt.learningRateDecay,
}
optimStateG = {
    learningRate = opt.learningRate,
    weightDecay = opt.weightDecay,
    momentum = opt.momentum,
    learningRateDecay = opt.learningRateDecay,
}


-- Hacks to enable/disable optimization
function optimIsEnabled(optimState)
    return optimState.learningRateSaved == nil
end

function optimDisable(optimState)
    if not optimIsEnabled(optimState) then return end
    optimState.learningRateSaved = optimState.learningRate
    optimState.learningRate = 0
end

function optimEnable(optimState)
    if optimIsEnabled(optimState) then return end
    optimState.learningRate = optimState.learningRateSaved
    optimState.learningRateSaved = nil
end

function trainNaive()
    model.G:training()
    epoch = epoch or 1

    -- drop learning rate every "epoch_step" epochs
    if epoch % opt.epoch_step == 0 then
        optimStateG.learningRate = optimStateG.learningRate/2
    end

    print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

    local noiseInputs = torch.CudaTensor(opt.batchSize, opt.noiseDim)
    local inputs = torch.CudaTensor(opt.batchSize, 1, 32, 32)
    local targets = torch.CudaTensor(opt.batchSize, 2, 32, 32)
    local indices = torch.randperm(provider.trainData.gray:size(1)):long():split(opt.batchSize)
    -- remove last element so that all the batches have equal size
    indices[#indices] = nil

    -- FIXME? Disable noise
    noiseInputs:fill(0)

    local trainError = 0
    local numTrained = 0
    for t,v in ipairs(indices) do
        xlua.progress(t, #indices)

        -- get y and uv images
        --noiseInputs:normal(0, opt.noiseStd)
        inputs:copy(provider.trainData.gray:index(1,v))
        targets:copy(provider.trainData.data:index(1,v):index(2,torch.LongTensor{2,3}))

        local feval = function(x)
            if x ~= parametersG then parametersG:copy(x) end
            gradParametersG:zero()

            -- Get UV output
            local outputs = model.G:forward({inputs, noiseInputs})

            -- Compute loss
            local f = criterionNaive:forward(outputs, targets)
            if f ~= f then
                print("nan detected in error! skipping minibatch..")
            else
                -- compute gradients
                local df_do = criterionNaive:backward(outputs, targets)
                model.G:backward({inputs, noiseInputs}, df_do)

                -- Update error
                trainError = trainError + f
                numTrained = numTrained + opt.batchSize
            end

            return f, gradParametersG
        end

        optim.adam(feval, parametersG, optimStateG)
    end
    trainError = trainError / numTrained

    print('Train average MSE error:', trainError)

    epoch = epoch + 1
end


function train()
    model.D:training()
    model.G:training()
    epoch = epoch or 1

    print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

    -- Drop learning rate every "epoch_step" epochs
    -- if epoch % opt.epoch_step == 0 then
    --     optimStateD.learningRate = optimStateD.learningRate / 2
    --     optimStateG.learningRate = optimStateG.learningRate / 2
    -- end


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
    for t, chunkG in ipairs(indicesG) do
        print(optimIsEnabled(optimStateD))
        print(optimIsEnabled(optimStateG))
        local chunkD = indicesD[t]
        xlua.progress(t, #indicesG)
        
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
            local errFake = criterion:forward(dOutputs:narrow(1, halfBatchSize + 1, halfBatchSize),
                                              dTargets:narrow(1, halfBatchSize + 1, halfBatchSize))

            -- Compute heuristics on whether or not to optimize G/D
            local margin = 0.5
            print(errFake)
            print(errReal)
            optimEnable(optimStateD)
            optimEnable(optimStateG)
            if errFake < margin or errReal < margin then
                optimDisable(optimStateD)
            end
            --if errFake > (1.0 - margin) then --or errReal > (1.0 - margin) then
            --    optimDisable(optimStateG)
            --end
            if (not optimIsEnabled(optimStateG)) and (not optimIsEnabled(optimStateD)) then
                optimEnable(optimStateD)
                optimEnable(optimStateG)
            end

            -- Compute loss
            local f = criterion:forward(dOutputs, dTargets)

            -- Backward pass
            local df_dOutputs = criterion:backward(dOutputs, dTargets)
            model.D:backward(yuvInputs, df_dOutputs)

            -- Update confusion matrix
            for i = 1, opt.batchSize do
                local c
                if dOutputs[i][1] > 0.5 then c = 2 else c = 1 end
                confusion:add(c, dTargets[i] + 1)
            end

            print('|gradParametersD|=', gradParametersD:norm())
            
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
            local df_dyuvSamples = model.D:backward(yuvSamples, df_dOutputs)
            model.G:backward({grayInputs, noiseInputs}, df_dyuvSamples)

            print('|df_dyuvSamples|=', df_dyuvSamples:norm())
            print('|gradParametersG|=', gradParametersG:norm())

            return f, gradParametersG
        end

        -----------------------------------------------------------------------
        -- Update G network: maximize log(D(G(z)))
        -----------------------------------------------------------------------
        grayInputs:copy(provider.trainData.gray:index(1, chunkG))
        noiseInputs:normal(0, opt.noiseStd)
        dTargets:fill(1)  -- goal is to fool the discriminator
        optim.rmsprop(fevalG, parametersG, optimStateG)

        -----------------------------------------------------------------------
        -- Update D network once (K=1): maximize log(D(x)) + log(1 - D(G(z)))
        -- Get half a minibatch of real, half fake
        -----------------------------------------------------------------------
        noiseInputs:normal(0, opt.noiseStd)
        grayInputs:copy(provider.trainData.gray:index(1, chunkD))

        -- Real color samples
        uvInputs[{{1, halfBatchSize}}] = (provider.trainData.data:index(1, chunkD[{{1, halfBatchSize}}])
                                          :index(2, torch.LongTensor{2, 3}))
        dTargets[{{1, halfBatchSize}}]:fill(1)

        -- Generated color samples: forward half of the gray inputs through the generator
        uvInputs[{{halfBatchSize + 1, opt.batchSize}}] = model.G:forward({
            grayInputs[{{halfBatchSize + 1, opt.batchSize}}],
            noiseInputs[{{halfBatchSize + 1, opt.batchSize}}]
        })
        dTargets[{{halfBatchSize + 1, opt.batchSize}}]:fill(0)

        optim.rmsprop(fevalD, parametersD, optimStateD)

        -----------------------------------------------------------------------
        -- Some logging
        -----------------------------------------------------------------------
        print(confusion)
        confusion:zero()

    end

    -- Save model every epoch step
    if epoch % opt.epoch_step == 0 then
        local filename = paths.concat(opt.save, 'model.net')
        print('==> saving model to '..filename)
        torch.save(filename, model)
    end

    epoch = epoch + 1
end


function test()
    model.G:evaluate()
    model.D:evaluate()

    print(c.blue '==>'.." testing")
    local bs = 125
    local testError = 0
    for i=1,provider.testData.data:size(1),bs do
        -- Copy into Cuda
        local noiseInputs = torch.CudaTensor(bs, opt.noiseDim)
        local inputs = torch.CudaTensor(bs, 1, 32, 32)
        local targets = torch.CudaTensor(bs, 2, 32, 32)
        noiseInputs:normal(0, opt.noiseStd)
        inputs:copy(provider.testData.gray:narrow(1, i, bs))
        targets:copy(provider.testData.data:narrow(1, i, bs):index(2, torch.LongTensor{2, 3}))

        -- forward through network
        local outputs = model.G:forward({inputs, noiseInputs})

        -- add to running average
        local f = criterionNaive:forward(outputs, targets)
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


if opt.generator ~= "" then
    -- Load pretrained generator model
    model.G = torch.load(opt.generator)
    parametersG, gradParametersG = model.G:getParameters()
else
    -- Train using L2 loss first
    print(c.blue'==>' ..' training with L2 loss')
    optimStateG.learningRate = 1
    for i=1,25 do
        trainNaive()
    end

    -- Save naively trained generator model
    local filename = paths.concat(opt.save, 'naiveModel.net')
    print('==> saving naive model to '..filename)
    torch.save(filename, model.G)
end


-- Train adversarial game
print(c.blue'==>' ..' training with adversarial game')
epoch = 1
optimStateG.learningRate = opt.learningRate * 10
for i=1,opt.max_epoch do
    train()
    --test()
end


