dofile './provider.lua'

provider = Provider()
provider:normalize()
torch.save('/mnt/provider.t7', provider)
