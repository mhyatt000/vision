import torch
import torch.nn.functional as F

# a = torch.Tensor([3, 3, 2, 2, 3, 3, 2, 2, 2, 2, 3, 3, 2, 2, 3, 3]).view(1, 1, 4, 4)
a = torch.Tensor([3,3,3,2,2,2,3,3,3,2,2,2,3,3,3,2,2,2,2,2,2,3,3,3,2,2,2,3,3,3,2,2,2,3,3,3]).view(1,1,6,6)
# a = torch.rand(1,1,9,9)
x = torch.cat([a, a, a], 1)

x = torch.rand(1,3,6,6)

print(x[0,0])
print()

x = F.pad(x,(1,1,1,1),'reflect')
b, c, w, h = x.shape

unfold = torch.nn.Unfold(3, padding=1)
windows = unfold(x).permute(0, 2, 1)
d = windows.shape[1]
windows = windows.reshape(b,d,c,-1)

medians ,_= windows.median(-1)
medians = medians.permute(0,2,1)
n = int(medians.shape[2]**0.5)

out = medians.reshape(b,c,n,n)
out = (x-out)[:,:,1:-1,1:-1]


print(out[0,0])
print(out.shape)
print()

# out[:,:,:,0] = 0
# out[:,:,:,-1] = 0
# out[:,:,0,:] = 0
# out[:,:,-1,:] = 0

print(x[0,0])
print(x.shape)
