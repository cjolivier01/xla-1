import torch
import lazy_xla
import lazy_tensor_core.debug.metrics as metrics

os.environ['XLA_USE_XRT'] = '=1'
os.environ['XRT_DEVICE_MAP'] = "CPU:0;/job:localservice/replica:0/task:0/device:XLA_CPU:0"
os.environ['XRT_WORKERS'] = "localservice:0;grpc://localhost:40935"

torch.manual_seed(42)

device = 'xla'
dtype = torch.float32

x = torch.randn(2, 3, 4, device=device, dtype=dtype)
y = torch.randn(2, 3, 4, device=device, dtype=dtype)
z = torch.randn(2, 1, 1, device=device, dtype=dtype)
t = torch.randn(2, 3, 4, device=device, dtype=dtype)

print((x / y + z))
print(x.type_as(t))
print(x.relu())
print(x.sign())
print((x <= y))
print(x.reciprocal())
print(x.sigmoid())
print(x.sinh())
print(torch.where(x <= y, z, t))
print(torch.addcmul(x, y, z, value=0.1))
print(torch.remainder(x, y))

print(metrics.metrics_report())
