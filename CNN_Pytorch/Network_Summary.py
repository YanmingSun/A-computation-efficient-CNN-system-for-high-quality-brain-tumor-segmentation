import Network_Old
import torch
from torchsummary import summary

cnn = Network_Old.Net_Old_version()
cnn.cpu()
cnn.eval()

summary(cnn, (4, 168, 200))