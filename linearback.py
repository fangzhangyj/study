import torch
import time
x_data=torch.Tensor([[1.0],[2.0],[3.0]])
y_data=torch.Tensor([[2.0],[4.0],[6.0]])
class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel,self).__init__()   #调用父类构造函数，just do it
        self.linear=torch.nn.Linear(1,1)  #构造线性模型，输入和输出维度
    def forward(self,x):
        y_pred=self.linear(x)
        return y_pred
model=LinearModel()
criterion=torch.nn.MSELoss(reduction='sum')
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
timestamp1 = time.time()
for epoch in range(1000):
    y_pred=model(x_data)
    loss=criterion(y_pred,y_data)
    print(epoch,loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print('w=',model.linear.weight.item())
print('b=',model.linear.bias.item())
timestamp2 = time.time()
print(timestamp2-timestamp1)
x_test=torch.Tensor([[4.0]])
y_test=model(x_test)
print('y_pred',y_test.item())
