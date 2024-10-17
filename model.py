
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import ViTModel





# Set for ResNet Model
model_Res = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)

# Remove the last layer of the model Res
layers_Res = list(model_Res.children())
model_Res = nn.Sequential(*layers_Res[:-1])

# Set the top layers to be not trainable
count = 0
for child in model_Res.children():
    count += 1
    if count < 8:
        for param in child.parameters():
            param.requires_grad = False
 
# Set for ViT Model
model_trans = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
count = 0
for child in model_trans.children():
    count += 1
    if count < 4:
        for param in child.parameters():
            param.requires_grad = False

layers_trans = list(model_trans.children()) # Get all the layers from the Transformer model
model_trans_top = nn.Sequential(*layers_trans[:-2]) # Remove the normalization layer and pooler layer
trans_layer_norm = list(model_trans.children())[2] # Get the normalization layer



class model_final(nn.Module):
    def __init__(self, model_trans_top, trans_layer_norm, model_Res, dp_rate = 0.3):
        super().__init__()
        # All the trans model layers
        self.model_trans_top = model_trans_top
        self.trans_layer_norm = trans_layer_norm
        self.trans_flatten = nn.Flatten()
        self.trans_linear = nn.Linear(150528, 2048)

        # All the ResNet model
        self.model_Res = model_Res

        # Merge the result and pass the
        self.dropout = nn.Dropout(dp_rate)
        self.linear1 = nn.Linear(4096, 500)
        self.linear2 = nn.Linear(500,1)

    def forward(self, trans_b, res_b):
        # Get intermediate outputs using hidden layer
        result_trans = self.model_trans_top(trans_b)
        patch_state = result_trans.last_hidden_state[:,1:,:] # Remove the classification token and get the last hidden state of all patchs
        result_trans = self.trans_layer_norm(patch_state)
        result_trans = self.trans_flatten(patch_state)
        result_trans = self.dropout(result_trans)
        result_trans = self.trans_linear(result_trans)

        result_res = self.model_Res(res_b)
        # result_res = result_res.squeeze() # Batch size cannot be 1
        result_res = torch.reshape(result_res, (result_res.shape[0], result_res.shape[1]))

        result_merge = torch.cat((result_trans, result_res),1)
        result_merge = self.dropout(result_merge)
        result_merge = self.linear1(result_merge)
        result_merge = self.dropout(result_merge)
        result_merge = self.linear2(result_merge)

        return result_merge
    


model = model_final(model_trans_top, trans_layer_norm, model_Res, dp_rate = 0.3)

# Set up optimizer and learing rate scheduel
params = [param for param in list(model.parameters()) if param.requires_grad]
optimizer = torch.optim.SGD(params, lr=1e-7, momentum=0.2)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.1, 
    patience=2, 
    verbose=True)

# Fit funciton to train the model
def fit(epochs, model, train_dl):   
    opt = optimizer
    sched = lr_scheduler
    loss_func = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        batch_num = 1
        for x_trans, x_res, yb in train_dl:
            # Pass the opt so that funciton will get trained
            total_loss = 0
            preds = model(x_trans, x_res)
            loss = loss_func(preds.squeeze(), yb)
            loss.backward()
            opt.step()
            opt.zero_grad()
            print('\r', f'batch #{batch_num}: {loss}', end='')
            batch_num += 1
            total_loss += loss.item()
        sched.step(total_loss)
        print('\n', f'Epoch: ({epoch+1}/{epochs}) Loss = {total_loss}')
  
epochs = 100
fit(epochs, model, train_dl)