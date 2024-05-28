import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from augmentations import embed_data_mask
from data_openml import DataSetCatCon, data_prep_custom, load_df
from models import SAINT
from torch import nn
from torch.utils.data import DataLoader
from utils import count_parameters, mean_ab_error, mean_sq_error

# Arguments
opt_set_seed = 1
opt_use_dist = False
opt_use_cat = True
opt_embedding_size = 32
opt_batchsize = 256
opt_attentiontype = "colrow"
opt_transformer_depth = 6
opt_attention_heads = 8
opt_attention_dropout = 0.1
opt_ff_dropout = 0.1
opt_dtask = "reg"
opt_task = "regression"
opt_use_sep = True
opt_cont_embeddings = "MLP"
opt_final_mlp_style = "sep"
opt_vision_dset = "store_true"
opt_lr = 0.0001
opt_epochs = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device is {device}.")

torch.manual_seed(opt_set_seed)

print("Processing the dataset, it might take some time.")
raw_df = pd.read_csv("dataset/delivus_feature_dataset.csv", sep=",")

df, feat_name_dict = load_df(opt_use_cat, opt_use_dist, raw_df)

(
    cat_dims,
    cat_idxs,
    con_idxs,
    X_train,
    y_train,
    X_valid,
    y_valid,
    X_test,
    y_test,
    train_mean,
    train_std,
    train_uuid,
    valid_uuid,
    test_uuid,
) = data_prep_custom(df, "deltime", 42, "regression", feat_name_dict)
np.save("train/cat_dims.npy", np.array(cat_dims))

continuous_mean_std = np.array([train_mean, train_std]).astype(np.float32)

### Setting some hyperparams based on inputs and dataset
_, nfeat = X_train["data"].shape
if nfeat > 100:
    opt_embedding_size = min(8, opt_embedding_size)
    opt_batchsize = min(64, opt_batchsize)
if opt_attentiontype != "col":
    opt_transformer_depth = 1
    opt_attention_heads = min(4, opt_attention_heads)
    opt_attention_dropout = 0.8
    opt_embedding_size = min(32, opt_embedding_size)
    opt_ff_dropout = 0.8


train_ds = DataSetCatCon(
    X_train, y_train, cat_idxs, opt_dtask, continuous_mean_std, train_uuid
)
trainloader = DataLoader(
    train_ds, batch_size=opt_batchsize, shuffle=True, num_workers=0
)

valid_ds = DataSetCatCon(
    X_valid, y_valid, cat_idxs, opt_dtask, continuous_mean_std, valid_uuid
)
validloader = DataLoader(
    valid_ds, batch_size=opt_batchsize, shuffle=False, num_workers=0
)

test_ds = DataSetCatCon(
    X_test, y_test, cat_idxs, opt_dtask, continuous_mean_std, test_uuid
)
testloader = DataLoader(test_ds, batch_size=opt_batchsize, shuffle=False, num_workers=0)

if opt_task == "regression":
    y_dim = 1
else:
    y_dim = len(np.unique(y_train["data"][:, 0]))

# Appending 1 for CLS token, this is later used to generate embeddings.
cat_dims = np.append(np.array([1]), np.array(cat_dims)).astype(int)

model = SAINT(
    categories=tuple(cat_dims),
    num_continuous=len(con_idxs),
    dim=opt_embedding_size,
    dim_out=1,
    depth=opt_transformer_depth,
    heads=opt_attention_heads,
    opt_use_dist=opt_use_dist,
    opt_use_sep=opt_use_sep,
    attn_dropout=opt_attention_dropout,
    ff_dropout=opt_ff_dropout,
    mlp_hidden_mults=(4, 2),
    cont_embeddings=opt_cont_embeddings,
    attentiontype=opt_attentiontype,
    final_mlp_style=opt_final_mlp_style,
    y_dim=y_dim,
    nfeats=int(
        (len(con_idxs) / 2) + len(cat_dims) - len(feat_name_dict["trg_feat_cat"])
    ),
)

vision_dset = opt_vision_dset
criterion = nn.MSELoss().to(device)
model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=opt_lr)

best_valid_auroc = 0
best_valid_accuracy = 0
best_test_auroc = 0
best_test_accuracy = 0
best_valid_rmse = 100000
print("Training begins now.")

for epoch in range(opt_epochs):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        optimizer.zero_grad()
        x_categ, x_cont, y_gts, cat_mask, con_mask, x_uuid = (
            data[0].to(device),
            data[1].to(device),
            data[2].to(device),
            data[3].to(device),
            data[4].to(device),
            data[5],
        )

        # We are converting the data to embeddings in the next step
        _, x_categ_enc, x_cont_enc = embed_data_mask(
            x_categ, x_cont, cat_mask, con_mask, model, vision_dset
        )

        B, C, D = x_cont_enc.shape
        _, F, _ = x_categ_enc.shape

        temp_dim = int(C / 2)

        if opt_use_sep:
            if not opt_use_dist:
                len_c = len(feat_name_dict["common_feat_con"])
                len_s = len(feat_name_dict["src_feat_con"])
                len_t = len(feat_name_dict["trg_feat_con"])

                temp_tensor_1 = torch.cat(
                    [
                        x_cont_enc[:, :len_c, :],
                        x_cont_enc[:, len_c : (len_c + len_s), :],
                    ],
                    dim=1,
                )
                temp_tensor_2 = torch.cat(
                    [x_cont_enc[:, :len_c, :], x_cont_enc[:, (len_c + len_s) :, :]],
                    dim=1,
                )
                x_cont_enc = torch.cat([temp_tensor_1, temp_tensor_2], dim=0)

        # separate src / dst feature
        if opt_use_sep:
            if F != 1:
                len_c = len(feat_name_dict["common_feat_cat"]) + 1  # 1 is for cls token
                len_s = len(feat_name_dict["src_feat_cat"])
                len_t = len(feat_name_dict["trg_feat_cat"])
                temp_tensor_1 = torch.cat(
                    [
                        x_categ_enc[:, :len_c, :],
                        x_categ_enc[:, len_c : (len_c + len_s), :],
                    ],
                    dim=1,
                )
                temp_tensor_2 = torch.cat(
                    [x_categ_enc[:, :len_c, :], x_categ_enc[:, (len_c + len_s) :, :]],
                    dim=1,
                )
                x_categ_enc = torch.cat([temp_tensor_1, temp_tensor_2], dim=0)
            else:
                x_categ_enc = torch.cat([x_categ_enc, x_categ_enc], dim=0)

        reps = model.transformer(x_categ_enc, x_cont_enc)

        # select only the representations corresponding to CLS token and apply mlp on it in the next step to get the predictions.
        if opt_use_sep:
            y_reps = torch.cat([reps[:B, :1, :], reps[B:, :1, :]], dim=2)
        elif not opt_use_sep:
            y_reps = reps[:, 0, :]

        y_outs = model.mlpfory(y_reps)
        loss = criterion(y_outs.squeeze(), y_gts.squeeze())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    if epoch % 5 == 0:
        model.eval()
        with torch.no_grad():
            valid_rmse = mean_sq_error(
                model, validloader, device, vision_dset, opt_use_sep, opt_use_dist
            )
            test_rmse = mean_sq_error(
                model, testloader, device, vision_dset, opt_use_sep, opt_use_dist
            )

            valid_mae = mean_ab_error(
                model, validloader, device, vision_dset, opt_use_sep, opt_use_dist
            )
            test_mae = mean_ab_error(
                model, testloader, device, vision_dset, opt_use_sep, opt_use_dist
            )

            print("[EPOCH %d] VALID RMSE: %.3f" % (epoch + 1, valid_rmse))
            print("[EPOCH %d] TEST RMSE: %.3f" % (epoch + 1, test_rmse))
            print("[EPOCH %d] VALID MAE: %.3f" % (epoch + 1, valid_mae))
            print("[EPOCH %d] TEST MAE: %.3f" % (epoch + 1, test_mae))

            if valid_rmse < best_valid_rmse:
                best_valid_rmse = valid_rmse
                best_test_rmse = test_rmse
                best_valid_mae = valid_mae
                best_test_mae = test_mae
                torch.save(model.state_dict(), "train/bestmodel.pth")
        model.train()

total_parameters = count_parameters(model)
print("TOTAL NUMBER OF PARAMS: %d" % (total_parameters))
if opt_task == "binary":
    print("AUROC on best model:  %.3f" % (best_test_auroc))
elif opt_task == "multiclass":
    print("Accuracy on best model:  %.3f" % (best_test_accuracy))
else:
    print("RMSE on best model:  %.3f" % (best_test_rmse))
    print("MAE on best model:  %.3f" % (best_test_mae))
