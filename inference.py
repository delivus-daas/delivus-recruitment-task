import argparse
import os
import tarfile

import numpy as np
import pandas as pd
import torch
from augmentations import embed_data_mask
from data_openml import DataSetCatCon, load_df_inference
from models import SAINT
from torch import nn
from torch.utils.data import DataLoader

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

print("Started Inference")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device is {device}.")

torch.manual_seed(opt_set_seed)
np.random.seed(42)

raw_dataset = pd.read_csv("dataset/delivus_feature_dataset.csv", sep=",")

df, feat_name_dict = load_df_inference(opt_use_cat, opt_use_dist, raw_dataset)
X = df

uuid_all = X["shippingitem_uuid"]
X = X.drop(["shippingitem_uuid"], axis=1)

temp_con = X[feat_name_dict["trg_feat_con"]]
temp_cat = X[feat_name_dict["trg_feat_cat"]]
temp_con.columns = [x + "_2" for x in temp_con.columns]
temp_cat.columns = [x + "_2" for x in temp_cat.columns]

X = pd.concat(
    [
        X[feat_name_dict["trg_feat_con"]],
        temp_con,
        X[feat_name_dict["common_feat_cat"]],
        X[feat_name_dict["trg_feat_cat"]],
        temp_cat,
    ],
    axis=1,
)

categorical_columns = (
    feat_name_dict["common_feat_cat"]
    + feat_name_dict["trg_feat_cat"]
    + temp_cat.columns.tolist()
)
cont_columns = (
    feat_name_dict["common_feat_con"]
    + feat_name_dict["trg_feat_con"]
    + temp_con.columns.tolist()
)

cat_idxs = [X.columns.get_loc(c) for c in categorical_columns if c in X]
con_idxs = list(set(range(len(X.columns))) - set(cat_idxs))

## Assuming NaNs are already handled, so no need for nan_mask in this custom data.
cat_dims = np.load("train/cat_dims.npy")

for col in cont_columns:
    X.fillna(X.loc[:, col].mean(), inplace=True)

y = {"data": np.zeros([X.values.shape[0], 1])}
X = {"data": X.values}

train_mean, train_std = np.array(X["data"][:, con_idxs], dtype=np.float32).mean(
    0
), np.array(X["data"][:, con_idxs], dtype=np.float32).std(0)
train_std = np.where(train_std < 1e-6, 1e-6, train_std)

continuous_mean_std = np.array([train_mean, train_std]).astype(np.float32)

##### Setting some hyperparams based on inputs and dataset
_, nfeat = X["data"].shape
if nfeat > 100:
    opt_embedding_size = min(8, opt_embedding_size)
    opt_batchsize = min(64, opt_batchsize)
if opt_attentiontype != "col":
    opt_transformer_depth = 1
    opt_attention_heads = min(4, opt_attention_heads)
    opt_attention_dropout = 0.8
    opt_embedding_size = min(32, opt_embedding_size)
    opt_ff_dropout = 0.8

train_ds = DataSetCatCon(X, y, cat_idxs, opt_dtask, continuous_mean_std, uuid_all)
trainloader = DataLoader(
    train_ds, batch_size=opt_batchsize, shuffle=False, num_workers=0
)

y_dim = 1
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

model.load_state_dict(torch.load("train/bestmodel.pth"))
model.to(device)

vision_dset = opt_vision_dset
criterion = nn.MSELoss().to(device)

model.eval()
for i, data in enumerate(trainloader, 0):
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
            len_t = len(feat_name_dict["trg_feat_con"])
            x_cont_enc = torch.cat(
                [x_cont_enc[:, : (len_c + len_t)], x_cont_enc[:, (len_c + len_t) :]],
                dim=0,
            )

    # separate src / dst feature
    if opt_use_sep:
        if F != 1:
            len_c = len(feat_name_dict["common_feat_cat"]) + 1
            len_t = len(feat_name_dict["trg_feat_cat"])

            temp_tensor_1 = torch.cat(
                [x_categ_enc[:, :len_c, :], x_categ_enc[:, len_c : (len_c + len_t), :]],
                dim=1,
            )
            temp_tensor_2 = torch.cat(
                [x_categ_enc[:, :len_c, :], x_categ_enc[:, (len_c + len_t) :, :]], dim=1
            )
            x_categ_enc = torch.cat([temp_tensor_1, temp_tensor_2], dim=0)
        else:
            x_categ_enc = torch.cat([x_categ_enc, x_categ_enc], dim=0)

    with torch.no_grad():
        reps = model.transformer(x_categ_enc, x_cont_enc)

    # select only the representations corresponding to CLS token and apply mlp on it in the next step to get the predictions.
    if opt_use_sep:
        y_reps = torch.cat([reps[:B, :1, :], reps[B:, :1, :]], dim=2)
    elif not opt_use_sep:
        y_reps = reps[:, 0, :]

    if i == 0:
        final_var = y_reps.detach().cpu().numpy()
        final_uuid = np.array(x_uuid)
    else:
        final_var = np.concatenate([final_var, y_reps.detach().cpu().numpy()], axis=0)
        final_uuid = np.concatenate([final_uuid, np.array(x_uuid)])

np.save("inference/uuid.npy", final_uuid)
np.save("inference/reps_dst.npy", final_var[:, :, 32:])
print("Inference Finished")
