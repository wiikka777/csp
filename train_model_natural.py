import time
import torch
import pandas as pd
import numpy as np
from scipy.stats import norm
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.nn import BCELoss, BCEWithLogitsLoss, MSELoss
from model.dcn import My_DeepCrossNetworkModel_withCommentsRanking # (确保 model/dcn.py 是我们修正过的“快速版”)
from utils.set_seed import setup_seed
from utils.summary_dat import cal_comments_dims, make_feature_with_comments, cal_field_dims
from utils.data_wrapper import Wrap_Dataset, Wrap_Dataset4
from utils.early_stop import EarlyStopping2
from utils.loss import ListMLELoss
from utils.evaluate import cal_group_metric, cal_reg_metric
from preprocessing.cal_ground_truth import cal_ground_truth

class Learner(object):
    
    def __init__(self, args):
        self.dat_name = args.dat_name
        self.model_name = args.model_name
        self.label_name = args.label_name

        self.group_num = args.group_num
        self.windows_size = args.windows_size
        self.eps = args.eps

        self.batch_size = args.batch_size
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.patience = args.patience
        self.use_cuda = args.use_cuda
        self.epoch_num = args.epoch_num
        self.seed = args.randseed
        self.fout = args.fout

        self.noise_point = args.noise_point
        self.bias_point = args.bias_point
        if args.dat_name == 'KuaiComt':
            if args.label_name == 'WLR':
                self.label_name = 'long_view2'
                self.weight_name = 'weighted_st'
                self.label2_name = 'comments_score'
                self.label1_name = 'user_clicked'

        self.load_to_eval = args.load_to_eval
        # (关键!) 设置你找到的最佳参数
        self.lambda1 = 0.001 
        self.lambda2 = 0.1
        print(f"Using Hyperparameters: lambda1={self.lambda1}, lambda2={self.lambda2}")

        # -----------------------------
        # (关键!) 嵌入表加载开关
        # -----------------------------
        # 切换这个变量来运行两个实验
        # '1.8B' = 运行 1.8B 实验组
        # '7B'   = 运行 7B 新基线
        self.EMBEDDING_MODE = '1.8B' # <-- 在这里切换
        
        print(f"--- RUNNING IN {self.EMBEDDING_MODE} MODE ---")
        
        if self.EMBEDDING_MODE == '1.8B':
            VIDEO_EMB_PATH = '/user/zhuohang.yu/u24922/LCU-main/finetune/embeddings/video_embeddings_qwen1.8b_tiny.pt'
            COMMENT_EMB_PATH = '/user/zhuohang.yu/u24922/LCU-main/finetune/embeddings/comment_embeddings_qwen1.8b_tiny.pt'
        else: # '7B'
            # (注意!) 更改为你的 7B 原始嵌入表路径
            #VIDEO_EMB_PATH = '../rec_datasets/KuaiComt/video_embeddings_qwen7b_tiny.pt'
            #COMMENT_EMB_PATH = '../rec_datasets/KuaiComt/comment_embeddings_qwen7b_tiny.pt'
            raise ValueError(f"EMBEDDING_MODE '{self.EMBEDDING_MODE}' 不受支持，请使用 '1.8B' 运行。")

        print(f"Loading video embeddings (cpu) from: {VIDEO_EMB_PATH}")
        # (修正!) 你的 1.8B 嵌入保存为字典, 7B 保存为完整张量, 我们需要统一处理
        video_embeddings_data = torch.load(VIDEO_EMB_PATH)
        if isinstance(video_embeddings_data, dict):
            # 这是你的 1.8B 字典 (短 ID -> 嵌入)
            # (注意!) 我们假设 7B 也是字典 (短 ID -> 嵌入)。如果 7B 是张量, 这里的逻辑需要修改
            video_ids_sorted = sorted(video_embeddings_data.keys())
            video_embeddings_list = [video_embeddings_data[k] for k in video_ids_sorted]
            video_embeddings_tensor = torch.stack(video_embeddings_list).to(dtype=torch.float32).cpu()
            # (关键!) 你的 ID 已经是整数了
            self.video_id2idx = {vid: i for i, vid in enumerate(video_ids_sorted)}
        else:
            # 这是 7B 完整张量 (假设它很大)
            video_embeddings_tensor = video_embeddings_data.to(dtype=torch.float32).cpu()
            # (注意!) 这里的 ID 映射逻辑依赖 7B 嵌入表的原始格式
            # 我们假设 7B 嵌入的索引与 "短 ID" (行号) 一致
            self.video_id2idx = {i: i for i in range(len(video_embeddings_tensor))} 
            print("Warning: Assuming 7B video embedding index matches short ID.")

        video_embeddings_tensor.requires_grad = False
        self.photo_embeddings = video_embeddings_tensor
        print(f"Loaded video embeddings: {self.photo_embeddings.shape}")


        print(f"Loading comment embeddings (cpu) from: {COMMENT_EMB_PATH}")
        comment_embeddings_data = torch.load(COMMENT_EMB_PATH)
        if isinstance(comment_embeddings_data, dict):
            # 这是你的 1.8B 字典 (短 ID -> 嵌入)
            comment_ids_sorted = sorted(comment_embeddings_data.keys())
            comment_embeddings_list = [comment_embeddings_data[k] for k in comment_ids_sorted]
            comment_embeddings_tensor = torch.stack(comment_embeddings_list).to(dtype=torch.float32).cpu()
            self.comment_id2idx = {cid: i for i, cid in enumerate(comment_ids_sorted)}
        else:
            # 这是 7B 完整张量
            comment_embeddings_tensor = comment_embeddings_data.to(dtype=torch.float32).cpu()
            self.comment_id2idx = {i: i for i in range(len(comment_embeddings_tensor))}
            print("Warning: Assuming 7B comment embedding index matches short ID.")

        comment_embeddings_tensor.requires_grad = False
        self.comment_embeddings = comment_embeddings_tensor
        print(f"Loaded comment embeddings: {self.comment_embeddings.shape}")


    def train(self):
        setup_seed(self.seed)
        self.all_dat, self.train_dat, self.vali_dat, self.test_dat = self._load_and_spilt_dat()
        self.train_loader, self.vali_loader, self.test_loader = self._wrap_dat()
        self.model,self.c_model, self.optim, self.c_optim, self.early_stopping = self._init_train_env()
        if not self.load_to_eval:
            self._train_iteration()
        self._test_and_save()

    @staticmethod
    def _cal_correct_wt(row, sigma=1.0):
        d = row['duration_ms']
        wt = row['play_time_truncate']
        return wt - 1 * (norm.pdf((d - wt)/sigma) / norm.cdf((d - wt)/sigma))

    def _load_and_spilt_dat(self):
        if self.dat_name == 'KuaiComt':
            # (关键!) 加载你采样的 2% TINY 数据集
            print("--- LOADING TINY 2% DATASET ---")
            all_dat = pd.read_csv('/user/zhuohang.yu/u24922/LCU-main/rec_datasets/WM_KuaiComt/KuaiComt_TINY_subset.csv', sep=',')
            
            # (注意!) 你需要确保 cal_ground_truth 仍然有效
            all_dat = cal_ground_truth(all_dat, self.dat_name)
            
            # (注意!) 这里的日期划分在 TINY 数据集上可能导致验证/测试集为空
            # 我们将使用 80/10/10 的随机划分代替
            
            # all_dat = all_dat.sample(frac=1, random_state=self.seed).reset_index(drop=True)
            # n = len(all_dat)
            # n_train = int(n * 0.8)
            # n_vali = int(n * 0.1)
            # train_dat = all_dat.iloc[:n_train]
            # vali_dat = all_dat.iloc[n_train : n_train + n_vali]
            # test_dat = all_dat.iloc[n_train + n_vali :]

            # (更新!) 保持和基线一致的日期划分, 确保TINY数据集中仍然有数据
            print("Using original date splits on TINY dataset...")
            train_dat = all_dat[(all_dat['date'] <= 2023102199) & (all_dat['date'] >= 2023100100)]
            vali_dat = all_dat[(all_dat['date'] <= 2023102699) & (all_dat['date'] >= 2023102200)]
            test_dat = all_dat[(all_dat['date'] <= 2023103199) & (all_dat['date'] >= 2023102700)]
            
            print(f"Train samples: {len(train_dat)}, Vali samples: {len(vali_dat)}, Test samples: {len(test_dat)}")
            if len(vali_dat) == 0 or len(test_dat) == 0:
                print("--- WARNING! Validation or Test set is empty! Date splitting failed on tiny data. ---")
                print("--- Reverting to 80/10/10 random split. ---")
                all_dat = all_dat.sample(frac=1, random_state=self.seed).reset_index(drop=True)
                n = len(all_dat)
                n_train = int(n * 0.8)
                n_vali = int(n * 0.1)
                train_dat = all_dat.iloc[:n_train]
                vali_dat = all_dat.iloc[n_train : n_train + n_vali]
                test_dat = all_dat.iloc[n_train + n_vali :]
                print(f"Random Split -> Train: {len(train_dat)}, Vali: {len(vali_dat)}, Test: {len(test_dat)}")

        return all_dat, train_dat, vali_dat, test_dat

    # _wrap_dat (保持不变)
    def _wrap_dat(self):
        print("Wrapping data...")
        input_train = Wrap_Dataset4(make_feature_with_comments(self.train_dat, self.dat_name),
                                    self.train_dat[self.label_name].tolist(),
                                    self.train_dat[self.weight_name].tolist(),
                                    self.train_dat[self.label1_name].tolist(),
                                    self.train_dat[self.label2_name].tolist(), False)
        train_loader = DataLoader(input_train, 
                                  batch_size=self.batch_size, 
                                  shuffle=True)

        input_vali = Wrap_Dataset(make_feature_with_comments(self.vali_dat, self.dat_name),
                                  self.vali_dat[self.label_name].tolist(),
                                  self.vali_dat[self.weight_name].tolist())
        vali_loader = DataLoader(input_vali, 
                                 batch_size=2048, 
                                 shuffle=False)

        input_test = Wrap_Dataset(make_feature_with_comments(self.test_dat, self.dat_name),
                                  self.test_dat[self.label_name].tolist(),
                                  self.test_dat[self.weight_name].tolist())
        test_loader = DataLoader(input_test, 
                                 batch_size=2048, 
                                 shuffle=False)
        return train_loader, vali_loader, test_loader

    # _init_train_env (使用 "快速" dcn.py 的逻辑)
    def _init_train_env(self):
        print("Initializing model...")
        if self.model_name == 'DCN':
            # (关键!) 将 CPU 嵌入表 和 id->idx 映射表 打包传递
            text_embeddings_bundle = {
                "video_emb_tensor_cpu": self.photo_embeddings,
                "video_id2idx": self.video_id2idx,
                "comment_emb_tensor_cpu": self.comment_embeddings,
                "comment_id2idx": self.comment_id2idx
            }

            model = My_DeepCrossNetworkModel_withCommentsRanking(field_dims=cal_field_dims(self.all_dat, self.dat_name),
                                                                comments_dims=cal_comments_dims(self.all_dat, self.dat_name),
                                                                embed_dim=10, num_layers=3, mlp_dims=[64,64,64], dropout=0.2,
                                                                text_embeddings=text_embeddings_bundle) # <-- 传递 bundle
            c_model = My_DeepCrossNetworkModel_withCommentsRanking(field_dims=cal_field_dims(self.all_dat, self.dat_name),
                                                                comments_dims=cal_comments_dims(self.all_dat, self.dat_name),
                                                                embed_dim=10, num_layers=3, mlp_dims=[64,64,64], dropout=0.2,
                                                                text_embeddings=text_embeddings_bundle) # <-- 传递 bundle

        if self.use_cuda:
            model = model.cuda()
            c_model = c_model.cuda()

        lr = 1e-4
        optim = Adam(model.parameters(), lr=lr, weight_decay=self.weight_decay)
        c_optim = Adam(c_model.parameters(), lr=lr, weight_decay=self.weight_decay)

        early_stopping = EarlyStopping2(self.fout + '_temp', patience=self.patience, verbose=True)

        print(model)
        return model, c_model, optim, c_optim, early_stopping 

    # _train_iteration (保持不变)
    def _train_iteration(self):
        dur=[]
        for epoch in range(self.epoch_num):
            if epoch >= 0:
                t0 = time.time()
            loss_log = []
            c_loss_log = []

            self.model.train()
            self.c_model.train()

            for _id, batch in enumerate(self.train_loader):
                batch = [item.cuda() for item in batch]
                self.c_model.train()
                self.c_optim.zero_grad()
                BCELossfunc = BCEWithLogitsLoss()
                output_score = self.c_model(batch[0])
                output_score = output_score.view(batch[0].size(0))
                target = batch[1]
                train_loss = BCELossfunc(output_score, target)
                train_loss.backward()
                self.c_optim.step()
                c_loss_log.append(train_loss.item())

                self.model.train()
                self.optim.zero_grad()
                BCELossfunc = BCEWithLogitsLoss(weight=batch[2])
                BCELossfunc2 = BCELoss()
                ListMLEfunc = ListMLELoss()
                output_score = self.model(batch[0])
                comments_score = self.model.get_comment_probs()
                comments_score_ = self.model.get_comment_probs_()
                output_score = output_score.view(batch[0].size(0))
                comments_score = comments_score.view(batch[0].size(0), -1)
                comments_score_ = comments_score_.view(batch[0].size(0), -1)
                target = batch[1]
                train_loss = BCELossfunc(output_score, target)        
                label_sums = batch[3].sum(dim=1)
                mask = label_sums > 0
                masked_output = comments_score[mask]
                masked_target = batch[3][mask]
                if masked_output.numel() > 0: 
                    train_loss += self.lambda1 * BCELossfunc2(masked_output, masked_target)
                train_loss += self.lambda2 * ListMLEfunc(comments_score_, batch[4])
                train_loss.backward()
                self.optim.step()
                loss_log.append(train_loss.item())

            if self.weight_name == 'weighted_st':
                rmse, mae, xgauc, xauc = cal_reg_metric(self.vali_dat, self.model, self.vali_loader, self.all_dat, self.weight_name, self.c_model)
            else:
                rmse, mae, xgauc, xauc = 0, 0, 0, 0
            self.early_stopping(mae, self.model, self.c_model)

            if self.early_stopping.early_stop:
                print("Early stopping")
                break 

            if epoch >= 0:
                dur.append(time.time() - t0)

            print("Epoch {:05d} | Time(s) {:.4f} | Train_Loss {:.4f} | Train_c_Loss {:.4f} | "
                        "Vali_NDCG@1 {:.4f}| Vali_RMSE {:.4f}| Vali_MAE {:.4f}| Vali_GXAUC {:.4f}| Vali_XAUC {:.4f}|". format(epoch, np.mean(dur), np.mean(loss_log),np.mean(c_loss_log),
                                                                        0, rmse, mae, xgauc, xauc))
    
    # _test_and_save (使用 "快速" dcn.py 的逻辑)
    def _test_and_save(self):
        print("Testing...")
        text_embeddings_bundle = {
            "video_emb_tensor_cpu": self.photo_embeddings,
            "video_id2idx": self.video_id2idx,
            "comment_emb_tensor_cpu": self.comment_embeddings,
            "comment_id2idx": self.comment_id2idx
        }
        model = My_DeepCrossNetworkModel_withCommentsRanking(field_dims=cal_field_dims(self.all_dat, self.dat_name),
                                                            comments_dims=cal_comments_dims(self.all_dat, self.dat_name),
                                                            embed_dim=10, num_layers=3, mlp_dims=[64,64,64], dropout=0.2,
                                                            text_embeddings=text_embeddings_bundle)
        c_model = My_DeepCrossNetworkModel_withCommentsRanking(field_dims=cal_field_dims(self.all_dat, self.dat_name),
                                                            comments_dims=cal_comments_dims(self.all_dat, self.dat_name),
                                                            embed_dim=10, num_layers=3, mlp_dims=[64,64,64], dropout=0.2,
                                                            text_embeddings=text_embeddings_bundle)

        model = model.cuda()
        c_model = c_model.cuda()

        model.load_state_dict(torch.load(self.fout + '_temp_checkpoint.pt'))
        c_model.load_state_dict(torch.load(self.fout + '_temp_usr_checkpoint.pt'))

        ndcg_ls, pcr_ls, wt_ls, gauc_val, mrr_val= cal_group_metric(self.test_dat, c_model,[1,3,5], self.test_loader, dat_name=self.dat_name)

        if self.weight_name == 'weighted_st':
            rmse, mae, xgauc, xauc = cal_reg_metric(self.test_dat, model, self.test_loader, self.all_dat, self.weight_name, c_model)
        else:
            rmse, mae, xgauc, xauc = 0, 0, 0, 0

        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("{}_{} | Log_loss {:.4f} | AUC {:.4f} | GAUC {:.4f} | MRR {:.4f} | "
                    "nDCG@1 {:.4f}| nDCG@3 {:.4f}| nDCG@5 {:.4f}| "
                    "PCR@1 {:.4f}| PCR@3 {:.4f}| PCR@5 {:.4f}| WT@1 {:.4f}| WT@3 {:.4f}| WT@5 {:.4f}| RMSE {:.4f} | MAE {:.4f}| XGAUC {:.4f}| XAUC {:.4f}|". format(self.model_name, self.label_name, 0,0, gauc_val, mrr_val,
                                                                    ndcg_ls[0],ndcg_ls[1],ndcg_ls[2],pcr_ls[0],pcr_ls[1],pcr_ls[2],wt_ls[0],wt_ls[1],wt_ls[2], rmse, mae, xgauc, xauc))
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        
        df_result = pd.DataFrame([],columns=['GAUC','MRR','nDCG@1','nDCG@3','nDCG@5','PCR@1','PCR@3','PCR@5','WT@1','WT@3','WT@5','RMSE', 'MAE','XGAUC', 'XAUC'])
        df_result.loc[1] =  [gauc_val, mrr_val] + ndcg_ls + pcr_ls + wt_ls + [rmse, mae, xgauc, xauc]

        # (修正!) 保存结果时, 附加上 EMBEDDING_MODE
        result_filename = f'{self.fout}_result_{self.EMBEDDING_MODE}.csv'
        model_filename = f'{self.fout}_model_{self.EMBEDDING_MODE}.pt'
        
        df_result.to_csv(result_filename)
        torch.save(model.state_dict(), model_filename)
        print(f"Results saved to: {result_filename}")
        print(f"Model saved to: {model_filename}")

if __name__=="__main__":
    pass
