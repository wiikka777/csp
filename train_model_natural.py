import time
import torch
import pandas as pd
import numpy as np
from scipy.stats import norm
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.nn import BCELoss, BCEWithLogitsLoss, MSELoss
from model.dcn import My_DeepCrossNetworkModel_withCommentsRanking 
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
        
        # 针对 KuaiComt 数据集定义主标签、权重和辅助标签
        if args.dat_name == 'KuaiComt':
            if args.label_name == 'WLR':
                self.label_name = 'long_view2'      # 主任务: 深度观看 (隐式兴趣)
                self.weight_name = 'weighted_st'    # 权重: 观看时长权重
                self.label2_name = 'comments_score' # 辅助任务 2: 评论质量
                self.label1_name = 'user_clicked'   # 辅助任务 1: 用户点击评论/显式行为

        self.load_to_eval = args.load_to_eval
        
        # **【核心修改点 1】**：设置辅助损失权重为零，禁用辅助任务
        self.lambda1 = 0.0      # 禁用 BCE 辅助损失 (依赖 user_clicked)
        self.lambda2 = 0.0      # 禁用 ListMLE 辅助损失 (依赖 comments_score)
        print(f"Using Hyperparameters (NEUTRAL MODE): lambda1={self.lambda1}, lambda2={self.lambda2}")

        # -----------------------------
        # 嵌入表加载配置
        # -----------------------------
        self.EMBEDDING_MODE = '7B_NEUTRAL' # 标识当前为中立实验模式
        
        print(f"--- RUNNING IN {self.EMBEDDING_MODE} MODE ---")
        
        # 确保路径与你的实际文件路径一致 (假设都位于 LCU-main_backup/rec_datasets/WM_KuaiComt/)
        VIDEO_EMB_PATH = '../rec_datasets/WM_KuaiComt/video_embeddings_qwen7b_tiny.pt'
        COMMENT_EMB_PATH = '../rec_datasets/WM_KuaiComt/comment_embeddings_qwen7b_tiny.pt'
             
        print(f"Loading video embeddings (cpu) from: {VIDEO_EMB_PATH}")
        video_embeddings_data = torch.load(VIDEO_EMB_PATH)
        if isinstance(video_embeddings_data, dict):
            video_ids_sorted = sorted(video_embeddings_data.keys())
            video_embeddings_list = [video_embeddings_data[k] for k in video_ids_sorted]
            video_embeddings_tensor = torch.stack(video_embeddings_list).to(dtype=torch.float32).cpu()
            self.video_id2idx = {vid: i for i, vid in enumerate(video_ids_sorted)}
        else:
            video_embeddings_tensor = video_embeddings_data.to(dtype=torch.float32).cpu()
            self.video_id2idx = {i: i for i in range(len(video_embeddings_tensor))} 
            print("Warning: Assuming 7B video embedding index matches short ID.")

        video_embeddings_tensor.requires_grad = False
        self.photo_embeddings = video_embeddings_tensor
        print(f"Loaded video embeddings: {self.photo_embeddings.shape}")

        print(f"Loading comment embeddings (cpu) from: {COMMENT_EMB_PATH}")
        comment_embeddings_data = torch.load(COMMENT_EMB_PATH)
        if isinstance(comment_embeddings_data, dict):
            comment_ids_sorted = sorted(comment_embeddings_data.keys())
            comment_embeddings_list = [comment_embeddings_data[k] for k in comment_ids_sorted]
            comment_embeddings_tensor = torch.stack(comment_embeddings_list).to(dtype=torch.float32).cpu()
            self.comment_id2idx = {cid: i for i, cid in enumerate(comment_ids_sorted)}
        else:
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
            # **【核心修改点 2】**：加载新生成的不包含显式反馈的 NEUTRAL 数据集
            print("--- LOADING NEUTRAL DATASET ---")
            # 确保这里的路径与你创建脚本中的输出路径一致
            all_dat = pd.read_csv('../rec_datasets/WM_KuaiComt/KuaiComt_NEUTRAL_subset.csv', sep=',') 
            
            # cal_ground_truth 仍然需要运行，但由于 is_like 等列缺失，它会生成 NaN
            # 除非你修改 cal_ground_truth 让其只依赖于仍然存在的列
            # 此时的 user_clicked 和 comments_score 字段会缺失
            all_dat = cal_ground_truth(all_dat, self.dat_name)
            
            print("Using original date splits on NEUTRAL dataset...")
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

    # _wrap_dat (保持不变，但依赖于 self.label1/2_name 的存在性)
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

    # _init_train_env (保持不变)
    def _init_train_env(self):
        print("Initializing model...")
        if self.model_name == 'DCN':
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

        if self.use_cuda:
            model = model.cuda()
            c_model = c_model.cuda()

        lr = 1e-4
        optim = Adam(model.parameters(), lr=lr, weight_decay=self.weight_decay)
        c_optim = Adam(c_model.parameters(), lr=lr, weight_decay=self.weight_decay)

        early_stopping = EarlyStopping2(self.fout + '_temp', patience=self.patience, verbose=True)

        print(model)
        return model, c_model, optim, c_optim, early_stopping 

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
                
                # BCELossfunc2 和 ListMLEfunc 用于辅助损失，但在中立模式下，我们仅保留主任务。
                # 由于 lambda1=0 和 lambda2=0，以下辅助损失计算可以注释掉，
                # 但保留它们（即使权重为 0）可避免因 batch[3]/batch[4] 缺失而导致的崩溃。
                
                # ListMLEfunc = ListMLELoss() 
                
                output_score = self.model(batch[0])
                comments_logits = self.model.get_comment_probs()
                comments_logits_ = self.model.get_comment_probs_()
                output_score = output_score.view(batch[0].size(0))
                comments_logits = comments_logits.view(batch[0].size(0), -1)
                comments_logits_ = comments_logits_.view(batch[0].size(0), -1)
                target = batch[1]
                
                # **【核心修改点 3】**：主任务损失 (long_view2)
                train_loss = BCELossfunc(output_score, target) 

                
                # --- 辅助损失部分 (由于 lambda1=0, lambda2=0，此部分不会影响 train_loss, 
                # 但我们需要注意 batch[3] 和 batch[4] 可能缺失) ---
                
                # try/except 保护，以防 batch[3] 和 batch[4] 字段缺失导致的索引错误
                # 在 train_model 中，这部分代码是必须存在的，否则会因为缺失标签而失败
                try:
                    # 辅助损失 1: BCE loss for user_clicked (batch[3])
                    label_sums = batch[3].sum(dim=1)
                    mask = label_sums > 0
                    masked_output = comments_logits[mask] 
                    masked_target = batch[3][mask]

                    if masked_output.numel() > 0 and self.lambda1 > 0:
                        BCELossfunc2 = BCEWithLogitsLoss()
                        labels_norm = batch[4][mask].clamp(min=0)
                        labels_norm = labels_norm / (labels_norm.max(dim=1, keepdim=True).values + 1e-8)
                        bce_loss = BCELossfunc2(masked_output, labels_norm)
                        train_loss += self.lambda1 * bce_loss
                        
                    # 辅助损失 2: ListMLE loss for comments_score (batch[4])
                    if self.lambda2 > 0:
                        ListMLEfunc = ListMLELoss()
                        safe_labels = batch[4].clamp(min=0)
                        labels_mle = torch.log1p(safe_labels)
                        listmle_loss = ListMLEfunc(comments_logits_, labels_mle)
                        train_loss += self.lambda2 * listmle_loss

                except IndexError:
                    # 仅在 batch 索引发生错误时打印警告，因为我们预期 batch[3] 和 batch[4] 可能缺失
                    if self.lambda1 > 0 or self.lambda2 > 0:
                         print("Warning: Auxiliary loss components skipped due to missing batch indices (expected in NEUTRAL mode).")
                    
                
                train_loss.backward()
                if torch.isnan(train_loss).any():
                    print(f"FATAL: NaN detected in train_loss at batch {_id}. Stopping training.")
                    break 
                self.optim.step()
                loss_log.append(train_loss.item())

            # ... (评估和 Early Stopping 保持不变)
            # ...
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

    
    # _test_and_save (保存结果时附加上 EMBEDDING_MODE)
    def _test_and_save(self):
        print("Testing...")
        text_embeddings_bundle = {
            "video_emb_tensor_cpu": self.photo_embeddings,
            "video_id2idx": self.video_id2idx,
            "comment_emb_tensor_cpu": self.comment_embeddings,
            "comment_id2idx": self.comment_id2idx
        }
        # ... (模型加载和指标计算保持不变)
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
        df_result.loc[1] =  [gauc_val, mrr_val] + ndcg_ls + pcr_ls + wt_ls + [rmse, mae, xgauc, xauc]

        # 附加上 EMBEDDING_MODE
        result_filename = f'{self.fout}_result_{self.EMBEDDING_MODE}.csv'
        model_filename = f'{self.fout}_model_{self.EMBEDDING_MODE}.pt'
        
        df_result.to_csv(result_filename)
        torch.save(model.state_dict(), model_filename)
        print(f"Results saved to: {result_filename}")
        print(f"Model saved to: {model_filename}")

if __name__=="__main__":
    pass
