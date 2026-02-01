import os
os.environ["TOKENIZERS_PARALLELISM"] = "False"
import sys
current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_directory) 

import os
import torch
import numpy as np
import pandas
from scipy import linalg

import metric.models.retrieval_models.actor as Actor
import metric.models.retrieval_models.tmr as TMR

import metric.models.retrieval_models.actor_withheadpose as Actor_withheadpose
import metric.models.retrieval_models.tmr_withheadpose as TMR_withheadpose

from transformers import AutoTokenizer, AutoModel
from metric.models.word_vectorizer import POS_enumerator
 
from metric.models.wav2vec_model.wav2vec import Wav2Vec2Model

from metric.config.get_eval_option import get_opt



class MetricCalculator():
    def __init__(self, metric_root):
        
        # flame mask
        flame_mask = pandas.read_pickle(os.path.join(metric_root, 'data', 'FLAME_masks.pkl'))
        self.flame_mask = {'face': flame_mask['face'], 'lip': flame_mask['lips']}

        # mean&std
        self.dataset_mean = torch.Tensor(np.load(os.path.join(metric_root, 'data', 'train_test_val_mesh_mean.npy'))).cuda()
        self.dataset_std = torch.Tensor(np.load(os.path.join(metric_root, 'data', 'train_test_val_mesh_std.npy'))).cuda()

        # fid net
        dataset_opt_path = os.path.join(metric_root, 'config', 'vq_config.txt')
        wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
        self.eval_wrapper = EvaluatorModelWrapper(wrapper_opt, metric_root)


    def calculate_batch_lve(self, pred, gt, valid_len):
        '''
        lip vertex error
        pred, gt: (bs, max_len, 15069)
        valid_len: (bs,)
        '''
        bs, seq_len = pred.shape[0], pred.shape[1]

        # extract lip vertices
        pred_lip = pred.contiguous().view(bs, seq_len, 5023, 3)[:, :, self.flame_mask['lip'], :]
        gt_lip = gt.contiguous().view(bs, seq_len, 5023, 3)[:, :, self.flame_mask['lip'], :]

        lips_avg_dist = []
        for i in range(bs):
            l2_dist = torch.norm(pred_lip[i, :valid_len[i], :, :] - gt_lip[i, :valid_len[i], :, :], p=2, dim=-1, keepdim=True)
            l2_dist, _ = torch.max(l2_dist, dim = 1)
            l2_dist = torch.mean(l2_dist)           
            lips_avg_dist.append(l2_dist.cpu().numpy())
        
        lve = np.mean(lips_avg_dist)   # average over a batch
        return lve
    

    def calculate_batch_ve(self, pred, gt, valid_len):
        '''
        face vertex error
        pred, gt: (bs, max_len, 15069)
        valid_len: (bs,)
        '''
        bs, seq_len = pred.shape[0], pred.shape[1]

        pred_face = pred.contiguous().view(bs, seq_len, 5023, 3)[:, :, self.flame_mask['face'], :]
        gt_face = gt.contiguous().view(bs, seq_len, 5023, 3)[:, :, self.flame_mask['face'], :]

        avg_dist = []
        for i in range(bs):
            l2_dist = torch.norm(pred_face[i, :valid_len[i], :, :] - gt_face[i, :valid_len[i], :, :], p=2, dim=-1, keepdim=True)
            l2_dist = torch.mean(l2_dist)           
            avg_dist.append(l2_dist.cpu().numpy())
        
        ve = np.mean(avg_dist)
        return ve


    # (X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train
    def __euclidean_distance_matrix(self, matrix1, matrix2):
        """
            Params:
            -- matrix1: N1 x D
            -- matrix2: N2 x D
            Returns:
            -- dist: N1 x N2
            dist[i, j] == distance(matrix1[i], matrix2[j])
        """
        assert matrix1.shape[1] == matrix2.shape[1]
        d1 = -2 * np.dot(matrix1, matrix2.T)    # shape (num_test, num_train)
        d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    # shape (num_test, 1)
        d3 = np.sum(np.square(matrix2), axis=1)     # shape (num_train, )
        dists = np.sqrt(d1 + d2 + d3)  # broadcasting
        return dists

    def __calculate_top_k(self, mat, top_k):
        size = mat.shape[0]
        gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
        bool_mat = (mat == gt_mat)
        correct_vec = False
        top_k_list = []
        for i in range(top_k):
            correct_vec = (correct_vec | bool_mat[:, i])
            top_k_list.append(correct_vec[:, None])
        top_k_mat = np.concatenate(top_k_list, axis=1)
        return top_k_mat

    def __calculate_R_precision(self, embedding1, embedding2, top_k, sum_all=False):
        dist_mat = self.__euclidean_distance_matrix(embedding1, embedding2)
        matching_score = dist_mat.trace()
        argmax = np.argsort(dist_mat, axis=1)
        top_k_mat = self.__calculate_top_k(argmax, top_k)
        if sum_all:
            return top_k_mat.sum(axis=0), matching_score
        else:
            return top_k_mat, matching_score    
        

    def calculate_R_precision_and_matching_score_batch(self, embedding1, embedding2, top_k=3, sum_all=True):
        """
        calculate average R_precision and matching_score for a batch
        embedding1/2: tensor
        """
        bs = embedding1.shape[0]
        R_precision, matching_score = self.__calculate_R_precision(embedding1.cpu().numpy(), embedding2.cpu().numpy(), top_k=top_k, sum_all=sum_all)

        return R_precision/bs, matching_score/bs



    def __calculate_diversity(self, activation, diversity_times):
        assert len(activation.shape) == 2
        assert activation.shape[0] > diversity_times
        num_samples = activation.shape[0]

        first_indices = np.random.choice(num_samples, diversity_times, replace=False)
        second_indices = np.random.choice(num_samples, diversity_times, replace=False)
        dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
        return dist.mean()


    def __calculate_activation_statistics(self, activations):

        mu = np.mean(activations, axis=0)
        cov = np.cov(activations, rowvar=False)
        return mu, cov


    def __calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1)
                + np.trace(sigma2) - 2 * tr_covmean)
    

    def prepare_embs_batch(self, pred, gt, gt_param, mask, 
                           text, motion_annotation_list, motion_pred_list,
                           audio, have_headpose=False, pred_param=None):

        if have_headpose:
            assert pred_param is not None

        bs, seq_len = pred.shape[0], pred.shape[1]

        # normalize
        pred = (pred - self.dataset_mean) /  self.dataset_std
        gt = (gt - self.dataset_mean) / self.dataset_std

        # face vertices
        pred_face = pred.contiguous().view(bs, seq_len, 5023, 3)[:, :, self.flame_mask['face'], :].contiguous().view(bs, seq_len, -1)
        gt_face = gt.contiguous().view(bs, seq_len, 5023, 3)[:, :, self.flame_mask['face'], :].contiguous().view(bs, seq_len, -1)
        
        ## for text fid net
        # head pose
        pred_headpose = pred_param[..., 50:53] if have_headpose else torch.zeros_like(gt_param[..., 0:3])
        gt_headpose = gt_param[..., 50:53]

        et_pred, em_pred = self.eval_wrapper.get_co_embeddings(text, pred_face, pred_headpose, mask)
        et, em = self.eval_wrapper.get_co_embeddings(text, gt_face, gt_headpose, mask)

        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)


        ## for audio fid net
        seq_len_crop = 75
        audio_len_crop = 48000   # 75/25*16000
        pred_face_crop = torch.zeros_like(pred_face[:, :seq_len_crop, :])
        gt_face_crop =  torch.zeros_like(gt_face[:, :seq_len_crop, :])
        audio_crop = torch.zeros_like(audio[:, :audio_len_crop])
        pred_face_crop = pred_face[:, :seq_len_crop, :]
        gt_face_crop = gt_face[:, :seq_len_crop, :]
        audio_crop = audio[:, :audio_len_crop]

        ea_pred, eam_pred = self.eval_wrapper.get_audio_sync_embedding(pred_face_crop, audio_crop, mask[:,:seq_len_crop])
        ea_gt, eam_gt = self.eval_wrapper.get_audio_sync_embedding(gt_face_crop, audio_crop, mask[:,:seq_len_crop])

        result_dict = {}
        result_dict['gt'] = {'et': et, 'em': em, 'ea': ea_gt, 'eam': eam_gt}
        result_dict['pred'] = {'et': et_pred, 'em': em_pred, 'ea': ea_pred, 'eam': eam_pred}

        return result_dict

        


    def calculate_fid(self, motion_annotation_list, motion_pred_list):
        """
        motion_annotation_list, motion_pred_list contain the results for the entire test set
        """
        motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
        motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
        gt_mu, gt_cov  = self.__calculate_activation_statistics(motion_annotation_np)
        mu, cov= self.__calculate_activation_statistics(motion_pred_np)

        fid = self.__calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
        return fid
    

    def calculate_diversity(self, motion_annotation_list, motion_pred_list):
        """
        motion_annotation_list, motion_pred_list contain the results of the entire test set
        """
        nsample = len(motion_annotation_list)
        motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
        motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()

        diversity_real = self.__calculate_diversity(motion_annotation_np, 300 if nsample > 300 else 100)
        diversity = self.__calculate_diversity(motion_pred_np, 300 if nsample > 300 else 100)
        
        return diversity_real, diversity





######################################################################################################################
######################################################################################################################


def build_models(opt, metric_root):
    
    motion_encoder = Actor_withheadpose.ACTORStyleEncoder(nfeats=1787*3, vae=True, latent_dim=256, ff_size=1024, \
    num_layers=6, num_heads=4, dropout=0.1, activation='gelu')  #56

    text_encoder = Actor_withheadpose.ACTORStyleEncoder_text(nfeats=768, vae=True, latent_dim=256, ff_size=1024, \
        num_layers=6, num_heads=4, dropout=0.1, activation='gelu')

    motion_decoder = Actor_withheadpose.ACTORStyleDecoder(nfeats=1787*3, latent_dim=256, ff_size=1024, num_layers=6,\
        num_heads=4, dropout=0.1, activation='gelu')


    tmr_vae = TMR.tmr(motion_encoder=motion_encoder, text_encoder=text_encoder, motion_decoder=motion_decoder).cuda()


    

    ### loading T5 models
    t5_tokenizer = AutoTokenizer.from_pretrained(os.path.join(metric_root, 'checkpoints', 'distilbert-base-uncased'))
    t5_model = AutoModel.from_pretrained(os.path.join(metric_root, 'checkpoints', 'distilbert-base-uncased')).cuda()
    t5_model.eval()
    for p in t5_model.parameters():
        p.requires_grad = False

    
    tmr_checkpoint = torch.load(os.path.join(metric_root, 'checkpoints', 'tmr_text', 'tmr_vae_70.pth'), map_location=opt.device)


    # print("all checkpoints keys are", tmr_checkpoint.keys())
    
    # tmr_vae.load_state
    
    tmr_vae.load_state_dict(tmr_checkpoint['vae'])
    # print('Loading Evaluation Model Wrapper (Epoch %d) Completed!!' % (checkpoint['epoch']))
    return tmr_vae, t5_tokenizer, t5_model



def build_audio_matching_models(opt, metric_root):
    
    
    audio_model = Wav2Vec2Model.from_pretrained(os.path.join(metric_root, 'checkpoints', 'wav2vec2-base-960h')).cuda()
    audio_model.eval()
    for p in audio_model.parameters():
        p.requires_grad = False
    
    motion_encoder = Actor.ACTORStyleEncoder(nfeats=1787*3, vae=True, latent_dim=256, ff_size=1024, \
        num_layers=6, num_heads=4, dropout=0.1, activation='gelu')  # 56

    audio_encoder = Actor.ACTORStyleEncoder(nfeats=768, vae=True, latent_dim=256, ff_size=1024, \
        num_layers=6, num_heads=4, dropout=0.1, activation='gelu')

    motion_decoder = Actor.ACTORStyleDecoder(nfeats=1787*3, latent_dim=256, ff_size=1024, num_layers=6,\
        num_heads=4, dropout=0.1, activation='gelu')

    tmr_vae = TMR.tmr(motion_encoder=motion_encoder, text_encoder=audio_encoder, motion_decoder=motion_decoder).cuda()

 
    tmr_checkpoint = torch.load(os.path.join(metric_root, 'checkpoints', 'tmr_audio', 'tmr_vae_180.pth'))
    
    tmr_vae.load_state_dict(tmr_checkpoint['vae'])


    return tmr_vae, audio_model




class EvaluatorModelWrapper(object):

    def __init__(self, opt, metric_root):


        opt.dim_pose = 184

        opt.dim_word = 300
        opt.max_motion_length = 196
        opt.dim_pos_ohot = len(POS_enumerator)
        opt.dim_motion_hidden = 1024
        opt.max_text_len = 20
        opt.dim_text_hidden = 512
        opt.dim_coemb_hidden = 512

        # print(opt)

        self.text_tmr, self.t5_tokenizer, self.t5_model = build_models(opt, metric_root)
        self.opt = opt
        self.device = opt.device
        
        self.text_tmr.to(opt.device)
        self.text_tmr.eval()


        ###
        self.audio_tmr, self.audio_encoder = build_audio_matching_models(opt, metric_root)
        self.audio_tmr.to(opt.device)
        self.audio_tmr.eval()





    # Please note that the results does not following the order of inputs
    def get_co_embeddings(self, text_input, motions, headpose, m_mask):
        with torch.no_grad():
            input_ids = self.t5_tokenizer(list(text_input), return_tensors="pt", padding=True, max_length=300, truncation=True) # 120感觉有点短, max_length=200, 
            text_feat = self.t5_model(**input_ids.to("cuda:0")).last_hidden_state
            
            text_mask = input_ids['attention_mask'].bool()
        
        
            # input_ids = self.t5_tokenizer(list(text_input), return_tensors="pt", padding=True, max_length=300, truncation=True).input_ids.cuda() # truncation=True,
            # text_feat = self.t5_model(input_ids).last_hidden_state
            text_feat = text_feat.detach().to(self.device).float()
            # print("when getting", text_feat.shape)
            motions = motions.detach().to(self.device).float()

            '''Movement Encoding'''
            motion_embedding = self.text_tmr.encode_motion({'x': motions, 'headpose': headpose, 'mask': m_mask}, sample_mean = True)   #self.motion_encoder(motions)

            '''Text Encoding'''
            text_embedding = self.text_tmr.encode_text({'x': text_feat, 'mask': text_mask}, sample_mean = True)
            # print("text embe shape", text_embedding.shape)
        return text_embedding, motion_embedding

    # Please note that the results does not following the order of inputs
    def get_motion_embeddings(self, motions, m_lens):
        with torch.no_grad():
            motions = motions.detach().to(self.device).float()

            motion_embedding = self.motion_encoder(motions)
        return motion_embedding


    def get_audio_sync_embedding(self, motions, audio, m_mask):
        with torch.no_grad():
            m_mask[...] = 1
            m_mask = m_mask.bool()
            motions = motions.detach().to(self.device).float()
            motion_embedding = self.audio_tmr.encode_motion({'x': motions, 'mask': m_mask}, sample_mean = True)  
            # motion_embedding = self.audmot_encoder(motions)

            audio = audio.detach().to(self.device).float()
            audio_feat = self.audio_encoder(audio, "vocaset", frame_num=75).last_hidden_state.cuda() # 600
            audio_mask = (torch.arange(audio_feat.shape[1]) < 10000).unsqueeze(0).repeat(audio_feat.shape[0], 1).cuda()
            
            
            
            audio_embedding = self.audio_tmr.encode_text( {'x':audio_feat, 'mask':audio_mask},  sample_mean = True)
        
        return motion_embedding, audio_embedding
