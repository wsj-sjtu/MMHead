import torch
from tqdm import tqdm
import numpy as np
from FLAME.decalib.utils.config import cfg as deca_cfg
from FLAME.decalib.models.FLAME import FLAME
from metric.metrics import MetricCalculator


flame = FLAME(deca_cfg.model).to('cuda:0')
metric = MetricCalculator('./metric')


def calculate_metrics_batch(
        audio, 
        text, 
        gt_motion, 
        motion_len,
        mask,
        pred_motion=None, 
        pred_verts=None,
        motion_annotation_list=[],    # used to store data for the entire test set to compute fid and diversity
        motion_pred_list=[]
    ):
    """
    audio: (bs, seq_len)
    text: tuple (bs,)
    gt_motion: (bs, 200, 56)
    pred_motion: (bs, 200, 56)
    motion_len: (bs, 1)
    mask: (bs, 200)  indicate valid frames
    pred_verts: (bs, 200, 15069)
    
    Note that:
    1. All motion sequences are padded to 200 frames, with motion_len and mask indicating the valid frames.
    2. We use bs=32 in our paper.
    """

    ## flame parameters to mesh vertices sequence
    # gt
    bs, seq_len = gt_motion.shape[0], gt_motion.shape[1]
    motion_tmp = gt_motion.contiguous().view(bs*seq_len, -1)
    full_poses = torch.cat([torch.zeros_like(motion_tmp[:, :3]), torch.zeros_like(motion_tmp[:, 50:53]), motion_tmp[:, 53:56], torch.zeros_like(motion_tmp[..., :6])], dim=1).cuda()
    shape = torch.zeros((motion_tmp.shape[0], 100))
    exps = motion_tmp[:, :50]
    verts_p, _, _ = flame(shape_params=shape.cuda(),
                                expression_params=exps.cuda(),
                                full_pose=full_poses.cuda())
    gt_verts = verts_p.contiguous().view(bs, seq_len, 5023*3)

    # pred
    if pred_verts is None:   # only methods that predict FLAME parameters require a forward pass through FLAME
        assert(pred_motion is not None)

        motion_tmp = pred_motion.contiguous().view(bs*seq_len, -1)
        full_poses = torch.cat([torch.zeros_like(motion_tmp[:, :3]), torch.zeros_like(motion_tmp[:, 50:53]), motion_tmp[:, 53:56], torch.zeros_like(motion_tmp[..., :6])], dim=1).cuda()
        shape = torch.zeros((motion_tmp.shape[0], 100))
        exps = motion_tmp[:, :50]
        verts_p, _, _ = flame(shape_params=shape.cuda(),
                                    expression_params=exps.cuda(),
                                    full_pose=full_poses.cuda())
        pred_verts = verts_p.contiguous().view(bs, seq_len, 5023*3) 


    ## calculate metrics
    # vertices error
    lve_batch = metric.calculate_batch_lve(pred_verts, gt_verts, motion_len)
    ve_batch = metric.calculate_batch_ve(pred_verts, gt_verts, motion_len)

    # inference fid net
    emb_dict = metric.prepare_embs_batch(pred_verts, gt_verts, gt_motion, mask, text, motion_annotation_list, motion_pred_list, audio)
    
    # R_precision & matching_score for text match
    R_precision_real_batch, matching_score_real_batch = metric.calculate_R_precision_and_matching_score_batch(emb_dict['gt']['et'], emb_dict['gt']['em'])   # metrics for real(gt) motion
    R_precision_batch, matching_score_batch = metric.calculate_R_precision_and_matching_score_batch(emb_dict['pred']['et'], emb_dict['pred']['em'])

    # R_precision & matching_score for audio match
    audio_R_precision_real_batch, audio_matching_score_real_batch = metric.calculate_R_precision_and_matching_score_batch(emb_dict['gt']['ea'], emb_dict['gt']['eam'])   # metrics for real(gt) motion
    audio_R_precision_batch, audio_matching_score_batch = metric.calculate_R_precision_and_matching_score_batch(emb_dict['pred']['ea'], emb_dict['pred']['eam'])

    metric_dict = {
        'lve_batch': lve_batch,
        've_batch': ve_batch,
        'R_precision_batch': R_precision_batch[None],
        'matching_score_batch': matching_score_batch,
        'audio_R_precision_batch': audio_R_precision_batch[None],
        'audio_matching_score_batch': audio_matching_score_batch,
    }
    return metric_dict



if __name__=="__main__":

    metric_list_dict = {
        'lve_batch': [],
        've_batch': [],
        'R_precision_batch': [],
        'matching_score_batch': [],
        'audio_R_precision_batch': [],
        'audio_matching_score_batch': []
    }
    motion_annotation_list = []
    motion_pred_list = []


    ## calculate metrics for each batch
    for audio, text, gt_motion, motion_len, mask, pred_motion in tqdm(test_data_loader):    #TODO: define your data loader
        metric_dict = calculate_metrics_batch(
            audio, 
            text, 
            gt_motion, 
            motion_len,
            mask,
            pred_motion=pred_motion, 
            motion_annotation_list=motion_annotation_list,
            motion_pred_list=motion_pred_list
        )

        for key in metric_list_dict.keys():
            metric_list_dict[key].append(metric_dict[key])


    ## calculate final metrics
    # average over the entire test set
    lve = np.mean(metric_list_dict['lve_batch'])
    ve = np.mean(metric_list_dict['ve_batch'])
    R_precision = np.concatenate(metric_list_dict['R_precision_batch'], axis=0)
    R_precision = np.mean(R_precision, axis=0)
    matching_score = np.mean(metric_list_dict['matching_score_batch'])
    audio_R_precision = np.concatenate(metric_list_dict['audio_R_precision_batch'], axis=0)
    audio_R_precision = np.mean(audio_R_precision, axis=0)
    audio_matching_score = np.mean(metric_list_dict['audio_matching_score_batch'])

    # diversity and fid need to be calculated over the entire test set
    diversity_real, diversity = metric.calculate_diversity(motion_annotation_list, motion_pred_list)
    fid = metric.calculate_fid(motion_annotation_list, motion_pred_list)

    ## print metrics
    log = f"Metrics--LVE:{lve:.6f}, FVE:{ve:.6f}, FID:{fid:.4f},"
    print(log)
    log = f"R_precision:{R_precision}, matching_score:{matching_score:.4f}, diversity:{diversity:.4f},"
    print(log)
    log = f"audio_R_precision:{audio_R_precision}, audio_matching_score:{audio_matching_score:.4f}"
    print(log)