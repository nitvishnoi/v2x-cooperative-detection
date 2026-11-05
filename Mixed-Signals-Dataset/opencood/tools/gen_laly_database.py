import argparse
from tqdm import tqdm
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.tools.inference_utils import inference_late_fusion


def main(path_ckpt_late_fusion: str):
    hypes_yaml = 'opencood/hypes_yaml/point_pillar_late_multi.yaml'
    params = load_yaml(hypes_yaml)
    params['fusion']['colab_setting'] = 'all'
    params['gen_laly_database'] = True
    dataset = build_dataset(params, visualize=False, train=False, gen_laly_database=True)
    print(f"INFO | dataset._seq_names: {dataset._seq_names}")    
    dataloader = DataLoader(dataset, 
                            batch_size=1, 
                            num_workers=8,
                            collate_fn=dataset.collate_batch_test,
                            shuffle=False, pin_memory=False)
    
    model = train_utils.create_model(params)
    train_utils.openpcdet_load_weights(model, path_ckpt_late_fusion, strict=True)
    model.eval().cuda()
    device = torch.device('cuda')

    # laly_database_directory = Path(params['root_dir']) / "laly_database"
    laly_database_directory = Path("/share/nikola/v2x_acfr") / "laly_database"
    laly_database_directory.mkdir(exist_ok=True)
    seq_id_prev = -1

    for i, batch_data in tqdm(enumerate(dataloader), total=len(dataset)):
        batch_data = train_utils.to_device(batch_data, device)
        
        seq_id = int(batch_data['metadata']['seq_name'])
        if seq_id != seq_id_prev:
            print(f"INFO | seq_id: {seq_id}")
            seq_id_prev = seq_id
        idx_in_seq = batch_data['metadata']['idx_in_seq']
        sample_token = f"seq{seq_id}_idx{idx_in_seq}"

        # =======================
        # late fusion inference without the NMS
        # =======================
        output_dict = inference_late_fusion(batch_data, model, dataset, return_output_dict=True)

        # ---------
        # postprocess for individual agent
        # ---------
        total_preds = list()

        for agent_name in output_dict.keys():
            agent_output = {agent_name: output_dict[agent_name]}
            agent_data = {agent_name: batch_data[agent_name]}

            pred_boxes_in_top, pred_scores, pred_class = dataset.post_processor.post_process(agent_data, agent_output)
            # NOTE: pred_boxes is in TOP frame

            if isinstance(pred_boxes_in_top, torch.Tensor):
                agent_indicator = pred_boxes_in_top.new_zeros(pred_boxes_in_top.shape[0], 1) + \
                    dataset.usyd.convert_agent_name_to_agent_indicator(agent_name)
                
                pred_boxes_in_top = torch.cat([
                    pred_boxes_in_top, 
                    pred_scores.unsqueeze(-1), 
                    pred_class.unsqueeze(-1), 
                    agent_indicator], 
                    dim=1)
                total_preds.append(pred_boxes_in_top)

        # save
        total_preds = torch.cat(total_preds).cpu().numpy().astype(np.float32)
        path_total_preds_for_sample = laly_database_directory / f"{sample_token}.npy"
        np.save(str(path_total_preds_for_sample), total_preds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='blah')
    parser.add_argument('--ckpt', type=str)
    args = parser.parse_args()

    main(args.ckpt)
