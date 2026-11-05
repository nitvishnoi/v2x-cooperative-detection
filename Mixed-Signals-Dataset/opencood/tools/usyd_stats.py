from pathlib import Path
from typing import List
import pickle
from tqdm import tqdm
from pprint import pprint
from copy import deepcopy
import pickle

import numpy as np

from opencood.utils.v2x_sim_utils import apply_se3


class USydStatistics(object):
    def __init__(self, root_dir: str, is_debug: bool = False):
        self.root_dir = Path(root_dir)

        path_gt_infos = Path(self.root_dir) / "infos_gt.pkl"
        with open(path_gt_infos, 'rb') as f:
            self.gt_infos = pickle.load(f)

        if not is_debug:
            set_test_sequences = set(('4', '5', '17', '21'))
            self.list_train_seqs = list(set([str(i) for i in range(1, 38)]) - set_test_sequences)
            self.list_train_seqs.sort()
            self.list_test_seqs = list(set_test_sequences)
            self.list_test_seqs.sort()
        else:
            self.list_train_seqs = ['1', '2']
            self.list_test_seqs = ['3', '4']

        self.dict_seqID_to_seqLen = dict()

        for sample_token in self.gt_infos.keys():
            str_seq_name, str_time_idx = sample_token.split('_')
            seq_name = str_seq_name[len('seq'):]
            if seq_name not in self.dict_seqID_to_seqLen:
                self.dict_seqID_to_seqLen[seq_name] = 1
            else:
                self.dict_seqID_to_seqLen[seq_name] += 1

        self._map_se3_top = np.array([
            [1.0, 0.0, 0.0, -41.551],
            [0.0, 1.0, 0.0, -51.878],
            [0.0, 0.0, 1.0, -1.077],
            [0.0, 0.0, 0.0, 1.0],
        ])
        self._top_se3_map = np.linalg.inv(self._map_se3_top)

    def make_sample_token(self, seq_name: str, time_idx: int):
        return f"seq{seq_name}_idx{time_idx}"
    
    def run(self, save_stats_dict: bool = True):
        dataset_stats = {
            'classes_by_splits': self.classes_by_splits(),
            'classes_by_range': self.classes_by_range(),
            'classes_heading': self.classes_heading(),  # to make a histogram
            'classes_size': self.classes_size(),  # to make a histogram
            'classes_distribution_polar_coord': self.classes_distribution_polar_coord(),  # to make a histogram
        }

        if save_stats_dict:
            print(f"save dataset stats to {self.root_dir / 'usyd_dataset_stats.pkl'}")
            with open(self.root_dir / 'usyd_dataset_stats.pkl', 'wb') as f:
                pickle.dump(dataset_stats, f)

        return dataset_stats

    def classes_by_splits(self):
        def _class_in_1split(is_train: bool):
            seqs_name = self.list_train_seqs if is_train else self.list_test_seqs
            num_cars, num_bikes, num_peds = 0, 0, 0
            for name_seq in tqdm(seqs_name, total=len(seqs_name), desc=f'_class_in_1split is_train {is_train}'):
                len_seq = self.dict_seqID_to_seqLen[name_seq]
                for time_idx in range(len_seq):
                    sample_token = self.make_sample_token(name_seq, time_idx)
                    info = self.gt_infos[sample_token]

                    gt_class = info['gt_class'].astype(int)

                    num_cars += np.sum(gt_class == 1)
                    num_bikes += np.sum(gt_class == 2)
                    num_peds += np.sum(gt_class == 3)
                # ---
            # ---
            return {'num_cars': num_cars, 'num_bikes': num_bikes, 'num_peds': num_peds}

        out = {
            'train': _class_in_1split(True),
            'test': _class_in_1split(False),
        }
        return out
    
    def classes_by_range(self):
        def _classes_by_range_in_seqs(list_seqs: List[str]):
            def make_dict_cls_range():
                return {
                    '0-25': 0,
                    '25-50': 0,
                    '50-inf': 0}
            dict_ranges = {
                'car': make_dict_cls_range(),
                'bike': make_dict_cls_range(),
                'ped': make_dict_cls_range(),
            }

            dict_nums = {
                'car': 0,
                'bike': 0,
                'ped': 0,
            }

            for name_seq in tqdm(list_seqs, desc=f'_classes_by_range_in_seqs, num_seq {len(list_seqs)}'):
                len_seq = self.dict_seqID_to_seqLen[name_seq]

                for time_idx in range(len_seq):
                    sample_token = self.make_sample_token(name_seq, time_idx)
                    info = self.gt_infos[sample_token]

                    gt_class = info['gt_class'].astype(int)
                    
                    gt_boxes_in_map = info['gt_boxes_in_map']
                    gt_boxes_in_top = apply_se3(self._top_se3_map, boxes_=gt_boxes_in_map)
                    
                    for cls_idx, cls_name in zip(range(1, 4), dict_ranges.keys()):
                        mask_cls = gt_class == cls_idx
                        if not np.any(mask_cls):
                            continue

                        dict_nums[cls_name] += np.sum(mask_cls)

                        gt_dist = np.linalg.norm(gt_boxes_in_top[mask_cls, :2], axis=1)
                        
                        mask_0_to_25 = gt_dist < 25.0
                        dict_ranges[cls_name]['0-25'] += np.sum(mask_0_to_25)
        
                        mask_50_to_inf = gt_dist > 50
                        dict_ranges[cls_name]['50-inf'] += np.sum(mask_50_to_inf)

                        mask_25_to_50 = np.logical_and(gt_dist >= 25, gt_dist <= 50)
                        dict_ranges[cls_name]['25-50'] += np.sum(mask_25_to_50)
                    # ---
                # ---
            # ---

            dict_ranges_relative = deepcopy(dict_ranges)
            total_num = dict_nums['car'] + dict_nums['bike'] + dict_nums['ped']
            for cls_name in dict_ranges.keys():
                num_cls = float(dict_nums[cls_name])
                for val_range, num_in_range in dict_ranges_relative[cls_name].items():
                    dict_ranges_relative[cls_name][val_range] = float(num_in_range) / num_cls

            return dict_ranges, dict_ranges_relative
        
        train_abs, train_rel = _classes_by_range_in_seqs(self.list_train_seqs)
        test_abs, test_rel = _classes_by_range_in_seqs(self.list_test_seqs)
        all_abs, all_rel = _classes_by_range_in_seqs(self.list_train_seqs + self.list_test_seqs)


        out = {
            'train_absolute': train_abs,
            'train_relative': train_rel,
            'test_absolute': test_abs,
            'test_relative': test_rel,
            'all_absolute': all_abs,
            'all_relative': all_rel,
        }
        return out
    
    def classes_heading(self):
        """To make a histogram"""
        def _classes_heading_in_seqs(list_seqs: List[str]):
            dict_heading = {
                'car': list(),
                'bike': list(),
                'ped': list()}
            for name_seq in tqdm(list_seqs, desc=f'_classes_heading_in_seqs, num_seq {len(list_seqs)}'):
                len_seq = self.dict_seqID_to_seqLen[name_seq]

                for time_idx in range(len_seq):
                    sample_token = self.make_sample_token(name_seq, time_idx)
                    info = self.gt_infos[sample_token]

                    gt_class = info['gt_class'].astype(int)
                    
                    gt_boxes_in_map = info['gt_boxes_in_map']
                    gt_boxes_in_top = apply_se3(self._top_se3_map, boxes_=gt_boxes_in_map)
                    gt_heading = np.arctan2(np.sin(gt_boxes_in_top[:, 6]), np.cos(gt_boxes_in_top[:, 6]))
                    
                    for cls_idx, cls_name in zip(range(1, 4), dict_heading.keys()):
                        mask_cls = gt_class == cls_idx
                        if not np.any(mask_cls):
                            continue
                        
                        dict_heading[cls_name] += gt_heading[mask_cls].tolist()
                    # ---
                # ---
            # ---
            return dict_heading
        
        out = {
            'train': _classes_heading_in_seqs(self.list_train_seqs),
            'test': _classes_heading_in_seqs(self.list_test_seqs),
            'all': _classes_heading_in_seqs(self.list_train_seqs + self.list_test_seqs),
        }
        return out

    def classes_size(self):
        """To make a histogram"""
        def _classes_size_in_seqs(list_seqs: List[str]):
            def make_dict_cls_size():
                return {
                    'dx': list(),
                    'dy': list(),
                    'dz': list()}
            
            dict_size = {
                'car': make_dict_cls_size(),
                'bike': make_dict_cls_size(),
                'ped': make_dict_cls_size()}
            for name_seq in tqdm(list_seqs, desc=f'_classes_size_in_seqs, num_seq {len(list_seqs)}'):
                len_seq = self.dict_seqID_to_seqLen[name_seq]

                for time_idx in range(len_seq):
                    sample_token = self.make_sample_token(name_seq, time_idx)
                    info = self.gt_infos[sample_token]

                    gt_class = info['gt_class'].astype(int)
                    
                    gt_boxes_in_map = info['gt_boxes_in_map']
                    gt_boxes_in_top = apply_se3(self._top_se3_map, boxes_=gt_boxes_in_map)
                    
                    for cls_idx, cls_name in zip(range(1, 4), dict_size.keys()):
                        mask_cls = gt_class == cls_idx
                        if not np.any(mask_cls):
                            continue
                        
                        dict_size[cls_name]['dx'] += gt_boxes_in_top[mask_cls, 3].tolist()
                        dict_size[cls_name]['dy'] += gt_boxes_in_top[mask_cls, 4].tolist()
                        dict_size[cls_name]['dz'] += gt_boxes_in_top[mask_cls, 5].tolist()
                    # ---
                # ---
            # ---
            return dict_size
        
        out = {
            'train': _classes_size_in_seqs(self.list_train_seqs),
            'test': _classes_size_in_seqs(self.list_test_seqs),
            'all': _classes_size_in_seqs(self.list_train_seqs + self.list_test_seqs),
        }
        return out

    def classes_distribution_polar_coord(self):
        def _classes_distrib_polar_coord(list_seqs: List[str]):
            def make_dict_cls_polar():
                return {
                    'r': list(),
                    'theta': list()}
                        
            dict_polar = {
                'car': make_dict_cls_polar(),
                'bike': make_dict_cls_polar(),
                'ped': make_dict_cls_polar()}

            for name_seq in tqdm(list_seqs, desc=f'_classes_distrib_polar_coord, num_seq {len(list_seqs)}'):
                len_seq = self.dict_seqID_to_seqLen[name_seq]

                for time_idx in range(len_seq):
                    sample_token = self.make_sample_token(name_seq, time_idx)
                    info = self.gt_infos[sample_token]

                    gt_class = info['gt_class'].astype(int)
                    
                    gt_boxes_in_map = info['gt_boxes_in_map']
                    gt_boxes_in_top = apply_se3(self._top_se3_map, boxes_=gt_boxes_in_map)
                    
                    gt_heading = np.arctan2(np.sin(gt_boxes_in_top[:, 6]), np.cos(gt_boxes_in_top[:, 6]))
                    gt_dist = np.linalg.norm(gt_boxes_in_top[:, :2], axis=1)

                    for cls_idx, cls_name in zip(range(1, 4), dict_polar.keys()):
                        mask_cls = gt_class == cls_idx
                        if not np.any(mask_cls):
                            continue
                        
                        dict_polar[cls_name]['r'] += gt_dist[mask_cls].tolist()
                        dict_polar[cls_name]['theta'] += gt_heading[mask_cls].tolist()
                    # ---
                # ---
            # ---
            return dict_polar
        
        out = {
            'train': _classes_distrib_polar_coord(self.list_train_seqs),
            'test': _classes_distrib_polar_coord(self.list_test_seqs),
            'all': _classes_distrib_polar_coord(self.list_train_seqs + self.list_test_seqs),
        }
        return out

        

if __name__ == '__main__':
    usyd_stats_calculator = USydStatistics('data/usyd-v2x', is_debug=False)
    stats = usyd_stats_calculator.run(save_stats_dict=True)

    # # ==================
    # # display
    # stats.pop('classes_heading')
    # stats.pop('classes_size')
    # stats.pop('classes_distribution_polar_coord')
    # pprint(stats)
