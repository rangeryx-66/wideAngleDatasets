import json
import numpy as np
import cv2
import torch
import os
import h5py
import torch.nn.functional as F
import tqdm
import argparse

class overlaps_info:
    def __init__(
            self,
            data_root,
            image_paths,
            depth_paths,
            intrinsics,
            poses,
            pairs,
            ht=1920,
            wt=1080,
    ) -> None:
        self.data_root = data_root
        self.image_paths = image_paths
        self.depth_paths = depth_paths
        self.intrinsics = intrinsics
        self.poses = poses
        self.pairs = pairs
        self.ht = ht
        self.wt = wt
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def get_needed_info(self, idx1, idx2):
        self.K1 = torch.from_numpy(self.intrinsics[idx1]).unsqueeze(0).to(self.device)  # [1, 3, 3]
        self.K2 = torch.from_numpy(self.intrinsics[idx2]).unsqueeze(0).to(self.device)  # [1, 3, 3]
        T1 = np.expand_dims(self.poses[idx1], axis=0)  # [1, 4, 4]
        T2 = np.expand_dims(self.poses[idx2], axis=0)  # [1, 4, 4]
        self.T_1to2 = torch.tensor(np.matmul(T2, np.linalg.inv(T1)), dtype=torch.float)[:4, :4].to(self.device)
        self.depth1 = torch.from_numpy(np.array(h5py.File(os.path.join(self.data_root, self.depth_paths[idx1]))["depth"])).unsqueeze(
            0).to(self.device)
        self.depth2 = torch.from_numpy(np.array(h5py.File(os.path.join(self.data_root, self.depth_paths[idx2]))["depth"])).unsqueeze(
            0).to(self.device)

    def warp_kpts(self, kpts0, depth0, depth1, T_0to1, K0, K1):

        n, h, w = depth0.shape
        kpts0_depth = F.grid_sample(depth0[:, None], kpts0[:, :, None], mode="bilinear")[:, 0, :, 0]
        kpts0 = torch.stack((w * (kpts0[..., 0] + 1) / 2, h * (kpts0[..., 1] + 1) / 2), dim=-1)
        nonzero_mask = kpts0_depth != 0

        kpts0_h = torch.cat([kpts0, torch.ones_like(kpts0[:, :, [0]])], dim=-1) * kpts0_depth[..., None]
        kpts0_n = K0.inverse() @ kpts0_h.transpose(2, 1)
        kpts0_cam = kpts0_n

        w_kpts0_cam = T_0to1[:, :3, :3] @ kpts0_cam + T_0to1[:, :3, [3]]
        w_kpts0_depth_computed = w_kpts0_cam[:, 2, :]

        w_kpts0_h = (K1 @ w_kpts0_cam).transpose(2, 1)
        w_kpts0 = w_kpts0_h[:, :, :2] / (w_kpts0_h[:, :, [2]] + 1e-4)

        h, w = depth1.shape[1:3]
        covisible_mask = (w_kpts0[:, :, 0] > 0) * (w_kpts0[:, :, 0] < w - 1) * (w_kpts0[:, :, 1] > 0) * (w_kpts0[:, :, 1] < h - 1)
        w_kpts0 = torch.stack((2 * w_kpts0[..., 0] / w - 1, 2 * w_kpts0[..., 1] / h - 1), dim=-1)

        w_kpts0_depth = F.grid_sample(depth1[:, None], w_kpts0[:, :, None], mode="bilinear")[:, 0, :, 0]
        consistent_mask = ((w_kpts0_depth - w_kpts0_depth_computed) / w_kpts0_depth).abs() < 0.05
        valid_mask = nonzero_mask * covisible_mask * consistent_mask

        return valid_mask, w_kpts0

    def calculate_overlaps(self):
        b, h1, w1, d = 1, self.ht, self.wt, 2
        x1_n = torch.meshgrid(*[torch.linspace(-1 + 1 / n, 1 - 1 / n, n, device=self.device) for n in (b, h1, w1)])
        x1_n = torch.stack((x1_n[2], x1_n[1]), dim=-1).reshape(b, h1 * w1, 2)
        mask, x2 = self.warp_kpts(
            x1_n.double(),
            self.depth1.double(),
            self.depth2.double(),
            self.T_1to2.double(),
            self.K1.double(),
            self.K2.double(),
        )
        return mask.float().mean(dim=1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compute overlaps for a given scene")
    parser.add_argument("--scene_name", type=str, required=True, help="Name of the scene (e.g., tokyo)")
    parser.add_argument("--data_root", type=str, default="/root/autodl-tmp/wideAngleDatasets/", help="Root directory of the dataset")
    parser.add_argument("--output_file", type=str, default=None, help="Output file path (default: scene_name.npy)")
    args = parser.parse_args()


    scene_name = args.scene_name
    data_root = args.data_root
    output_file = args.output_file if args.output_file else os.path.join(data_root, f"{scene_name}.npy")


    camera_params_path = os.path.join(data_root, scene_name, "camera_params.json")
    with open(camera_params_path, 'r') as f:
        camera_params = json.load(f)


    camera_params.sort(key=lambda x: x['view_id'])


    for i, param in enumerate(camera_params):
        assert param['view_id'] == i, "view_id 必须连续且从 0 开始"


    intrinsics_list = []
    poses_list = []
    for param in camera_params:
        K = np.array(param['K'], dtype=np.float32)
        RT = np.array(param['RT'], dtype=np.float32)
        intrinsics_list.append(K)
        poses_list.append(RT)

    intrinsics = np.stack(intrinsics_list, axis=0)
    poses = np.stack(poses_list, axis=0)


    image_paths = np.array([f"{scene_name}/rgb/render_{i}.png" for i in range(len(camera_params))], dtype=np.str_)
    depth_paths = np.array([f"{scene_name}/depthh5/raw_depth_{i}_0001.h5" for i in range(len(camera_params))], dtype=np.str_)


    pairs = []
    for i, param in enumerate(camera_params):
        if 'parent_view_id' in param:
            parent_id = param['parent_view_id']
            if 0 <= parent_id < len(camera_params):
                pairs.append((i, parent_id))
    pairs = np.array(pairs, dtype=np.int32)


    overlaps = []
    getOverlaps = overlaps_info(data_root, image_paths, depth_paths, intrinsics, poses, pairs)
    for pair in tqdm.tqdm(pairs, desc="Processing pairs", ncols=100):
        idx1, idx2 = pair
        getOverlaps.get_needed_info(idx1, idx2)
        overlap = getOverlaps.calculate_overlaps()
        overlaps.append(overlap.to('cpu'))

    overlaps = np.array(overlaps, dtype=object)
    data = {
        'image_paths': image_paths,
        'depth_paths': depth_paths,
        'intrinsics': intrinsics,
        'poses': poses,
        'pairs': pairs,
        'overlaps': overlaps
    }


    parent_ids = np.unique(pairs[:, 1])
    np.random.seed(42)
    np.random.shuffle(parent_ids)

    split_idx = int(len(parent_ids) * 0.7)
    train_parents = parent_ids[:split_idx]
    test_parents = parent_ids[split_idx:]

    train_mask = np.isin(pairs[:, 1], train_parents)
    test_mask = np.isin(pairs[:, 1], test_parents)

    train_pairs = pairs[train_mask]
    test_pairs = pairs[test_mask]
    train_overlaps = overlaps[train_mask]
    test_overlaps = overlaps[test_mask]

    data.update({
        'train_pairs': train_pairs,
        'test_pairs': test_pairs,
        'train_overlaps': train_overlaps,
        'test_overlaps': test_overlaps
    })


    np.save(output_file, data)

    print(f"训练集配对数量: {len(train_pairs)}")
    print(f"测试集配对数量: {len(test_pairs)}")
    print(f"数据保存成功，包含键：{list(data.keys())}")
#python generate_npy.py --scene_name tokyo --data_root /path/to/datasets --output_file /path/to/output.npy