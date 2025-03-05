import json
import numpy as np
import cv2
import torch
import os
import h5py
import torch.nn.functional as F
import tqdm

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
        self.depth_paths=depth_paths
        self.intrinsics = intrinsics
        self.poses = poses
        self.pairs = pairs
        self.ht=ht
        self.wt=wt
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    def get_needed_info(self,idx1,idx2):
        self.K1 = torch.from_numpy(self.intrinsics[idx1]).unsqueeze(0).to(self.device) # [1, 3, 3]
        self.K2 = torch.from_numpy(self.intrinsics[idx2]).unsqueeze(0).to(self.device)  # [1, 3, 3]
        T1 = np.expand_dims(self.poses[idx1], axis=0)  # [1, 4, 4]
        T2 = np.expand_dims(self.poses[idx2], axis=0)  # [1, 4, 4]
        self.T_1to2 = torch.tensor(np.matmul(T2, np.linalg.inv(T1)), dtype=torch.float)[:4, :4].to(self.device)
        self.depth1 = torch.from_numpy(np.array(h5py.File(os.path.join(self.data_root,self.depth_paths[idx1]))["depth"])).unsqueeze(
            0).to(self.device)
        self.depth2 = torch.from_numpy(np.array(h5py.File(os.path.join(self.data_root,self.depth_paths[idx2]))["depth"])).unsqueeze(
            0).to(self.device)

    def warp_kpts(self,kpts0, depth0, depth1, T_0to1, K0, K1):
        """Warp kpts0 from I0 to I1 with depth, K and Rt
        Also check covisibility and depth consistency.
        Depth is consistent if relative error < 0.2 (hard-coded).
        # https://github.com/zju3dv/LoFTR/blob/94e98b695be18acb43d5d3250f52226a8e36f839/src/loftr/utils/geometry.py adapted from here
        Args:
            kpts0 (torch.Tensor): [N, L, 2] - <x, y>, should be normalized in (-1,1)
            depth0 (torch.Tensor): [N, H, W],
            depth1 (torch.Tensor): [N, H, W],
            T_0to1 (torch.Tensor): [N, 3, 4],
            K0 (torch.Tensor): [N, 3, 3],
            K1 (torch.Tensor): [N, 3, 3],
        Returns:
            calculable_mask (torch.Tensor): [N, L]
            warped_keypoints0 (torch.Tensor): [N, L, 2] <x0_hat, y1_hat>
        """
        (
            n,
            h,
            w,
        ) = depth0.shape
        kpts0_depth = F.grid_sample(depth0[:, None], kpts0[:, :, None], mode="bilinear")[
                      :, 0, :, 0
                      ]
        kpts0 = torch.stack(
            (w * (kpts0[..., 0] + 1) / 2, h * (kpts0[..., 1] + 1) / 2), dim=-1
        )  # [-1+1/h, 1-1/h] -> [0.5, h-0.5]
        # Sample depth, get calculable_mask on depth != 0
        nonzero_mask = kpts0_depth != 0

        # Unproject
        kpts0_h = (
                torch.cat([kpts0, torch.ones_like(kpts0[:, :, [0]])], dim=-1)
                * kpts0_depth[..., None]
        )  # (N, L, 3)
        kpts0_n = K0.inverse() @ kpts0_h.transpose(2, 1)  # (N, 3, L)
        kpts0_cam = kpts0_n

        # Rigid Transform
        w_kpts0_cam = T_0to1[:, :3, :3] @ kpts0_cam + T_0to1[:, :3, [3]]  # (N, 3, L)
        w_kpts0_depth_computed = w_kpts0_cam[:, 2, :]

        # Project
        w_kpts0_h = (K1 @ w_kpts0_cam).transpose(2, 1)  # (N, L, 3)
        w_kpts0 = w_kpts0_h[:, :, :2] / (
                w_kpts0_h[:, :, [2]] + 1e-4
        )  # (N, L, 2), +1e-4 to avoid zero depth

        # Covisible Check
        h, w = depth1.shape[1:3]
        covisible_mask = (
                (w_kpts0[:, :, 0] > 0)
                * (w_kpts0[:, :, 0] < w - 1)
                * (w_kpts0[:, :, 1] > 0)
                * (w_kpts0[:, :, 1] < h - 1)
        )
        w_kpts0 = torch.stack(
            (2 * w_kpts0[..., 0] / w - 1, 2 * w_kpts0[..., 1] / h - 1), dim=-1
        )  # from [0.5,h-0.5] -> [-1+1/h, 1-1/h]
        # w_kpts0[~covisible_mask, :] = -5 # xd

        w_kpts0_depth = F.grid_sample(
            depth1[:, None], w_kpts0[:, :, None], mode="bilinear"
        )[:, 0, :, 0]
        consistent_mask = (
                                  (w_kpts0_depth - w_kpts0_depth_computed) / w_kpts0_depth
                          ).abs() < 0.05
        valid_mask = nonzero_mask * covisible_mask * consistent_mask

        return valid_mask, w_kpts0


    def calculate_overlaps(self):
        b, h1, w1, d = 1,self.ht,self.wt,2
        x1_n = torch.meshgrid(
            *[
                torch.linspace(
                    -1 + 1 / n, 1 - 1 / n, n, device=self.device
                )
                for n in (b, h1, w1)
            ]
        )
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





# 加载JSON参数文件
with open('/root/autodl-tmp/wideAngleDatasets/tokyo/camera_params.json', 'r') as f:
    camera_params = json.load(f)

# 按view_id排序确保顺序
camera_params.sort(key=lambda x: x['view_id'])

# 验证view_id连续性
for i, param in enumerate(camera_params):
    assert param['view_id'] == i, "view_id必须连续且从0开始"

# 提取相机参数
intrinsics_list = []
poses_list = []

for param in camera_params:
    K = np.array(param['K'], dtype=np.float32)
    RT = np.array(param['RT'], dtype=np.float32)
    intrinsics_list.append(K)
    poses_list.append(RT)

# 将列表转换为数组，确保 dtype 为 float32
intrinsics = np.stack(intrinsics_list, axis=0)
poses = np.stack(poses_list, axis=0)

# 验证形状和 dtype
print("Intrinsics shape:", intrinsics.shape)
print("Intrinsics dtype:", intrinsics.dtype)
print("Supose shape:", poses.shape)
print("Supose dtype:", poses.dtype)

# 生成图像路径和配对
image_paths = np.array([f"tokyo/rgb/render_{i}.png" for i in range(len(camera_params))], dtype=np.str_)
depth_paths = np.array([f"tokyo/depthh5/raw_depth_{i}_0001.h5" for i in range(len(camera_params))], dtype=np.str_)

# 生成图像对
pairs = []
for i, param in enumerate(camera_params):
    if 'parent_view_id' in param:
        parent_id = param['parent_view_id']
        if 0 <= parent_id < len(camera_params):
            pairs.append((i, parent_id))
pairs = np.array(pairs, dtype=np.int32)
overlaps=[]
getOverlaps=overlaps_info("/root/autodl-tmp/wideAngleDatasets/",image_paths,depth_paths,intrinsics,poses,pairs)
for pair in tqdm.tqdm(pairs, desc="Processing pairs", ncols=100):
    idx1, idx2 = pair
    getOverlaps.get_needed_info(idx1, idx2)
    overlap = getOverlaps.calculate_overlaps()
    overlaps.append(overlap.to('cpu'))
# 创建结构化数据字典
overlaps = np.array(overlaps, dtype=object)
data = {
    'image_paths': image_paths,
    'depth_paths': depth_paths,
    'intrinsics': intrinsics,
    'poses': poses,
    'pairs': pairs,
    'overlaps': overlaps
}


# 划分训练集和测试集
parent_ids = np.unique(pairs[:, 1])  # 提取所有唯一的卫星图ID
np.random.seed(42)  # 设置随机种子以保证可重复性
np.random.shuffle(parent_ids)  # 打乱顺序

split_idx = int(len(parent_ids) * 0.7)  # 70%训练，30%测试
train_parents = parent_ids[:split_idx]
test_parents = parent_ids[split_idx:]

# 创建训练和测试的掩码
train_mask = np.isin(pairs[:, 1], train_parents)
test_mask = np.isin(pairs[:, 1], test_parents)

# 划分pairs和overlaps
train_pairs = pairs[train_mask]
test_pairs = pairs[test_mask]
train_overlaps = overlaps[train_mask]
test_overlaps = overlaps[test_mask]

# 更新数据字典
data.update({
    'train_pairs': train_pairs,
    'test_pairs': test_pairs,
    'train_overlaps': train_overlaps,
    'test_overlaps': test_overlaps
})

# 保存为.npy文件
np.save('/root/autodl-tmp/tokyo.npy', data)

print("训练集配对数量:", len(train_pairs))
print("测试集配对数量:", len(test_pairs))
print("数据保存成功，包含键：", list(data.keys()))