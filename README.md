# wideAngleDatasets

构造了一个大视角的图像匹配数据集，重点关注卫星视图和低空无人机视图的匹配问题

pipeline：

第一步：获取3d模型并载入到blender中

[wideAngleDatasets/3Dmodel/getmodel.md at main · rangeryx-66/wideAngleDatasets (github.com)](https://github.com/rangeryx-66/wideAngleDatasets/blob/main/3Dmodel/getmodel.md)

现有的一些模型：https://pan.baidu.com/s/18musguVBZevCI1lmpsDJNQ?pwd=cgpr 提取码: cgpr

现有的一些渲染结果：还在上传

第二步：blender对模型进行渲染

blender --background --python render.py -- --filepath /path/to/model.blend --savepath /output/path --height 200

第三步：对渲染结果进行处理得到数据集

先由exr的深度图文件转换成h5格式

python convert_exr_to_h5.py --input_dir /path/to/exr_files --output_dir /path/to/output_h5_files --channel R

再生成npy文件

python generate_npy.py --scene_name tokyo --data_root /path/to/datasets --output_file /path/to/output.npy

第四步：进行训练或者微调

[wideAngleDatasets/modelstore/dkm/experiments/dkm/train_DKMv3_outdoor.py at main · rangeryx-66/wideAngleDatasets (github.com)](https://github.com/rangeryx-66/wideAngleDatasets/blob/main/modelstore/dkm/experiments/dkm/train_DKMv3_outdoor.py)