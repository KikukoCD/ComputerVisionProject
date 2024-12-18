# INFOCV Project: Optical Flow Action Recognition

## Setup
* Run `datasets.sh` to download the datasets
* Run `preprocess.ipynb` to preprocess the data
* Train the frames model using `train_frames1.py`

## Usage
### Notebooks
To train / explore the models we suggest you to use the following notebooks:

* [Training of Single Frame Models on Stanford-40 dataset](https://github.com/EgorDm/INFOMCV/blob/master/notebooks/INFOCV_ActionRecogniton_frames_stanford.ipynb)
* [Training of Single Frame Models on TV-HI dataset](https://github.com/EgorDm/INFOMCV/blob/master/notebooks/INFOCV_ActionRecogniton_frames_hitv.ipynb)
* [Training of Optical Flow Models on TV-HI dataset](https://github.com/EgorDm/INFOMCV/blob/master/notebooks/INFOCV_ActionRecogniton_of_and_fusion.ipynb)
* [Training of Two-Stream Model on TV-HI dataset](https://github.com/EgorDm/INFOMCV/blob/master/notebooks/INFOCV_ActionRecogniton_of_and_fusion.ipynb)

### Manual Usage
To download the datasets use 
```bash
bash datasets.sh
```

To preprocess and split the datasets
```bash
python preprocess.py --process_tvhi=True --process_stanford=True
```

To augment datasets use
```bash
# Stanford-40
python bake.py --bake_input=datasets/Stanford40/data --bake_output=datasets/Stanford40/baked --bake_optical_flow=False --augment_repeat_count=25 
# TV-HI with optical flow
python bake.py --augment_repeat_count=25 --bake_input=datasets/TV-HI/data --bake_output=datasets/TV-HI/baked --bake_optical_flow=True
```

To train the models simply use train_frames command. Note that you may need to change the config.py
```bash
python train_frames.py
```

Example commands:
```bash
# Evaluate stanford model
python train_frames.py \
  --run_name=frames_stanford --model_name=frames_stanford_v2 \
  --load_weights=snapshots/stanford_frames__frames_stanford_v2.h5 \
  --evaluate=True --train=False
  
# Evaluate TV-HI frames model
python train_frames.py \
  --run_name=frames_tvhi --model_name=frames_stanford_v2 \
  --frames_dataset=tvhi \
  --load_weights=snapshots/tvhi_frames_transfer__frames_stanford_v2.h5 \
  --evaluate=True --train=False
  
# Evaluate TV-HI optical flow model
python train_of.py \
  --run_name=of_tvhi --model_name=frames_stanford_v2 \
  --frames_dataset=tvhi \
  --load_weights=snapshots/tvhi_of__frames_stanford_v2.h5 \
  --evaluate=True --train=False

# Evaluate TV-HI two-stream model
python train_fusion.py \
  --run_name=fusion_tvhi --model_name=fusion_b67 \
  --fusion_frames_model_name=frames_stanford_half_v2 \
  --fusion_frames_weights=snapshots/tvhi_frames_transfer__frames_stanford_v2.h5 \
  --fusion_of_model_name=frames_stanford_half_v2 \
  --fusion_of_weights=snapshots/tvhi_of__frames_stanford_v2.h5 \
  --load_weights=snapshots/tvhi_two_stream__fusion_b67.h5 \
  --evaluate=True --train=False
```

## Development
Updating requirements: `python -m piptools compile requirements.in`