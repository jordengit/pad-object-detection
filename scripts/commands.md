# Execute these commands to get started


## Creating model/validation records

For the orb data:

```
python scripts/create_pascal_tf_record_generic.py \
--data_dir=images \
--annotations_dir=annotations \
--output_path=training/inputs/orb_data \
--label_map_path=training/inputs/orb_data/orb_label_map.pbtxt
```

For the full label set:

```
python scripts/create_pascal_tf_record_generic.py \
--data_dir=images \
--annotations_dir=annotations \
--output_path=training/inputs/pad_data \
--label_map_path=training/inputs/pad_data/pad_label_map.pbtxt
```

## Compiling the protos in research/object_detection

If you're on Windows, globs don't work, try this command:

```
# from tensorflow/models/research/
for %P in (object_detection\protos\*.proto) do "C:\tmp\protoc-3.5.1-win32\bin\protoc.exe" %P --python_out=.
```

On Linux, follow the command specified in the object_detection docs:

```
# From tensorflow/models/research/
protoc object_detection/protos/*.proto --python_out=.
```

## Training

This step takes at least a couple of hours on my PC to get good results. If
you're impatient, you can kill the job after an hour, run evaluation to
check the results (or if your PC is beefy enough, run evaluation concurrently).

SSD Inception:

```
TF_RESEARCH=/path/to/tensorflow/models/research

python $TF_RESEARCH/object_detection/train.py \
--logtostderr \
--pipeline_config_path=model_configs/configured/ssd_inception_v2_coco.config \
--train_dir=training/checkpoints/ssd_inception
```

Faster RCNN:

```
TF_RESEARCH=/path/to/tensorflow/models/research

python $TF_RESEARCH/object_detection/train.py \
--logtostderr \
--pipeline_config_path=model_configs/configured/faster_rcnn_resnet101_coco.config \
--train_dir=training/checkpoints/faster_rcnn
```

## Evaluation

```
TF_RESEARCH=/path/to/tensorflow/models/research

python $TF_RESEARCH/object_detection/eval.py \
--logtostderr \
--pipeline_config_path=model_configs/configured/ssd_inception_v2_coco.config \
--checkpoint_dir=training/checkpoints/ssd_inception \
--eval_dir=training/checkpoints/ssd_inception


# In another window
tensorboard --logdir=training/checkpoints/ssd_inception
```

## Exporting the frozen graph

Replace '17822' in the command below with the latest checkpoint you
generated.

```
TF_RESEARCH=/path/to/tensorflow/models/research

python $TF_RESEARCH/object_detection/export_inference_graph.py \
--input_type image_tensor \
--pipeline_config_path=model_configs/configured/ssd_inception_v2_coco.config \
--train_dir=training/checkpoints/ssd_inception/model.ckpt-17822 \
--output_directory=training/exported/ssd_inception
```

## Running the resulting graph on some images
