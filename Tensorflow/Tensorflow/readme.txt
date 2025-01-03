Step 1: prepare dataset
Step 2: put them to images folder
Step 3: python xml_to_csv.py
Step 4: Generate tfrecord:
# python generate_tfrecord.py --csv_input=images/test_labels.csv --image_dir=images/test --output_path=test.record
# python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record
Step 5: Download a pre-trained model and modify config
change: num_classes, batch_size, num_steps, fine_tune_checkpoint (not include suffix), label_map_path, input_path
Step 6: Start training
# python model_main_tf2.py --pipeline_config_path=ssd_efficientdet_d0_512x512_coco17_tpu-8.config  --model_dir=training --alsologtostderr
Step 7: Export graph
python exporter_main_v2.py --trained_checkpoint_dir=training  --pipeline_config_path=ssd_efficientdet_d0_512x512_coco17_tpu-8.config --output_directory inference_graph


Environment Setting:
    cd models/research
    python use_protobuf.py object_detection/protos protoc
    cp object_detection/packages/tf2/setup.py .
    python -m pip install .

Test installation run:
    python object_detection/builders/model_builder_tf2_test.py

    cd models/research/object_detection
    python gpu.py

每一次訓練前需要設置(因爲路徑有空格):
    set XLA_FLAGS=--xla_gpu_cuda_data_dir="C:/Program Files/NVIDIA_GPU_Computing_Toolkit/CUDA/v11.8"
    set TF_XLA_FLAGS=--tf_xla_enable_xla_devices=false
    set CUDA_DIR=C:/Program Files/NVIDIA_GPU_Computing_Toolkit/CUDA/v11.8
    切換到cmd，而不是terminal
    
做過的change:
jit_compile=True -> jit_compile=False