from tftrt.examples.object_detection import download_model

config_path, checkpoint_path = download_model('ssd_inception_v2_coco',
                                              output_dir='/home/nvidia/models/tensorflow')
