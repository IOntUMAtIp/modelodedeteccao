def treinar():

    import numpy as np
    import os

    from tflite_model_maker.config import ExportFormat
    from tflite_model_maker import model_spec
    from tflite_model_maker import object_detector

    import tensorflow as tf
    assert tf.__version__.startswith('2')

    tf.get_logger().setLevel('ERROR')
    from absl import logging
    logging.set_verbosity(logging.ERROR)

    label_type = "coco"

    train_data = object_detector.DataLoader.from_pascal_voc("train", "train", label_map = label_type)
    validation_data = object_detector.DataLoader.from_pascal_voc("valid", "valid", label_map = label_type)
    test_data = object_detector.DataLoader.from_pascal_voc("test", "test", label_map = label_type)

    spec = model_spec.get('efficientdet_lite0')

    model = object_detector.create(train_data, model_spec=spec, batch_size=16, train_whole_model=True, validation_data=validation_data)

    model.evaluate(test_data)

    return 0

treinar()