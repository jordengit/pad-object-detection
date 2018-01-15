Contains all the labels that there are annotations for. Since the model will
only output 100 generated detections at max, it's probably a bad idea to use
this. For a 7x6 board full of locks and plusses, this value will be exceeded.

The generated train.tfrecord and val.tfrecord files will also be created here.
