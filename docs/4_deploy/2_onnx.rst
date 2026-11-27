.. _ONNX:

Compile to ONNX
===============

OpenVINO CPU deployment
-----------------------

The export script fuses YOLOv9 ``RepConv`` branches in evaluation mode before
export. For INT8 calibration, use images outside the evaluation subset:

.. code-block:: powershell

   .venv\Scripts\python.exe scripts\export_openvino.py `
     --model v9-t --weights weights\v9-t.pt --dataset coco --imgsz 640 `
     --int8 --calib-dir data\coco_val2017\images\val2017 --calib-samples 500 `
     --calib-exclude-ann data\coco_val2017\annotations\instances_val2017.json `
     --calib-exclude-first-images 500 --protect-heads --embed-preprocess --static-preprocess `
     --output-dir exports\coco_bench\v9t_int8_reexport --name v9t_int8_reexport

The exclusion count follows the COCO evaluator's image-ID ordering, so those
images are not used for calibration when evaluating the first 500 images.
NNCF's slower bias-correction mode is enabled for better quantization accuracy.
For a full validation-set evaluation, calibrate from the training split instead.

The stream runner uses four asynchronous requests by default, keeps frame
ordering for tracking/output, and processes every captured frame unless
``--drop-stale-frames`` is explicitly set:

.. code-block:: powershell

   .venv\Scripts\python.exe inference\demo_stream_count.py `
     --model-xml exports\coco_bench\v9t_int8_reexport\v9t_int8_reexport.xml `
     --source input.mp4 --threads 4 --async-requests 4

Use ``--async-requests 1`` for sequential low-latency inference. Run the
benchmark with the same request count to compare throughput and CPU-seconds
per 100 frames:

.. code-block:: powershell

   .venv\Scripts\python.exe scripts\benchmark_compare.py `
     --models v9t-int8=exports\coco_bench\v9t_int8_reexport\v9t_int8_reexport.xml `
     --val-images data\coco_val2017\images\val2017 `
     --val-ann data\coco_val2017\annotations\instances_val2017.json `
     --max-eval-images 500 --threads 4 --async-requests 4 `
     --output-md benchmark_report_runtime.md --output-json benchmark_report_runtime.json
