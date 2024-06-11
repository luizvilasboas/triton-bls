import sys
import numpy as np
import tritonclient.http as httpclient
from tritonclientutils import np_to_triton_dtype, InferenceServerException

model_name = "bls_sync"
shape = [1, 3, 640, 640]

with httpclient.InferenceServerClient("localhost:8000") as client:
    input0_data = np.random.rand(*shape).astype(np.float32)
    inputs = [
        httpclient.InferInput("images", input0_data.shape, np_to_triton_dtype(input0_data.dtype)),
        httpclient.InferInput("model", [1], np_to_triton_dtype(np.object_)),
    ]

    inputs[0].set_data_from_numpy(input0_data)
    inputs[1].set_data_from_numpy(np.array(["yolov8n"], dtype=np.object_))

    outputs = [
        httpclient.InferRequestedOutput("output0"),
    ]

    response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)

    result = response.get_response()
    output0_data = response.as_numpy("output0")
    print("=========='yolov8n' model result==========")
    print("images ({}) = output0 ({})".format(input0_data, output0_data))

    inputs[1].set_data_from_numpy(np.array(["yolov8n_mug"], dtype=np.object_))
    response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)

    result = response.get_response()
    output0_data = response.as_numpy("output0")
    print("\n")
    print("=========='yolov8n_mug' model result==========")
    print("images ({}) = output0 ({})".format(input0_data, output0_data))

    print("\n")
    print("=========='undefined' model result==========")
    try:
        inputs[1].set_data_from_numpy(np.array(["undefined_model"], dtype=np.object_))
        _ = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)
    except InferenceServerException as e:
        print(e.message())

    print("PASS: BLS Sync")
    sys.exit(0)
