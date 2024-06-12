import json
import numpy as np
import cv2
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded."""

        self.model_config = json.loads(args["model_config"])

        self.num_detections_dtype = self._get_output_dtype("num_detections")
        self.detection_boxes_dtype = self._get_output_dtype("detection_boxes")
        self.detection_scores_dtype = self._get_output_dtype("detection_scores")
        self.detection_classes_dtype = self._get_output_dtype("detection_classes")

        self.score_threshold = 0.25
        self.nms_threshold = 0.45

    def _get_output_dtype(self, output_name):
        output_config = pb_utils.get_output_config_by_name(self.model_config, output_name)

        return pb_utils.triton_string_to_numpy(output_config['data_type'])

    def execute(self, requests):
        """`execute` receives a list of pb_utils.InferenceRequest as the only argument."""
        responses = []
        for request in requests:
            input_tensors = self._get_input_tensors(request)
            detections = self._process_tensor(input_tensors)
            response = self._create_response(detections)
            responses.append(response)

        return responses

    def _get_input_tensors(self, request):
        in_0 = pb_utils.get_input_tensor_by_name(request, "images")
        model_name = pb_utils.get_input_tensor_by_name(request, "model")

        return [in_0, model_name]

    def _process_tensor(self, input_tensors):
        in_0 = input_tensors[0]
        model_name = input_tensors[1].as_numpy()[0]

        infer_request = pb_utils.InferenceRequest(
            model_name=model_name,
            requested_output_names=["output0"],
            inputs=[in_0]
        )

        infer_response = infer_request.exec()

        if infer_response.has_error():
            raise pb_utils.TritonModelException(infer_response.error().message())

        infer_output_tensors = infer_response.output_tensors()

        output = infer_output_tensors[0]

        all_boxes, all_scores, all_class_ids = [], [], []
        for i in range(output.shape[0]):
            box = output[i, :4]
            classes_scores = output[i, 4:]
            max_class_idx = np.argmax(classes_scores)
            max_score = classes_scores[max_class_idx]

            if max_score >= self.score_threshold:
                all_boxes.append(box)
                all_scores.append(max_score)
                all_class_ids.append(max_class_idx)

        if len(all_boxes) > 0:
            indices = cv2.dnn.NMSBoxes(all_boxes, all_scores, self.score_threshold, self.nms_threshold)
        else:
            indices = []

        detections = {'boxes': [], 'scores': [], 'class_ids': [], 'num_detections': 0}
        if len(indices) > 0:
            for i in indices:
                idx = i[0] if isinstance(i, np.ndarray) else i
                detections['boxes'].append(all_boxes[idx])
                detections['scores'].append(all_scores[idx])
                detections['class_ids'].append(all_class_ids[idx])

            detections['num_detections'] = len(detections['boxes'])

        return detections

    def _create_response(self, detections):
        num_detections_tensor = pb_utils.Tensor("num_detections", np.array([detections['num_detections']], dtype=self.num_detections_dtype))
        detection_boxes_tensor = pb_utils.Tensor("detection_boxes", np.array(detections['boxes'], dtype=self.detection_boxes_dtype))
        detection_scores_tensor = pb_utils.Tensor("detection_scores", np.array(detections['scores'], dtype=self.detection_scores_dtype))
        detection_classes_tensor = pb_utils.Tensor("detection_classes", np.array(detections['class_ids'], dtype=self.detection_classes_dtype))

        return pb_utils.InferenceResponse(output_tensors=[num_detections_tensor, detection_boxes_tensor, detection_scores_tensor, detection_classes_tensor])

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")
