import cv2
import numpy as np

import onnxruntime as ort
from .onnxdet import inference_detector
from .onnxpose import inference_pose


# Workaround: https://github.com/microsoft/onnxruntime/issues/7846
def init_session(model_path, providers=None):
    if providers is None:
        providers = [
            ('CUDAExecutionProvider', {
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'cudnn_conv_algo_search': 'DEFAULT',
                'do_copy_in_default_stream': True,
            }),
            'CPUExecutionProvider',
        ]
    sess = ort.InferenceSession(model_path, providers=providers)
    return sess

class PickableInferenceSession: # This is a wrapper to make the current InferenceSession class pickable.
    def __init__(self, model_path, providers=None):
        self.model_path = model_path
        self.sess = init_session(self.model_path, providers)

    def run(self, *args):
        return self.sess.run(*args)

    def get_inputs(self):
        return self.sess.get_inputs()

    def get_outputs(self):
        return self.sess.get_outputs()

    def __getstate__(self):
        return {'model_path': self.model_path}

    def __setstate__(self, values):
        self.model_path = values['model_path']
        self.sess = init_session(self.model_path)

class Wholebody:
    def __init__(self, onnx_path, providers=None):
        onnx_det = f'{onnx_path}/yolox_l.onnx'
        onnx_pose = f'{onnx_path}/dw-ll_ucoco_384.onnx'

        self.session_det = PickableInferenceSession(onnx_det, providers)
        self.session_pose = PickableInferenceSession(onnx_pose, providers)

    def _filter_one_person(self, det_result):
        if len(det_result) == 0:
            return det_result
        areas = [xywh[2] * xywh[3] for xywh in det_result]
        idx = np.argmax(areas)
        return np.asarray([det_result[idx]])
    
    def __call__(self, oriImg, only_one_person=False):
        det_result = inference_detector(self.session_det, oriImg)
        det_result = self._filter_one_person(det_result) if only_one_person else det_result
        keypoints, scores = inference_pose(self.session_pose, det_result, oriImg)

        keypoints_info = np.concatenate(
            (keypoints, scores[..., None]), axis=-1)
        # compute neck joint
        neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
        # neck score when visualizing pred
        neck[:, 2:4] = np.logical_and(
            keypoints_info[:, 5, 2:4] > 0.3,
            keypoints_info[:, 6, 2:4] > 0.3).astype(int)
        new_keypoints_info = np.insert(
            keypoints_info, 17, neck, axis=1)
        mmpose_idx = [
            17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3
        ]
        openpose_idx = [
            1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17
        ]
        new_keypoints_info[:, openpose_idx] = \
            new_keypoints_info[:, mmpose_idx]
        keypoints_info = new_keypoints_info

        keypoints, scores = keypoints_info[
            ..., :2], keypoints_info[..., 2]
        
        # I guessed that the empty list means a whole
        return keypoints, scores, det_result


