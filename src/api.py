
import torch
import dlib
import numpy as np
import cv2
from paddleocr import PaddleOCR
import os
import base64

from ppocr.utils.logging import get_logger
import logging
logger = get_logger()
logger.setLevel(logging.INFO)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.abspath("data/shape_predictor_68_face_landmarks.dat"))

# load custom model.
model = torch.hub.load(os.path.abspath('yolov5'), 'custom', path=os.path.abspath('data/best.onnx'), source='local', force_reload=True)
model.conf = 0.25  # confidence threshold (0-1)
model.iou = 0.45   # NMS IoU threshold (0-1)

ocr = PaddleOCR(use_angle_cls=True, lang="ch")

def find_elemnt(detections, img, name='face'):
  for idx, row in detections.iterrows():
    if (row['name'] == name):
      xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
      cropped = img[ymin:ymin + (ymax - ymin), xmin:xmin + (xmax - xmin)].copy()
      return cropped
  return None



def calculate_rotation_angle(landmarks):
  """
  根据人眼位置计算旋转角度
  :param landmarks: 人脸关键点列表（68个点）
  :return: 旋转角度（单位：度）
  """
  # 左眼中心点（关键点 36-41）
  left_eye = np.mean([landmarks[i] for i in range(36, 42)], axis=0)
  # 右眼中心点（关键点 42-47）
  right_eye = np.mean([landmarks[i] for i in range(42, 48)], axis=0)
  
  # 计算两眼连线的斜率
  dy = right_eye[1] - left_eye[1]
  dx = right_eye[0] - left_eye[0]
  angle = np.degrees(np.arctan2(dy, dx))  # 计算角度
  
  # 目标角度是水平方向（0度），因此需要矫正的角度为 `angle`
  return angle


def rotate_image(image, angle):
  """
  根据角度旋转图片（保持完整内容，避免裁剪）
  :param image: 输入图片
  :param angle: 旋转角度（度）
  :return: 旋转后的图片
  """
  (h, w) = image.shape[:2]
  center = (w // 2, h // 2)
  M = cv2.getRotationMatrix2D(center, angle, 1.0)
  
  # 计算旋转后的图像边界
  cos = np.abs(M[0, 0])
  sin = np.abs(M[0, 1])
  new_w = int((h * sin) + (w * cos))
  new_h = int((h * cos) + (w * sin))
  
  # 调整旋转矩阵的平移参数
  M[0, 2] += (new_w - w) // 2
  M[1, 2] += (new_h - h) // 2
  
  rotated = cv2.warpAffine(image, M, (new_w, new_h), borderMode=cv2.BORDER_REPLICATE)
  return rotated




def detect_face_landmarks(image):
  """
  检测人脸和关键点
  :param image: 输入图片（BGR格式）
  :return: 人脸关键点坐标列表（68个点）
  """
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  faces = detector(gray)
  if len(faces) == 0:
    return None
  # 取第一个人脸的关键点
  landmarks = predictor(gray, faces[0])
  return [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]



def find_face_angle(faceImg): 
  """
  """
  for angle in range(0, 360, 10):
    rotateImg = rotate_image(faceImg, angle)
    dets = detector(rotateImg, 0)
    if len(dets) > 0:
      image_points = detect_face_landmarks(rotateImg)
      eyeAngle = calculate_rotation_angle(image_points)
      
      finallyAngle = angle + eyeAngle
      if finallyAngle > 360:
        return finallyAngle - 360
      elif finallyAngle < -360:
        return finallyAngle + 360
      return finallyAngle
  return None


def to_base64(image):
  _, buffer = cv2.imencode('.jpg', image)
  b64str = base64.b64encode(buffer).decode('utf-8')
  return b64str


def detect_image(original_img, auto_detect_direction=True, result_include_image=True):
  results = model(original_img)
  dets = results.pandas().xyxy[0]
  face = find_elemnt(dets, original_img, name='face')
  if face is None:
    raise Exception('May not be an ID-Card image. [-100]')

  angle = 0
  if auto_detect_direction == True:
    angle = find_face_angle(face)

  normal_image = original_img if angle == 0 else rotate_image(original_img, angle)
  results = results if angle == 0 else model(normal_image)
  dets = results.pandas().xyxy[0]
  front_image = find_elemnt(dets, normal_image, name='front')
  if front_image is None:
    raise Exception('May not be an ID-Card image. [-101]')

  ocr_result =  ocr.ocr(front_image, cls=True)
  return ocr_result, \
    to_base64(front_image) if result_include_image == True else None, \
    to_base64(find_elemnt(dets, normal_image, name='face')) if result_include_image == True else None


    

