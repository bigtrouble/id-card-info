FROM python:3.9-slim

# Set the environment variables、
ARG PIP_ARG="-i https://mirrors.aliyun.com/pypi/simple/"
ARG GHP_RPOXY="https://ghfast.top/"

# Set the working directory
WORKDIR /app

# Install the build tool for dlib
RUN cat /etc/os-release
COPY sources.list /etc/apt/sources.list
RUN apt update
RUN apt -y install cmake gcc g++ ffmpeg libsm6 libxext6 libgl1 unzip wget bzip2 git

# Copy the requirements file
RUN python -m pip install --upgrade pip

# Copy the rest of the application code
COPY . .

# Install the dependencies
RUN pip install -r requirements.txt $PIP_ARG

# clone yolov5
RUN git clone ${GHP_RPOXY}https://github.com/ultralytics/yolov5.git
RUN pip install -r yolov5/requirements.txt $PIP_ARG

RUN wget https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 
RUN bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
RUN mv shape_predictor_68_face_landmarks.dat data/shape_predictor_68_face_landmarks.dat

# Download the PaddleOCR model
RUN mkdir -p /root/.paddleocr/whl/det/ch/ch_PP-OCRv4_det_infer/
RUN mkdir -p /root/.paddleocr/whl/rec/ch/ch_PP-OCRv4_rec_infer/
RUN mkdir -p /root/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer/
RUN wget https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar         -O /root/.paddleocr/whl/det/ch/ch_PP-OCRv4_det_infer/ch_PP-OCRv4_det_infer.tar
RUN wget https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar         -O /root/.paddleocr/whl/rec/ch/ch_PP-OCRv4_rec_infer/ch_PP-OCRv4_rec_infer.tar
RUN wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar -O /root/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer/ch_ppocr_mobile_v2.0_cls_infer.tar

# Specify the command to run the application
EXPOSE 5000
CMD ["python", "src/app.py"]
