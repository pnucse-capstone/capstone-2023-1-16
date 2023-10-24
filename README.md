# 실재감 증대를 위한 증강현실 시각 효과 연구개발

## 1. 프로젝트 소개

기존에 존재하는 실시간 image segmentation 모델과 이미지나 동영상에서 특정 부분을 지우고 채워주는 inpainting 모델을 결합하여 실시간으로 객체를 검출하고 지우는 프로그램 구현한다.

## 2. 팀소개

### 드루와 유니티의 숲 팀

<table>
  <tr>
    <td>
      천형주
    </td>
    <td>
      <ul>
        <li>Yolo v8과 OpenCV를 이용하여 객체를 직접 선택해 Image Segmentation하는 코드 구현</li>
        <li>Deepfillv2, MAT 모델 적용</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td>
      민예진
    </td>
    <td>
      <ul>
        <li>마우스 클릭 로직 구현</li>
        <li>Inpainting 모델 서칭 및 LaMa 모델 적용</li>
      </ul>
    </td>
  </tr>
</table>

## 3. 구성도
### 전체 구성도
![image](https://github.com/pnucse-capstone/capstone-2023-1-16/assets/68144657/cfe965d4-b00c-496b-ac3d-d6914bca6927)

## 4. 시연 영상

## 5. 설치 및 사용법
0. pretrained model 준비
1. 구동환경 : pytorch 2.0.1 / CUDA 11.8
2. 패키지
- pyyaml 
- tqdm
- numpy
- easydict
- scikit-image
- opencv-python
- tensorflow
- joblib
- matplotlib
- pandas
- albumentations
- hydra-core
- pytorch-lightning
- tabulate
- kornia
- webdataset
- packaging
- future
- scipy
- click
- requests
- pyspng
- Pillow
- tensorboard
- ninja
- imageio-ffmpeg
- timm
- psutil
- scikit-learn
- Pillow
- tensorboard
- pyyaml

3. 사용법  
```
python modi.py <<Inpainting Model 이름>>
```
Inpainting Model 이름 : LAMA, DEEPFILLV2, MAT 중 하나를 적는다.
