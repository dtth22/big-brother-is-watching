from mmpose.apis import MMPoseInferencer

img_path = 'webcam'   # replace this with your own image path

# instantiate the inferencer using the model alias
inferencer = MMPoseInferencer(pose3d='human3d')

# The MMPoseInferencer API employs a lazy inference approach,
# creating a prediction generator when given input
result_generator = inferencer(img_path, show=True)

while True:
    result = next(result_generator)
