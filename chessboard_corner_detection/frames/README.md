# Frames
#### A python 3 package for converting a video files to a directory of frames (1.png, 2.png, ..., n.png) and vice versa.

## Usage
Clone
```
git clone https://github.com/browlm13/frames.git
```
Change directory
```
cd frames
```
Install
```
sudo python setup.py install
```
Importing the package
```
import frames
```
Converting a video file to a directory of frames
```
input_video_path = 'test_video.mp4'
output_frames_directory_path = 'video_frames/test_video'
frames.to_frames_directory(input_video_path, output_frames_directory_path)
```
Converting a directory of frames into a video file
```
input_frames_directory_path = 'video_frames/test_video'
output_video_path = 'reconstructed_test_video.mp4'
frames.to_video(input_frames_directory_path, output_video_path)
```

### Additional features TODO:
- frame_path_dict