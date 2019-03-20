import logging
import os
import glob

# external
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_frame_path_dict(input_frames_directory_path):
    """
    :param input_frames_directory_path: String path to directory full of frame images with name format "i.jpg"
    :return: python dictionary of frame_path_dict format {frame_number:"path/to/image, ...}
    """

    # get path to each frame in video frames directory
    image_path_list = glob.glob(input_frames_directory_path + "/*")

    frame_path_dict = []
    for path in image_path_list:
        # get filename
        filename = os.path.basename(path)

        # strip extension
        filename_wout_ext = filename.split('.')[0]

        # frame_number
        frame = int(filename_wout_ext)

        frame_path_dict.append((frame, path))

    return dict(frame_path_dict)


def min_max_frames(frame_path_dict):
    """
    return minimum and maximum frame number for frame path dict as well as continuous boolean value
    :param frame_path_dict: Python dictionary format {frame_number:"path/to/image, ...}
    :return: minimum frame, maximum frame, and a boolean of whether or not a continuous range exists
    """
    frames, paths = list(frame_path_dict.keys()), list(frame_path_dict.values())

    min_frame, max_frame = min(frames), max(frames)
    continuous = set(range(min_frame, max_frame + 1)) == set(frames)

    return min_frame, max_frame, continuous


def _write_mp4_video(ordered_image_paths, output_mp4_filepath):
    """
    source: http://tsaith.github.io/combine-images-into-a-video-with-python-3-and-opencv-3.html
    :param ordered_image_paths: array of image path strings to combine into mp4 video file
    :param output_mp4_file path: output file name without extension
    """
    # create output video file directory if it does not exist
    if not os.path.exists(os.path.split(output_mp4_filepath)[0]):
        os.makedirs(os.path.split(output_mp4_filepath)[0])

    # Determine the width and height from the first image
    image_path = ordered_image_paths[0]
    frame = cv2.imread(image_path)
    cv2.imshow('video', frame)
    height, width, channels = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    out = cv2.VideoWriter(output_mp4_filepath, fourcc, 20.0, (width, height))

    for image_path in ordered_image_paths:
        frame = cv2.imread(image_path)
        out.write(frame)  # Write out frame to video

    # Release everything if job is finished
    out.release()


def to_frames_directory(input_video_path, output_frames_directory_path):
    """
    Convert an mp4 video file to a directory of frames (currently .jpg)
    :param input_video_path:
    :param output_frames_directory_path:
    :return: exit code
    """

    # individual frame file name template
    frame_i_file_path_template = "%d.jpg"

    # ensure input video exists
    assert os.path.isfile(input_video_path)

    # create resource/frame directory for video if it does not exist
    logger.info("\n\tCreating frames directory...\n\tInput Video: %s\n\tOutput Frames Directory: %s" % (
        input_video_path, output_frames_directory_path))

    if not os.path.exists(output_frames_directory_path):
        os.makedirs(output_frames_directory_path)

    # write frame jpg's
    logger.info('\n\tWriting frames...')
    video_cap = cv2.VideoCapture(input_video_path)
    success, image = video_cap.read()
    count = 0
    while success:
        if success:
            output_file = frame_i_file_path_template % count
            output_file_path = os.path.join(output_frames_directory_path, output_file)
            logger.debug('Writing a new frame: %s ' % os.path.join(output_frames_directory_path, output_file))
            cv2.imwrite(output_file_path, image)  # save frame as JPEG file
            count += 1
        success, image = video_cap.read()

    logger.info('Finished writing frames.')
    return 0


def to_video(input_frames_directory_path, output_video_path):
    """
    :param input_frames_directory_path:
    :param output_video_path:
    :return: exit code
    """
    # write video
    logger.info('\n\tWriting Video...')
    output_frame_paths_dict = get_frame_path_dict(input_frames_directory_path)
    min_frame, max_frame, continuous = min_max_frames(output_frame_paths_dict)

    if continuous:
        ordered_frame_paths = []
        for frame in range(min_frame, max_frame + 1):
            ordered_frame_paths.append(output_frame_paths_dict[frame])
        _write_mp4_video(ordered_frame_paths, output_video_path)
    else:
        logger.error("Video Frames Directory %s Not continuous")

    logger.info('Finished writing video.')
    return 0
