# The codes below partially refer to the PySceneDetect. According
# to its BSD 3-Clause License, we keep the following.
#
#          PySceneDetect: Python-Based Video Scene Detector
#   ---------------------------------------------------------------
#     [  Site: http://www.bcastell.com/projects/PySceneDetect/   ]
#     [  Github: https://github.com/Breakthrough/PySceneDetect/  ]
#     [  Documentation: http://pyscenedetect.readthedocs.org/    ]
#
# Copyright (C) 2014-2020 Brandon Castellano <http://www.bcastell.com>.
#
# PySceneDetect is licensed under the BSD 3-Clause License; see the included
# LICENSE file, or visit one of the following pages for details:
#  - https://github.com/Breakthrough/PySceneDetect/
#  - http://www.bcastell.com/projects/PySceneDetect/
#
# This software uses Numpy, OpenCV, click, tqdm, simpletable, and pytest.
# See the included LICENSE files or one of the above URLs for more information.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import logging
import math
import os
import subprocess
import time
from string import Template
from .custom_platform import tqdm


def is_mkvmerge_available():
    """Is mkvmerge Available: Gracefully checks if mkvmerge command is
    available.

    Returns:
        (bool) True if the mkvmerge command is available, False otherwise.
    """
    ret_val = None
    try:
        ret_val = subprocess.call(['mkvmerge', '--quiet'])
    except OSError:
        return False
    if ret_val is not None and ret_val != 2:
        return False
    return True


def is_ffmpeg_available():
    """Is ffmpeg Available: Gracefully checks if ffmpeg command is available.

    Returns:
        (bool) True if the ffmpeg command is available, False otherwise.
    """
    ret_val = None
    try:
        ret_val = subprocess.call(['ffmpeg', '-v', 'quiet'])
    except OSError:
        return False
    if ret_val is not None and ret_val != 1:
        return False
    return True


def split_video_mkvmerge(input_video_paths,
                         shot_list,
                         output_file_prefix,
                         video_name,
                         suppress_output=False):
    """Split video.

    Calls the mkvmerge command on the input video(s), splitting it at
    the passed timecodes, where each shot is written in sequence from
    001.
    """

    if not input_video_paths or not shot_list:
        return

    logging.info(
        'Splitting input video%s using mkvmerge, output path template:\n  %s',
        's' if len(input_video_paths) > 1 else '', output_file_prefix)

    ret_val = None
    # mkvmerge automatically appends '-$SHOT_NUMBER'.
    output_file_name = output_file_prefix.replace('-${SHOT_NUMBER}', '')
    output_file_name = output_file_prefix.replace('-$SHOT_NUMBER', '')
    output_file_template = Template(output_file_name)
    output_file_name = output_file_template.safe_substitute(
        VIDEO_NAME=video_name, SHOT_NUMBER='')

    try:
        call_list = ['mkvmerge']
        if suppress_output:
            call_list.append('--quiet')
        call_list += [
            '-o', output_file_name, '--split',
            'parts:%s' % ','.join([
                '%s-%s' % (start_time.get_timecode(), end_time.get_timecode())
                for start_time, end_time in shot_list
            ]), ' +'.join(input_video_paths)
        ]
        total_frames = shot_list[-1][1].get_frames(
        ) - shot_list[0][0].get_frames()
        processing_start_time = time.time()
        ret_val = subprocess.call(call_list)
        if not suppress_output:
            print('')
            logging.info(
                'Average processing speed %.2f frames/sec.',
                float(total_frames) / (time.time() - processing_start_time))
    except OSError:
        logging.error(
            'mkvmerge could not be found on the system.'
            ' Please install mkvmerge to enable video output support.')
        raise
    if ret_val is not None and ret_val != 0:
        logging.error('Error splitting video (mkvmerge returned %d).', ret_val)


def split_video_ffmpeg(input_video_paths,
                       shot_list,
                       output_dir,
                       output_file_template=('${OUTPUT_DIR}/'
                                             'shot_${SHOT_NUMBER}.mp4'),
                       arg_override='-crf 21',
                       hide_progress=False,
                       suppress_output=False,
                       output_names=None):
    """Calls the ffmpeg command on the input video(s), generating a new video
    for each shot based on the start/end timecodes.

    Args:
        input_video_paths (List[str])
        shot_list (List[Tuple[FrameTimecode, FrameTimecode]],)
    """

    os.makedirs(output_dir, exist_ok=True)
    if not input_video_paths or not shot_list:
        return

    logging.info(
        'Splitting input video%s using ffmpeg, output path template:\n  %s',
        's' if len(input_video_paths) > 1 else '', output_file_template)
    if len(input_video_paths) > 1:
        # TODO: Add support for splitting multiple/appended input videos.
        # https://trac.ffmpeg.org/wiki/Concatenate#samecodec
        # Requires generating a temporary file list for ffmpeg.
        logging.error(
            'Sorry, splitting multiple appended/concatenated input videos with'
            ' ffmpeg is not supported yet. This feature will be added to a'
            ' future version of ShotDetect. In the meantime, you can try using'
            ' the -c / --copy option with the split-video to use mkvmerge,'
            ' which generates less accurate output,'
            ' but supports multiple input videos.')
        raise NotImplementedError()

    arg_override = arg_override.replace('\\"', '"')

    ret_val = None
    if len(arg_override) > 0:
        arg_override = arg_override.split(' ')
    else:
        arg_override = []
    filename_template = Template(output_file_template)
    shot_num_format = '%0'
    shot_num_format += str(
        max(4,
            math.floor(math.log(len(shot_list), 10)) + 1)) + 'd'
    try:
        progress_bar = None
        total_frames = shot_list[-1][1].get_frames(
        ) - shot_list[0][0].get_frames()
        if tqdm and not hide_progress:
            progress_bar = tqdm(
                total=total_frames,
                unit='frame',
                miniters=1,
                desc='Split Video')
        processing_start_time = time.time()
        for i, (start_time, end_time) in enumerate(shot_list):
            end_time = end_time.__sub__(1)
            # Fix the last frame of a shot to be 1 less than the first frame
            # of the next shot
            duration = (end_time - start_time)
            # an alternative way to do it
            # duration = (end_time.get_frames()-1)/end_time.framerate -
            #    (start_time.get_frames())/start_time.framerate
            # duration_frame = end_time.get_frames() - 1 - \
            # start_time.get_frames()
            call_list = ['ffmpeg']
            if suppress_output:
                call_list += ['-v', 'quiet']
            elif i > 0:
                # Only show ffmpeg output for the first call, which will
                # display any errors if it fails, and then break the loop.
                # We only show error messages for the remaining calls.
                call_list += ['-v', 'error']
            call_list += [
                '-y', '-ss',
                start_time.get_timecode(), '-i', input_video_paths[0]
            ]
            call_list += arg_override  # compress
            call_list += ['-map_chapters', '-1']  # remove meta stream
            if output_names is None:
                output_name = filename_template.safe_substitute(OUTPUT_DIR=output_dir, SHOT_NUMBER=shot_num_format % (i))
            else:
                output_name = output_names[i]
            call_list += [
                '-strict', '-2', '-t',
                duration.get_timecode(), '-sn',
                output_name
            ]
            ret_val = subprocess.call(call_list)
            if not suppress_output and i == 0 and len(shot_list) > 1:
                logging.info(
                    'Output from ffmpeg for shot 1 shown above, splitting \
                        remaining shots...')
            if ret_val != 0:
                break
            if progress_bar:
                progress_bar.update(
                    duration.get_frames() +
                    1)  # to compensate the missing one frame caused above
        if progress_bar:
            print('')
            logging.info(
                'Average processing speed %.2f frames/sec.',
                float(total_frames) / (time.time() - processing_start_time))
    except OSError:
        logging.error('ffmpeg could not be found on the system.'
                      ' Please install ffmpeg to enable video output support.')
    if ret_val is not None and ret_val != 0:
        logging.error('Error splitting video (ffmpeg returned %d).', ret_val)
