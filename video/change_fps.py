import moviepy.editor as mp

def change_frame_rate(input_file, output_file, new_frame_rate):
    # 读取视频文件
    video = mp.VideoFileClip(input_file)

    # 使用新帧率创建一个新的视频剪辑
    new_video = video.set_fps(new_frame_rate)

    # 写入新的视频文件
    new_video.write_videofile(output_file, codec='libx264')

if __name__ == "__main__":
    # 输入视频文件路径
    input_file = "parking_video/delivery_env1.mp4"
    # 输出视频文件路径
    output_file = "parking_video/delivery_env1_15fps.mp4"
    # 新帧率
    new_frame_rate = 15  # 例如，将帧率改为 30 FPS

    # 调用函数改变帧率
    change_frame_rate(input_file, output_file, new_frame_rate)

    print(f"save changed video as: {output_file}")
