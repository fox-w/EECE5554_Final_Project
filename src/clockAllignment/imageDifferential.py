import rosbag
with rosbag.Bag("EECE5554_Final_Project/data/MH_01_easy.bag", 'r') as bag:
    cam0_timestamps = []
    cam1_timestamps = []
    for topic, msg, t in bag.read_messages(topics=['/cam0/image_raw']):
        cam0_timestamps.append(t.to_nsec())
    for topic, msg, t in bag.read_messages(topics=['/cam1/image_raw']):
        cam1_timestamps.append(t.to_nsec())
print("Cam0 Timestamps:", cam0_timestamps[:1000])
print("Cam1 Timestamps:", cam1_timestamps[:1000])
time_diffs = [abs(c0 - c1) for c0, c1 in zip(cam0_timestamps, cam1_timestamps)]
print("average time difference (ns):", (sum(time_diffs) / len(time_diffs)))
