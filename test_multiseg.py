import pixellib
from pixellib.instance import custom_segmentation

test_video = custom_segmentation()
test_video.inferConfig(num_classes=  7, class_names=["BG","base_link","link_s", "link_l", "link_u","link_r","link_b"])
test_video.load_model("models/segmentation/multi/A.h5")
test_video.process_video("data/set10/og_vid.avi", show_bboxes = True,  output_video_name="output/multiseg_test.avi", frames_per_second=15)