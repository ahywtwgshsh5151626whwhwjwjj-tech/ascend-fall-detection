# test_model_load.py
import acl

acl.init()
acl.rt.set_device(0)

model_id, ret = acl.mdl.load_from_file("models/yolov8n_pose.om")
print("model_id:", model_id)
