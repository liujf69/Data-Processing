import cv2
import numpy as np

# Channel-by-channel accumulation
# 将events按时间戳累积到不同通道内
def gen_events_array(events_ori, C_event, duration, event_h, event_w):
    events = np.zeros((event_h, event_w, C_event))
    C_inter = duration / C_event
    for i in range(events_ori.shape[0]):
        W, H, t, p = events_ori[i]
        p = -1 if p == 0 else p
        events[int(H), int(W), min(int(t // C_inter), C_event - 1)] += p
    return events

def vis_event(gray):
    h,w = gray.shape
    out = 255*np.ones([h, w, 3])
    pos_weight = gray.copy()
    neg_weight = gray.copy()

    pos_weight[pos_weight < 0] = 0
    pos_weight = pos_weight * 2 * 255

    neg_weight[neg_weight > 0] = 0
    neg_weight = abs(neg_weight) * 2 * 255
    out[..., 1] = out[...,1] - pos_weight - neg_weight
    out[..., 0] -= pos_weight
    out[..., 2] -= neg_weight
    out = out.clip(0, 255)
    return out.astype(np.uint8)

if __name__ == "__main__":
    file_name = "./A0P10C0-2021_11_04_12_30_08.npy"
    events_ori = np.load(file_name, allow_pickle = True) # N W H T P
    event_w = int(np.max(events_ori[:, 0]) + 1) # max_W 
    event_h = int(np.max(events_ori[:, 1]) + 1) # max_H
    C_event = 5
    
    events_ori[:, 2] = events_ori[:, 2] - events_ori[:, 2].min() # T
    events_ori[:, 3] = (events_ori[:, 3] - 0.5)*2 # P
    duration = events_ori[:, 2].max() - events_ori[:, 2].min() # T
    
    events1 = gen_events_array(events_ori, C_event, duration, event_h, event_w) # event_h event_w C_event
    events1_vis = events1.clip(-5, 5) / 10
    
    cv2.imwrite('./events1_vis.jpg', vis_event(events1_vis[..., 2]))