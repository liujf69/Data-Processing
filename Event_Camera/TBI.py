import cv2
import torch
import numpy as np

# Voxel grid using temporal bilinear interpolation
def events_to_arr(xs, ys, ps, events_h, events_w, accumulate=True):
    """
    Accumulate events into an image.
    """

    img_size = [events_h, events_w]
    img = torch.zeros(img_size, dtype=ys.dtype)

    if xs.dtype is not torch.long:
        xs = xs.long()
    if ys.dtype is not torch.long:
        ys = ys.long()

    img.index_put_((ys, xs), ps, accumulate=accumulate)
    return img

def gen_events_voxel(events_ori, num_bins, duration, events_h, events_w, round_ts=False):
    """
    Generate a voxel grid from input events using temporal bilinear interpolation.
    """
    events_ori = torch.from_numpy(events_ori.copy())
    xs, ys, ts, ps = events_ori[:, 0], events_ori[:, 1], events_ori[:, 2], events_ori[:, 3]
    voxel = []
    ts = ts * (num_bins - 1) / duration

    if round_ts:
        ts = torch.round(ts)

    zeros = torch.zeros(ts.shape)
    for b_idx in range(num_bins):
        weights = torch.max(zeros, 1.0 - torch.abs(ts - b_idx))
        voxel_bin = events_to_arr(xs, ys, ps * weights, events_h, events_w)
        voxel.append(voxel_bin)
    return torch.stack(voxel).numpy().transpose(1, 2, 0)

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
    
    events2 = gen_events_voxel(events_ori.astype(np.float32), C_event, duration, event_h, event_w, round_ts=False)
    events2_vis = events2.clip(-5,5) / 10
    
    cv2.imwrite('events2_vis.jpg',vis_event(events2_vis[...,2]))
    
    print("All Done!")