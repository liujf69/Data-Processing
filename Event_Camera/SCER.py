import cv2
import torch
import numpy as np

# Symmetric Cumulative Event Representation
def events_to_image_torch(xs, ys, ps,
        device=None, sensor_size=(180, 240), clip_out_of_range=True,
        interpolation=None, padding=True, default=0):

    if device is None:
        device = xs.device
    if interpolation == 'bilinear' and padding:
        img_size = (sensor_size[0]+1, sensor_size[1]+1)
    else:
        img_size = list(sensor_size)

    mask = torch.ones(xs.size(), device=device)
    if clip_out_of_range:
        zero_v = torch.tensor([0.], device=device)
        ones_v = torch.tensor([1.], device=device)
        clipx = img_size[1] if interpolation is None and padding==False else img_size[1]-1
        clipy = img_size[0] if interpolation is None and padding==False else img_size[0]-1
        mask = torch.where(xs>=clipx, zero_v, ones_v)*torch.where(ys>=clipy, zero_v, ones_v)

    img = (torch.ones(img_size)*default).to(device)
    img = img.float()
    
    if xs.dtype is not torch.long:
        xs = xs.long().to(device)
    if ys.dtype is not torch.long:
        ys = ys.long().to(device)
    try:
        mask = mask.long().to(device)
        xs, ys = xs*mask, ys*mask
        img.index_put_((ys, xs), ps, accumulate=True)
        # print("able to put tensor {} positions ({}, {}) into {}. Range = {},{}".format(
        #     ps.shape, ys.shape, xs.shape, img.shape,  torch.max(ys), torch.max(xs)))
    except Exception as e:
        print("Unable to put tensor {} positions ({}, {}) into {}. Range = {},{}".format(
            ps.shape, ys.shape, xs.shape, img.shape,  torch.max(ys), torch.max(xs)))
        raise e
    return img

def gen_events_voxel_v2(events_ori, B, device=None, sensor_size=(180, 240), keep_middle=True):
    events_ori = torch.from_numpy(events_ori)
    events_ori = events_ori.float()
    # xs, ys, ts, ps = events_ori[:,1], events_ori[:,2], events_ori[:,0], events_ori[:,3]
    xs, ys, ts, ps = events_ori[:, 0], events_ori[:, 1], events_ori[:, 2], events_ori[:, 3]
    if device is None:
        device = xs.device
    assert(len(xs)==len(ys) and len(ys)==len(ts) and len(ts)==len(ps))
    bins = []
    dt = ts[-1]-ts[0]
    t_mid = ts[0] + (dt/2)
    
    # left of the mid -
    tend = t_mid
    end = binary_search_torch_tensor(ts, 0, len(ts)-1, tend)
    for bi in range(int(B/2)):
        tstart = ts[0] + (dt/B)*bi
        beg = binary_search_torch_tensor(ts, 0, len(ts)-1, tstart)
        vb = events_to_image_torch(xs[beg:end], ys[beg:end],
                ps[beg:end], device, sensor_size=sensor_size,
                clip_out_of_range=False)
        bins.append(-vb) # !
    # self
    if keep_middle:
        bins.append(torch.zeros_like(vb))  # TODO!!!
    # right of the mid +
    tstart = t_mid
    beg = binary_search_torch_tensor(ts, 0, len(ts)-1, tstart)
    for bi in range(int(B/2), B):
        tend = ts[0] + (dt/B)*(bi+1)
        end = binary_search_torch_tensor(ts, 0, len(ts)-1, tend)
        vb = events_to_image_torch(xs[beg:end], ys[beg:end],
                ps[beg:end], device, sensor_size=sensor_size,
                clip_out_of_range=False)
        bins.append(vb)

    bins = torch.stack(bins)
    bins = bins.numpy().transpose(1,2,0)
    return bins

def binary_search_torch_tensor(t, l, r, x, side='left'):

    if r is None:
        r = len(t)-1
    while l <= r:
        mid = l + (r - l)//2
        midval = t[mid]
        if midval == x:
            return mid
        elif midval < x:
            l = mid + 1
        else:
            r = mid - 1
    if side == 'left':
        return l
    return r

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
    
    events3 = gen_events_voxel_v2(events_ori, C_event, device=None, sensor_size=(event_h, event_w), keep_middle=False)
    events3_vis = events3.clip(-5,5) / 10
    
    cv2.imwrite('events3_vis.jpg',vis_event(events3_vis[...,1]))
    print("All Done!")