"""Metric Z-up boxes: [cx, cy, cz, dx, dy, dz, yaw], CCW yaw about +Z."""
import numpy as np

CLASSES = ['bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
           'night_stand', 'bookshelf', 'bathtub']
EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]


def corners(box):
    signs = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                      [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]])
    p = signs * np.asarray(box[3:6]) / 2
    c, s = np.cos(box[6]), np.sin(box[6])
    return p @ np.array([[c,s,0],[-s,c,0],[0,0,1]]) + box[:3]


def inside(points, box):
    c, s = np.cos(box[6]), np.sin(box[6])
    local = (points - box[:3]) @ np.array([[c,-s,0],[s,c,0],[0,0,1]])
    return (np.abs(local) <= np.asarray(box[3:6])/2 + 1e-6).all(1)


def cross(a, b): return a[0]*b[1]-a[1]*b[0]


def clip_polygon(subject, clip):
    output = list(subject)
    for a, b in zip(clip, np.roll(clip, -1, axis=0)):
        inp, output = output, []
        if not inp: break
        p = inp[-1]
        for q in inp:
            dp, dq = cross(b-a, p-a), cross(b-a, q-a)
            if (dp >= -1e-10) != (dq >= -1e-10):
                output.append(p + (q-p) * dp / (dp-dq))
            if dq >= -1e-10: output.append(q)
            p = q
    return np.asarray(output)


def iou3d(a, b):
    height = min(a[2]+a[5]/2, b[2]+b[5]/2)-max(a[2]-a[5]/2, b[2]-b[5]/2)
    if height <= 0: return 0.0
    poly = clip_polygon(corners(a)[:4,:2], corners(b)[:4,:2])
    area = 0.0 if len(poly)<3 else abs(sum(cross(p,q) for p,q in zip(poly,np.roll(poly,-1,axis=0))))/2
    inter = area * height
    return float(np.clip(inter / max(np.prod(a[3:6])+np.prod(b[3:6])-inter,1e-10),0,1))


def visualize(path, points, colors, boxes, labels, scores=None, title='GT boxes'):
    import plotly.graph_objects as go
    ids = np.linspace(0,len(points)-1,min(len(points),12000),dtype=int)
    fig = go.Figure(go.Scatter3d(x=points[ids,0],y=points[ids,1],z=points[ids,2],
        mode='markers',marker=dict(size=1,color=colors[ids].astype(np.uint8)),name='RGB points'))
    for i,(box,label) in enumerate(zip(boxes,labels)):
        xyz = corners(box)
        lines = np.array([v for a,b in EDGES for v in (xyz[a],xyz[b],[np.nan]*3)])
        name = f'{i}: {CLASSES[int(label)]}' + (f' {scores[i]:.3f}' if scores is not None else '')
        fig.add_trace(go.Scatter3d(x=lines[:,0],y=lines[:,1],z=lines[:,2],mode='lines',
            line=dict(width=5),name=name))
    fig.update_layout(title=title,scene=dict(aspectmode='data',xaxis_title='X right (m)',
        yaxis_title='Y forward (m)',zaxis_title='Z up (m)'),margin=dict(l=0,r=0,b=0,t=50))
    fig.write_html(str(path),include_plotlyjs=True)

