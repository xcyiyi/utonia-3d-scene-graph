"""Differentiable oriented IoU in Z-up coordinates, implemented in current PyTorch."""
import torch


def cross(a,b): return a[...,0]*b[...,1]-a[...,1]*b[...,0]


def rectangle(boxes):
    signs=boxes.new_tensor([[-1,-1],[1,-1],[1,1],[-1,1]])
    p=signs*boxes[...,None,3:5]/2
    c,s=boxes[...,6,None].cos(),boxes[...,6,None].sin()
    return torch.stack((p[...,0]*c-p[...,1]*s,p[...,0]*s+p[...,1]*c),-1)+boxes[...,None,:2]


def convex_hull_area(points):
    """Area of the convex hull of eight rectangle corners, piecewise differentiable.

    A directed hull edge has every valid point on its left. Collinear boundary
    points are handled by keeping only extreme endpoints; duplicates are removed.
    Topology is discrete; area gradients flow through the original hull vertices.
    """
    with torch.no_grad():
        p=points.detach()
        duplicate=((p[..., :,None,:]-p[...,None,:,:]).square().sum(-1)<1e-14)
        earlier=torch.tril(torch.ones(8,8,dtype=torch.bool,device=p.device),diagonal=-1)
        valid=~(duplicate&earlier).any(-1)
        edge=p[...,None,:,:]-p[..., :,None,:]
        to_point=p[...,None,None,:,:]-p[..., :,None,None,:]
        signed=cross(edge[...,None,:],to_point)
        valid_k=valid[...,None,None,:]
        left=((signed>=-1e-7)|~valid_k).all(-1)
        projection=(edge[...,None,:]*to_point).sum(-1)
        length=edge.square().sum(-1)
        within=(projection>=-1e-7)&(projection<=length[...,None]+1e-7)
        extreme=((signed.abs()>1e-7)|within|~valid_k).all(-1)
        hull_edges=left&extreme&valid[..., :,None]&valid[...,None,:]&(length>1e-14)
    area=(cross(points[..., :,None,:],points[...,None,:,:])*hull_edges).sum((-1,-2)).abs()/2
    return area


def pairwise_iou_giou(a,b):
    """[N,7] x [M,7] -> IoU/GIoU [N,M]. Exact rotated intersection.

    GIoU uses the convex-hull XY enclosure and the enclosing Z interval.
    Piecewise differentiable polygon clipping via interior corners/edge intersections.
    """
    a,b=a.float(),b.float()
    pa,pb=rectangle(a)[:,None],rectangle(b)[None]
    ea=pa.roll(-1,-2)-pa; eb=pb.roll(-1,-2)-pb
    ina=(cross(eb[...,None,:,:],pa[..., :,None,:]-pb[...,None,:,:])>=-1e-7).all(-1)
    inb=(cross(ea[...,None,:,:],pb[..., :,None,:]-pa[...,None,:,:])>=-1e-7).all(-1)
    delta=pb[...,None,:,:]-pa[..., :,None,:]
    denom=cross(ea[..., :,None,:],eb[...,None,:,:])
    safe=torch.where(denom.abs()>1e-8,denom,torch.ones_like(denom))
    t=cross(delta,eb[...,None,:,:])/safe
    u=cross(delta,ea[..., :,None,:])/safe
    valid=(denom.abs()>1e-8)&(t>=0)&(t<=1)&(u>=0)&(u<=1)
    intersections=pa[..., :,None,:]+t[...,None]*ea[..., :,None,:]
    n,m=len(a),len(b)
    points=torch.cat((pa.expand(n,m,4,2),pb.expand(n,m,4,2),intersections.reshape(n,m,16,2)),-2)
    mask=torch.cat((ina,inb,valid.reshape(n,m,16)),-1)
    count=mask.sum(-1)
    center=(points*mask[...,None]).sum(-2)/count.clamp_min(1)[...,None]
    angle=torch.atan2(points[...,1]-center[...,None,1],points[...,0]-center[...,None,0])
    order=torch.where(mask,angle,angle.new_full((),10)).argsort(-1)
    ordered=points.gather(-2,order[...,None].expand(-1,-1,-1,2))
    pos=torch.arange(24,device=a.device).view(1,1,-1)
    nextpos=torch.where(pos+1<count[...,None],pos+1,0).expand(n,m,-1)
    nxt=ordered.gather(-2,nextpos[...,None].expand(-1,-1,-1,2))
    area=(cross(ordered,nxt)*(pos<count[...,None])).sum(-1).abs()/2
    height=(torch.minimum(a[:,None,2]+a[:,None,5]/2,b[None,:,2]+b[None,:,5]/2)-
            torch.maximum(a[:,None,2]-a[:,None,5]/2,b[None,:,2]-b[None,:,5]/2)).clamp_min(0)
    intersection=area*height
    volume_a=a[:,3:6].prod(-1)[:,None]; volume_b=b[:,3:6].prod(-1)[None]
    union=(volume_a+volume_b-intersection).clamp_min(1e-7)
    iou=(intersection/union).clamp(0,1)
    hull_area=convex_hull_area(torch.cat((pa.expand(n,m,4,2),pb.expand(n,m,4,2)),-2))
    zlo=torch.minimum(a[:,None,2]-a[:,None,5]/2,b[None,:,2]-b[None,:,5]/2)
    zhi=torch.maximum(a[:,None,2]+a[:,None,5]/2,b[None,:,2]+b[None,:,5]/2)
    enclosing=(hull_area*(zhi-zlo)).clamp_min(1e-7)
    giou=iou-(enclosing-union).clamp_min(0)/enclosing
    return iou,giou
