"""True pretrained Utonia encoder + a small, modern PyTorch DETR-style head.

No PointNet++/PointNet2 dependencies. This is not an exact reproduction of 3DETR.
"""
from contextlib import nullcontext
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
import utonia


@torch.no_grad()
def fps(xyz, count):
    """Deterministic farthest point sampling using plain PyTorch."""
    count = min(count,len(xyz))
    selected = torch.empty(count,dtype=torch.long,device=xyz.device)
    dist = torch.full((len(xyz),),float('inf'),device=xyz.device)
    farthest = ((xyz-xyz.mean(0))**2).sum(-1).argmax()
    for i in range(count):
        selected[i] = farthest
        dist = torch.minimum(dist,((xyz-xyz[farthest])**2).sum(-1))
        farthest = dist.argmax()
    return selected


class UtoniaDetectionAdapter(nn.Module):
    def __init__(self, dim=192):
        super().__init__()
        self.projection = nn.Sequential(nn.LayerNorm(1008),nn.Linear(1008,dim),nn.GELU(),nn.LayerNorm(dim))

    def forward(self, features): return self.projection(features)


class MultiScaleDetectionAdapter(nn.Module):
    def __init__(self,dim=192,channels=(216,432,576)):
        super().__init__(); self.channels=channels
        self.projections=nn.ModuleList([nn.Sequential(nn.LayerNorm(c),nn.Linear(c,dim),nn.GELU()) for c in channels])
        self.scale_logits=nn.Parameter(torch.zeros(len(channels)))
        self.fusion=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,dim),nn.GELU(),nn.LayerNorm(dim))

    def forward(self,features):
        levels=features.split(self.channels,dim=-1)
        fused=sum(w*proj(f) for w,proj,f in zip(self.scale_logits.softmax(0),self.projections,levels))
        return self.fusion(fused)


class PositionEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer('frequencies',2**torch.arange(5,dtype=torch.float32)*torch.pi)
        self.project = nn.Linear(33,dim)

    def forward(self, xyz):
        angles = xyz[...,None]*self.frequencies
        return self.project(torch.cat((xyz,angles.sin().flatten(-2),angles.cos().flatten(-2)),dim=-1))


def mlp(dim,out): return nn.Sequential(nn.Linear(dim,dim),nn.GELU(),nn.Linear(dim,out))


def gather_multiscale_features(stage_features,pooling_inverses,keep):
    """Align selected fine-stage points to coarser stages without a full dense concat.

    This is exactly equivalent to aligning every fine-stage point, concatenating
    all levels, and indexing the result by ``keep``.  Applying ``keep`` first
    avoids materializing the large temporary that otherwise dominates memory.
    """
    if len(stage_features)!=len(pooling_inverses)+1:
        raise ValueError('Each coarser feature stage requires one pooling inverse')
    indices=keep
    selected=[stage_features[0][indices]]
    for inverse,features in zip(pooling_inverses,stage_features[1:]):
        indices=inverse[indices]
        selected.append(features[indices])
    return torch.cat(selected,dim=-1)


class DetectionHead(nn.Module):
    def __init__(self, dim=192, queries=64, layers=3, query_type='fps', attention_heads=6,
                 checkpoint_layers=False):
        super().__init__()
        self.num_queries = queries
        self.query_type=query_type
        self.checkpoint_layers=checkpoint_layers
        if query_type not in ('fps','learned'): raise ValueError(query_type)
        if query_type=='learned': self.reference_points=nn.Embedding(queries,3)
        self.position = PositionEncoding(dim)
        self.queries = nn.Embedding(queries,dim)
        self.decoder = nn.ModuleList([nn.TransformerDecoderLayer(dim,attention_heads,dim*2,dropout=0,
            activation='gelu',batch_first=True,norm_first=True) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.classifier,self.center,self.size,self.heading = mlp(dim,11),mlp(dim,3),mlp(dim,3),mlp(dim,2)
        nn.init.zeros_(self.center[-1].weight); nn.init.zeros_(self.center[-1].bias)
        nn.init.zeros_(self.heading[-1].bias)
        with torch.no_grad(): self.heading[-1].bias[1] = 1.0

    def forward(self, memory, xyz, mask, query_indices, bounds):
        low, extent = bounds[:,0],(bounds[:,1]-bounds[:,0]).clamp_min(1)
        position = self.position((xyz-low[:,None])/extent[:,None])
        gather = query_indices[...,None].expand(-1,-1,xyz.shape[-1])
        anchors = torch.gather(xyz,1,gather)
        anchor_position = self.position((anchors-low[:,None])/extent[:,None])
        query_memory = torch.gather(memory,1,query_indices[...,None].expand(-1,-1,memory.shape[-1]))
        query = self.queries.weight[None] + query_memory
        if self.query_type=='learned':
            anchors=low[:,None]+self.reference_points.weight.sigmoid()[None]*extent[:,None]
            anchor_position=self.position((anchors-low[:,None])/extent[:,None])
            query=self.queries.weight[None].expand(len(memory),-1,-1)
        outputs = []
        for layer in self.decoder:
            if self.checkpoint_layers and self.training and torch.is_grad_enabled():
                def run_decoder(q,mem,qpos,mpos,padding,decoder=layer):
                    return decoder(q+qpos,mem+mpos,memory_key_padding_mask=padding)
                query=checkpoint(run_decoder,query,memory,anchor_position,position,mask,use_reentrant=False)
            else:
                query = layer(query+anchor_position,memory+position,memory_key_padding_mask=mask)
            h = self.norm(query)
            heading = self.heading(h).float()
            # A cuboid is invariant under yaw+pi; regress sin(2*yaw), cos(2*yaw).
            yaw = 0.5*torch.atan2(heading[...,0],heading[...,1])
            center = anchors+self.center(h).float()*extent[:,None]/4
            size = F.softplus(self.size(h).float())+0.02
            outputs.append({'logits':self.classifier(h).float(),'boxes':torch.cat((center,size,yaw[...,None]),-1),
                            'heading_vector':heading})
        outputs[-1]['aux_outputs'] = outputs[:-1]
        return outputs[-1]


class UtoniaDetector(nn.Module):
    def __init__(self, checkpoint, freeze_backbone=True, dim=192, queries=64, layers=3, max_tokens=1024,
                 adapter_type='legacy',query_type='fps',pretrained=True,attention_heads=6):
        super().__init__()
        if pretrained:
            self.backbone = utonia.load(str(checkpoint),custom_config={'shuffle_orders':False})
        else:
            config=utonia.load(str(checkpoint),ckpt_only=True)['config']; config['shuffle_orders']=False
            self.backbone=utonia.model.PointTransformerV3(**config)
        self.adapter_type=adapter_type
        if adapter_type not in ('legacy','multiscale'): raise ValueError(adapter_type)
        for module in self.backbone.modules():
            if hasattr(module,'shuffle_orders'): module.shuffle_orders=False
        self.freeze_backbone = freeze_backbone
        self.backbone.requires_grad_(not freeze_backbone)
        self.adapter = UtoniaDetectionAdapter(dim) if adapter_type=='legacy' else MultiScaleDetectionAdapter(dim)
        # Full-backbone B/C training is activation-heavy. Decoder checkpointing
        # is exact here because its dropout is zero, and changes no state keys.
        self.head = DetectionHead(dim,queries,layers,query_type,attention_heads,
                                  checkpoint_layers=not freeze_backbone)
        self.max_tokens = max_tokens
        if freeze_backbone: self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone: self.backbone.eval()
        return self

    def encode(self, points, bounds):
        with torch.no_grad() if self.freeze_backbone else nullcontext():
            # spconv wheel uses FP32 sparse convolution; attention keeps its own
            # official mixed-precision implementation. The detector uses AMP.
            with torch.autocast(device_type='cuda',enabled=False):
                out = self.backbone(points)
            stage3 = out.pooling_parent
            coords = stage3.origin_coord
            spatial=stage3
            stage_features=(stage3.feat,out.feat)
            pooling_inverses=(out.pooling_inverse,)
            if self.adapter_type=='multiscale':
                stage2=stage3.pooling_parent
                spatial=stage2; coords=stage2.origin_coord
                stage_features=(stage2.feat,stage3.feat,out.feat)
                pooling_inverses=(stage3.pooling_inverse,out.pooling_inverse)
            records = []
            for b in range(len(bounds)):
                ix = torch.where(spatial.batch==b)[0]
                keep = ix[fps(coords[ix],self.max_tokens)]
                features=gather_multiscale_features(stage_features,pooling_inverses,keep)
                records.append({'features':features,'xyz':coords[keep],
                    'query_indices':fps(coords[keep],self.head.num_queries),
                    'bounds':bounds[b], 'stage3_points':int((stage3.batch==b).sum()),
                    'stage4_points':int((out.batch==b).sum())})
            return records

    def decode(self, records):
        count = max(len(r['xyz']) for r in records)
        features,xyz,masks,queries = [],[],[],[]
        for r in records:
            n=len(r['xyz']); pad=count-n
            features.append(F.pad(r['features'],(0,0,0,pad)))
            xyz.append(F.pad(r['xyz'],(0,0,0,pad)))
            masks.append(torch.arange(count,device=r['xyz'].device)>=n)
            q=r['query_indices']
            if len(q)<self.head.num_queries: q=q.repeat((self.head.num_queries+len(q)-1)//len(q))[:self.head.num_queries]
            queries.append(q)
        memory = self.adapter(torch.stack(features))
        return self.head(memory,torch.stack(xyz),torch.stack(masks),torch.stack(queries),torch.stack([r['bounds'] for r in records]))

    def forward(self, points, bounds): return self.decode(self.encode(points,bounds))

    def optimizer_groups(self, head_lr=1e-4, backbone_lr=1e-5):
        groups=[{'params':list(self.adapter.parameters())+list(self.head.parameters()),'lr':head_lr,'name':'adapter_head'}]
        if not self.freeze_backbone:
            groups.append({'params':[p for p in self.backbone.parameters() if p.requires_grad],
                           'lr':backbone_lr,'name':'backbone'})
        return groups
