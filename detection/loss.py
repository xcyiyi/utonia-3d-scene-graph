"""Hungarian set matching; supervised class, metric center/size and periodic yaw losses."""
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from .box_ops import pairwise_iou_giou


class DetectionCriterion(nn.Module):
    def __init__(self,weights=None,matcher=None,background_weight=.1,aux_weight=.3):
        super().__init__()
        self.weights=weights or {'class':1.,'center':5.,'size':2.,'heading':1.,'iou':0.}
        self.costs=matcher or {'class':2.,'center':5.,'size':2.,'heading':.5,'giou':0.}
        self.background_weight=background_weight; self.aux_weight=aux_weight
    @torch.no_grad()
    def match(self, output, targets):
        matches=[]
        for b,target in enumerate(targets):
            pred,gt=output['boxes'][b],target['boxes']
            if len(gt)>len(pred): raise ValueError('More GT objects than queries')
            cost = self.costs['center']*torch.cdist(pred[:,:3],gt[:,:3],p=1)+self.costs['size']*torch.cdist(pred[:,3:6],gt[:,3:6],p=1)
            cost += self.costs['heading']*(1-torch.cos(2*(pred[:,None,6]-gt[None,:,6])))
            cost -= self.costs['class']*output['logits'][b].softmax(-1)[:,target['labels']]
            if self.costs['giou'] and len(gt): cost-=self.costs['giou']*pairwise_iou_giou(pred,gt)[1]
            i,j=linear_sum_assignment(cost.cpu().numpy())
            matches.append((torch.as_tensor(i,device=pred.device),torch.as_tensor(j,device=pred.device)))
        return matches

    def single(self, output, targets):
        matches=self.match(output,targets)
        logits=output['logits']
        classes=torch.full(logits.shape[:2],10,device=logits.device,dtype=torch.long)
        pred,gt,hvec=[],[],[]
        for b,(i,j) in enumerate(matches):
            classes[b,i]=targets[b]['labels'][j]
            pred.append(output['boxes'][b,i]); gt.append(targets[b]['boxes'][j]); hvec.append(output['heading_vector'][b,i])
        weight=logits.new_ones(11); weight[-1]=self.background_weight
        losses={'class':F.cross_entropy(logits.transpose(1,2),classes,weight=weight)}
        pred,gt,hvec=torch.cat(pred),torch.cat(gt),torch.cat(hvec)
        if len(gt):
            losses['center']=F.l1_loss(pred[:,:3],gt[:,:3])
            losses['size']=F.l1_loss(pred[:,3:6],gt[:,3:6])
            angle_target=torch.stack(((2*gt[:,6]).sin(),(2*gt[:,6]).cos()),-1)
            losses['heading']=F.mse_loss(hvec,angle_target)
            if self.weights.get('iou',0):
                iou,giou=pairwise_iou_giou(pred,gt)
                losses['iou']=(1-giou.diag()).mean()
                losses['matched_iou']=iou.diag().mean().detach()
            else:
                losses['iou']=pred.sum()*0
                with torch.no_grad(): losses['matched_iou']=pairwise_iou_giou(pred,gt)[0].diag().mean()
            losses['matched_center_distance']=(pred[:,:3]-gt[:,:3]).norm(dim=-1).mean().detach()
        else:
            for k in ('center','size','heading','iou','matched_iou','matched_center_distance'): losses[k]=output['boxes'].sum()*0
        losses['positive_query_ratio']=logits.new_tensor(len(gt)/(logits.shape[0]*logits.shape[1]))
        total=sum(self.weights[k]*losses[k] for k in self.weights)
        return total,losses

    def forward(self, output, targets):
        total,losses=self.single(output,targets)
        for aux in output.get('aux_outputs',[]): total=total+self.aux_weight*self.single(aux,targets)[0]
        losses['total']=total
        return total,losses
