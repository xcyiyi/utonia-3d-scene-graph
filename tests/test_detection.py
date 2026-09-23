import unittest
import numpy as np
import torch
from detection.geometry import corners,inside,iou3d
from detection.evaluate import evaluate
from detection.loss import DetectionCriterion
from detection.data import prepare_batch
from detection.model import DetectionHead


class DetectionTests(unittest.TestCase):
    def test_oriented_iou_and_inside(self):
        a=np.array([0.,0.,0.,4.,2.,2.,0.])
        self.assertAlmostEqual(iou3d(a,a),1)
        b=a.copy(); b[6]=np.pi/2
        self.assertAlmostEqual(iou3d(a,b),1/3,places=6)
        b=a.copy(); b[2]=3
        self.assertEqual(iou3d(a,b),0)
        b=a.copy(); b[6]=0.37
        self.assertTrue(inside(corners(b),b).all())
        c=b.copy(); c[6]+=np.pi
        self.assertAlmostEqual(iou3d(b,c),1)
        self.assertAlmostEqual(iou3d(a,b),iou3d(b,a))

    def test_ap_duplicate_and_absent_classes(self):
        b=np.array([[0.,0.,0.,2.,1.,1.,0.]])
        gt={1:{'boxes':b,'labels':np.array([0])}}
        pred={1:{'boxes':np.repeat(b,2,axis=0),'labels':np.array([0,0]),'scores':np.array([.9,.8])}}
        metric=evaluate(pred,gt)['thresholds']['0.5']
        self.assertEqual(metric['TP'],1); self.assertEqual(metric['FP'],1)
        self.assertEqual(metric['per_class']['bed']['AP'],1)
        self.assertIsNone(metric['per_class']['chair']['AP'])

    def test_match_permutation_and_empty_scene_backward(self):
        criterion=DetectionCriterion()
        boxes=torch.tensor([[[0.,0,0,1,1,1,0],[2.,0,0,2,1,1,.4],[5.,0,0,1,1,1,0]]],requires_grad=True)
        logits=torch.randn(1,3,11,requires_grad=True)
        heading=torch.randn(1,3,2,requires_grad=True)
        output={'boxes':boxes,'logits':logits,'heading_vector':heading}
        target={'boxes':boxes.detach()[0,:2],'labels':torch.tensor([0,3])}
        loss,_=criterion(output,[target])
        reversed_target={k:v.flip(0) for k,v in target.items()}
        loss2,_=criterion(output,[reversed_target])
        torch.testing.assert_close(loss,loss2)
        loss.backward(); self.assertTrue(torch.isfinite(logits.grad).all())
        empty={'boxes':torch.empty(0,7),'labels':torch.empty(0,dtype=torch.long)}
        loss,_=criterion(output,[empty]); self.assertTrue(torch.isfinite(loss))

    def test_preprocess_preserves_metric_coordinates_and_batch(self):
        def sample(n):
            xyz=np.arange(n*3,dtype=np.float32).reshape(n,3)*.1+np.array([2.,3.,-1.],np.float32)
            return {'coord':xyz,'color':np.zeros((n,3),np.uint8),'normal':np.zeros((n,3),np.float32),
                    'boxes':np.zeros((0,7),np.float32),'labels':np.zeros(0,np.int64)}
        samples=[sample(20),sample(15)]
        points,bounds,targets=prepare_batch(samples,'cpu')
        self.assertEqual(points['offset'].tolist(),[20,35])
        torch.testing.assert_close(points['origin_coord'][:20],torch.from_numpy(samples[0]['coord']))
        self.assertEqual(points['feat'].shape,(35,9))

    def test_decoder_padding_does_not_change_predictions(self):
        torch.manual_seed(3)
        head=DetectionHead(dim=48,queries=4,layers=1).eval()
        memory=torch.randn(1,12,48); xyz=torch.randn(1,12,3)
        bounds=torch.tensor([[[-3.,-3,-3],[3.,3,3]]]); q=torch.tensor([[0,1,2,3]])
        with torch.no_grad():
            a=head(memory,xyz,torch.zeros(1,12,dtype=torch.bool),q,bounds)
            b=head(torch.cat((memory,torch.randn(1,5,48)*100),1),
                   torch.cat((xyz,torch.randn(1,5,3)*100),1),
                   torch.tensor([[False]*12+[True]*5]),q,bounds)
        torch.testing.assert_close(a['boxes'],b['boxes'],atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(a['logits'],b['logits'],atol=1e-5,rtol=1e-5)


if __name__=='__main__': unittest.main()
