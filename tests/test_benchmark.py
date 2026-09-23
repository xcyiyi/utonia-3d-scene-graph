import unittest
import copy
import numpy as np
import torch
from detection.box_ops import pairwise_iou_giou
from detection.geometry import iou3d,corners
from detection.augmentation import geometric_transform,augment
from detection.evaluate import evaluate
from detection.model import MultiScaleDetectionAdapter,DetectionHead,gather_multiscale_features
from detection.loss import DetectionCriterion
from scripts.train_benchmark import describe_nonfinite_gradients,split_microbatches,validate_resume_compatibility


class BenchmarkTests(unittest.TestCase):
    def test_microbatch_split_preserves_order_and_remainder(self):
        samples=list(range(23))
        chunks=split_microbatches(samples,10)
        self.assertEqual([len(x) for x in chunks],[10,10,3])
        self.assertEqual(sum(chunks,[]),samples)
        self.assertEqual(split_microbatches(samples),[samples])
        with self.assertRaisesRegex(ValueError,'positive'): split_microbatches(samples,0)

    def test_selected_multiscale_gather_matches_full_alignment_and_gradient(self):
        inverse23=torch.tensor([0,1,1,2,3,3,4,5])
        inverse34=torch.tensor([0,0,1,1,2,2])
        keep=torch.tensor([0,3,6,7])
        f2=torch.randn(8,3); f3=torch.randn(6,4); f4=torch.randn(3,5)
        def run(kind):
            xs=[x.clone().requires_grad_() for x in (f2,f3,f4)]
            if kind=='optimized': y=gather_multiscale_features(xs,(inverse23,inverse34),keep)
            else: y=torch.cat((xs[0],xs[1][inverse23],xs[2][inverse34[inverse23]]),-1)[keep]
            y.square().sum().backward(); return y.detach(),[x.grad for x in xs]
        selected,selected_grad=run('optimized'); full,full_grad=run('reference')
        torch.testing.assert_close(selected,full)
        for actual,expected in zip(selected_grad,full_grad): torch.testing.assert_close(actual,expected)

    def test_nonfinite_gradient_diagnostics_are_json_safe(self):
        model=torch.nn.Linear(3,2)
        model.weight.grad=torch.tensor([[1.,float('nan'),float('inf')],[-float('inf'),2.,3.]])
        model.bias.grad=torch.ones_like(model.bias)
        details=describe_nonfinite_gradients(model)
        self.assertEqual(details,[{'parameter':'weight','shape':[2,3],'nonfinite':3,
                                   'nan':1,'posinf':1,'neginf':1}])

    def test_resume_migration_preserves_effective_batch_and_schedule(self):
        config={'training':{'batch_size_per_gpu':4},'unchanged':'value'}
        checkpoint={'epoch_complete':True,'world_size':2,'config':copy.deepcopy(config),'epoch':90,
                    'global_step':60151,'rng_by_rank':[{},{}]}
        migrated=copy.deepcopy(config); migrated['training']['batch_size_per_gpu']=8
        info=validate_resume_compatibility(checkpoint,migrated,1,661,True)
        self.assertEqual(info['effective_batch_size'],8)
        self.assertEqual(info['steps_per_epoch'],661)
        self.assertFalse(info['bit_exact'])
        with self.assertRaisesRegex(ValueError,'identical world size'):
            validate_resume_compatibility(checkpoint,migrated,1,661,False)
        incompatible=copy.deepcopy(migrated); incompatible['unchanged']='different'
        with self.assertRaisesRegex(ValueError,'only permits'):
            validate_resume_compatibility(checkpoint,incompatible,1,661,True)
        too_large=copy.deepcopy(config); too_large['training']['batch_size_per_gpu']=16
        with self.assertRaisesRegex(ValueError,'effective batch size'):
            validate_resume_compatibility(checkpoint,too_large,1,331,True)

    def test_rotated_iou_against_independent_clipper_and_gradient(self):
        rng=np.random.default_rng(2)
        a=np.concatenate((rng.normal(0,.5,(12,3)),rng.uniform(.3,2,(12,3)),rng.uniform(-3,3,(12,1))),1).astype(np.float32)
        b=np.concatenate((rng.normal(0,.5,(9,3)),rng.uniform(.3,2,(9,3)),rng.uniform(-3,3,(9,1))),1).astype(np.float32)
        x=torch.tensor(a,requires_grad=True); y=torch.tensor(b)
        i,g=pairwise_iou_giou(x,y)
        expected=np.array([[iou3d(aa,bb) for bb in b] for aa in a])
        np.testing.assert_allclose(i.detach().numpy(),expected,atol=2e-5,rtol=1e-4)
        (1-g).mean().backward(); self.assertTrue(torch.isfinite(x.grad).all()); self.assertGreater(float(x.grad.abs().sum()),0)
        eps=1e-3; ap=a.copy(); am=a.copy(); ap[0,0]+=eps; am[0,0]-=eps
        numeric=((1-pairwise_iou_giou(torch.tensor(ap),y)[1]).mean()-(1-pairwise_iou_giou(torch.tensor(am),y)[1]).mean())/(2*eps)
        self.assertAlmostEqual(float(x.grad[0,0]),float(numeric),delta=3e-4)
        same_iou,same_giou=pairwise_iou_giou(torch.tensor(a),torch.tensor(a))
        torch.testing.assert_close(same_iou.diag(),torch.ones(len(a)),atol=2e-5,rtol=1e-5)
        torch.testing.assert_close(same_giou.diag(),torch.ones(len(a)),atol=2e-5,rtol=1e-5)

    def test_geometric_augmentations_transform_boxes_and_normals(self):
        box=np.array([[1.,2.,-.5,2.,1.,.7,.4]],np.float32)
        sample={'coord':corners(box[0]).astype(np.float32),'boxes':box,'normal':np.tile([1.,0,0],(8,1)).astype(np.float32),'color':np.ones((8,3),np.uint8)*100}
        for flip,angle,scale in [(True,0,1),(False,.3,1),(False,0,.87),(True,-.4,1.13)]:
            result=geometric_transform(sample,flip,angle,scale)
            a=result['coord']; b=corners(result['boxes'][0]); dist=np.linalg.norm(a[:,None]-b[None],axis=-1)
            self.assertLess(float(dist.min(1).max()),2e-6)
            np.testing.assert_allclose(np.linalg.norm(result['normal'],axis=1),1,atol=1e-6)
        np.testing.assert_array_equal(sample['boxes'],box)

    def test_fixed_ten_classes(self):
        b=np.array([[0.,0,0,1,1,1,0]],np.float32)
        p={0:{'boxes':b,'labels':np.array([0]),'scores':np.array([1.])}}
        t={0:{'boxes':b,'labels':np.array([0])}}
        m=evaluate(p,t,fixed_classes=True)['thresholds']['0.5']
        self.assertEqual(m['mAP'],.1); self.assertEqual(m['class_count'],10)
        self.assertNotIn('mAP_present_classes',m)
        p[1]={'boxes':np.empty((0,7),np.float32),'labels':np.array([],dtype=int),'scores':np.array([])}
        t[1]={'boxes':np.empty((0,7),np.float32),'labels':np.array([],dtype=int)}
        self.assertEqual(evaluate(p,t,fixed_classes=True)['thresholds']['0.5']['mAP'],.1)

    def test_all_scales_receive_gradient_and_query_modes(self):
        x=torch.randn(2,30,1224,requires_grad=True); adapter=MultiScaleDetectionAdapter(48)
        y=adapter(x); y.square().mean().backward()
        for g in x.grad.split((216,432,576),dim=-1): self.assertGreater(float(g.abs().sum()),0)
        for kind in ['fps','learned']:
            head=DetectionHead(48,8,1,query_type=kind)
            output=head(y.detach(),torch.randn(2,30,3),torch.zeros(2,30,dtype=torch.bool),torch.arange(8).repeat(2,1),torch.tensor([[[-3.,-3,-3],[3,3,3]]]*2))
            self.assertEqual(output['boxes'].shape,(2,8,7)); self.assertTrue((output['boxes'][...,3:6]>0).all())

    def test_decoder_checkpointing_preserves_output_and_gradient(self):
        regular=DetectionHead(48,8,2,attention_heads=6,checkpoint_layers=False).train()
        recomputed=copy.deepcopy(regular); recomputed.checkpoint_layers=True
        xyz=torch.randn(2,12,3); mask=torch.zeros(2,12,dtype=torch.bool)
        queries=torch.arange(8).repeat(2,1); bounds=torch.tensor([[[-2.,-2,-2],[2,2,2]]]*2)
        def run(head):
            memory=torch.randn(2,12,48,requires_grad=True)
            output=head(memory,xyz,mask,queries,bounds)
            loss=output['logits'].sum()+output['boxes'].sum()
            for aux in output['aux_outputs']: loss=loss+aux['logits'].sum()+aux['boxes'].sum()
            loss.backward()
            return output,memory.grad,[p.grad for p in head.parameters()]
        torch.manual_seed(5); expected,expected_input,expected_parameters=run(regular)
        torch.manual_seed(5); actual,actual_input,actual_parameters=run(recomputed)
        torch.testing.assert_close(actual['logits'],expected['logits'])
        torch.testing.assert_close(actual['boxes'],expected['boxes'])
        torch.testing.assert_close(actual_input,expected_input)
        for got,want in zip(actual_parameters,expected_parameters):
            if want is None: self.assertIsNone(got)
            else: torch.testing.assert_close(got,want)

    def test_color_dropout_and_jitter(self):
        s={'coord':np.ones((16,3),np.float32),'boxes':np.ones((1,7),np.float32),'normal':np.ones((16,3),np.float32),'color':np.ones((16,3),np.float32)*255}
        c={'flip_probability':0,'rotation_degrees':0,'scale_range':[1,1],'brightness_range':[.8,1.2], 'rgb_shift':.05,'point_color_jitter':.025,'color_dropout':1.}
        out=augment(s,np.random.default_rng(37),c); self.assertEqual(out['color'].sum(),0)
        c['color_dropout']=0; out=augment(s,np.random.default_rng(37),c)
        self.assertTrue(((out['color']>=0)&(out['color']<=255)).all())


if __name__=='__main__': unittest.main()
