import os
import sys
import warnings
import logging

warnings.filterwarnings("ignore", message="TORCH_CUDA_ARCH_LIST is not set")

logging.getLogger("torch._inductor").setLevel(logging.ERROR)
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("torch.fx.experimental.symbolic_shapes").setLevel(logging.ERROR)

import torch
import time
from typing import Dict, Tuple

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from pos_enc.rayrope import RayRoPE_DotProductAttention
from pos_enc.rayrope_cuda import RayRoPE_DotProductAttention_CUDA

class RayRoPETester:
    def __init__(self, config: Dict, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        self.config = config
        self.device = torch.device(device)
        self.dtype = dtype
        
        model_params = [
            'head_dim', 'patches_x', 'patches_y', 'image_width', 'image_height',
            'pos_enc_type', 'num_rays_per_patch', 'depth_type', 'denc_type',
            'freq_base', 'apply_vo'
        ]
        
        layer_config = {k: v for k, v in config.items() if k in model_params}
        
        self.ref_model = RayRoPE_DotProductAttention(**layer_config).to(self.device).to(self.dtype)
        self.cuda_model = RayRoPE_DotProductAttention_CUDA(**layer_config).to(self.device).to(self.dtype)
        
        self._init_geometry()

    def _init_geometry(self):
        batch = self.config['batch']
        num_cams = self.config['num_cameras']
        self.w2cs = torch.eye(4, device=self.device, dtype=self.dtype).unsqueeze(0).expand(batch, num_cams, -1, -1).contiguous()
        self.Ks = torch.eye(3, device=self.device, dtype=self.dtype).unsqueeze(0).expand(batch, num_cams, -1, -1).contiguous()
        
        self.ref_model._precompute_and_cache_apply_fns(self.w2cs, self.Ks)
        self.cuda_model._precompute_and_cache_apply_fns(self.w2cs, self.Ks)

    def generate_inputs(self, requires_grad: bool = True) -> Tuple[torch.Tensor, ...]:
        c = self.config
        seqlen = c['num_cameras'] * c['patches_x'] * c['patches_y']
        
        def create_tensor(*shape):
            return torch.randn(*shape, device=self.device, dtype=self.dtype, requires_grad=requires_grad)

        q = create_tensor(c['batch'], c['num_heads'], seqlen, c['head_dim'])
        k = create_tensor(c['batch'], c['num_heads'], seqlen, c['head_dim'])
        v = create_tensor(c['batch'], c['num_heads'], seqlen, c['head_dim'])
        pd = create_tensor(c['batch'], seqlen, 2)
        return q, k, v, pd

    def measure_runtime(self, func, iters: int = 100) -> float:
        # Warmup
        for _ in range(10): func()
        torch.cuda.synchronize()
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(iters): func()
        end_event.record()
        torch.cuda.synchronize()
        
        return start_event.elapsed_time(end_event) / iters

    def run_benchmark(self, iters: int = 100):
        print(f"\n{'='*20} Benchmarking (Time & VRAM), {iters} iters{'='*20}")
        q, k, v, pd = self.generate_inputs()
        grad_out = torch.randn_like(q)

        # --- Forward ---
        torch.cuda.reset_peak_memory_stats()
        fwd_ref_time = self.measure_runtime(lambda: self.ref_model(q, k, v, predicted_d=pd), iters=iters)
        fwd_ref_mem = torch.cuda.max_memory_allocated() / 1024**2

        torch.cuda.reset_peak_memory_stats()
        fwd_cuda_time = self.measure_runtime(lambda: self.cuda_model(q, k, v, predicted_d=pd), iters=iters)
        fwd_cuda_mem = torch.cuda.max_memory_allocated() / 1024**2
        
        # --- Backward Setup ---
        out_ref = self.ref_model(q, k, v, predicted_d=pd)
        out_cuda = self.cuda_model(q, k, v, predicted_d=pd)

        def bwd_fn(model_out):
            q.grad = k.grad = v.grad = pd.grad = None
            model_out.backward(grad_out, retain_graph=True)

        # --- Backward ---
        torch.cuda.reset_peak_memory_stats()
        bwd_ref_time = self.measure_runtime(lambda: bwd_fn(out_ref))
        bwd_ref_mem = torch.cuda.max_memory_allocated() / 1024**2

        torch.cuda.reset_peak_memory_stats()
        bwd_cuda_time = self.measure_runtime(lambda: bwd_fn(out_cuda))
        bwd_cuda_mem = torch.cuda.max_memory_allocated() / 1024**2

        # 결과 출력
        print(f"{'Pass':10} | {'PyTorch (Time/Mem)':^25} | {'CUDA (Time/Mem)':^25} | Speedup")
        print("-" * 80)
        print(f"Forward    | {fwd_ref_time:6.3f}ms / {fwd_ref_mem:6.1f}MB | {fwd_cuda_time:6.3f}ms / {fwd_cuda_mem:6.1f}MB | {fwd_ref_time/fwd_cuda_time:6.2f}x")
        print(f"Backward   | {bwd_ref_time:6.3f}ms / {bwd_ref_mem:6.1f}MB | {bwd_cuda_time:6.3f}ms / {bwd_cuda_mem:6.1f}MB | {bwd_ref_time/bwd_cuda_time:6.2f}x")
    
    def run_equivalence_test(self):
        print(f"\n{'='*20} Equivalence Test {'='*20}")
        q, k, v, pd = self.generate_inputs()
        
        # 데이터 복사 (정밀한 비교를 위해)
        inputs_ref = [t.clone().detach().requires_grad_(True) for t in (q, k, v, pd)]
        inputs_cuda = [t.clone().detach().requires_grad_(True) for t in (q, k, v, pd)]

        # Forward
        out_ref = self.ref_model(*inputs_ref[:3], predicted_d=inputs_ref[3])
        out_cuda = self.cuda_model(*inputs_cuda[:3], predicted_d=inputs_cuda[3])

        fwd_diff = (out_ref - out_cuda).abs().max().item()
        print(f"Forward Max Diff: {fwd_diff:.6e}")

        # Backward
        grad_out = torch.randn_like(out_ref)
        out_ref.backward(grad_out)
        out_cuda.backward(grad_out)

        print("\n--- Gradient Max Differences ---")
        names = ["Q_grad", "K_grad", "V_grad", "Pd_grad"]
        for i, name in enumerate(names):
            diff = (inputs_ref[i].grad - inputs_cuda[i].grad).abs().max().item()
            print(f"{name:8}: {diff:.6e}")

if __name__ == "__main__":    
    device = "cuda"
    dtype = torch.bfloat16 # torch.float32
    
    config = {
        "batch": 4,
        "num_cameras": 4,
        "patches_x": 16,
        "patches_y": 16,
        "image_width": 256,
        "image_height": 256,
        "head_dim": 48,
        "num_heads": 8,
        "num_rays_per_patch": 3,
        "pos_enc_type": 'd_pj+0_3d',
        "freq_base": 3.0,
        "apply_vo": True,
    }

    print(f"\n{'='*20} Environment {'='*20}")
    print(f"device: {device}, dtype: {dtype}")
    tester = RayRoPETester(config, device, dtype)
    tester.run_equivalence_test()
    tester.run_benchmark()