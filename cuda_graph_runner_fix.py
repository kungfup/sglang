
# 关键修复1：恢复stream管理
def replay_prepare(self, forward_batch, pp_proxy_tensors=None):
    """修复版：使用正确的stream管理"""
    raw_bs = forward_batch.batch_size
    raw_num_token = forward_batch.input_ids.shape[0]
    bs = self.get_padded_batch_size(raw_bs)
    
    # 填充默认值
    if bs != raw_bs:
        self.seq_lens.fill_(self.seq_len_fill_value)
        self.out_cache_loc.zero_()
    
    # 关键修复：在捕获的stream上执行所有拷贝操作
    # 这避免了跨stream同步等待
    with torch.cuda.stream(self.stream):
        # 优化1：对大模型使用异步拷贝
        if self.model_size_gb > 30:  # 32B模型
            # 使用non_blocking=True加速拷贝
            self.input_ids[:raw_num_token].copy_(
                forward_batch.input_ids, non_blocking=True
            )
            self.req_pool_indices[:raw_bs].copy_(
                forward_batch.req_pool_indices, non_blocking=True
            )
            self.seq_lens[:raw_bs].copy_(
                forward_batch.seq_lens, non_blocking=True
            )
            self.out_cache_loc[:raw_num_token].copy_(
                forward_batch.out_cache_loc, non_blocking=True
            )
            self.positions[:raw_num_token].copy_(
                forward_batch.positions, non_blocking=True
            )
        else:
            # 小模型使用同步拷贝
            self.input_ids[:raw_num_token].copy_(forward_batch.input_ids)
            self.req_pool_indices[:raw_bs].copy_(forward_batch.req_pool_indices)
            self.seq_lens[:raw_bs].copy_(forward_batch.seq_lens)
            self.out_cache_loc[:raw_num_token].copy_(forward_batch.out_cache_loc)
            self.positions[:raw_num_token].copy_(forward_batch.positions)
        
        # 其他必要的拷贝...
        if forward_batch.seq_lens_cpu is not None:
            if bs != raw_bs:
                self.seq_lens_cpu.fill_(self.seq_len_fill_value)
            self.seq_lens_cpu[:raw_bs].copy_(forward_batch.seq_lens_cpu)
        
        # 在stream上初始化attention metadata
        self.model_runner.attn_backend.init_forward_metadata_replay_cuda_graph(
            bs,
            self.req_pool_indices[:bs],
            self.seq_lens[:bs],
            forward_batch.seq_lens_sum + (bs - raw_bs) * self.seq_len_fill_value,
            self.encoder_lens[:bs] if self.is_encoder_decoder else None,
            self.capture_forward_mode,
            forward_batch.spec_info,
            seq_lens_cpu=self.seq_lens_cpu[:bs],
        )
    
    # 存储字段
    self.raw_bs = raw_bs
    self.raw_num_token = raw_num_token
    self.bs = bs

# 关键修复2：优化replay函数
def replay(self, forward_batch, pp_proxy_tensors=None):
    """修复版：移除不必要的日志和同步"""
    if self.update_mode == CudaGraphRunnerMode.UPDATE_TORCH_INPUT:
        self.input_ids[:self.raw_num_token].copy_(forward_batch.input_ids)
        self.positions[:self.raw_num_token].copy_(forward_batch.positions)
    
    # 关键修复：确保在正确的stream上replay
    with torch.cuda.stream(self.stream):
        self.graphs[self.bs].replay()
    
    # 对于32B模型，添加stream同步点管理
    if self.model_size_gb > 30 and os.getenv("SGLANG_32B_SYNC_AFTER_REPLAY"):
        # 只在必要时同步，而不是每次都同步
        self.stream.synchronize()
    
    output = self.output_buffers[self.bs]
    return output

# 关键修复3：优化warmup过程
def capture(self):
    """修复版：减少同步开销"""
    # ...capture代码...
    
    def run_once():
        # 运行一次forward
        return self.model_runner.forward(...)
    
    # 优化：只对小模型进行warmup同步
    if self.model_size_gb < 30:  # 非32B模型
        for _ in range(2):
            torch.cuda.synchronize()
            self.model_runner.tp_group.barrier()
            run_once()
    else:
        # 32B模型：减少warmup同步
        run_once()  # 只运行一次，不同步
    
    # 开始捕获
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=self.stream):
        out = run_once()
    
    return graph, out
