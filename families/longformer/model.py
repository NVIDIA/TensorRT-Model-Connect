# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Longformer token features with the reference's default local-only attention."""

import json
from pathlib import Path
import numpy as np
import tensorrt as trt
from . import graph as g


def local_attention_mask(length, window):
    if window <= 0 or window % 2:
        raise ValueError('Longformer attention window must be positive and even')
    positions = np.arange(length)
    return np.where(np.abs(positions[:,None]-positions[None,:]) <= window//2,
                    0.0,-1e9).astype(np.float32).reshape(1,1,length,length)


def build_engine(weights, cfg, length, verbose=False):
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    options = builder.create_builder_config()
    options.builder_optimization_level = 3
    options.clear_flag(trt.BuilderFlag.TF32)
    width = cfg['hidden_size']
    def constant(key):
        value = weights[key]
        return g.add_constant(network,value.shape,value)
    def linear(x,key):
        return g.add_linear(network,x,weights[key+'.weight'],weights[key+'.bias'])
    def norm(x,key):
        return g.add_layer_norm(network,x,weights[key+'.weight'],weights[key+'.bias'],cfg['layer_norm_eps'])
    ids = network.add_input('input_ids',trt.int32,(length,))
    valid = network.add_input('attention_mask',trt.int32,(length,))
    word = network.add_gather(constant('longformer.embeddings.word_embeddings.weight'),ids,0).get_output(0)
    # Requests contain unpadded token sequences. Only trailing engine-profile
    # padding is inserted by the runtime, so valid positions are contiguous.
    positions = weights['longformer.embeddings.position_embeddings.weight'][2:length+2]
    hidden = g.add_sum(network,word,g.add_constant(network,positions.shape,positions))
    types = weights['longformer.embeddings.token_type_embeddings.weight'][:1]
    hidden = g.add_sum(network,hidden,g.add_constant(network,types.shape,types))
    hidden = norm(hidden,'longformer.embeddings.LayerNorm')
    valid = network.add_cast(valid,trt.float32).get_output(0)
    padding = network.add_elementwise(g.add_constant(network,(1,),np.ones(1)),valid,
                                     trt.ElementWiseOperation.SUB).get_output(0)
    padding = network.add_elementwise(padding,g.add_constant(network,(1,),np.array([-1e9])),
                                     trt.ElementWiseOperation.PROD).get_output(0)
    reshape = network.add_shuffle(padding);reshape.reshape_dims=(1,1,1,length)
    windows = cfg['attention_window']
    for index in range(cfg['num_hidden_layers']):
        prefix = f'longformer.encoder.layer.{index}'
        projections = [linear(hidden,prefix+'.attention.self.'+name) for name in ('query','key','value')]
        window = local_attention_mask(length,windows[index])
        mask = g.add_sum(network,g.add_constant(network,window.shape,window),reshape.get_output(0))
        attention = g.add_attention(network,*projections,num_heads=cfg['num_attention_heads'],
                                     head_dim=width//cfg['num_attention_heads'],mask=mask)
        hidden = norm(g.add_sum(network,hidden,linear(attention,prefix+'.attention.output.dense')),
                       prefix+'.attention.output.LayerNorm')
        inner = g.add_gelu(network,linear(hidden,prefix+'.intermediate.dense'))
        hidden = norm(g.add_sum(network,hidden,linear(inner,prefix+'.output.dense')),
                       prefix+'.output.LayerNorm')
    hidden.name='hidden_states';network.mark_output(hidden)
    plan=builder.build_serialized_network(network,options)
    if plan is None: raise RuntimeError('Longformer engine build failed')
    return bytes(plan)


def build(request,writer):
    if request.task != 'text_to_token_features' or request.backend != 'trt':
        raise ValueError('Longformer requires text_to_token_features and backend=trt')
    if (request.precision != 'fp32' or request.tensor_parallel_size != 1
        or request.context_parallel_size != 1 or request.max_batch_size != 1
        or request.quantization not in (None,'none') or request.fp32_layers
        or request.dynamic_kv_cache or request.mtp_seq_len is not None
        or request.image_height is not None or request.image_width is not None
        or request.video_num_frames is not None):
        raise NotImplementedError('Longformer supports single-sequence fp32 token features')
    directory = Path(request.model_dir)
    cfg = json.loads((directory/'config.json').read_text())
    if cfg.get('model_type') != 'longformer' or cfg.get('hidden_act') != 'gelu':
        raise ValueError('Expected the GELU Longformer backbone')
    length = request.max_sequence_length or 512
    if length < 1 or length > cfg['max_position_embeddings']-2:
        raise ValueError('Longformer sequence length exceeds its position table')
    windows = cfg['attention_window']
    if len(windows) != cfg['num_hidden_layers']:
        raise ValueError('Longformer requires one attention window per layer')
    import torch
    weights = {name:value.float().numpy() for name,value in torch.load(
        directory/'pytorch_model.bin',map_location='cpu',weights_only=True).items()}
    writer.set_header(family='longformer',task=request.task,backend=request.backend)
    writer.add_bytes('model.plan',build_engine(weights,cfg,length,request.verbose))
    writer.add_json('runtime.json',{'sequence_length':length,'hidden_size':cfg['hidden_size'],
        'vocab_size':cfg['vocab_size'],'pad_token_id':cfg['pad_token_id']})
    writer.add_bytes('tokenizer.json',(directory/'tokenizer.json').read_bytes())
