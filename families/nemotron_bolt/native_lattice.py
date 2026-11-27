# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bolt LiLiCorr's learned chain transition scores, not independent top-K proposals."""

import numpy as np

from .native_language import Graph


class Lattice(Graph):
    def fp16(self, x):
        return self.cast(x, self.t.float16)

    def weight(self, key):
        # The provider materializes the dense draft parameters in FP16.
        return self.w.f32("lilicorr." + key).astype(np.float16)

    def sum(self, a, b):
        return self.ew(a, b, "SUM")

    def prod(self, a, b):
        return self.ew(a, b, "PROD")

    def mm(self, a, b, transpose=True):
        op = self.t.MatrixOperation
        return self.n.add_matrix_multiply(
            a, op.NONE, b, op.TRANSPOSE if transpose else op.NONE
        ).get_output(0)

    def linear(self, x, key):
        prefix = "lilicorr." + key
        x = self.fp16(x)
        if prefix + ".weight_scale" in self.w.index:
            value = self.w4a16(x, prefix)
        else:
            value = self.mm(x, self.const(self.weight(key + ".weight")))
        return self.f(self.sum(value, self.const(self.weight(key + ".bias").reshape(1, -1))))

    def rms(self, x, key, eps):
        var = self.reduce(self.prod(x, x), "AVG", 1)
        denom = self.n.add_unary(
            self.sum(var, self.scalar(eps, 2)), self.t.UnaryOperation.SQRT
        ).get_output(0)
        w = self.weight(key + ".weight").astype(np.float32)
        return self.prod(self.ew(x, denom, "DIV"), self.const(w[None, :]))

    def unit(self, x, eps):
        energy = self.reduce(self.prod(x, x), "SUM", 1)
        norm = self.n.add_unary(energy, self.t.UnaryOperation.SQRT).get_output(0)
        return self.ew(x, self.ew(norm, self.scalar(eps, 2), "MAX"), "DIV")

    def mlp(self, x, k):
        return self.linear(self.silu(self.linear(x, k + ".up_proj")), k + ".down_proj")

    def lattice_scores(self, c):
        n, t = self.n, self.t
        spec = c["dflash_config"]
        S, K, H, D = (
            c["block_size"] - 1,
            spec["lilicorr_candidate_topk"],
            spec["lilicorr_hidden_size"],
            c["hidden_size"],
        )
        L, heads = S * K, spec["lilicorr_num_heads"]
        e = c["rms_norm_eps"]
        embeddings = n.add_input("candidate_embeddings", t.float16, (L, D))
        lp = n.add_input("candidate_log_probs", t.float32, (S, K))
        ph = n.add_input("pass_hidden", t.float16, (S, D))
        anchor = n.add_input("anchor_hidden", t.float32, (1, D))
        valid = n.add_input("anchor_valid", t.int32, (1, 1))
        valid = self.ew(valid, self.const(np.zeros((1, 1), np.int32)), "GREATER")
        rank = np.broadcast_to(np.arange(K, dtype=np.float32)[None, :], (S, K)).reshape(L, 1)
        cols = [
            self.sh(x, L, 1)
            for x in (
                lp,
                n.add_unary(lp, t.UnaryOperation.EXP).get_output(0),
                self.ew(lp, self.reduce(lp, "MAX", 1), "SUB"),
            )
        ]
        cat = n.add_concatenation(
            cols + [self.const(rank / (K - 1)), self.const((rank == 0).astype(np.float32))]
        )
        cat.axis = 1
        feats = cat.get_output(0)
        centered = self.ew(feats, self.reduce(feats, "AVG", 1), "SUB")
        var = self.reduce(self.prod(centered, centered), "AVG", 1)
        denom = n.add_unary(self.sum(var, self.scalar(1e-5, 2)), t.UnaryOperation.SQRT).get_output(
            0
        )
        x = self.ew(centered, denom, "DIV")
        x = self.sum(
            self.prod(x, self.const(self.weight("feature_norm.weight")[None].astype(np.float32))),
            self.const(self.weight("feature_norm.bias")[None].astype(np.float32)),
        )
        token = self.sh(self.linear(embeddings, "token_proj"), S, K, H)
        token = self.sum(token, self.sh(self.linear(ph, "pass_hidden_proj"), S, 1, H))
        token = self.sum(token, self.sh(self.mlp(x, "feature_mlp"), S, K, H))
        for k, shape in [("slot_embedding", (S, 1, H)), ("rank_embedding", (1, K, H))]:
            token = self.sum(token, self.const(self.weight(k).astype(np.float32).reshape(shape)))
        x = self.sh(token, L, H)
        slots = np.arange(L) // K
        relative = self.weight("relative_slot_bias").astype(np.float32)
        same = self.weight("same_slot_bias").astype(np.float32)
        bias = self.const(
            relative[:, slots[:, None] - slots[None, :] + S]
            + same[:, None, None] * (slots[:, None] == slots[None, :])
        )
        for i in range(spec["lilicorr_num_layers"]):
            k = f"layers.{i}."
            a = self.rms(x, k + "attn_norm", e)
            qkv = self.mm(self.fp16(a), self.const(self.weight(k + "attn.in_proj_weight")))
            qkv = self.f(self.sum(qkv, self.const(self.weight(k + "attn.in_proj_bias")[None])))
            vals = []
            for j in range(3):
                v = n.add_slice(qkv, (0, j * H), (L, H), (1, 1)).get_output(0)
                vals.append(self.transpose(self.sh(v, L, heads, H // heads), (1, 0, 2)))
            q, key, v = vals
            logits = self.sum(
                self.prod(self.mm(q, key), self.scalar((H // heads) ** -0.5, 3)), bias
            )
            sm = n.add_softmax(logits)
            sm.axes = 4
            a = self.sh(self.transpose(self.mm(sm.get_output(0), v, False), (1, 0, 2)), L, H)
            x = self.sum(x, self.linear(a, k + "attn.out_proj"))
            x = self.sum(x, self.mlp(self.rms(x, k + "mlp_norm", e), k + "mlp"))
        x = self.rms(x, "output_norm", e)
        a = self.linear(anchor, "context_proj")
        a = n.add_select(valid, a, self.const(np.zeros((1, H), np.float32))).get_output(0)
        a = self.rms(a, "anchor_norm", e)
        weight = self.weight("factor_input_proj.weight")
        f = self.f(
            self.sum(
                self.mm(self.fp16(x), self.const(weight[:, :H])),
                self.const(self.weight("factor_input_proj.bias")[None]),
            )
        )
        f = self.sum(f, self.f(self.mm(self.fp16(a), self.const(weight[:, H : 2 * H]))))
        f = self.sum(f, self.f(self.mm(self.fp16(self.prod(x, a)), self.const(weight[:, 2 * H :]))))
        f = self.silu(f)
        eps = spec["lilicorr_vector_eps"]
        F = spec["lilicorr_factor_dim"]
        u = self.sh(self.unit(self.linear(f, "out_head"), eps), S, K, F)
        v = self.sh(self.unit(self.linear(f, "in_head"), eps), S, K, F)
        av = self.unit(self.linear(a, "anchor_out_head"), eps)
        first = n.add_slice(v, (0, 0, 0), (1, K, F), (1, 1, 1)).get_output(0)
        start = self.mm(self.sh(first, K, F), av)
        start = self.sum(self.sh(start, 1, 1, K), self.const(np.zeros((1, K, 1), np.float32)))
        prev = n.add_slice(u, (0, 0, 0), (S - 1, K, F), (1, 1, 1)).get_output(0)
        nxt = n.add_slice(v, (1, 0, 0), (S - 1, K, F), (1, 1, 1)).get_output(0)
        joined = n.add_concatenation([start, self.mm(prev, nxt)])
        joined.axis = 0
        self.output(
            self.prod(joined.get_output(0), self.scalar(spec["lilicorr_logit_scale"], 3)),
            "transition_scores",
        )


def build_lattice(weights, config, *, logger):
    import tensorrt as trt

    b = trt.Builder(logger)
    n = b.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    graph = Lattice(n, weights)
    graph.lattice_scores(config)
    bc = b.create_builder_config()
    bc.builder_optimization_level = 3
    bc.clear_flag(trt.BuilderFlag.TF32)
    plan = b.build_serialized_network(n, bc)
    if plan is None:
        raise RuntimeError("TensorRT could not build Bolt LiLiCorr lattice")
    return plan
