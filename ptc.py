"""Prototype-guided Text Calibration (PTC).

This module separates PTC from concrete segmentor implementations.
The segmentor only needs to provide a `forward_feature` callback with the
following interface when `return_tokens=True`:

    image_features, query_scores, (grid_h, grid_w), (patch_h, patch_w) = \
        forward_feature(region_img, query_override=None, return_tokens=True)

where:
    image_features: [B, N, D], normalized visual token features.
    query_scores:   [B, N, Q], visual-token to original-query similarities.

The returned calibrated query features have shape [B, Q, D] and can be passed
back to the segmentor via its `query_override` argument.
"""


from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypeGuidedTextCalibrator(nn.Module):
    """Training-free PTC module"""
    def __init__(self, query_features, query_idx, forward_feature, *, logit_scale=40.0, slide_stride=112,
                 slide_crop=336,
                 ptc_proto_mode="auto", ptc_proto_crop=224, ptc_mu=0.10, ptc_seed_ratio=0.10, ptc_min_seeds=40,
                 ptc_skip_bg=False, ptc_border=0, ptc_debug=False,
                 pad_divisor=56, allow_crop_without_slide=True):
        super().__init__()

        if ptc_proto_mode not in {"auto", "crop", "image"}:
            raise ValueError(f"ptc_proto_mode must be in {{'auto', 'crop', 'image'}}, got {ptc_proto_mode}")
        if isinstance(ptc_proto_crop, int) and ptc_proto_crop <= 0:
            raise ValueError(f"ptc_proto_crop must be > 0, got {ptc_proto_crop}")
        if ptc_min_seeds <= 0:
            raise ValueError(f"ptc_min_seeds must be > 0, got {ptc_min_seeds}")
        if pad_divisor <= 0:
            raise ValueError(f"pad_divisor must be > 0, got {pad_divisor}")
        self.forward_feature = forward_feature
        self.logit_scale = logit_scale
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop
        self.ptc_proto_mode = ptc_proto_mode
        self.ptc_proto_crop = ptc_proto_crop
        self.ptc_mu = ptc_mu
        self.ptc_seed_ratio = ptc_seed_ratio
        self.ptc_min_seeds = ptc_min_seeds
        self.ptc_skip_bg = ptc_skip_bg
        self.ptc_border = ptc_border
        self.ptc_debug = ptc_debug
        self.pad_divisor = pad_divisor
        self.allow_crop_without_slide = allow_crop_without_slide
        self.ptc_step = 0

        query_features = query_features.detach().clone()
        query_idx = query_idx.detach().long().clone()

        unique_idx = torch.unique(query_idx, sorted=True)
        expected_idx = torch.arange(
            int(query_idx.max().item()) + 1,
            device=query_idx.device,
            dtype=query_idx.dtype,
        )
        if not torch.equal(unique_idx, expected_idx):
            raise ValueError("query_idx must contain contiguous class indices starting from 0.")

        if not bool(torch.all(query_idx[:-1] <= query_idx[1:])):
            raise ValueError("query_idx must be arranged in ascending class order.")

        if (query_features.shape[0] == expected_idx.numel() and not torch.equal(query_idx, expected_idx)):
            raise ValueError("query_idx must follow ascending class order when each class has one query.")

        self.register_buffer("query_features", query_features, persistent=False)
        self.register_buffer("query_idx", query_idx, persistent=False)

        self.num_queries = int(query_features.shape[0])
        self.num_classes = int(query_idx.max().item()) + 1

        _ = self.resolve_ptc_proto_mode(raise_error=True)

    def set_query_features(self, query_features):
        if query_features.shape != self.query_features.shape:
            raise ValueError(
                f"query_features shape mismatch: expected {tuple(self.query_features.shape)}, "
                f"got {tuple(query_features.shape)}"
            )

        query_features = query_features.detach().to(
            device=self.query_features.device,
            dtype=self.query_features.dtype
        )
        self.query_features.copy_(query_features)

    def resolve_ptc_proto_mode(self, raise_error=True):
        """Resolve how PTC collects visual evidence.

        - auto: uses crop prototypes for sliding inference;
          full-image inference -> image prototypes.
        - crop: uses fixed local windows for prototype construction.
        - image: uses the full image for prototype construction.
        """
        mode = self.ptc_proto_mode

        if mode == "auto":
            if self.slide_crop > 0:
                return "crop"
            return "image"

        if mode == "crop" and self.slide_crop <= 0 and not self.allow_crop_without_slide:
            if raise_error:
                raise ValueError(
                    "ptc_proto_mode='crop' requires slide_crop > 0 when "
                    "allow_crop_without_slide=False."
                )
            return None

        return mode

    @torch.no_grad()
    def forward(self, img, return_info=False):
        """Build calibrated query features for a batch of images."""
        mode = self.resolve_ptc_proto_mode(raise_error=True)

        if mode == "image":
            return self.ptc_build_calibrated_queries(img, mode="image", return_info=return_info)

        if mode == "crop":
            return self.ptc_build_calibrated_queries(img, mode="crop", stride=self.slide_stride,
                                                     ptc_proto_crop=self.ptc_proto_crop,
                                                     return_info=return_info)

        raise ValueError(f"Unsupported resolved proto mode: {mode}")

    @torch.no_grad()
    def ptc_build_calibrated_queries(self, img, *, mode, stride=None, ptc_proto_crop=None, return_info=False):
        """Construct class prototypes and return calibrated query features.

        Returns:
            Tensor [B, Q, D] if return_info=False.
            (Tensor [B, Q, D], list[dict]) if return_info=True.
        """
        if isinstance(img, list):
            img = img[0].unsqueeze(0)

        batch_size = int(img.shape[0])
        D = int(self.query_features.shape[1])

        calibrated_batch = []
        info_batch = []

        for b in range(batch_size):
            img_one = img[b:b + 1]
            cls_fea_sum = torch.zeros((self.num_classes, D), device=img.device, dtype=torch.float32)
            cls_reg_count = torch.zeros((self.num_classes,), device=img.device, dtype=torch.float32)
            cls_seed_count = torch.zeros((self.num_classes,), device=img.device, dtype=torch.int32)

            regions = list(
                self.ptc_iter_proto_regions(img_one, mode=mode, stride=stride, ptc_proto_crop=ptc_proto_crop))
            num_regions = len(regions)

            for region_img in regions:
                self.ptc_accumulate_proto_for_region(
                    region_img,
                    cls_fea_sum,
                    cls_reg_count,
                    cls_seed_count,
                )

            calibrated = self.ptc_calibrate_queries_with_prototypes(
                cls_fea_sum,
                cls_reg_count,
                cls_seed_count,
                num_crops=num_regions,
                return_info=return_info,
            )
            if return_info:
                calibrated, info = calibrated
                info_batch.append(info)
            calibrated_batch.append(calibrated)

        calibrated_batch = torch.cat(calibrated_batch, dim=0).detach()
        if return_info:
            return calibrated_batch, info_batch
        return calibrated_batch

    def ptc_iter_proto_regions(self, img_one, *, mode, stride=None, ptc_proto_crop=None):
        """Yield image regions used for visual-prototype construction."""
        if mode == "image":
            yield img_one
            return

        if stride is None:
            stride = self.slide_stride
        if ptc_proto_crop is None:
            ptc_proto_crop = self.ptc_proto_crop

        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(ptc_proto_crop, int):
            ptc_proto_crop = (ptc_proto_crop, ptc_proto_crop)

        h_stride, w_stride = stride
        proto_h_crop, proto_w_crop = ptc_proto_crop
        _, _, h_img, w_img = img_one.shape
        proto_h_crop = min(int(proto_h_crop), int(h_img))
        proto_w_crop = min(int(proto_w_crop), int(w_img))

        h_grids = max(h_img - proto_h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - proto_w_crop + w_stride - 1, 0) // w_stride + 1

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + proto_h_crop, h_img)
                x2 = min(x1 + proto_w_crop, w_img)
                y1 = max(y2 - proto_h_crop, 0)
                x1 = max(x2 - proto_w_crop, 0)
                yield img_one[:, :, y1:y2, x1:x2]

    def ptc_accumulate_proto_for_region(self, region_img, cls_fea_sum, cls_reg_count,
                                        cls_seed_count):
        """Extract visual evidence from one region and update prototype statistics."""
        H0, W0 = region_img.shape[2:]
        pad = self.compute_padsize(H0, W0, self.pad_divisor)
        region_pad = F.pad(region_img, pad) if any(pad) else region_img

        feats, query_scores, (I, J), (ph, pw) = self.forward_feature(
            region_pad,
            query_override=None,
            return_tokens=True,
        )

        valid = self.ptc_token_filter(I, J, ph, pw, pad, H0, W0, device=feats.device)
        self.ptc_accumulate_proto_stats(
            cls_fea_sum, cls_reg_count, cls_seed_count,
            feats, query_scores, valid)

    def ptc_accumulate_proto_stats(self, cls_fea_sum, cls_reg_count, cls_seed_count,
                                   image_features, query_scores, valid_mask):
        """Select per-class evidence tokens by score margin and accumulate crop-level prototypes."""

        img_feat = image_features[0]
        query_score = query_scores[0]
        if valid_mask is not None:
            img_feat = img_feat[valid_mask]
            query_score = query_score[valid_mask]

        img_feat = img_feat.float()
        query_score = query_score.float()

        cls_scores = self.query_to_cls_scores(query_score.unsqueeze(0))[0]
        cls_scores = cls_scores * self.logit_scale

        pred_c = cls_scores.argmax(dim=-1)
        if self.num_classes >= 2:
            top2 = cls_scores.topk(k=2, dim=-1).values
            margin = (top2[:, 0] - top2[:, 1]).clamp_min(0.0)
        else:
            margin = cls_scores[:, 0].clamp_min(0.0)

        for c in range(self.num_classes):
            if self.ptc_skip_bg and c == 0:
                continue

            cls_mask = pred_c == c
            n_cls = int(cls_mask.sum().item())
            if n_cls == 0:
                continue

            cls_idx = cls_mask.nonzero(as_tuple=False).squeeze(1)

            target_k = max(self.ptc_min_seeds, int(n_cls * self.ptc_seed_ratio))
            target_k = min(n_cls, target_k)

            cls_rank_score = margin[cls_mask]
            top_local = torch.topk(cls_rank_score, k=target_k, largest=True).indices
            seed_idx = cls_idx[top_local]

            seed_mask = torch.zeros_like(cls_mask)
            seed_mask[seed_idx] = True

            n_seed = int(seed_mask.sum().item())
            if n_seed < self.ptc_min_seeds:
                continue

            selected_margins = margin[seed_mask]
            margin_sum = selected_margins.sum()
            if margin_sum <= 1e-6:
                continue
            seed_weights = selected_margins / margin_sum
            cls_fea_sum[c] += (img_feat[seed_mask] * seed_weights.unsqueeze(-1)).sum(dim=0)
            cls_reg_count[c] += 1.0
            cls_seed_count[c] += n_seed

    def ptc_calibrate_queries_with_prototypes(self, cls_fea_sum, cls_reg_count, cls_seed_count, *,
                                              num_crops, return_info=False):
        """Generate prototypes and calibrate the corresponding query features"""
        base_q = self.query_features.to(device=cls_fea_sum.device)
        query_idx = self.query_idx.to(device=cls_fea_sum.device)

        calibrated_query_features = base_q.unsqueeze(0).clone()
        D = int(base_q.shape[1])
        class_prototypes = torch.zeros((self.num_classes, D), device=base_q.device, dtype=torch.float32)

        # When only one crop is available, allow single-crop support
        if num_crops == 1:
            min_crops = 1
        else:
            # When there are many crops or classes, require cross-crop support to avoid constructing semantically incorrect prototypes
            compact_regime = (num_crops <= 16) and (self.num_classes <= 30)
            min_crops = 1 if compact_regime else 2

        ptc_min_seeds = self.ptc_min_seeds
        valid_classes = (cls_reg_count >= min_crops) & (cls_seed_count >= ptc_min_seeds)

        if self.ptc_skip_bg and self.num_classes > 0:
            valid_classes = valid_classes.clone()
            valid_classes[0] = False

        # Average valid crop-level prototypes and normalize the resulting image-level prototype.
        if bool(valid_classes.any()):
            class_prototypes[valid_classes] = (
                    cls_fea_sum[valid_classes] / (cls_reg_count[valid_classes].unsqueeze(-1) + 1e-6)
            )
            class_prototypes[valid_classes] = F.normalize(class_prototypes[valid_classes], dim=-1)

        for c in range(self.num_classes):
            if not bool(valid_classes[c].item()):
                continue

            class_query_mask = query_idx == c
            if not bool(class_query_mask.any()):
                continue

            # Adapt the calibration strength to the total amount of evidence.
            n = float(cls_seed_count[c].item())
            denom = float(ptc_min_seeds) * 10.0
            rel = min(1.0, math.log1p(n) / math.log1p(denom))
            mu_c = self.ptc_mu * rel

            class_query_features = calibrated_query_features[0, class_query_mask].float()
            class_query_features = (
                    (1.0 - mu_c) * class_query_features + mu_c * class_prototypes[c].unsqueeze(0)
            )
            calibrated_query_features[0, class_query_mask] = F.normalize(
                class_query_features, dim=-1
            ).to(base_q.dtype)

        need_info = return_info or self.ptc_debug
        if not need_info:
            return calibrated_query_features

        ptc_info = {
            "updated_classes": int(valid_classes.sum().item()),
            "avg_seed": (
                float(cls_seed_count[valid_classes].float().mean().item())
                if bool(valid_classes.any()) else 0.0
            ),
            "delta_q": float((calibrated_query_features - base_q.unsqueeze(0)).abs().mean().item()),
            "min_crops": int(min_crops),
            "ptc_min_seeds": int(ptc_min_seeds),
        }

        if self.ptc_debug:
            self.ptc_log_info("PTC-anchor", ptc_info)

        if return_info:
            return calibrated_query_features, ptc_info

        return calibrated_query_features

    def query_to_cls_scores(self, query_scores):
        """Merge query-level scores into class-level scores by max pooling."""
        if self.num_classes == self.num_queries:
            return query_scores

        query_idx = self.query_idx.to(device=query_scores.device)
        cls_score_list = []
        for c in range(self.num_classes):
            qmask = query_idx == c
            cls_score_list.append(query_scores[..., qmask].max(dim=-1).values)
        return torch.stack(cls_score_list, dim=-1)

    def ptc_token_filter(self, I, J, ph, pw, pad, H, W, *, device):
        """Mask out padded and optional border tokens before evidence selection."""
        l, _, t, _ = pad
        l0, t0 = l // pw, t // ph
        I0, J0 = H // ph, W // pw

        y1, y2 = t0, t0 + I0
        x1, x2 = l0, l0 + J0

        if self.ptc_border is None:
            bd = int(round(16.0 / float(min(ph, pw))))
            bd = max(0, min(2, bd))
        else:
            bd = int(self.ptc_border)

        yi1, yi2 = y1 + bd, y2 - bd
        xi1, xi2 = x1 + bd, x2 - bd
        if yi2 <= yi1 or xi2 <= xi1:
            yi1, yi2, xi1, xi2 = y1, y2, x1, x2

        mask = torch.zeros((I, J), device=device, dtype=torch.bool)
        mask[yi1:yi2, xi1:xi2] = True
        return mask.view(-1)

    @staticmethod
    def compute_padsize(H, W, patch_size):
        l, r, t, b = 0, 0, 0, 0
        if W % patch_size:
            lr = patch_size - (W % patch_size)
            l = lr // 2
            r = lr - l
        if H % patch_size:
            tb = patch_size - (H % patch_size)
            t = tb // 2
            b = tb - t
        return l, r, t, b

    def ptc_log_info(self, tag, ptc_info):
        if not self.ptc_debug:
            return
        self.ptc_step += 1
        if self.ptc_step % 100 == 0:
            print(
                f"[{tag}] step={self.ptc_step} "
                f"updated_classes={ptc_info['updated_classes']}/{self.num_classes}, "
                f"min_crops={ptc_info['min_crops']}, "
                f"avg_seed={ptc_info['avg_seed']:.2f}, "
                f"delta_q={ptc_info['delta_q']:.6f}"
            )