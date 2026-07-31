import torch
import torch.nn as nn
import sys

sys.path.append("..")

from prompts.imagenet_template import openai_imagenet_template, sub_imagenet_template

from mmseg.models.segmentors import BaseSegmentor
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmengine.structures import PixelData

from mmseg.registry import MODELS

from torchvision import transforms
import torch.nn.functional as F
from einops import rearrange

from open_clip import create_model, tokenizer
from segment_anything import sam_model_registry

from myutils import UnNormalize
from ptc import PrototypeGuidedTextCalibrator


@MODELS.register_module()
class PTCSegmentation(BaseSegmentor):
    def __init__(self, clip_type, model_type, vfm_model, name_path, checkpoint=None, device=torch.device('cuda'),
                 prob_thd=0.0, logit_scale=40, beta=1.2, gamma=3.0, slide_stride=112, slide_crop=336,

                 # ---- PTC configuration ----
                 ptc_enable=False,
                 ptc_proto_mode='auto',  # 'auto' | 'crop' | 'image'
                 ptc_proto_crop=224,
                 ptc_mu=0.1,
                 ptc_seed_ratio=0.10,
                 ptc_min_seeds=40,
                 ptc_skip_bg=False,
                 ptc_border=0,
                 ptc_debug=False):

        data_preprocessor = SegDataPreProcessor(
            mean=[122.771, 116.746, 104.094],
            std=[68.501, 66.632, 70.323],
            bgr_to_rgb=True
        )
        super().__init__(data_preprocessor=data_preprocessor)

        self.clip = create_model(model_type, pretrained=clip_type, precision='fp16')
        self.clip.eval().to(device)
        self.tokenizer = tokenizer.tokenize

        self.vfm_model = vfm_model
        if vfm_model == 'sam':
            self.vfm = sam_model_registry["vit_b"](checkpoint=checkpoint)

        elif vfm_model == 'dino':
            self.vfm = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')

        elif vfm_model == 'dinov2':
            self.vfm = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')

        self.vfm = self.vfm.half()
        for p in self.vfm.parameters():
            p.requires_grad = False
        self.vfm.eval().to(device)

        self.unnorm = UnNormalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        self.norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        query_words, self.query_idx = get_cls_idx(name_path)
        self.num_queries = len(query_words)
        self.num_classes = max(self.query_idx) + 1
        self.query_idx = torch.Tensor(self.query_idx).to(torch.int64).to(device)

        query_features = []
        with torch.no_grad():
            for qw in query_words:
                query = self.tokenizer([temp(qw) for temp in openai_imagenet_template]).to(device)
                feature = self.clip.encode_text(query)
                feature /= feature.norm(dim=-1, keepdim=True)
                feature = feature.mean(dim=0)
                feature /= feature.norm()
                query_features.append(feature.unsqueeze(0))
        self.query_features = torch.cat(query_features, dim=0).detach()

        self.dtype = self.query_features.dtype
        self.logit_scale = logit_scale
        self.prob_thd = prob_thd
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop
        self.beta = beta
        self.gamma = gamma

        # PTC calibrates text features without changing ProxyCLIP feature extraction.
        self.ptc_enable = bool(ptc_enable)
        self.ptc = None
        if self.ptc_enable:
            self.ptc = PrototypeGuidedTextCalibrator(
                self.query_features,
                self.query_idx,
                self.forward_feature,
                logit_scale=self.logit_scale,
                slide_stride=self.slide_stride,
                slide_crop=self.slide_crop,
                ptc_proto_mode=ptc_proto_mode,
                ptc_proto_crop=ptc_proto_crop,
                ptc_mu=ptc_mu,
                ptc_seed_ratio=ptc_seed_ratio,
                ptc_min_seeds=ptc_min_seeds,
                ptc_skip_bg=ptc_skip_bg,
                ptc_border=ptc_border,
                ptc_debug=ptc_debug,
                pad_divisor=56,
                allow_crop_without_slide=True,
            )
            if ptc_debug:
                print(
                    f"[PTC-init] enable={self.ptc_enable}, "
                    f"min_seeds={self.ptc.ptc_min_seeds}, "
                    f"seed_ratio={self.ptc.ptc_seed_ratio}, "
                    f"mu={self.ptc.ptc_mu}"
                )

    @torch.no_grad()
    def forward_feature(self, img, logit_size=None, query_override=None, return_tokens=False):
        """Extract ProxyCLIP token features and compute query logits.

        PTC-related arguments only affect the text matching head:
        - return_tokens=True returns normalized visual tokens and original query scores
          for prototype construction.
        - query_override replaces the original query features only when computing
          final logits. The visual feature propagation path remains unchanged.
        """
        if type(img) == list:
            img = img[0]

        clip_token_size = (img.shape[-2] // self.clip.visual.patch_size[0], img.shape[-1] // self.clip.visual.patch_size[1])

        imgs_norm = [self.norm(self.unnorm(img[i])) for i in range(len(img))]
        imgs_norm = torch.stack(imgs_norm, dim=0)

        imgs_norm = imgs_norm.half()

        if self.vfm_model == 'sam':
            patch_size = self.vfm.image_encoder.patch_embed.proj.kernel_size
            imgs_norm = F.interpolate(imgs_norm, size=(1024, 1024), mode='bilinear', align_corners=False)
            I, J = imgs_norm.shape[-2] // patch_size[0], imgs_norm.shape[-2] // patch_size[1]
            ex_feats = self.vfm.image_encoder(imgs_norm)

        elif self.vfm_model == 'dino':
            feat_out = {}

            def hook_fn_forward_qkv(module, input, output):
                feat_out["qkv"] = output

            handle = self.vfm._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(
                hook_fn_forward_qkv)

            feat = self.vfm.get_intermediate_layers(imgs_norm)[0]
            handle.remove()

            nb_im = feat.shape[0]
            nb_tokens = feat.shape[1]
            nh = self.vfm.blocks[0].attn.num_heads

            qkv = (
                feat_out["qkv"]
                .reshape(nb_im, nb_tokens, 3, nh, -1 // nh)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv[0], qkv[1], qkv[2]
            k = k.transpose(1, 2).reshape(nb_im, nb_tokens, -1)[:, 1:, :]
            q = q.transpose(1, 2).reshape(nb_im, nb_tokens, -1)[:, 1:, :]
            v = v.transpose(1, 2).reshape(nb_im, nb_tokens, -1)[:, 1:, :]

            patch_size = self.vfm.patch_embed.patch_size
            I, J = imgs_norm[0].shape[-2] // patch_size, imgs_norm[0].shape[-2] // patch_size

            ex_feats = feat[:, 1:, :].reshape(nb_im, I, J, -1).permute(0, 3, 1, 2)

        elif self.vfm_model == 'dinov2':
            patch_size = self.vfm.patch_embed.patch_size
            I, J = imgs_norm.shape[-2] // patch_size[0], imgs_norm.shape[-2] // patch_size[1]
            ex_feats = self.vfm.get_intermediate_layers(imgs_norm, reshape=True)[0]

        else:
            I, J = clip_token_size
            ex_feats = None

        image_features = self.clip.encode_image(
            img.half(),
            external_feats=ex_feats,
            beta=self.beta,
            gamma=self.gamma
        )
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Use original text queries to construct PTC visual evidence.
        query_scores = image_features @ self.query_features.T

        if return_tokens:
            return image_features, query_scores, (I, J), self._get_proxy_patch_size(img, I, J)

        if query_override is not None:
            if query_override.dim() == 2:
                query_override = query_override.unsqueeze(0).expand(image_features.shape[0], -1, -1)
            query_override = query_override.to(device=image_features.device, dtype=image_features.dtype)
            query_override = F.normalize(query_override, dim=-1)
            logits = torch.einsum('bnd,bqd->bnq', image_features, query_override)
        else:
            logits = query_scores

        logits = logits.permute(0, 2, 1).reshape(-1, logits.shape[-1], I, J)

        if logit_size == None:
            logits = nn.functional.interpolate(logits, size=img.shape[-2:], mode='bilinear')
        else:
            logits = nn.functional.interpolate(logits, size=logit_size, mode='bilinear')

        return logits

    @staticmethod
    def _get_proxy_patch_size(img, grid_h, grid_w):
        """Return the image-space stride corresponding to the current token grid.

        ProxyCLIP may reshape logits according to the external VFM grid
        (e.g., DINO, DINOv2, SAM), so this value must follow the
        actual output grid rather than CLIP's native patch size.
        """
        ph = max(1, int(img.shape[-2] // max(1, int(grid_h))))
        pw = max(1, int(img.shape[-1] // max(1, int(grid_w))))
        return ph, pw

    def forward_slide(self, img, img_metas, stride=112, crop_size=224, query_override=None):
        """Inference by sliding-window with overlap.
        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        if type(img) == list:
            img = img[0].unsqueeze(0)
        if type(stride) == int:
            stride = (stride, stride)
        if type(crop_size) == int:
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = img.shape
        out_channels = self.num_queries
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]

                # pad image when (image_size % patch_size != 0)
                H, W = crop_img.shape[2:]  # original image shape
                pad = self.compute_padsize(H, W, 56)

                if any(pad):
                    crop_img = nn.functional.pad(crop_img, pad)  # zero padding
                crop_seg_logit = self.forward_feature(crop_img, query_override=query_override).detach()

                torch.cuda.empty_cache()

                # mask cutting for padded image
                if any(pad):
                    l, t = pad[0], pad[2]
                    crop_seg_logit = crop_seg_logit[:, :, t:t + H, l:l + W]

                preds += nn.functional.pad(crop_seg_logit,
                                           (int(x1), int(preds.shape[3] - x2), int(y1),
                                            int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0

        preds = preds / count_mat
        img_size = img_metas[0]['ori_shape'][:2]
        logits = nn.functional.interpolate(preds, size=img_size, mode='bilinear')

        return logits

    def predict(self, inputs, data_samples):
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                                  dict(
                                      ori_shape=inputs.shape[2:],
                                      img_shape=inputs.shape[2:],
                                      pad_shape=inputs.shape[2:],
                                      padding_size=[0, 0, 0, 0])
                              ] * inputs.shape[0]

        calibrated_query_features = None
        if self.ptc_enable and self.ptc is not None:
            calibrated_query_features = self.ptc(inputs)

        if self.slide_crop > 0:
            seg_logits = self.forward_slide(
                inputs,
                batch_img_metas,
                self.slide_stride,
                self.slide_crop,
                query_override=calibrated_query_features
            )
        else:
            seg_logits = self.forward_feature(
                inputs,
                batch_img_metas[0]['ori_shape'],
                query_override=calibrated_query_features
            )

        return self.postprocess_result(seg_logits, data_samples)

    def postprocess_result(self, seg_logits, data_samples):
        batch_size = seg_logits.shape[0]
        for i in range(batch_size):
            seg_logits = seg_logits[i] * self.logit_scale
            seg_logits = seg_logits.softmax(0)  # n_queries * w * h

            num_cls, num_queries = max(self.query_idx) + 1, len(self.query_idx)
            if num_cls != num_queries:
                seg_logits = seg_logits.unsqueeze(0)
                cls_index = nn.functional.one_hot(self.query_idx)
                cls_index = cls_index.T.view(num_cls, num_queries, 1, 1)
                seg_logits = (seg_logits * cls_index).max(1)[0]

            seg_pred = seg_logits.argmax(0, keepdim=True)
            seg_pred[seg_logits.max(0, keepdim=True)[0] < self.prob_thd] = 0

            if data_samples is None:
                return seg_pred
            else:
                data_samples[i].set_data({
                    'seg_logits':
                        PixelData(**{'data': seg_logits}),
                    'pred_sem_seg':
                        PixelData(**{'data': seg_pred})
                })
        return data_samples

    def compute_padsize(self, H: int, W: int, patch_size: int):
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

    def _forward(data_samples):
        """
        """

    def inference(self, img, batch_img_metas):
        """
        """

    def encode_decode(self, inputs, batch_img_metas):
        """
        """

    def extract_feat(self, inputs):
        """
        """

    def loss(self, inputs, data_samples):
        """
        """


def get_cls_idx(path):
    with open(path, 'r') as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)

    class_names, class_indices = [], []
    for idx in range(num_cls):
        names_i = name_sets[idx].split('; ')
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace('\n', '') for item in class_names]
    return class_names, class_indices