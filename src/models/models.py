"""
Multimodal action recognition models.

@author Barun Das <barun.das.23@gmail.com>
"""

from typing import Iterable
import torch
from torch import nn as nn
from sentence_transformers import SentenceTransformer
from constants import (
    MULTIMODAL_FEATURE_SIZE,
    NUM_NOUNS,
    NUM_VERBS,
    SENTENCE_TRANSFORMER_MODEL,
    WORD_EMBEDDING_SIZE,
)
from utils import get_loggers, strip_model_prefix, vector_gather
from models.tsm import TSM

LOG = get_loggers(__name__)


# WORD EMBEDDINGS
class WordEmbeddings(nn.Module):
    """ Compute sentence embeddings using Sentence Transformers
    """
    def __init__(self, embed_size=WORD_EMBEDDING_SIZE, device="cpu") -> None:
        super().__init__()
        self.device = device
        self.model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
        if embed_size != WORD_EMBEDDING_SIZE:
            self.fc_out = nn.Linear(WORD_EMBEDDING_SIZE, embed_size)
        else:
            self.fc_out = None

    def forward(self, text):
        embeddings = self.model.encode(text, convert_to_tensor=True)
        return embeddings if self.fc_out is None else self.fc_out(embeddings)
        # return torch.from_numpy(embeddings)


class EmbeddingModel(nn.Module):
    def __init__(
        self,
        cfg,
        dropout,
        device,
        embed_size=WORD_EMBEDDING_SIZE,
        n_conv=[100],
        dim_size=MULTIMODAL_FEATURE_SIZE,
    ):
        super().__init__()
        self.device = device
        self.cfg = cfg
        self.narration_model = self._load_narration_model()
        self.rgb_model = self._load_vision_model(modality="rgb")
        self.flow_model = self._load_vision_model(modality="flow")

        # if len(bn) > 0:
        #     layers = [
        #         [
        #             nn.Conv1d(ni, nc, kf, bias=False if i in bn else True),
        #             nn.ReLU(),
        #             nn.Dropout(dropout),
        #         ]
        #         for i, (ni, nc, kf) in enumerate(zip(in_feat_dim, n_conv, k_sizes))
        #     ]
        #     for i in bn:
        #         layers[i].insert(2, nn.BatchNorm2d(n_conv[i]))

        # else:
        #     layers = [
        #         [nn.Conv1d(ni, nc, kf, bias=True), nn.ReLU(), nn.Dropout(dropout),]
        #         for ni, nc, kf in zip(in_feat_dim, n_conv, k_sizes)
        #     ]

        # self.linear_layer = nn.Sequential(*layers)

        # for nc, ks in zip(n_conv, k_sizes):
        #     layers.append()

        self.linear_layer = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(8, 32, 5, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout),
            nn.Conv1d(32, 10, 3, bias=True),
            nn.ReLU(),
            nn.Linear(dim_size, embed_size),
        )

    def _load_vision_model(self, modality, output_dim=0):
        assert (
            self.cfg.model.type == "TSM"
        ), f"Unknown model type {self.cfg.model_type}. Only TSM models have been configured"
        model = TSM(
            num_class=output_dim if output_dim > 0 else self.cfg.model.num_class,
            num_segments=self.cfg.data.frame_count,
            modality=modality,
            base_model=self.cfg.model.backbone,
            segment_length=self.cfg["data"][modality]["segment_length"],
            consensus_type="avg",
            dropout=self.cfg.model.dropout,
            partial_bn=self.cfg.model.partial_bn,
            pretrained=self.cfg.model.get("pretrained", None),
            shift_div=self.cfg.model.shift_div,
            non_local=self.cfg.model.non_local,
            temporal_pool=self.cfg.model.temporal_pool,
            freeze_train_layers=self.cfg.model.freeze_pretrain_layers,
        )

        if self.cfg.model.get("weights", None) is not None:
            if self.cfg.model.pretrained is not None:
                LOG.warning(
                    f"model.pretrained was set to {self.cfg.model.pretrained!r} but "
                    f"you also specified to load weights from {self.cfg.model.weights}."
                    "The latter will take precedence."
                )
            weight_loc = self.cfg.model.weights[modality]
            LOG.info(f"Loading weights from {weight_loc}")
            state_dict = torch.load(weight_loc, map_location=torch.device("cpu"))
            if "state_dict" in state_dict:
                # Person is trying to load a checkpoint with a state_dict key, so we pull
                # that out.
                LOG.info("Stripping 'model' prefix from pretrained state_dict keys")
                sd = strip_model_prefix(state_dict["state_dict"])
                # Change shape of final linear layer
                sd["new_fc.weight"] = torch.rand([1024, 2048], requires_grad=True)
                sd["new_fc.bias"] = torch.rand(1024, requires_grad=True)
                missing, unexpected = model.load_state_dict(sd, strict=False)
                if len(missing) > 0:
                    LOG.warning("Missing keys in checkpoint: %s", missing)
                if len(unexpected) > 0:
                    LOG.warning("Unexpected keys in checkpoint: %s", unexpected)

        return model

    def _load_narration_model(self):
        model = WordEmbeddings(device=self.device)
        narr_cfg = self.cfg.model.get("narration_model", False)
        if narr_cfg and narr_cfg.get("pretrained", False):
            LOG.info("Using pretrained WordEmbeddings")
            for param in model.parameters():
                param.requires_grad = False

        return model

    def forward(self, rgb, flow, narration, permute_dims=True, fp16=True):
        # (rgb, metadata), (flow, _) = x
        # narration = metadata['narration']
        rgb_feats = self.rgb_model(rgb)
        flow_feats = self.flow_model(flow)
        narration_feats = self.narration_model(narration).to(flow_feats.device)
        # narration_feats = torch.unsqueeze(narration_feats, 1).expand(-1, F, -1) # (B, F, word_embedding_dim)
        # print(f'narration_feats = {narration_feats.shape}, rgb_feats = {rgb_feats.shape}, flow_feats = {flow_feats.shape} ')
        feats = (
            torch.hstack([rgb_feats, flow_feats, narration_feats])
            .contiguous()
            .unsqueeze(1)
        )
        # print('feats shape =', feats.shape)
        if fp16:
            feats = feats.to(torch.float16)
        # feats = feats[:, None, :].to(torch.float32)
        feats = self.linear_layer(feats)
        return feats.permute((0, 2, 1)) if permute_dims else feats


class HaarModel(nn.Module):
    def __init__(
        self,
        cfg,
        action_embeddings,
        dropout,
        device,
        linear_out=(NUM_VERBS, NUM_NOUNS),
        embed_size=WORD_EMBEDDING_SIZE,
        n_conv=10,
    ):
        super().__init__()
        # self.cfg = cfg
        self.num_verbs, self.num_nouns = linear_out
        self.device = device
        self.action_embeddings = action_embeddings
        self.feature_model = EmbeddingModel(cfg, dropout, device, embed_size, n_conv)
        self.transformer = nn.Transformer(
            d_model=cfg.model.transformer.d_model,
            nhead=cfg.model.transformer.nhead,
            batch_first=True,
            dropout=cfg.model.transformer.dropout,
            device=device,
        )
        self.fc_out_verb = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.num_verbs * embed_size, linear_out[0])
        )
        self.fc_out_noun = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.num_nouns * embed_size, linear_out[1])
        )
        self.cls_head = nn.Softmax(dim=-1)

    def _get_target_mask(self, feats):
        trg_len = feats.shape[1]
        trg_mask = torch.tril(torch.ones(trg_len, trg_len))
        return trg_mask.to(feats.device)

    def _get_task_output(self, task, logits):
        assert task in ["default", "inference"], f"invalid task {task}"

        if task == 'default':
            return logits
        elif task == 'inference':
            with torch.no_grad():
                return self.cls_head(logits).argmax(dim=-1)

    def forward(self, x, task='default', fp16=True, verb=True, noun=True):
        ''' Runs the forward pass on a batch of data

        __args__
            x: input batch
            task='default': task to perform. Choose
                between `default` (returns raw logits)
                and `inference` (returns label with 
                highest probability)
            fp16=True: whether to use fp16 or not
            verb=True: whether to compute verb logits
            noun=True: whether to compute noun logits

        '''
        (rgb, metadata), (flow, _) = x
        N = rgb.shape[0]

        feats = self.feature_model(
            rgb, flow, metadata["narration"], permute_dims=False, fp16=fp16
        )  # B, 10, d_model
        
        # batch size must be same for src and tgt
        target = torch.unsqueeze(self.action_embeddings, 0).expand(N, -1, -1).to(feats.device) 
        target_mask = self._get_target_mask(target)
        if fp16:
            target = target.to(torch.float16)
            target_mask = target_mask.to(torch.float16)
        # print(f'feats on {feats.device}, target on {target.device}, target mask on {target_mask.device}') 
        feats = self.transformer(feats, target, tgt_mask=target_mask)
        # feats = feats.reshape((N, -1))
        # print("feats.shape =", feats.shape)

        res_noun = None
        res_verb = None
        if noun:
            noun_logits = self.fc_out_noun(feats[:,  self.num_verbs:, :])
            res_noun = self._get_task_output(task, noun_logits)
        if verb:
            verb_logits = self.fc_out_verb(feats[:, :self.num_verbs, :])
            res_verb = self._get_task_output(task, verb_logits)
        
        return (res_verb, res_noun)


class AttentionModel(nn.Module):
    def __init__(self, word_map):
        super().__init__()
        self.cardinality = len(word_map)
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=100, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(MULTIMODAL_FEATURE_SIZE, WORD_EMBEDDING_SIZE),
        )
        self.linear_layer = nn.Linear(WORD_EMBEDDING_SIZE, self.cardinality, bias=True)
        self.softmax = nn.Softmax(dim=-1)

    def _predictions(self, frame_features, key, embeddings):
        """Takes the frame_features and returns the predictions
        for action (verb/noun)

        ------ Shape logic ------
        f = [b, D, 100]
        w1 = [b, C, D]
        A = w1@f = [b, C, 100]
        Ai = A.T[:, verb, :] = [b, 1, 100]
        F_i = Ai @ f.permute(0, 2, 1) = [b, 1, D]
        P = W2(F_i) = [b, 1, C]
        res = softmax(P) = [b, 1, C]
        ----------------------------

        Args:
            frame_features (torch.Tensor): the multimodal features
            key (torch.Tensor): tensor of verb/noun class indices
                used to collect class-aware attention

        Raises:
            Exception: Invalid mode. It has to be 'verb' or 'noun'

        Returns:
            torch.Tensor: prediction probabilities for each verb/noun
                class
        """
        if embeddings.ndim == 1:
            embeddings = embeddings[:, None]  # Convert to shape [b, K]
        A = torch.sigmoid(
            torch.matmul(embeddings, frame_features)
        )  # shape: [b, C, 100]

        #! class-aware attention should only be done in training, figure out different flow for testing
        #! This is true for training only
        y = None
        if key is not None:
            A = vector_gather(A, key)
            y = torch.einsum("ijk, ik -> ij", frame_features, A)
            y = torch.div(y, torch.sum(A, dim=-1).reshape((-1, 1)))
            y = self.linear_layer(y)
            y = self.softmax(y)
        del A
        return y

    def _evaluate(self):
        pass

    def forward(self, x: torch.Tensor, label: int, embeddings):
        x = x[:, None, :].to(torch.float32)
        x = self.layer1(x).permute((0, 2, 1))
        return self._predictions(x, label, embeddings)


"""
(4, 32, 3, 224, 224)  # B, F, C, H, W
(4, 3, 224, 224)      # B, C, H, W  temporal pooling of features
"""
