import copy
import functools
import math

import torch
from torch import Tensor, nn

from models.bricks.misc import Conv2dNormActivation
from models.bricks.base_transformer import TwostageTransformer
from models.bricks.basic import MLP
from models.bricks.ms_deform_attn import MultiScaleDeformableAttention
from models.bricks.position_encoding import get_sine_pos_embed_morph, get_sine_pos_embed
from util.misc import inverse_sigmoid
import torch.nn.functional as F

class MorphologyAwarePositionalEncoding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        # 形态梯度计算（示例：膨胀与腐蚀的差）
        self.morph_grad = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.AvgPool2d(3, stride=1, padding=1),
        )
        # 结构对称性（水平翻转相关）
        # 主方向（使用PCA简化）
        self.orientation_conv = nn.Conv2d(embed_dim, 2, kernel_size=3, padding=1)
        
    def compute_morphology_map(self, x):
        # 形态梯度
        dilated = self.morph_grad[0](x)
        eroded = self.morph_grad[1](x)
        grad = dilated - eroded
        
        # 对称性（水平翻转相关）
        flipped = torch.flip(x, [3])
        sym = x * flipped
        
        # 主方向（使用卷积模拟方向响应）
        orient = self.orientation_conv(x)
        
        # 拼接特征
        morph_map = torch.cat([grad, sym, orient], dim=1)
        return morph_map
    
    def forward(self, features):
        pe = []
        for f in features:
            b, c, h, w = f.shape
            
            # 确保特征图尺寸有效
            assert h >= 1 and w >= 1, f"无效的特征图尺寸：h={h}, w={w}"
            
            # 计算形态响应坐标（修正版）
            with torch.no_grad():
                # 安全生成网格坐标
                y_coord = torch.linspace(0, 1, h, device=f.device).view(-1, 1)  # [h, 1]
                x_coord = torch.linspace(0, 1, w, device=f.device).view(1, -1)   # [1, w]
                
                # 计算形态权重（避免除零）
                morph_weight = f.mean(dim=1, keepdim=True)  # [b, 1, h, w]
                morph_weight = morph_weight - morph_weight.min()
                morph_weight = morph_weight / (morph_weight.max() + 1e-6)
                
                # 加权平均坐标
                y_pos = (morph_weight * y_coord).sum(dim=(2,3))  # [b, 1]
                x_pos = (morph_weight * x_coord).sum(dim=(2,3))  # [b, 1]
            
            # 安全生成位置编码
            grid_y = get_sine_pos_embed_morph(y_pos, self.embed_dim//2)  # [b, 1, d/2]
            grid_x = get_sine_pos_embed_morph(x_pos, self.embed_dim//2)  # [b, 1, d/2]
            pe.append(torch.cat([grid_y, grid_x], dim=-1))
        
        return pe

class TopologyAwareAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.lambda_param = nn.Parameter(torch.tensor(0.1))
        # 修改投影网络结构
        self.topology_proj = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_heads)
        )
    
    def compute_topology_matrix(self, boxes):
        """ boxes: [batch, num_queries, 4] (cx, cy, w, h) """
        centers = boxes[..., :2]
        dist = torch.cdist(centers, centers)  # [b, n, n]
        mean_dist = dist.mean(dim=(1,2), keepdim=True)
        topology = dist / (mean_dist + 1e-6)
        return topology.unsqueeze(1)  # [b, 1, n, n]

    def process_reference_points(self, ref_pts):
        """
        将4D参考点转换为3D格式
        输入: [batch, num_queries, height, width]
        输出: [batch, num_queries, 2] (归一化坐标)
        """
        assert ref_pts.dim() == 4, f"输入应为4D张量，实际得到{ref_pts.dim()}D"
        
        # 方案1：取空间平均值
        if False:  # 根据需求选择
            # 生成网格坐标 (归一化到0-1)
            h, w = ref_pts.shape[-2:]
            y_coord = torch.linspace(0, 1, h, device=ref_pts.device).view(1, 1, -1, 1)
            x_coord = torch.linspace(0, 1, w, device=ref_pts.device).view(1, 1, 1, -1)
            
            # 计算加权平均坐标
            weights = ref_pts.softmax(dim=-1).softmax(dim=-2)  # 双softmax确保空间归一化
            y_pos = (weights * y_coord).sum(dim=(-2, -1))      # [batch, num_queries, 1]
            x_pos = (weights * x_coord).sum(dim=(-2, -1))      # [batch, num_queries, 1]
            return torch.cat([x_pos, y_pos], dim=-1)           # [batch, num_queries, 2]
        
        # 方案2：取最大值位置 (更高效)
        else:
            batch, nq, h, w = ref_pts.shape
            # 展平空间维度并找最大值索引
            flat_pts = ref_pts.view(batch, nq, -1)              # [batch, nq, h*w]
            max_indices = flat_pts.argmax(dim=-1)               # [batch, nq]
            
            # 转换为坐标
            y_coord = (max_indices // w) / (h - 1)              # 归一化y [0,1]
            x_coord = (max_indices % w) / (w - 1)               # 归一化x [0,1]
            return torch.stack([x_coord, y_coord], dim=-1)      # [batch, nq, 2]
    
    def forward(self, attn_scores, reference_points):
        
        # 输入维度检查
        reference_points = self.process_reference_points(reference_points)
        assert reference_points.dim() == 3, f"参考点应为3维，实际得到{reference_points.dim()}"
        batch, num_queries, _ = reference_points.shape
        num_heads = 8
        
        # 转换参考点为bbox格式
        boxes = self.refpoints_to_boxes(reference_points)  # [b, n, 4]
        
        # 计算拓扑矩阵
        topology = self.compute_topology_matrix(boxes)  # [b, 1, n, n]
        
        # 维度调整（关键修正）
        if topology.dim() == 4:
            # 标准情况：[b, 1, n, n] -> [b, n, n, 1]
            topology = topology.permute(0, 2, 3, 1)
        else:
            # 异常情况处理
            raise ValueError(f"意外的拓扑矩阵维度：{topology.shape}")
        
        # 投影得到注意力偏置
        topology_bias = self.topology_proj(topology)  # [b, n, n, h]
        topology_bias = topology_bias.permute(0, 3, 1, 2)  # [b, h, n, n]
        
        # 应用偏置
        output = attn_scores + self.lambda_param * topology_bias.mean(dim=1)
        
        # 根据原始输入形状调整输出
        return output
    
    @staticmethod
    def refpoints_to_boxes(reference_points):
        """ 将参考点转换为伪bbox格式 """
        # 假设reference_points是sigmoid后的归一化坐标 [0,1]范围
        wh = torch.ones_like(reference_points) * 0.05  # 默认小区域
        return torch.cat([reference_points, wh], dim=-1)

class PriorGuidedQuery(nn.Module):
    def __init__(self, embed_dim, num_queries):
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        # 使用轻量级网络预测关键点
        self.conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim//2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim//2, num_queries, kernel_size=1)
        )
        # 可学习的默认Query
        self.default_query = nn.Parameter(torch.randn(1, num_queries, embed_dim))
        
    def forward(self, features):
        if features is None or features[-1] is None:
            return self.default_query.expand(features[0].shape[0], -1, -1)
            
        # 取最高层特征图
        f = features[-1]  # [b, c, h, w]
        b, _, h, w = f.shape
        
        # 生成热图
        heatmap = self.conv(f)  # [b, num_queries, h, w]
        
        # 空间softmax获取概率分布
        heatmap = heatmap.view(b, self.num_queries, -1)  # [b, nq, h*w]
        prob = F.softmax(heatmap, dim=-1)  # [b, nq, h*w]
        
        # 生成坐标网格
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, h, device=f.device),
            torch.linspace(0, 1, w, device=f.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1)  # [h, w, 2]
        grid = grid.view(-1, 2)  # [h*w, 2]
        
        # 计算期望坐标 (修正后的einsum)
        coords = torch.einsum('bnk,kd->bnd', prob, grid)  # [b, nq, 2]
        
        # 坐标转嵌入
        pos_embed = get_sine_pos_embed_morph(coords, self.embed_dim)  # [b, nq, embed_dim]
        return pos_embed


class MorphologyTransformer(TwostageTransformer):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        num_classes: int,
        num_feature_levels: int = 4,
        two_stage_num_proposals: int = 900,
        hybrid_num_proposals: int = 900,
    ):
        super().__init__(num_feature_levels, encoder.embed_dim)
        # model parameters
        self.two_stage_num_proposals = two_stage_num_proposals
        self.hybrid_num_proposals = hybrid_num_proposals
        self.num_classes = num_classes

        # model structure
        self.encoder = encoder
        self.decoder = decoder
        self.tgt_embed = nn.Embedding(two_stage_num_proposals, encoder.embed_dim)
        self.encoder_class_head = nn.Linear(self.embed_dim, num_classes)
        self.encoder_bbox_head = MLP(self.embed_dim, self.embed_dim, 4, 3)
        self.hybrid_tgt_embed = nn.Embedding(hybrid_num_proposals, encoder.embed_dim)
        self.hybrid_class_head = nn.Linear(self.embed_dim, num_classes)
        self.hybrid_bbox_head = MLP(self.embed_dim, self.embed_dim, 4, 3)

        # 形态学模块
        self.mpe = MorphologyAwarePositionalEncoding(encoder.embed_dim)
        self.taa = TopologyAwareAttention(encoder.embed_dim, decoder.num_heads)
        self.pgd = PriorGuidedQuery(encoder.embed_dim, two_stage_num_proposals)
        
        # 初始化占位符
        # self._init_query_embedding(two_stage_num_proposals, hybrid_num_proposals)

        self.init_weights()

    def _init_query_embedding(self, num_queries, hybrid_queries):
        # 标准Query (实际由PGD生成)
        self.register_buffer('tgt_embed', torch.zeros(1, num_queries, self.embed_dim))
        # 混合Query
        self.register_buffer('hybrid_tgt_embed', 
                           torch.zeros(1, hybrid_queries, self.embed_dim))

    def init_weights(self):
        # initialize embedding layers
        nn.init.normal_(self.tgt_embed.weight)
        nn.init.normal_(self.hybrid_tgt_embed.weight)
        # initilize encoder and hybrid classification layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.encoder_class_head.bias, bias_value)
        nn.init.constant_(self.hybrid_class_head.bias, bias_value)
        # initiailize encoder and hybrid regression layers
        nn.init.constant_(self.encoder_bbox_head.layers[-1].weight, 0.0)
        nn.init.constant_(self.encoder_bbox_head.layers[-1].bias, 0.0)
        nn.init.constant_(self.hybrid_bbox_head.layers[-1].weight, 0.0)
        nn.init.constant_(self.hybrid_bbox_head.layers[-1].bias, 0.0)

    def forward(
        self,
        multi_level_feats,
        multi_level_masks,
        multi_level_pos_embeds,
        noised_label_query=None,
        noised_box_query=None,
        attn_mask=None,
    ):


        # 动态生成Query（每次forward更新）
        with torch.no_grad():
            self.tgt_embed.data = self.pgd(multi_level_feats)
            if self.training:
                self.hybrid_tgt_embed.data = self.pgd(multi_level_feats)
        
        # 使用形态位置编码
        morphology_pos = self.mpe(multi_level_feats)
        lvl_pos_embed_flatten = self.get_lvl_pos_embed(morphology_pos)
        
        # 生成先验Query
        # self.tgt_embed = self.pgd(multi_level_feats)

        # get input for encoder
        feat_flatten = self.flatten_multi_level(multi_level_feats)
        mask_flatten = self.flatten_multi_level(multi_level_masks)
        lvl_pos_embed_flatten = self.get_lvl_pos_embed(multi_level_pos_embeds)
        spatial_shapes, level_start_index, valid_ratios = self.multi_level_misc(multi_level_masks)
        reference_points, proposals = self.get_reference(spatial_shapes, valid_ratios)

        # transformer encoder
        memory = self.encoder(
            query=feat_flatten,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reference_points=reference_points,
        )

        # get encoder output, classes and coordinates
        output_memory, output_proposals = self.get_encoder_output(memory, proposals, mask_flatten)
        enc_outputs_class = self.encoder_class_head(output_memory)
        enc_outputs_coord = self.encoder_bbox_head(output_memory) + output_proposals
        enc_outputs_coord = enc_outputs_coord.sigmoid()

        # get topk output classes and coordinates
        topk, num_classes = self.two_stage_num_proposals, self.num_classes
        topk_index = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)[1].unsqueeze(-1)
        enc_outputs_class = enc_outputs_class.gather(1, topk_index.expand(-1, -1, num_classes))
        enc_outputs_coord = enc_outputs_coord.gather(1, topk_index.expand(-1, -1, 4))

        # get target and reference points
        reference_points = enc_outputs_coord.detach()
        target = self.tgt_embed.weight.expand(multi_level_feats[0].shape[0], -1, -1)

        topk = self.hybrid_num_proposals if self.training else 0
        if self.training:
            # get hybrid classes and coordinates, target and reference points
            hybrid_enc_class = self.hybrid_class_head(output_memory)
            hybrid_enc_coord = self.hybrid_bbox_head(output_memory) + output_proposals
            hybrid_enc_coord = hybrid_enc_coord.sigmoid()
            topk_index = torch.topk(hybrid_enc_class.max(-1)[0], topk, dim=1)[1].unsqueeze(-1)
            hybrid_enc_class = hybrid_enc_class.gather(
                1, topk_index.expand(-1, -1, self.num_classes)
            )
            hybrid_enc_coord = hybrid_enc_coord.gather(1, topk_index.expand(-1, -1, 4))
            hybrid_reference_points = hybrid_enc_coord.detach()
            hybrid_target = self.hybrid_tgt_embed.weight.expand(
                multi_level_feats[0].shape[0], -1, -1
            )
        else:
            hybrid_enc_class = None
            hybrid_enc_coord = None

        # combine with noised_label_query and noised_box_query for denoising training
        if noised_label_query is not None and noised_box_query is not None:
            target = torch.cat([noised_label_query, target], 1)
            reference_points = torch.cat([noised_box_query.sigmoid(), reference_points], 1)

        outputs_classes, outputs_coords = self.decoder(
            query=target,
            value=memory,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            attn_mask=attn_mask,
        )

        if self.training:
            hybrid_classes, hybrid_coords = self.decoder(
                query=hybrid_target,
                value=memory,
                key_padding_mask=mask_flatten,
                reference_points=hybrid_reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                skip_relation=True,
            )
        else:
            hybrid_classes = hybrid_coords = None

        return (
            outputs_classes,
            outputs_coords,
            enc_outputs_class,
            enc_outputs_coord,
            hybrid_classes,
            hybrid_coords,
            hybrid_enc_class,
            hybrid_enc_coord,
        )


class MorphologyTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers: int = 6):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.embed_dim = encoder_layer.embed_dim
        self.memory_fusion = nn.Sequential(
            nn.Linear((num_layers + 1) * self.embed_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )

        self.init_weights()

    def init_weights(self):
        # initialize encoder layers
        for layer in self.layers:
            if hasattr(layer, "init_weights"):
                layer.init_weights()

    def forward(
        self,
        query,
        spatial_shapes,
        level_start_index,
        reference_points,
        query_pos=None,
        query_key_padding_mask=None,
    ):
        queries = [query]
        for layer in self.layers:
            query = layer(
                query,
                query_pos,
                reference_points,
                spatial_shapes,
                level_start_index,
                query_key_padding_mask,
            )
            queries.append(query)
        query = torch.cat(queries, -1)
        query = self.memory_fusion(query)
        return query


class MorphologyTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        d_ffn=1024,
        dropout=0.1,
        n_heads=8,
        activation=nn.ReLU(inplace=True),
        n_levels=4,
        n_points=4,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # self attention
        self.self_attn = MultiScaleDeformableAttention(embed_dim, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        # ffn
        self.linear1 = nn.Linear(embed_dim, d_ffn)
        self.activation = activation
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, embed_dim)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.init_weights()

    def init_weights(self):
        # initialize Linear layer
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, query):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(query))))
        query = query + self.dropout3(src2)
        query = self.norm2(query)
        return query

    def forward(
        self,
        query,
        query_pos,
        reference_points,
        spatial_shapes,
        level_start_index,
        query_key_padding_mask=None,
    ):
        # self attention
        src2 = self.self_attn(
            query=self.with_pos_embed(query, query_pos),
            reference_points=reference_points,
            value=query,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            key_padding_mask=query_key_padding_mask,
        )
        query = query + self.dropout1(src2)
        query = self.norm1(query)

        # ffn
        query = self.forward_ffn(query)

        return query


class MorphologyTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, num_classes):
        super().__init__()
        # parameters
        self.embed_dim = decoder_layer.embed_dim
        self.num_heads = decoder_layer.num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes

        # decoder layers and embedding
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.ref_point_head = MLP(2 * self.embed_dim, self.embed_dim, self.embed_dim, 2)
        self.query_scale = MLP(self.embed_dim, self.embed_dim, self.embed_dim, 2)

        # iterative bounding box refinement
        class_head = nn.Linear(self.embed_dim, num_classes)
        bbox_head = MLP(self.embed_dim, self.embed_dim, 4, 3)
        self.class_head = nn.ModuleList([copy.deepcopy(class_head) for _ in range(num_layers)])
        self.bbox_head = nn.ModuleList([copy.deepcopy(bbox_head) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(self.embed_dim)

        # relation embedding
        self.position_relation_embedding = PositionRelationEmbedding(16, self.num_heads)

        self.init_weights()

    def init_weights(self):
        # initialize decoder layers
        for layer in self.layers:
            if hasattr(layer, "init_weights"):
                layer.init_weights()
        # initialize decoder classification layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for class_head in self.class_head:
            nn.init.constant_(class_head.bias, bias_value)
        # initialize decoder regression layers
        for bbox_head in self.bbox_head:
            nn.init.constant_(bbox_head.layers[-1].weight, 0.0)
            nn.init.constant_(bbox_head.layers[-1].bias, 0.0)

    def forward(
        self,
        query,
        reference_points,
        value,
        spatial_shapes,
        level_start_index,
        valid_ratios,
        key_padding_mask=None,
        attn_mask=None,
        skip_relation=False,
    ):
        outputs_classes, outputs_coords = [], []
        valid_ratio_scale = torch.cat([valid_ratios, valid_ratios], -1)[:, None]

        pos_relation = attn_mask  # fallback pos_relation to attn_mask
        for layer_idx, layer in enumerate(self.layers):
            reference_points_input = reference_points.detach()[:, :, None] * valid_ratio_scale
            query_sine_embed = get_sine_pos_embed(
                reference_points_input[:, :, 0, :], self.embed_dim // 2
            )
            query_pos = self.ref_point_head(query_sine_embed)
            query_pos = query_pos * self.query_scale(query) if layer_idx != 0 else query_pos

            # relation embedding
            query = layer(
                query=query,
                query_pos=query_pos,
                reference_points=reference_points_input,
                value=value,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=key_padding_mask,
                self_attn_mask=pos_relation,
            )

            # get output, reference_points are not detached for look_forward_twice
            output_class = self.class_head[layer_idx](self.norm(query))
            output_coord = self.bbox_head[layer_idx](self.norm(query))
            output_coord = output_coord + inverse_sigmoid(reference_points)
            output_coord = output_coord.sigmoid()
            outputs_classes.append(output_class)
            outputs_coords.append(output_coord)

            if layer_idx == self.num_layers - 1:
                break

            # calculate position relation embedding
            # NOTE: prevent memory leak like denoising, or introduce totally separate groups?
            if not skip_relation:
                src_boxes = tgt_boxes if layer_idx >= 1 else reference_points
                tgt_boxes = output_coord
                pos_relation = self.position_relation_embedding(src_boxes, tgt_boxes).flatten(0, 1)
                if attn_mask is not None:
                    pos_relation.masked_fill_(attn_mask, float("-inf"))

            # iterative bounding box refinement
            reference_points = inverse_sigmoid(reference_points.detach())
            reference_points = self.bbox_head[layer_idx](query) + reference_points
            reference_points = reference_points.sigmoid()

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        return outputs_classes, outputs_coords


class MorphologyTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        d_ffn=1024,
        n_heads=8,
        dropout=0.1,
        activation=nn.ReLU(inplace=True),
        n_levels=4,
        n_points=4,
        use_topology_attention=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = n_heads
        # cross attention
        self.cross_attn = MultiScaleDeformableAttention(embed_dim, n_levels, n_heads, n_points)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        # self attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        # ffn
        self.linear1 = nn.Linear(embed_dim, d_ffn)
        self.activation = activation
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, embed_dim)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(embed_dim)

        # 新增拓扑感知注意力
        if use_topology_attention:
            self.taa = TopologyAwareAttention(embed_dim, n_heads)  # 正确初始化
        else:
            self.taa = None

        self.init_weights()

    def init_weights(self):
        # initialize self_attention
        nn.init.xavier_uniform_(self.self_attn.in_proj_weight)
        nn.init.xavier_uniform_(self.self_attn.out_proj.weight)
        # initialize Linear layer
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        if self.taa is not None: 
            # nn.init.xavier_uniform_(self.taa.lambda_param)
            nn.init.xavier_uniform_(self.taa.topology_proj[0].weight)
            nn.init.xavier_uniform_(self.taa.topology_proj[2].weight)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(
        self,
        query,
        query_pos,
        reference_points,
        value,
        spatial_shapes,
        level_start_index,
        self_attn_mask=None,
        key_padding_mask=None,
    ):
        tgt = query
        # 自注意力计算
        q = k = self.with_pos_embed(query, query_pos)
        attn_output, attn_weights = self.self_attn(
            q, k, value=query, need_weights=True)
        
        # 加入拓扑感知偏置
        if self.taa is not None:
            attn_weights = self.taa(attn_weights, reference_points)
            tgt = tgt + self.dropout(torch.bmm(attn_weights, tgt))
        else:
            tgt = tgt + self.dropout(attn_output)
        # attn_output = torch.matmul(attn_weights, query)
        
        # self attention
        query_with_pos = key_with_pos = self.with_pos_embed(query, query_pos)
        query2 = self.self_attn(
            query=query_with_pos,
            key=key_with_pos,
            value=query,
            attn_mask=self_attn_mask,
            need_weights=False,
        )[0]
        query = query + self.dropout2(query2)
        query = self.norm2(query)

        # cross attention
        query2 = self.cross_attn(
            query=self.with_pos_embed(query, query_pos),
            reference_points=reference_points,
            value=value,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            key_padding_mask=key_padding_mask,
        )
        query = query + self.dropout1(query2)
        query = self.norm1(query)

        # ffn
        query = self.forward_ffn(query)

        return query


def box_rel_encoding(src_boxes, tgt_boxes, eps=1e-5):
    # construct position relation
    xy1, wh1 = src_boxes.split([2, 2], -1)
    xy2, wh2 = tgt_boxes.split([2, 2], -1)
    delta_xy = torch.abs(xy1.unsqueeze(-2) - xy2.unsqueeze(-3))
    delta_xy = torch.log(delta_xy / (wh1.unsqueeze(-2) + eps) + 1.0)
    delta_wh = torch.log((wh1.unsqueeze(-2) + eps) / (wh2.unsqueeze(-3) + eps))
    pos_embed = torch.cat([delta_xy, delta_wh], -1)  # [batch_size, num_boxes1, num_boxes2, 4]

    return pos_embed


class PositionRelationEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        temperature=10000.0,
        scale=100.0,
        activation_layer=nn.ReLU,
        inplace=True,
    ):
        super().__init__()
        self.pos_proj = Conv2dNormActivation(
            embed_dim * 4,
            num_heads,
            kernel_size=1,
            inplace=inplace,
            norm_layer=None,
            activation_layer=activation_layer,
        )
        self.pos_func = functools.partial(
            get_sine_pos_embed,
            num_pos_feats=embed_dim,
            temperature=temperature,
            scale=scale,
            exchange_xy=False,
        )

    def forward(self, src_boxes: Tensor, tgt_boxes: Tensor = None):
        if tgt_boxes is None:
            tgt_boxes = src_boxes
        # src_boxes: [batch_size, num_boxes1, 4]
        # tgt_boxes: [batch_size, num_boxes2, 4]
        torch._assert(src_boxes.shape[-1] == 4, f"src_boxes much have 4 coordinates")
        torch._assert(tgt_boxes.shape[-1] == 4, f"tgt_boxes must have 4 coordinates")
        with torch.no_grad():
            pos_embed = box_rel_encoding(src_boxes, tgt_boxes)
            pos_embed = self.pos_func(pos_embed).permute(0, 3, 1, 2)
        pos_embed = self.pos_proj(pos_embed)

        return pos_embed.clone()
