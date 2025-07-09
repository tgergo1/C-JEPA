import numpy as np
from .energy_based import EnergyNetwork

class JointEmbeddingNetwork:
    """Minimal JEPA-style network with shared encoder and predictor."""

    def __init__(self, encoder_sizes, predictor_sizes=None):
        self.encoder = EnergyNetwork(encoder_sizes)
        pred_sizes = predictor_sizes if predictor_sizes is not None else [encoder_sizes[-1], encoder_sizes[-1]]
        self.predictor = EnergyNetwork(pred_sizes)

    def forward(self, context, target):
        """Return predicted and target embeddings."""
        ctx_z = self.encoder.forward(context)
        tgt_z = self.encoder.forward(target)
        pred_z = self.predictor.forward(ctx_z)
        return pred_z, tgt_z

    def loss(self, context, target):
        pred_z, tgt_z = self.forward(context, target)
        diff = pred_z - tgt_z
        return 0.5 * float(np.sum(diff ** 2))

    def train_step(self, context, target, lr=0.01):
        """Update network parameters to minimize prediction loss."""
        ctx_z, ctx_acts = self.encoder.forward_activations(context)
        tgt_z, tgt_acts = self.encoder.forward_activations(target)
        pred_z, pred_acts = self.predictor.forward_activations(ctx_z)

        diff = pred_z - tgt_z

        pred_grads, grad_ctx = self.predictor.backprop(pred_acts, diff)
        self.predictor.apply_grads(pred_grads, lr)

        enc_grads_ctx, _ = self.encoder.backprop(ctx_acts, grad_ctx)
        enc_grads_tgt, _ = self.encoder.backprop(tgt_acts, -diff)
        enc_grads = [gc + gt for gc, gt in zip(enc_grads_ctx, enc_grads_tgt)]
        self.encoder.apply_grads(enc_grads, lr)

        return 0.5 * float(np.sum(diff ** 2))
