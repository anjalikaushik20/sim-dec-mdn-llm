import sys
import random
import torch
import torch.nn as nn
from tools import feature_list
from models.mdn_model import MDNHead, gmm_loss

class S_SimDec(nn.Module):
    def __init__(self, env):
        super(S_SimDec, self).__init__()
        self.env = env
        self.c_num = len(feature_list.product_info[self.env.args.dataset] + feature_list.order_info[self.env.args.dataset] +\
                           feature_list.customer_info[self.env.args.dataset] + feature_list.shipping_info[self.env.args.dataset] )
        
        
        self.c_transform4p = nn.Linear(len(feature_list.product_info[self.env.args.dataset]), self.env.args.embed_dim)
        self.c_transform4o = nn.Linear(len(feature_list.order_info[self.env.args.dataset]), self.env.args.embed_dim)
        self.c_transform4c = nn.Linear(len(feature_list.customer_info[self.env.args.dataset]), self.env.args.embed_dim)
        self.c_transform4s = nn.Linear(len(feature_list.shipping_info[self.env.args.dataset]), self.env.args.embed_dim)

        # Project pooled scalar feature to embedding dim
        self.pooling_fc = nn.Linear(1, self.env.args.embed_dim)

        self.embedding = nn.Embedding(5, self.env.args.embed_dim)

        
        
        
        

        
        self.fc = nn.Linear(self.env.args.embed_dim, self.env.args.embed_dim)
        # Optional projection for decoder feedback to stabilize scale
        self.dec_in = nn.Linear(self.env.args.embed_dim, self.env.args.embed_dim)
        # Projection to turn sampled MDN output (D) into decoder input embedding (E)
        # Note: MDN output_dim is 1 in this project
        self.sample_proj = nn.Linear(1, self.env.args.embed_dim)

        
        self.encoder_lstm = nn.LSTM(input_size=self.env.args.embed_dim, hidden_size=self.env.args.embed_dim, 
                                    num_layers=self.env.args.encoder_num_layers, batch_first=True, bidirectional=True)

        self.decoder_lstm = nn.LSTM(input_size=self.env.args.embed_dim, hidden_size=self.env.args.embed_dim, 
                                    num_layers=self.env.args.decoder_num_layers, batch_first=True)
        
        # self.output_layer = nn.Linear(self.env.args.embed_dim, 2)


        # self.output_layer = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(self.env.args.embed_dim, num_classes),  
        #     )
        #     for num_classes in self.env.feature_classes
        # ])
        
        self.mdn_head = MDNHead(
            input_dim=self.env.args.embed_dim,
            output_dim=1,
            gaussians=getattr(self.env.args, 'mdn_components', 5),
            temperature=getattr(self.env.args, 'mdn_temperature', 1.0),
            sigma_floor=getattr(self.env.args, 'mdn_sigma_floor', 1e-3),
        )

        self.debug_logpi = []          # list of tensors [B,K]
        # self._debug_epoch_cache = []
        
        self.decision_maker = nn.Sequential(
                nn.Linear(self.c_num, self.c_num),
                nn.ReLU(),
                nn.Linear(self.c_num, 4)
        )

        self.to(self.env.device)
        
    def forward(self, c_input, shipping_mode, tgt):
        
        
        c_out4p = self.c_transform4p(c_input[:, :len(feature_list.product_info[self.env.args.dataset])])
        c_out4o = self.c_transform4o(c_input[:, len(feature_list.product_info[self.env.args.dataset]):len(feature_list.product_info[self.env.args.dataset]) + len(feature_list.order_info[self.env.args.dataset])])
        c_out4c = self.c_transform4c(c_input[:, len(feature_list.product_info[self.env.args.dataset]) + len(feature_list.order_info[self.env.args.dataset]):len(feature_list.product_info[self.env.args.dataset]) + \
                                    len(feature_list.order_info[self.env.args.dataset]) +len(feature_list.customer_info[self.env.args.dataset])])
        c_out4s = self.c_transform4s(c_input[:, -len(feature_list.shipping_info[self.env.args.dataset]):])


        c_out = torch.cat((c_out4p.unsqueeze(1), c_out4o.unsqueeze(1)), dim=1)
        c_out = torch.cat((c_out, c_out4c.unsqueeze(1)), dim=1)
        c_out = torch.cat((c_out, c_out4s.unsqueeze(1)), dim=1)

        
        # Per-sample pooled feature (row-wise mean), then project
        pooled_features = c_input[:, :self.c_num].mean(dim=1, keepdim=False)  # [B]
        pooled_features = pooled_features.unsqueeze(1)                        # [B,1]
        pooled_features = self.pooling_fc(pooled_features)                    # [B, embed_dim]

        
        if len(shipping_mode.shape) == 1:
            
            sm_embed = self.embedding(shipping_mode).unsqueeze(1)
        else:
            
            sm_embed = shipping_mode.unsqueeze(1)

        combined = torch.cat((c_out,  pooled_features.unsqueeze(1), sm_embed), dim=1) 
        
        combined = torch.relu(self.fc(combined))
            
        
        _, (h_n, c_n) = self.encoder_lstm(combined)

        
        h_n_forward = h_n[0:h_n.size(0):2]  
        h_n_backward = h_n[1:h_n.size(0):2]  

        
        h_n_combined = h_n_forward + h_n_backward
        c_n_combined = c_n[0:c_n.size(0):2] + c_n[1:c_n.size(0):2]

        decoder_hidden = (h_n_combined, c_n_combined)

        
        batch_size = c_input.shape[0]
        SOS_token = torch.full((batch_size, 1), 1, dtype=torch.long).to(self.env.device)

        # generated_tokens = []
        generated_mdn_params = []

        # Initialize autoregressive input with SOS token embedding
        next_tgt_embed = self.embedding(SOS_token)

        seq_len = tgt.size(-1)
        feedback_alpha = float(getattr(self.env.args, 'mdn_feedback_alpha', 0.8))  # blend strength for sampled feedback
        for t in range(seq_len):
            # Use the prepared embedding from previous step
            tgt_embed = next_tgt_embed

            # Run one decoder step
            decoder_output, decoder_hidden = self.decoder_lstm(tgt_embed, decoder_hidden)

            # MDN over current decoder output
            mus, sigmas, logpi = self.mdn_head(decoder_output.squeeze(1))

            # Optional: cache mixture weights for inspection
            if getattr(self.env.args, 'inspect_logpi', 0):
                if len(self.debug_logpi) < getattr(self.env.args, 'max_logpi_batches', 5):
                    self.debug_logpi.append(logpi.detach().cpu())

            generated_mdn_params.append((mus, sigmas, logpi))

            # Sample from the predicted mixture and feed back as next input
            with torch.no_grad():
                pi = logpi.exp()  # [B, K]
                comp = torch.distributions.Categorical(pi).sample()  # [B]
                idx = comp.view(-1, 1, 1).expand(-1, 1, mus.size(-1))  # [B,1,D]
                mu_sel = mus.gather(1, idx).squeeze(1)      # [B, D]
                sigma_sel = sigmas.gather(1, idx).squeeze(1)  # [B, D]
                sample = mu_sel + sigma_sel * torch.randn_like(mu_sel)  # [B, D]

            # Project sampled scalar/vector to embedding and blend with projected decoder output for stability
            proj_sample = self.sample_proj(sample).unsqueeze(1)          # [B,1,E]
            proj_dec = self.dec_in(decoder_output)                       # [B,1,E]
            next_tgt_embed = feedback_alpha * proj_sample + (1.0 - feedback_alpha) * proj_dec

        
        # return generated_tokens
        return generated_mdn_params
        
      
    def decision_process(self, c_input):
        decision_output = self.decision_maker(c_input)
        return decision_output
    
        # Add compute_loss method for GMM loss
    def compute_loss(self, targets, generated_mdn_params):
        # targets: [batch_size, seq_len, output_dim]
        # generated_mdn_params: list of (mus, sigmas, logpi) for each time step
        total_loss = 0
        ent_coeff = max(0.0, float(getattr(self.env.args, 'mdn_entropy_coeff', 1e-3)))
        mse_coeff = max(0.0, float(getattr(self.env.args, 'mdn_mse_coeff', 1e-2)))
        smooth_coeff = max(0.0, float(getattr(self.env.args, 'mdn_smooth_coeff', 1e-3)))
        prev_mus = None
        for t, (mus, sigmas, logpi) in enumerate(generated_mdn_params):
            tgt_t = targets[:, t, :]  # [batch_size, output_dim]
            nll = gmm_loss(tgt_t, mus, sigmas, logpi)
            # Entropy of mixture weights pi to discourage collapse
            with torch.no_grad():
                pass
            pi = logpi.exp()  # [B, K]
            entropy = -(pi * (pi.clamp_min(1e-8)).log()).sum(dim=-1).mean()
            # Expected value over mixture for regression surrogate
            y_pred = (pi.unsqueeze(-1) * mus).sum(dim=1)  # [B, D]
            mse = ((y_pred - tgt_t) ** 2).mean()
            # Temporal smoothness prior on mixture means to regularize dynamics
            smoothness = 0.0
            if prev_mus is not None and smooth_coeff > 0.0:
                smoothness = ((mus - prev_mus.detach()) ** 2).mean()
            total_loss += nll - ent_coeff * entropy + mse_coeff * mse + smooth_coeff * smoothness
            prev_mus = mus
        return total_loss / len(generated_mdn_params)  # Average over sequence
    
    def reset_logpi_debug(self):
        self.debug_logpi = []