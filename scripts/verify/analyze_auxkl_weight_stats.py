from pathlib import Path
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualWeightNetwork(nn.Module):
    def __init__(self, L, state_dim, action_dim, vis_dim, te_coeff=0.01,
                 d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.L = L
        self.chunk_proj = nn.Linear(action_dim, d_model)
        self.state_proj = nn.Linear(state_dim, d_model)
        self.image_proj = nn.Sequential(
            nn.LayerNorm(vis_dim),
            nn.Linear(vis_dim, d_model),
            nn.Dropout(p=0.1),
        )
        self.queries = nn.Embedding(L, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, n_layers)
        self.head = nn.Linear(d_model, 1)

        self.register_buffer(
            "te_logits",
            -te_coeff * torch.arange(L, dtype=torch.float32)
        )

    def forward(self, state, image_feat, contribs):
        B = state.shape[0]
        state_tok = self.state_proj(state).unsqueeze(1)
        image_tok = self.image_proj(image_feat).unsqueeze(1)
        chunk_toks = self.chunk_proj(contribs)

        memory = torch.cat([state_tok, image_tok, chunk_toks], dim=1)

        q = self.queries.weight.unsqueeze(0).expand(B, -1, -1)
        x = self.decoder(q, memory)

        delta = self.head(x).squeeze(-1)
        final_logits = self.te_logits.unsqueeze(0) + delta
        return final_logits, delta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_path", type=str, required=True)
    parser.add_argument("--weight_net_path", type=str, required=True)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"[LOAD CACHE] {args.cache_path}")
    data = np.load(args.cache_path, allow_pickle=True)
    print("[CACHE KEYS]", data.files)

    chunks = data["chunks"]       # [N, L, action_dim]
    states = data["states"]       # [N, state_dim]
    actions = data["actions"]     # [N, action_dim]
    vis_feats = data["vis_feats"] # [N, vis_dim]

    print("[SHAPE] chunks   :", chunks.shape)
    print("[SHAPE] states   :", states.shape)
    print("[SHAPE] actions  :", actions.shape)
    print("[SHAPE] vis_feats:", vis_feats.shape)

    print(f"[LOAD CKPT] {args.weight_net_path}")
    ckpt = torch.load(args.weight_net_path, map_location=device)

    L = ckpt["L"]
    model = ResidualWeightNetwork(
        L=L,
        state_dim=ckpt["state_dim"],
        action_dim=ckpt["action_dim"],
        vis_dim=ckpt["vis_dim"],
        te_coeff=ckpt["te_coeff"],
        d_model=ckpt["d_model"],
        n_heads=ckpt["n_heads"],
        n_layers=ckpt["n_layers"],
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    print(
        f"[CKPT INFO] L={ckpt['L']}, step={ckpt.get('step')}, "
        f"te_coeff={ckpt['te_coeff']}, vis_dim={ckpt['vis_dim']}, "
        f"lambda_delta_reg={ckpt.get('lambda_delta_reg')}, "
        f"lambda_aux={ckpt.get('lambda_aux')}, tau_aux={ckpt.get('tau_aux')}"
    )

    N = chunks.shape[0]

    weighted_errors = []
    te_errors = []
    uniform_errors = []
    best_errors = []
    best_matches = []
    entropies = []
    entropy_ratios = []
    max_weights = []
    delta_mses = []
    delta_abs_means = []
    kl_to_teachers = []
    target_entropies = []

    with torch.no_grad():
        for start in range(0, N, args.batch_size):
            end = min(start + args.batch_size, N)

            c = torch.tensor(chunks[start:end], dtype=torch.float32, device=device)
            s = torch.tensor(states[start:end], dtype=torch.float32, device=device)
            t = torch.tensor(actions[start:end], dtype=torch.float32, device=device)
            v = torch.tensor(vis_feats[start:end], dtype=torch.float32, device=device)

            final_logits, delta = model(s, v, c)
            weights = F.softmax(final_logits, dim=1)  # [B, L]

            # TE prior weights
            te_logits = model.te_logits.unsqueeze(0).expand_as(final_logits)
            te_weights = F.softmax(te_logits, dim=1)

            # per-chunk target error
            errors = (c - t.unsqueeze(1)).abs().mean(dim=-1)  # [B, L]

            # aux KL teacher
            target_weights = torch.softmax(-errors / args.tau, dim=1)

            weighted_error = (weights * errors).sum(dim=1)
            te_error = (te_weights * errors).sum(dim=1)
            uniform_error = errors.mean(dim=1)
            best_error = errors.min(dim=1).values

            pred_best_idx = weights.argmax(dim=1)
            oracle_best_idx = errors.argmin(dim=1)
            best_match = (pred_best_idx == oracle_best_idx).float()

            entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=1)
            entropy_ratio = entropy / np.log(L)
            max_weight = weights.max(dim=1).values

            target_entropy = -(target_weights * torch.log(target_weights + 1e-8)).sum(dim=1)
            kl_to_teacher = (
                target_weights
                * (torch.log(target_weights + 1e-8) - torch.log(weights + 1e-8))
            ).sum(dim=1)

            weighted_errors.append(weighted_error.cpu())
            te_errors.append(te_error.cpu())
            uniform_errors.append(uniform_error.cpu())
            best_errors.append(best_error.cpu())
            best_matches.append(best_match.cpu())
            entropies.append(entropy.cpu())
            entropy_ratios.append(entropy_ratio.cpu())
            max_weights.append(max_weight.cpu())
            delta_mses.append((delta ** 2).mean(dim=1).cpu())
            delta_abs_means.append(delta.abs().mean(dim=1).cpu())
            kl_to_teachers.append(kl_to_teacher.cpu())
            target_entropies.append(target_entropy.cpu())

    def mean(xs):
        return torch.cat(xs).mean().item()

    we = mean(weighted_errors)
    te = mean(te_errors)
    ue = mean(uniform_errors)
    be = mean(best_errors)
    bm = mean(best_matches)
    ent = mean(entropies)
    er = mean(entropy_ratios)
    mw = mean(max_weights)
    dm = mean(delta_mses)
    da = mean(delta_abs_means)
    klt = mean(kl_to_teachers)
    tent = mean(target_entropies)

    print("\n========== AUX-KL WEIGHT ANALYSIS ==========")
    print(f"N                         : {N}")
    print(f"L                         : {L}")
    print(f"tau                       : {args.tau}")
    print("--------------------------------------------")
    print(f"weighted_error            : {we:.6f}")
    print(f"te_error                  : {te:.6f}")
    print(f"uniform_error             : {ue:.6f}")
    print(f"best_error_oracle         : {be:.6f}")
    print("--------------------------------------------")
    print(f"weighted - TE             : {we - te:.6f}")
    print(f"weighted - uniform        : {we - ue:.6f}")
    print(f"weighted - oracle_best    : {we - be:.6f}")
    print("--------------------------------------------")
    print(f"best_match                : {bm:.6f}")
    print(f"random_match 기준         : {1.0 / L:.6f}")
    print(f"max_weight                : {mw:.6f}")
    print(f"entropy                   : {ent:.6f}")
    print(f"entropy_ratio             : {er:.6f}")
    print(f"target_entropy            : {tent:.6f}")
    print(f"kl_to_aux_teacher          : {klt:.6f}")
    print(f"delta_mse                 : {dm:.8f}")
    print(f"delta_abs_mean            : {da:.8f}")
    print("============================================")

    print("\n[판단]")
    if we < ue:
        print("- weighted_error < uniform_error: 평균보다는 좋은 chunk 쪽으로 weight가 갔음.")
    else:
        print("- weighted_error >= uniform_error: 좋은 chunk 선택 신호가 거의 안 먹혔을 가능성.")

    if we < te:
        print("- weighted_error < te_error: offline chunk-quality 기준으로는 TE보다 좋아짐.")
    else:
        print("- weighted_error >= te_error: offline에서도 TE보다 좋은 weighting이 아님.")

    if bm > (1.0 / L) * 2:
        print("- best_match가 random보다 꽤 높음: oracle-best chunk를 어느 정도 따라감.")
    elif bm > (1.0 / L):
        print("- best_match가 random보다 조금 높음: 약한 선택 신호는 있음.")
    else:
        print("- best_match가 random 수준: max weight가 oracle-best를 거의 못 따라감.")

    if er > 0.95:
        print("- entropy_ratio가 매우 높음: weight가 거의 TE/uniform처럼 퍼져 있음.")
    elif er < 0.70:
        print("- entropy_ratio가 낮음: selection이 강해졌고 TE smoothing을 깼을 수 있음.")
    else:
        print("- entropy_ratio가 중간: 어느 정도 선택성은 생김.")

    print("\n[결론 해석 가이드]")
    print("1) weighted_error < te_error인데 success가 낮다 → chunk 품질은 맞췄지만 rollout은 TE smoothing이 더 중요.")
    print("2) weighted_error >= te_error이고 success도 낮다 → aux KL이 현재 task에서 도움 안 됨.")
    print("3) best_match 낮고 entropy 높다 → 사실상 좋은 chunk 선택을 못 배웠거나 weight가 거의 TE.")
    print("4) best_match 높고 entropy 낮다 → 선택은 했지만 그 선택성이 control 안정성을 깼을 가능성.")


if __name__ == '__main__':
    main()
