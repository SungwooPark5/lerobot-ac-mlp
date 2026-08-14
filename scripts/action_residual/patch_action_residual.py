from pathlib import Path

train_path = Path("train_action_residual_act.py")
eval_path = Path("eval_action_residual_act.py")

# =========================
# 1. 공통 코드 블록
# =========================
ACTION_RESIDUAL_CLASS = r'''
class ActionResidualNetwork(nn.Module):
    """
    TE fixed + action residual only.

    Input:
        states:   [B, state_dim]
        contribs: [B, L, action_dim]
    Output:
        delta_a:  [B, action_dim]
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        max_delta: float = 1.0,
    ):
        super().__init__()
        self.max_delta = max_delta

        self.state_proj = nn.Linear(state_dim, d_model)
        self.chunk_proj = nn.Linear(action_dim, d_model)

        self.query = nn.Embedding(1, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, action_dim),
        )

        # zero-init: 시작점이 pure TE가 되도록 함
        last = self.head[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, states: torch.Tensor, contribs: torch.Tensor) -> torch.Tensor:
        B = states.shape[0]

        state_tok = self.state_proj(states).unsqueeze(1)   # [B, 1, D]
        chunk_toks = self.chunk_proj(contribs)             # [B, L, D]
        memory = torch.cat([state_tok, chunk_toks], dim=1) # [B, 1+L, D]

        q = self.query.weight.unsqueeze(0).expand(B, -1, -1) # [B, 1, D]
        x = self.decoder(q, memory).squeeze(1)               # [B, D]

        raw_delta = self.head(x)                             # [B, action_dim]
        delta_a = torch.tanh(raw_delta) * self.max_delta
        return delta_a


def make_fixed_te_weights(
    L: int,
    te_coeff: float,
    device: torch.device,
    fresh_index: str = "last",
):
    """
    fresh_index:
        "last"  -> contribs[:, -1]이 최신 chunk
        "first" -> contribs[:, 0]이 최신 chunk
    """
    if fresh_index == "last":
        age = torch.arange(L - 1, -1, -1, device=device).float()
    elif fresh_index == "first":
        age = torch.arange(0, L, device=device).float()
    else:
        raise ValueError(f"fresh_index must be 'first' or 'last', got {fresh_index}")

    weights = torch.exp(-te_coeff * age)
    weights = weights / weights.sum()
    return weights
'''


# =========================
# 2. train 파일 패치
# =========================
s = train_path.read_text()

# 기존 ResidualWeightNetwork 클래스 이름을 새 클래스 이름으로 강제 교체하기보다는
# 새 클래스를 파일에 추가하고, 생성부/forward부를 수동 교체하기 쉽게 marker 출력
if "class ActionResidualNetwork" not in s:
    # nn.Module 클래스들 뒤에 넣기 어렵기 때문에 import 구역 이후에 삽입
    insert_at = s.find("def ")
    class_at = s.find("class ")
    candidates = [x for x in [insert_at, class_at] if x != -1]
    pos = min(candidates) if candidates else 0
    s = s[:pos] + ACTION_RESIDUAL_CLASS + "\n\n" + s[pos:]

# argparse 추가
arg_block = '''
    parser.add_argument("--alpha_residual", type=float, default=0.1)
    parser.add_argument("--lambda_residual", type=float, default=0.1)
    parser.add_argument("--max_delta", type=float, default=1.0)
    parser.add_argument("--te_coeff", type=float, default=0.01)
    parser.add_argument("--fresh_index", type=str, default="last", choices=["first", "last"])
'''
if "--alpha_residual" not in s:
    # 가장 마지막 parser.add_argument 근처 뒤에 추가
    idx = s.rfind("parser.add_argument")
    end = s.find("\n", idx)
    if idx != -1 and end != -1:
        s = s[:end+1] + arg_block + s[end+1:]
    else:
        print("[WARN] parser.add_argument 위치를 못 찾음. argparse는 직접 추가 필요.")

# 모델 생성부 힌트용 치환
# 흔한 패턴들을 최대한 교체
s = s.replace(
    "ResidualWeightNetwork(",
    "ActionResidualNetwork("
)

# lambda_delta_reg 이름이 남아있으면 lambda_residual로 유도
s = s.replace("args.lambda_delta_reg", "args.lambda_residual")

# 저장 dict에 model_type 정보가 없으면 torch.save 직전에라도 들어가게는 어려움
# 대신 checkpoint 저장부는 아래 안내에 따라 확인 필요

train_path.write_text(s)
print(f"[OK] patched base train file: {train_path}")


# =========================
# 3. eval 파일 패치
# =========================
s = eval_path.read_text()

if "class ActionResidualNetwork" not in s:
    insert_at = s.find("def ")
    class_at = s.find("class ")
    candidates = [x for x in [insert_at, class_at] if x != -1]
    pos = min(candidates) if candidates else 0
    s = s[:pos] + ACTION_RESIDUAL_CLASS + "\n\n" + s[pos:]

arg_block_eval = '''
    parser.add_argument("--alpha_residual", type=float, default=0.1)
    parser.add_argument("--max_delta", type=float, default=1.0)
    parser.add_argument("--te_coeff", type=float, default=0.01)
    parser.add_argument("--fresh_index", type=str, default="last", choices=["first", "last"])
'''
if "--alpha_residual" not in s:
    idx = s.rfind("parser.add_argument")
    end = s.find("\n", idx)
    if idx != -1 and end != -1:
        s = s[:end+1] + arg_block_eval + s[end+1:]
    else:
        print("[WARN] eval parser.add_argument 위치를 못 찾음. argparse는 직접 추가 필요.")

s = s.replace(
    "ResidualWeightNetwork(",
    "ActionResidualNetwork("
)

eval_path.write_text(s)
print(f"[OK] patched base eval file: {eval_path}")

print("""
[DONE]

이 스크립트는 클래스/인자/모델명 기본 패치만 함.
이제 반드시 아래 두 부분을 직접 확인해서 교체해야 함:

1) train_action_residual_act.py
   기존:
      delta_logits = model(...)
      final_logits = te_logits + delta_logits
      weights = softmax(...)
      pred_action = ...

   변경:
      a_te = fixed TE action
      delta_a = model(...)
      pred_action = a_te + alpha_residual * delta_a

2) eval_action_residual_act.py
   기존 weight 계산 부분도 동일하게
      action = a_te + alpha_residual * delta_a
   로 교체.
""")
