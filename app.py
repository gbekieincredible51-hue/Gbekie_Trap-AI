import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

st.set_page_config(page_title="Gbekie Trap AI", layout="wide")

st.title("Gbekie Trap Demonstrator")
st.markdown("""
Transformer learns Ramadan study performance → multi-agent system shows **Language Trap** (oscillation)  
→ **Gbekie Condition** (auto-damping) brings stability + humility score.
""")

# ────────────────────────────────────────────────
# Data upload / fallback
# ────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload ramadan_full_model.csv", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("CSV uploaded successfully!")
else:
    st.info("Using built-in sample data (30 days placeholder)")
    df = pd.DataFrame({
        'Sleep_hours': np.random.normal(5.0, 1.2, 30),
        'Fasting_drag_F': np.linspace(1.0, 0.88, 30),
        'Prayer_penalty_P': np.random.normal(0.92, 0.04, 30),
        'Hydration_drag_H': np.linspace(1.0, 0.78, 30),
        'Study_performance': np.random.uniform(1.0, 5.5, 30)
    })

features = ['Sleep_hours', 'Fasting_drag_F', 'Prayer_penalty_P', 'Hydration_drag_H']
X = torch.tensor(df[features].values, dtype=torch.float32)
y = torch.tensor(df['Study_performance'].values, dtype=torch.float32).unsqueeze(1)

# ────────────────────────────────────────────────
# Tiny Transformer
# ────────────────────────────────────────────────
class GbekieTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=4, nhead=2, dim_feedforward=16, dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.fc = nn.Linear(4, 1)
        self.attn_weights = None

    def forward(self, x):
        def hook(module, inp, out):
            self.attn_weights = module.self_attn.attn_output_weights.detach().mean(dim=1).cpu().numpy()

        h = self.encoder.layers[0].self_attn.register_forward_hook(hook)
        x = x.unsqueeze(1)
        x = self.encoder(x)
        out = self.fc(x.squeeze(1))
        h.remove()
        return out

if 'model' not in st.session_state:
    st.session_state.model = GbekieTransformer()
    st.session_state.optimizer = torch.optim.Adam(st.session_state.model.parameters(), lr=0.005)
    st.session_state.trained = False

model = st.session_state.model

# Train button
if st.button("Train Transformer (5 epochs)") or not st.session_state.trained:
    with st.spinner("Training..."):
        criterion = nn.MSELoss()
        for _ in range(5):
            model.train()
            st.session_state.optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            st.session_state.optimizer.step()
        st.session_state.trained = True
    st.success("Training complete!")

# Show predictions
with torch.no_grad():
    preds = model(X).squeeze().numpy()

st.subheader("Transformer Predictions vs Actual")
fig, ax = plt.subplots()
ax.plot(df['Study_performance'], label="Actual", color="lime")
ax.plot(preds, label="Predicted", color="cyan", linestyle="--")
ax.set_xlabel("Day")
ax.set_ylabel("Study Performance")
ax.legend()
st.pyplot(fig)

# ────────────────────────────────────────────────
# Attention Visualization
# ────────────────────────────────────────────────
st.subheader("Attention Weights (Layer 1)")
with torch.no_grad():
    _ = model(X)  # trigger hook
if model.attn_weights is not None:
    fig, ax = plt.subplots()
    im = ax.imshow(model.attn_weights, cmap='viridis')
    plt.colorbar(im, ax=ax)
    ax.set_title("Attention Heatmap")
    ax.set_xlabel("Position")
    ax.set_ylabel("Position")
    st.pyplot(fig)
else:
    st.info("Attention weights not captured yet — run prediction again.")

# ────────────────────────────────────────────────
# Multi-Agent Language Trap
# ────────────────────────────────────────────────
st.subheader("Multi-Agent Language Trap Simulation")

N_AGENTS = st.slider("Number of Agents", 2, 20, 10)
PRED_STEPS = st.slider("Prediction Steps", 10, 60, 30)
osc_threshold = st.slider("Oscillation Trigger Threshold", 0.1, 1.0, 0.4)
auto_damp = st.checkbox("Auto-Damping (Gbekie Condition)", value=True)

if st.button("Run Language Trap Simulation"):
    alpha = 0.0
    history = np.zeros((PRED_STEPS + 1, len(df), N_AGENTS))
    initial = preds  # from Transformer
    history[0] = np.tile(initial, (N_AGENTS, 1)).T

    damping_log = []

    for step in range(PRED_STEPS):
        current = history[step]
        new_preds = np.zeros_like(current)

        for i in range(N_AGENTS):
            others_mean = np.mean(np.delete(current, i, axis=1), axis=1)
            new_preds[:, i] = (1 - alpha) * current[:, i] + alpha * (1 - others_mean)

        if auto_damp:
            osc = np.std(new_preds - current)
            if osc > osc_threshold:
                alpha = min(0.5, alpha + 0.08)
                damping_log.append(f"Step {step}: Oscillation {osc:.3f} → α = {alpha:.2f}")
            else:
                alpha = max(0.0, alpha - 0.015)

        history[step + 1] = new_preds

    humility = 1 - (alpha / 0.5)
    st.metric("Final Damping α", f"{alpha:.2f}")
    st.metric("Humility Score", f"{humility:.2f}", help="1 = fully damped & stable, 0 = trapped in oscillation")

    if damping_log:
        st.text("Damping Events:\n" + "\n".join(damping_log[:8]) + ("..." if len(damping_log)>8 else ""))

    # Plot
    fig, ax = plt.subplots(figsize=(12,6))
    for i in range(N_AGENTS):
        ax.plot(history[:, :, i].mean(axis=1), alpha=0.7, label=f'Agent {i+1}')
    ax.axhline(2.0, color='red', linestyle='--', label='Critical')
    ax.set_title('Multi-Agent Predictions (Average Across Days)')
    ax.set_xlabel('Prediction Step')
    ax.set_ylabel('Predicted Study Performance')
    ax.legend(ncol=3)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

st.markdown("---")
st.caption("Gbekie Trap AI — Transformer prediction meets multi-agent instability & humility damping")
