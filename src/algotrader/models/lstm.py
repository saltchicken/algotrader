import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import copy
from algotrader.logger import get_logger

logger = get_logger(__name__)


class LSTMTradingNet(nn.Module):
    # FIX 1: Reduced model capacity (hidden_size 64 -> 32, num_layers 2 -> 1)
    def __init__(self, input_size, hidden_size=32, num_layers=1):
        super(LSTMTradingNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Batch first because our data will be [batch_size, sequence_length, features]
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            # PyTorch expects dropout=0 if num_layers=1, so we handle it dynamically
            dropout=0.3 if num_layers > 1 else 0.0,
        )

        # FIX 2: Added LayerNorm and explicit Dropout to heavily regularize the output
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.5)

        # Final decision layer maps the LSTM's memory to a single binary output
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        out, _ = self.lstm(x)

        # We only care about the network's state at the very last time step
        last_step_out = out[:, -1, :]

        # Apply normalization and dropout
        last_step_out = self.layer_norm(last_step_out)
        last_step_out = self.dropout(last_step_out)

        # Output raw logits (no sigmoid) because we use BCEWithLogitsLoss
        return self.fc(last_step_out)


def train_lstm_with_early_stopping(
    model, X_train, y_train, X_val, y_val, epochs=100, batch_size=128, patience=15
):
    """Trains the LSTM and stops early if validation performance degrades."""

    train_data = TensorDataset(X_train, y_train)
    val_data = TensorDataset(X_val, y_val)

    # Increased batch size to 128 to smooth out gradient updates on noisy data
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # Handle Class Imbalance mathematically (Assuming 0s are more common than 1s)
    num_ones = max(y_train.sum().item(), 1)
    num_zeros = len(y_train) - num_ones
    imbalance_ratio = num_zeros / num_ones

    logger.info(f"Class imbalance ratio (0s to 1s): {imbalance_ratio:.2f}")

    # Cap the positive weight to prevent the loss function from becoming unstable
    capped_weight = min(imbalance_ratio, 3.0)
    pos_weight = torch.tensor([capped_weight])

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # FIX 3: Lowered learning rate (0.001 -> 0.0005) and increased L2 penalty (1e-5 -> 1e-3)
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-3)

    # FIX 4: Add a Learning Rate Scheduler to reduce LR when Validation Loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_weights = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                predictions = model(batch_X)
                v_loss = criterion(predictions, batch_y)
                val_loss += v_loss.item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)

        # Step the scheduler based on validation loss
        scheduler.step(avg_val)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch+1:03d}/{epochs} | LR: {current_lr:.6f} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}"
            )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            epochs_no_improve = 0
            best_weights = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            logger.info(
                f"Early Stopping triggered at Epoch {epoch+1}. Best Val Loss: {best_val_loss:.4f}"
            )
            break

    if best_weights:
        model.load_state_dict(best_weights)

    return model
