import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

from etna.models.base import BaseAdapter, NonPredictionIntervalContextIgnorantAbstractModel
from etna.models.mixins import PerSegmentModelMixin
from etna.models.mixins import NonPredictionIntervalContextIgnorantModelMixin 

import re

class LSTMModule(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_steps: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_steps = output_steps
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_size, output_steps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.linear(out[:, -1, :])  # Many-to-many: out[:, :, :]
        return out

class LSTMModel:
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_steps: int,
        seq_len: int = 10,
        num_epochs: int = 100,
        learning_rate: float = 0.001,
        autoregressive: bool = False
    ):
        self.model = LSTMModule(input_size, hidden_size, num_layers, output_steps)
        self.seq_len = seq_len
        self.output_steps = output_steps
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.autoregressive = autoregressive
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def _create_sequences(self, data: np.ndarray, target: np.ndarray) -> tuple:
        sequences = []
        targets = []
        for i in range(len(data) - self.seq_len - self.output_steps + 1):
            seq = data[i:i + self.seq_len]
            target_ = target[i + self.seq_len : i + self.seq_len + self.output_steps]
            sequences.append(seq)
            targets.append(target_)
        return torch.FloatTensor(sequences), torch.FloatTensor(targets)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_tensor, y_tensor = self._create_sequences(X, y)
        self.model.train()
        
        for epoch in range(self.num_epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            loss.backward()
            self.optimizer.step()

    def predict(self, X: np.ndarray, horizon: Optional[int] = None) -> np.ndarray:
        self.model.eval()
        horizon = horizon or self.output_steps
        
        if self.autoregressive:
            return self._predict_autoregressive(X, horizon)
        else:
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X[-self.seq_len:]).unsqueeze(0)
                prediction = self.model(X_tensor).numpy().flatten()
            return prediction[:horizon]

    def _predict_autoregressive(self, X: np.ndarray, horizon: int) -> np.ndarray:
        predictions = []
        current_input = torch.FloatTensor(X[-self.seq_len:]).unsqueeze(0)
        
        with torch.no_grad():
            for _ in range(horizon):
                pred = self.model(current_input).numpy().flatten()[0]
                predictions.append(pred)
                current_input = torch.cat([current_input[:, 1:, :], torch.cat([current_input[:, -1, 1:], torch.tensor([[pred]])], dim=1)])
        return np.array(predictions)

class LSTMModelAdapter(BaseAdapter):
    def __init__(
        self,
        input_size: int,  # УДАЛИТЬ ЭТОТ ПАРАМЕТР
        hidden_size: int,
        output_steps: int,
        num_layers: int = 1,
        autoregressive: bool = False,
        **kwargs
    ):
        self.model_params = {
            # УДАЛИТЬ 'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'output_steps': output_steps,
            'autoregressive': autoregressive,
            **kwargs
        }
        self.model: Optional[LSTMModel] = None

    def fit(self, df: pd.DataFrame, regressors: List[str]) -> "LSTMModelAdapter":
        target = df['target'].values
        features = df.drop(columns=["timestamp", "target"]).values
        
        input_data = features if features is not None else target.reshape(-1, 1)
        input_size = input_data.shape[1]  # Рассчитываем динамически
        
        # Удаляем input_size из model_params перед распаковкой
        model_params = {k: v for k, v in self.model_params.items() if k != 'input_size'}
        
        self.model = LSTMModel(
            input_size=input_size,  # Передаем рассчитанное значение
            **model_params
        )
        self.model.fit(input_data, target)
        return self

    def predict(self, df: pd.DataFrame, prediction_interval: bool = False, quantiles: List[float] = None) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not fitted!")
            
        input_data = df.drop(columns=["timestamp", "target"]).values
        
        if input_data.ndim == 1:
            input_data = input_data.reshape(-1, 1)

        horizon = len(df)

        return self.model.predict(input_data, horizon=horizon)

    def get_model(self) -> Any:
        return self.model

class LSTMPerSegmentModel(
    PerSegmentModelMixin,
    NonPredictionIntervalContextIgnorantModelMixin,
    NonPredictionIntervalContextIgnorantAbstractModel
):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_steps: int,
        num_layers: int = 1,
        seq_len: int = 10,
        num_epochs: int = 100,
        learning_rate: float = 0.001,
        autoregressive: bool = False, 
        **kwargs
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_steps = output_steps
        self.seq_len = seq_len
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.autoregressive = autoregressive
        self.kwargs = kwargs
        super().__init__(
            base_model=LSTMModelAdapter(
                input_size=input_size,
                hidden_size=hidden_size,
                output_steps=output_steps,
                num_layers=num_layers,
                autoregressive=autoregressive,
                **kwargs,
            )
        )
