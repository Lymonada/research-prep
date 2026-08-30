
# LSTM vs GRU Stock Price Prediction

## 1. Experiment Purpose
동일한 삼성전자 주가 예측 task에서
LSTM과 GRU의 구조 차이가 parameter 수,
학습 시간, 예측 결과에 어떤 차이를 만드는지 확인한다.

## 2. Common Settings
- Data: Samsung Electronics
- Input: Open, High, Low, Volume
- Target: next-day Close
- Sequence length: 5
- Hidden size: 4
- Layers: 1
- Batch size: 20
- Epochs: 200
- Optimizer: Adam
- Learning rate: 0.001

## 3. Model Difference

LSTM:
- hidden state + cell state
- 3 gates
- parameter 수가 상대적으로 많음

GRU:
- hidden state만 사용
- 2 gates
- parameter 수가 상대적으로 적음

## 4. Results

| Model | Parameters | Train Time | Test Loss |
|---|---:|---:|---:|
| LSTM | ... | ... | ... |
| GRU | ... | ... | ... |

## 5. Prediction Comparison
Actual / LSTM / GRU 그래프

## 6. Observation
짧은 해석

## 7. Limitations
한 번의 실행, 작은 hidden size, 특정 종목/기간 등
