# Step to step

## Installations

```python
pip3 install -r requirements.txt
```

## Download checkpoints

```python
python3 download_checkpoints.py
```

## Convert TF model to PyTorch

```python
python3 converter.py
```

## Inference

```python
# With TF
python3 tf_inference.py

# With Torch
python3 torch_inference.py
```
