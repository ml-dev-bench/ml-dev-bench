import datetime
import json
import random


def generate_training_logs():
    # Initial setup logs
    logs = [
        '[2024-01-06 10:00:01] INFO: Starting training process',
        '[2024-01-06 10:00:02] INFO: Loading dataset from /path/to/dataset',
        '[2024-01-06 10:00:03] INFO: Model configuration:',
        '[2024-01-06 10:00:03] INFO: '
        + json.dumps(
            {
                'batch_size': 32,
                'learning_rate': 0.001,
                'optimizer': 'Adam',
                'epochs': 100,
                'model_type': 'ResNet50',
            },
            indent=4,
        ),
        '',
    ]

    # Generate logs for 9999 epochs
    base_time = datetime.datetime(2024, 1, 6, 10, 0, 0)
    loss = 3.0
    accuracy = 0.1

    for epoch in range(1, 10000):
        # Add some random variation
        loss_change = random.uniform(0.02, 0.04)
        acc_change = random.uniform(0.005, 0.015)

        # Improve metrics
        loss -= loss_change
        accuracy += acc_change

        # Add some noise
        val_loss = loss + random.uniform(-0.1, 0.2)
        val_accuracy = accuracy + random.uniform(-0.02, 0.02)

        # Ensure metrics stay in reasonable ranges
        loss = max(0.3, min(3.0, loss))
        accuracy = max(0.1, min(0.93, accuracy))
        val_loss = max(0.3, min(3.0, val_loss))
        val_accuracy = max(0.1, min(0.93, val_accuracy))

        # Add timestamp
        timestamp = base_time + datetime.timedelta(minutes=epoch)

        # Generate log line
        log_line = (
            f'[{timestamp.strftime("%Y-%m-%d %H:%M:%S")}] INFO: Epoch {epoch}/100 - '
        )
        log_line += f'loss: {loss:.3f} - accuracy: {accuracy:.3f} - '
        log_line += f'val_loss: {val_loss:.3f} - val_accuracy: {val_accuracy:.3f}'

        # Add some random additional logs
        if random.random() < 0.1:
            logs.append(
                f'[{timestamp.strftime("%Y-%m-%d %H:%M:%S")}] INFO: Saving checkpoint...'
            )
        if random.random() < 0.05:
            logs.append(
                f'[{timestamp.strftime("%Y-%m-%d %H:%M:%S")}] WARNING: Learning rate adjusted to {random.uniform(0.0001, 0.001):.6f}'
            )

        logs.append(log_line)

        # Add empty line occasionally for readability
        if random.random() < 0.2:
            logs.append('')

    # Add the final failure
    final_time = base_time + datetime.timedelta(minutes=100)
    logs.extend(
        [
            f'[{final_time.strftime("%Y-%m-%d %H:%M:%S")}] WARNING: Detected unusual memory spike during forward pass',
            f'[{final_time.strftime("%Y-%m-%d %H:%M:%S")}] ERROR: RuntimeError: CUDA out of memory. Tried to allocate 3.2 GB on device with 8GB total capacity',
            f'[{final_time.strftime("%Y-%m-%d %H:%M:%S")}] ERROR: Training failed during epoch 100',
            f'[{final_time.strftime("%Y-%m-%d %H:%M:%S")}] ERROR: Stack trace:',
            'Traceback (most recent call last):',
            '  File "train.py", line 245, in train_step',
            '    outputs = model(inputs)',
            '  File "torch/nn/modules/module.py", line 1190, in _call_impl',
            '    return forward_call(*input, **kwargs)',
            'RuntimeError: CUDA out of memory. Tried to allocate 3.2 GB on device with 8GB total capacity',
        ]
    )

    # Write to file
    with open('training/training_logs.txt', 'w') as f:
        f.write('\n'.join(logs))


if __name__ == '__main__':
    generate_training_logs()
