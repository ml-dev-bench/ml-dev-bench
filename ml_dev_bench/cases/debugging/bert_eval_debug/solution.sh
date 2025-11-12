#!/bin/bash
# Solution for bert_eval_debug task
# Fix missing token_type_ids in evaluation script

# Ensure Poetry is in PATH
export PATH="/opt/poetry/bin:$PATH"

# Add token_type_ids extraction after attention_mask line
sed -i.bak '/attention_mask = batch\['\''attention_mask'\''\]\.to(device)/a\
            token_type_ids = batch['\''token_type_ids'\''].to(device)' evaluate.py

# Update the model forward call to include token_type_ids
sed -i.bak 's/outputs = model(input_ids=input_ids, attention_mask=attention_mask)/outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)/' evaluate.py

echo "✓ Fix applied: Added token_type_ids to model forward pass"

