#!/bin/bash
# Solution for vit_debugging task
# Fix multiple bugs in Vision Transformer implementation

echo "Fixing Vision Transformer bugs..."

# Bug 1: Fix patch_embed projection dimension (embed_dim // 2 should be embed_dim)
sed -i.bak 's/in_chans, embed_dim \/\/ 2, kernel_size=patch_size/in_chans, embed_dim, kernel_size=patch_size/' dino_v2/layers/patch_embed.py

# Bug 2: Fix patch_size not being converted to tuple in PatchEmbed
sed -i.bak 's/self\.patch_size = patch_size$/self.patch_size = make_2tuple(patch_size)/' dino_v2/layers/patch_embed.py

# Bug 3: Fix register_tokens expand - missing dimension specifications
sed -i.bak 's/self\.register_tokens\.expand(x\.shape\[0\])/self.register_tokens.expand(x.shape[0], -1, -1)/' dino_v2/models/vision_transformer.py

# Bug 4: Add dynamic_patch_size parameter to prepare_tokens_with_masks
sed -i.bak 's/def prepare_tokens_with_masks(self, x, masks=None):/def prepare_tokens_with_masks(self, x, masks=None, dynamic_patch_size=None):/' dino_v2/models/vision_transformer.py

# Bug 5: Update patch_embed call to handle dynamic_patch_size
# First, we need to modify the patch embedding call to create a new PatchEmbed if dynamic_patch_size is provided
# Replace the line: x = self.patch_embed(x)
sed -i.bak '/^        x = self\.patch_embed(x)$/c\
        if dynamic_patch_size is not None:\
            from ..layers import PatchEmbed\
            patch_embed_dyn = PatchEmbed(\
                img_size=w,\
                patch_size=dynamic_patch_size,\
                in_chans=nc,\
                embed_dim=self.embed_dim,\
            ).to(x.device)\
            x = patch_embed_dyn(x)\
        else:\
            x = self.patch_embed(x)' dino_v2/models/vision_transformer.py

# Bug 6: Fix masking to happen AFTER positional embeddings are added
# The current code masks before adding pos embeddings, but tests expect mask_token + pos_embed
# We need to restructure the prepare_tokens_with_masks to:
# 1. Get patch embeddings
# 2. Add cls token
# 3. Add positional embeddings
# 4. THEN apply masking (so masked = mask_token + pos_embed)

# Create a Python script to fix the masking logic
cat > /tmp/fix_masking.py << 'PYTHON_SCRIPT'
import re

with open('dino_v2/models/vision_transformer.py', 'r') as f:
    content = f.read()

# Find and replace the prepare_tokens_with_masks method
old_method = r'''def prepare_tokens_with_masks\(self, x, masks=None, dynamic_patch_size=None\):
        B, nc, w, h = x\.shape
        if dynamic_patch_size is not None:
            from \.\.layers import PatchEmbed
            patch_embed_dyn = PatchEmbed\(
                img_size=w,
                patch_size=dynamic_patch_size,
                in_chans=nc,
                embed_dim=self\.embed_dim,
            \)\.to\(x\.device\)
            x = patch_embed_dyn\(x\)
        else:
            x = self\.patch_embed\(x\)
        if masks is not None:
            x = torch\.where\(
                masks\.unsqueeze\(-1\), self\.mask_token\.to\(x\.dtype\)\.unsqueeze\(0\), x
            \)

        x = torch\.cat\(\(self\.cls_token\.expand\(x\.shape\[0\], -1, -1\), x\), dim=1\)
        x = x \+ self\.interpolate_pos_encoding\(x, w, h\)

        if self\.register_tokens is not None:
            x = torch\.cat\(
                \(
                    x\[:, :1\],
                    self\.register_tokens\.expand\(x\.shape\[0\], -1, -1\),
                    x\[:, 1:\],
                \),
                dim=1,
            \)

        return x'''

new_method = '''def prepare_tokens_with_masks(self, x, masks=None, dynamic_patch_size=None):
        B, nc, w, h = x.shape
        if dynamic_patch_size is not None:
            from ..layers import PatchEmbed
            patch_embed_dyn = PatchEmbed(
                img_size=w,
                patch_size=dynamic_patch_size,
                in_chans=nc,
                embed_dim=self.embed_dim,
            ).to(x.device)
            x = patch_embed_dyn(x)
        else:
            x = self.patch_embed(x)

        # Add cls token and positional embeddings BEFORE masking
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        pos_embed = self.interpolate_pos_encoding(x, w, h)
        
        # Apply masking AFTER positional embeddings (masked = mask_token + pos_embed)
        if masks is not None:
            # Extract patch tokens (skip cls token)
            patch_tokens = x[:, 1:, :]
            patch_pos_embed = pos_embed[:, 1:, :]
            
            # Apply mask: masked positions = mask_token + pos_embed, unmasked = patch + pos_embed
            mask_token_with_pos = self.mask_token.to(x.dtype).unsqueeze(0) + patch_pos_embed
            patch_tokens = torch.where(
                masks.unsqueeze(-1), mask_token_with_pos, patch_tokens + patch_pos_embed
            )
            
            # Reconstruct with cls token (cls gets its pos embed)
            x = torch.cat((x[:, :1] + pos_embed[:, :1], patch_tokens), dim=1)
        else:
            # No masking, just add pos embeddings normally
            x = x + pos_embed

        if self.register_tokens is not None:
            x = torch.cat(
                (
                    x[:, :1],
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x[:, 1:],
                ),
                dim=1,
            )

        return x'''

content = re.sub(old_method, new_method, content, flags=re.DOTALL)

with open('dino_v2/models/vision_transformer.py', 'w') as f:
    f.write(content)

print("Fixed masking logic")
PYTHON_SCRIPT

python /tmp/fix_masking.py

# Bug 7: Fix forward to pass dynamic_patch_size correctly
sed -i.bak 's/ret = self\.forward_features(\*args, dynamic_patch_size=None, \*\*kwargs)/ret = self.forward_features(*args, dynamic_patch_size=dynamic_patch_size, **kwargs)/' dino_v2/models/vision_transformer.py

# Bug 8: Fix forward_features to pass dynamic_patch_size to prepare_tokens_with_masks
sed -i.bak 's/x = self\.prepare_tokens_with_masks(x, masks)/x = self.prepare_tokens_with_masks(x, masks, dynamic_patch_size)/' dino_v2/models/vision_transformer.py

# Bug 9: Fix get_intermediate_layers methods to handle dynamic_patch_size
sed -i.bak 's/x = self\.prepare_tokens_with_masks(x)$/x = self.prepare_tokens_with_masks(x, dynamic_patch_size=dynamic_patch_size)/' dino_v2/models/vision_transformer.py

# Bug 10: Update patch_size for reshape when using dynamic_patch_size in get_intermediate_layers
# Create another Python script to fix the reshape logic
cat > /tmp/fix_reshape.py << 'PYTHON_SCRIPT'
with open('dino_v2/models/vision_transformer.py', 'r') as f:
    content = f.read()

# Find the get_intermediate_layers method and fix the reshape logic
old_reshape = r'''if reshape:
            B, _, w, h = x\.shape
            outputs = \[
                out\.reshape\(B, w // self\.patch_size, h // self\.patch_size, -1\)
                \.permute\(0, 3, 1, 2\)
                \.contiguous\(\)
                for out in outputs
            \]'''

new_reshape = '''if reshape:
            B, _, w, h = x.shape
            patch_size_to_use = dynamic_patch_size if dynamic_patch_size is not None else self.patch_size
            outputs = [
                out.reshape(B, w // patch_size_to_use, h // patch_size_to_use, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
                for out in outputs
            ]'''

import re
content = re.sub(old_reshape, new_reshape, content, flags=re.DOTALL)

with open('dino_v2/models/vision_transformer.py', 'w') as f:
    f.write(content)

print("Fixed reshape logic")
PYTHON_SCRIPT

python /tmp/fix_reshape.py

# Clean up temporary files
rm /tmp/fix_masking.py /tmp/fix_reshape.py

echo "✓ All fixes applied successfully"

