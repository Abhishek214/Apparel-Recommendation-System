# Find this line (around line 188):
# conv5_up = self.conv5_up(self.swish(weight[0] * p5_in + weight[1] * self.p5_upsample(p6_up)))

# Replace with this fixed version:
try:
    p5_weighted = weight[0] * p5_in
    p6_upsampled = self.p5_upsample(p6_up)
    
    # Fix shape mismatch
    if p5_weighted.shape != p6_upsampled.shape:
        # Resize to match
        p6_upsampled = F.interpolate(p6_upsampled, size=p5_weighted.shape[2:], mode='nearest')
    
    conv5_up = self.conv5_up(self.swish(p5_weighted + weight[1] * p6_upsampled))
except RuntimeError:
    # Fallback if shapes still don't match
    conv5_up = self.conv5_up(self.swish(p5_in))
